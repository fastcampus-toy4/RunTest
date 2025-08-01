import uuid
import traceback
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

# 새로운 스키마에 맞게 import 구문 수정
from models.schemas import ChatResponse, UserRequestInfo, FinalRecommendation, StartChatResponse, HistorySummary
from . import db_service, filter_service, crawl_service, nlu_service

# --- 핵심 로직: 새로운 채팅 시작 및 첫 메시지 처리 ---
async def start_new_chat_session(user_id: str, initial_message: Optional[str], db: AsyncSession) -> StartChatResponse:
    session_id = str(uuid.uuid4())

    state = {
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": [],
        "user_info": UserRequestInfo().model_dump()
    }

    if initial_message:
        print(f"[{session_id}] 새 채팅 시작과 함께 첫 메시지 처리: {initial_message}")
        chat_response = await process_chat_message(state, initial_message, db)
        return StartChatResponse(
            state=chat_response.state,
            bot_response=chat_response.response
        )
    else:
        initial_bot_response = "안녕하세요! 무엇을 도와드릴까요?"
        state["conversation_history"].append(f"Bot: {initial_bot_response}")
        await db_service.log_interaction(db, session_id, "chat_start", "bot_response", initial_bot_response, user_id)
        return StartChatResponse(state=state, bot_response=initial_bot_response)

# --- 핵심 로직: 메시지 처리 및 대화 진행 ---
async def process_chat_message(state: dict, user_input: str, db: AsyncSession) -> ChatResponse:
    user_info = UserRequestInfo(**state.get("user_info", {}))
    session_id = state.get("session_id")
    user_id = state.get("user_id")

    # 1. 대화 기록 및 정보 업데이트
    state["conversation_history"].append(f"User: {user_input}")
    await db_service.log_interaction(db, session_id, "user_interaction", "user_input", user_input, user_id)

    await _update_user_info(user_info, user_input, state["conversation_history"])
    state["user_info"] = user_info.model_dump()

    # 2. 필수 정보 수집 여부 확인
    if not user_info.is_ready():
        next_question = _get_next_question(user_info)
        state["conversation_history"].append(f"Bot: {next_question}")
        await db_service.log_interaction(db, session_id, "user_interaction", "bot_response", next_question, user_id)
        return ChatResponse(response=next_question, session_id=session_id, state=state)

    # 3. 모든 정보 수집 완료 -> 추천 프로세스 시작
    try:
        print(f"[{session_id}] 모든 정보 수집 완료. 추천 프로세스 시작.")
        print(f"    - 최종 수집 정보: {user_info.model_dump_json(indent=2)}")

        # [수정] 1. 지역명을 '구' 단위로 변환
        print(f"    - 원본 위치 입력: '{user_info.location}'")
        location_gu = await nlu_service.extract_gu_from_location(user_info.location)
        if location_gu == "알 수 없음":
            print(f"[DEBUG] 종료: 위치 '{user_info.location}'에서 '구'를 추출하지 못했습니다.")
            return ChatResponse(response=f"'{user_info.location}' 지역을 정확히 이해하지 못했어요. '강남구', '송파구' 처럼 다시 말씀해주시겠어요?", session_id=session_id, is_final=True, state=state)
        print(f"    - 변환된 위치 (구): '{location_gu}'")

        restaurants_step1 = await db_service.get_restaurants_from_db(db, location_gu, user_info.amenities)
        if not restaurants_step1:
            print("[DEBUG] 종료: 1단계에서 후보 음식점을 찾지 못했습니다.")
            return ChatResponse(response="해당 조건에 맞는 음식점이 없습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 1단계 통과 (위치/편의시설). 후보 음식점: {len(restaurants_step1)}개")

        menus_by_restaurant = await db_service.get_normalized_menus_for_restaurants(db, restaurants_step1)
        # ▼▼▼▼▼ [수정] 메뉴 정보가 있는 음식점만 후보로 다시 정의 ▼▼▼▼▼
        restaurants_step2 = [r for r in restaurants_step1 if f"{r['name']} {r.get('branch_name', '')}".strip() in menus_by_restaurant]
        if not restaurants_step2:
            print("[DEBUG] 종료: 2단계에서 메뉴 정보가 있는 음식점을 찾지 못했습니다.")
            return ChatResponse(response="조건에 맞는 음식점은 찾았지만, 메뉴 정보가 없어 추천할 수 없네요.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 2단계 통과 (메뉴 보유). 후보 음식점: {len(restaurants_step2)}개")
        
        all_menus = {menu for menus in menus_by_restaurant.values() for menu in menus}

        # [수정] 3. 질병명 정규화
        print(f"    - 원본 질병 입력: '{user_info.disease}'")
        normalized_disease = await nlu_service.normalize_disease_name(user_info.disease)
        print(f"    - 정규화된 질병명: '{normalized_disease}'")

        suitable_menus = await filter_service.filter_menus_by_health(all_menus, normalized_disease, user_info.dietary_restriction, db=db)
        if not suitable_menus:
            print("[DEBUG] 종료: 3단계(건강 필터링)에서 적합한 메뉴를 찾지 못했습니다.")
            return ChatResponse(response="고객님의 건강 조건에 맞는 메뉴를 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 3단계 통과 (건강 메뉴). 적합 메뉴: {len(suitable_menus)}개")

        # ▼▼▼▼▼ [수정] restaurants_step2를 기준으로 필터링 ▼▼▼▼▼
        restaurants_step4 = [r for r in restaurants_step2 if any(menu in suitable_menus for menu in menus_by_restaurant.get(f"{r['name']} {r.get('branch_name', '')}".strip(), set()))]
        if not restaurants_step4:
            print("[DEBUG] 종료: 4단계(메뉴 판매점 필터링)에서 적합한 음식점을 찾지 못했습니다.")
            return ChatResponse(response="건강에 좋은 메뉴를 파는 음식점을 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 4단계 통과 (메뉴 판매점). 후보 음식점: {len(restaurants_step4)}개")

        restaurants_step5 = await filter_service.filter_restaurants_by_review(restaurants_step4, user_info.other_requests)
        print(f"[DEBUG] 5단계 통과 (리뷰 필터링). 후보 음식점: {len(restaurants_step5)}개")

        final_recommendations = await crawl_service.get_final_recommendations_with_crawling(restaurants_step5[:10], user_info.time)
        print(f"[DEBUG] 6단계 통과 (최종 필터링). 최종 추천: {len(final_recommendations)}개")

        if not final_recommendations:
            return ChatResponse(response="후보 맛집은 찾았지만, 아쉽게도 요청하신 시간에 영업 중인 곳이 없네요.", session_id=session_id, is_final=True, state=state)

        bot_response = "🎉 고객님을 위한 최종 맛집 추천 목록입니다!"
        await db_service.log_interaction(db, session_id, "final_result", "success", [rec.model_dump() for rec in final_recommendations], user_id)
        return ChatResponse(response=bot_response, session_id=session_id, is_final=True, recommendations=final_recommendations, state=state)

    except Exception as e:
        print(f"[Orchestrator Error] {e}")
        traceback.print_exc()
        await db_service.log_interaction(db, session_id, "pipeline_error", "error", str(e), user_id)
        return ChatResponse(response="추천 과정 중 오류가 발생했습니다.", session_id=session_id, is_final=True, state=state)

# --- 보조 함수들 ---
async def _update_user_info(user_info: UserRequestInfo, user_input: str, history: List[str]):
    """NLU 서비스를 호출하여 사용자 정보를 업데이트합니다."""
    
    extracted_info: UserRequestInfo
    if len(history) <= 2:
        extracted_info = await nlu_service.extract_initial_request(user_input)
    else:
        last_bot_question = history[-2] if len(history) > 1 else ""
        extracted_info = await nlu_service.extract_info_from_message(history, last_bot_question)
    
    # ▼▼▼▼▼ [수정] 업데이트 로직 강화 ▼▼▼▼▼
    # 추출된 값이 의미가 있을 때만 (None이 아니고, 빈 문자열/리스트가 아닐 때만) 업데이트합니다.
    for field, value in extracted_info.model_dump(exclude_unset=True).items():
        if value: # None, "", [] 등 비어있는 값은 모두 False로 처리되어 업데이트되지 않음
            setattr(user_info, field, value)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def _get_next_question(user_info: UserRequestInfo) -> str:
    """다음에 물어봐야 할 질문을 반환합니다."""
    # ▼▼▼▼▼ [수정] 모든 검사 조건을 'is None'으로 통일하여 일관성 확보 ▼▼▼▼▼
    if user_info.location is None: return "어느 지역의 음식점을 찾으시나요?"
    if user_info.time is None: return "언제 방문하실 예정인가요?"
    if user_info.dietary_restrictions is None: return "특별히 피해야 할 음식이 있나요? (없으면 '없음')"
    if user_info.disease is None: return "앓고 있는 질환이 있으신가요? (없으면 '없음')"
    if user_info.other_requests is None: return "선호하는 분위기나 편의시설이 있나요? (없으면 '없음')"
    return "" # 모든 정보 수집 완료
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

async def get_user_history(user_id: str, db: AsyncSession) -> List[HistorySummary]:
    # TODO: DB에서 user_id에 해당하는 세션 요약 정보 조회
    return []
