# services/chat_orchestrator.py
import uuid
import json
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI

from models.schemas import ChatResponse, UserRequestInfo, FinalRecommendation, StartChatResponse, HistorySummary
from fastapi import HTTPException
from typing import List
from datetime import datetime
from . import db_service, filter_service, crawl_service
from core.config import settings

# 세션별 사용자 요청 정보를 메모리에 저장 (프로덕션에서는 Redis 등 사용 권장)
SESSION_STATE = {}

async def process_chat_message(state: dict, user_input: str, db: AsyncSession) -> ChatResponse:
    session_id = state.get("session_id")
    if not session_id or session_id not in SESSION_STATE:
        # 기존 세션이 없거나 유효하지 않으면 새로운 세션 시작
        session_id = str(uuid.uuid4())
        SESSION_STATE[session_id] = UserRequestInfo()
        # 초기 상태 복원 (선택 사항: 필요한 경우 state에서 다른 정보도 복원)
        if "user_info" in state:
            SESSION_STATE[session_id] = UserRequestInfo(**state["user_info"])

    user_info = SESSION_STATE[session_id]

    # 1. 사용자 정보 업데이트
    await _update_user_info(user_info, user_input)

    # 대화 기록 업데이트
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(f"User: {user_input}")
    # 2. 필수 정보가 모두 수집되었는지 확인
    if not user_info.is_ready():
        next_question = _get_next_question(user_info)
        return ChatResponse(response=next_question, session_id=session_id)

    # 3. 모든 정보 수집 완료 -> 추천 프로세스 시작
    try:
        print(f"[{session_id}] 모든 정보 수집 완료. 추천 프로세스 시작.")
        print(f"[{session_id}] 사용자 요청: {user_info.model_dump_json(indent=2)}")

        # 1단계: DB에서 위치 기반 후보군 조회 (I/O Bound - 비동기)
        restaurants_step1 = await db_service.get_restaurants_from_db(db, user_info.location, user_info.amenities)
        if not restaurants_step1:
            return ChatResponse(response="해당 조건에 맞는 음식점이 없습니다. 다른 지역으로 다시 시도해보세요.", session_id=session_id, is_final=True)

        # 2단계: 후보 음식점들의 정규화된 메뉴 조회 (I/O Bound - 비동기)
        menus_by_restaurant = await db_service.get_normalized_menus_for_restaurants(db, restaurants_step1)
        all_menus = {menu for menus in menus_by_restaurant.values() for menu in menus}

        # 3단계: 하이브리드 건강/식단 제약 필터링 (사전 판단 + LLM 정밀 검증)
        suitable_menus = await filter_service.filter_menus_with_hybrid_approach(
            db=db,
            session_id=session_id,
            standard_dishes=all_menus,
            disease=user_info.disease,
            dietary_restrictions=user_info.dietary_restrictions
        )
        
        if not suitable_menus:
             return ChatResponse(response="고객님의 건강 조건에 맞는 메뉴를 찾지 못했습니다.", session_id=session_id, is_final=True)
        
        # 4단계: 적합 메뉴를 판매하는 음식점만 남기기 (CPU Bound - 동기)
        restaurants_step4 = [
            r for r in restaurants_step1 
            if any(menu in suitable_menus for menu in menus_by_restaurant.get(f"{r['name']} {r.get('branch_name', '')}".strip(), set()))
        ]
        if not restaurants_step4:
            return ChatResponse(response="건강에 좋은 메뉴를 파는 음식점을 찾지 못했습니다.", session_id=session_id, is_final=True)

        # 5단계: 기타 요청사항(분위기 등)으로 리뷰 기반 필터링 (I/O Bound - 비동기)
        restaurants_step5 = await filter_service.filter_restaurants_by_review(restaurants_step4, user_info.other_requests)

        # 6단계: 실시간 정보 확인 (크롤링 - 비동기 Task Queue 시뮬레이션)
        crawled_infos = await crawl_service.get_restaurants_realtime_info(restaurants_step5[:10])
        
        # 7단계: 영업 여부 최종 판단
        visitable_restaurants = []
        for info in crawled_infos:
            if await crawl_service.check_visitable(info, user_info.time):
                 visitable_restaurants.append(FinalRecommendation(**info))

        if not visitable_restaurants:
            return ChatResponse(response="후보 맛집은 찾았지만, 아쉽게도 요청하신 시간에 영업 중인 곳이 없네요.", session_id=session_id, is_final=True)

        # 최종 추천 결과 반환
        bot_response = "🎉 고객님을 위한 최종 맛집 추천 목록입니다!"
        state["conversation_history"].append(f"Bot: {bot_response}")
        state["user_info"] = user_info.model_dump() # 최신 user_info 상태 저장

        return ChatResponse(
            response=bot_response,
            session_id=session_id,
            is_final=True,
            recommendations=visitable_restaurants[:5],
            state=state # 업데이트된 state 반환
        )
    except Exception as e:
        print(f"[Orchestrator Error] {e}")
        return ChatResponse(response="추천 과정 중 오류가 발생했습니다. 다시 시도해주세요.", session_id=session_id, is_final=True, state=state)

async def _update_user_info(user_info: UserRequestInfo, message: str):
    """LLM을 사용해 사용자 메시지에서 정보를 추출하여 상태를 업데이트"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(UserRequestInfo)
    
    # 현재까지 수집된 정보와 새로운 메시지를 함께 전달하여 컨텍스트 유지
    prompt = f"""
    아래는 현재까지 수집된 사용자 정보와 새로운 사용자 메시지입니다.
    메시지를 분석하여 UserRequestInfo JSON 객체를 업데이트해주세요.
    기존에 값이 있던 필드는 유지하고, 새로운 정보만 추가하거나 수정하세요.
    특히, 사용자가 '강남', '홍대' 등을 말하면 '강남구', '마포구'와 같이 '구' 단위로 변환해주세요.
    
    [기존 정보]
    {user_info.model_dump_json()}
    
    [새 메시지]
    {message}
    """
    
    updated_info = await structured_llm.ainvoke(prompt)
    
    # 기존 정보 위에 새로 추출된 정보 덮어쓰기 (None이 아닌 값만)
    for field, value in updated_info.model_dump().items():
        if value is not None and (isinstance(value, list) and value or not isinstance(value, list)):
            setattr(user_info, field, value)


def _get_next_question(user_info: UserRequestInfo) -> str:
    """다음에 물어봐야 할 질문을 반환"""
    if not user_info.location: return "어느 지역(구 단위)의 음식점을 찾으시나요?"
    if not user_info.time: return "언제 방문하실 예정인가요? (예: 오늘 저녁 7시)"
    if not user_info.dietary_restrictions: return "특별히 피해야 할 음식이나 식단(채식 등)이 있으신가요? (없으면 '없음')"
    if not user_info.disease: return "혹시 앓고 계신 질환이 있으신가요? (없으면 '없음')"
    if not user_info.other_requests: return "선호하는 분위기나 '주차', '놀이방' 같은 편의시설 요청이 있으신가요? (없으면 '없음')"
    return "알 수 없는 상태입니다. 다시 시작해주세요."


async def start_new_chat_session(user_id: str, db: AsyncSession) -> StartChatResponse:
    session_id = str(uuid.uuid4())
    SESSION_STATE[session_id] = UserRequestInfo()
    initial_bot_response = "안녕하세요! 무엇을 도와드릴까요?"
    
    # 세션 정보와 초기 봇 응답을 포함하는 state 생성
    state = {
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": [f"Bot: {initial_bot_response}"],
        "user_info": SESSION_STATE[session_id].model_dump()
    }
    
    return StartChatResponse(state=state, bot_response=initial_bot_response)


async def get_chat_history(session_id: str, db: AsyncSession) -> ChatResponse:
    if session_id not in SESSION_STATE:
        raise HTTPException(status_code=404, detail="Session not found")
    
    user_info = SESSION_STATE[session_id]
    # 여기서는 간단히 conversation_history만 반환하지만, 실제로는 DB에서 가져와야 합니다.
    # 현재는 SESSION_STATE에 저장된 정보만 활용합니다.
    
    # 예시: 가상의 대화 기록 생성
    # 실제 구현에서는 DB에서 해당 session_id의 모든 대화 기록을 조회해야 합니다.
    # 현재는 메모리상의 user_info와 초기 봇 응답을 기반으로 재구성합니다.
    
    # conversation_history는 process_chat_message에서 업데이트되므로, 그대로 사용
    conversation_history = SESSION_STATE[session_id].get("conversation_history", [])
    
    # 마지막 봇 응답을 찾아서 ChatResponse의 response 필드에 넣습니다.
    last_bot_response = ""
    for msg in reversed(conversation_history):
        if msg.startswith("Bot: "):
            last_bot_response = msg.replace("Bot: ", "")
            break

    return ChatResponse(
        response=last_bot_response, # 마지막 봇 응답
        session_id=session_id,
        is_final=user_info.is_ready(), # 정보 수집 완료 여부로 is_final 판단
        # recommendations는 필요에 따라 추가
    )

async def get_user_history(user_id: str, db: AsyncSession) -> List[HistorySummary]:
    # 실제 구현에서는 DB에서 user_id에 해당하는 모든 세션 기록을 조회해야 합니다.
    # 현재는 메모리상의 SESSION_STATE에서 user_id가 일치하는 세션들을 필터링합니다.
    
    history_list = []
    for session_id, user_info in SESSION_STATE.items():
        # user_id가 일치하고, 대화 기록이 있는 경우에만 추가
        if user_info.get("user_id") == user_id and "conversation_history" in user_info and user_info["conversation_history"]:
            # 첫 사용자 메시지를 content로 사용
            first_user_message = ""
            for msg in user_info["conversation_history"]:
                if msg.startswith("User: "):
                    first_user_message = msg.replace("User: ", "")
                    break
            
            if first_user_message:
                history_list.append(HistorySummary(
                    session_id=session_id,
                    content=first_user_message,
                    created_at=datetime.now() # 실제로는 세션 생성 시간을 사용해야 합니다.
                ))
    return history_list