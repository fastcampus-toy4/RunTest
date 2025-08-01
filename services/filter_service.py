# services/filter_service.py
import asyncio
import json
from typing import Set, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_community import GoogleSearchAPIWrapper
from collections import defaultdict

from . import data_loader  # 사전 로드된 DB 및 데이터 사용
from core.config import settings

# 질병 키워드 맵
DISEASE_KEYWORD_MAP = {
    "위식도 역류질환": ["위식도 역류질환", "GERD", "식도염", "식도역류증"], 
    "위염 및 소화성 궤양": ["위염", "소화성 궤양", "위궤양"],
    "염증성 장질환": ["염증성 장질환", "IBD", "만성궤양성 대장염", "크론병"], 
    "과민성 대장 증후군": ["과민성 대장 증후군", "IBS"],
    "변비 및 게실 질환": ["변비", "게실 질환", "게실염"], 
    "간질환": ["간질환", "간염", "간경변", "B형 간염", "원발성 간암", "간암"],
    "담낭질환": ["담낭질환", "담석증", "담낭염"], 
    "췌장염": ["췌장염", "Pancreatitis"],
    "고혈압 및 심혈관 질환": ["고혈압", "혈압", "심혈관", "심장", "동맥경화", "심부전", "심근경색증", "협심증", "관상동맥질환", "심장판막질환"],
    "고지혈증": ["고지혈증", "Hyperlipidemia", "콜레스테롤", "피가 탁함"], 
    "뇌졸중": ["뇌졸중", "Stroke", "뇌경색", "뇌출혈", "중풍"],
    "당뇨병": ["당뇨병", "당뇨", "Diabetes", "혈당"], 
    "통풍": ["통풍", "Gout"], 
    "갑상선 질환": ["갑상선", "갑상선 기능 항진증", "갑상선 기능 저하증", "갑상선 결절"],
    "만성 신장질환": ["신장질환", "CKD", "신장", "콩팥", "신부전", "신증후군", "사구체콩팥염", "신장병"],
    "요로계 질환": ["요로계", "신결석", "요로결석", "요로감염"], 
    "셀리악병": ["셀리악병", "Celiac Disease", "실리악 스푸루", "글루텐"],
    "유당 불내증": ["유당 불내증", "Lactose Intolerance"], 
    "삼킴곤란": ["삼킴곤란", "연하곤란", "Dysphagia"],
    "빈혈": ["빈혈", "Anemia", "겸상 적혈구 빈혈증"], 
    "암": ["암", "Cancer", "폐암", "백혈병", "유방암", "대장암", "림프종", "악성 종양"],
    "골격계 질환": ["골격계", "뼈", "골다공증", "골연화증", "구루병"], 
    "알레르기": ["알레르기", "두드러기", "천식"],
    "급성 감염성 질환": ["감염", "감기", "폐렴", "코로나", "장염", "기관지염", "수두", "설사"],
}

# ========================================
# 메인 인터페이스 함수 (chat_orchestrator에서 호출)
# ========================================

async def filter_menus_by_health(
    standard_dishes: Set[str], 
    disease: str, 
    dietary_restrictions: str,
    db: AsyncSession  # <--- MySQL 조회를 위한 DB 세션 추가
) -> Set[str]:
    """
    건강/식단 제약 기반 메뉴 필터링 - 요구사항에 맞춘 새로운 2단계 하이브리드 방식
    1단계: 사전 계산된 데이터로 빠른 필터링 (후보군 선별)
    2단계: 하이브리드 심층 필터링 (정량 분석 + 정성 분석)
    """
    print(f"\n--- 건강 기반 메뉴 필터링 시작 ---")
    print(f"대상 메뉴: {len(standard_dishes)}개, 질병: {disease}, 식단 제약: {dietary_restrictions}")
    
    if not disease or disease.strip().lower() in ['없음', '없어요', '해당 없음']:
        print("-> 질병 정보 없음. 모든 메뉴를 적합으로 간주합니다.")
        return standard_dishes

    try:
        # --- 1단계: 사전 판단 데이터 기반 필터링 (후보군 선별) ---
        pre_approved_dishes = await _filter_with_precomputed_data(standard_dishes, disease)
        if not pre_approved_dishes:
            print("-> 1단계 필터링 결과, 적합한 메뉴 후보가 없습니다.")
            return set()
        print(f"-> 1단계 통과. {len(pre_approved_dishes)}개 메뉴가 2단계 심층 필터링 대상으로 선정.")

        # --- 2단계: 하이브리드 심층 필터링 ---
        
        # 2-1. 동적 수치 기반 필터링 (정량 분석)
        nutritionally_suitable_dishes = await _filter_by_dynamic_nutrition(pre_approved_dishes, disease, db)
        if not nutritionally_suitable_dishes:
            print("-> 2-1단계(영양성분 기준) 필터링 결과, 적합한 메뉴가 없습니다.")
            return set()
        print(f"-> 2-1단계(영양 기준) 통과. {len(nutritionally_suitable_dishes)}개 메뉴가 최종 검증 대상으로 선정.")

        # 2-2. RAG 자가 교정 필터링 (정성 분석)
        final_suitable_dishes = await _filter_by_rag_self_correction(
            standard_dishes=nutritionally_suitable_dishes,
            disease=disease,
            dietary_restrictions=dietary_restrictions
        )
        
        print(f"-> 최종 건강 필터링 완료. {len(final_suitable_dishes)}개 메뉴 통과.")
        return final_suitable_dishes
        
    except Exception as e:
        print(f"[전체 건강 필터링 오류] {e}")
        # 오류 발생 시 안전하게 1단계 통과 메뉴라도 반환
        return pre_approved_dishes if 'pre_approved_dishes' in locals() else standard_dishes

# ========================================
# 1단계: 사전 계산된 데이터 기반 빠른 필터링 (기존 로직 활용)
# ========================================
async def _filter_with_precomputed_data(standard_dishes: Set[str], disease: str) -> Set[str]:
    """사전 계산된 클러스터 및 건강 판단 데이터를 사용한 빠른 필터링"""
    print("\n--- 1단계: 사전 계산된 데이터로 빠른 필터링 시작 ---")
    print("-> 사전 계산된 데이터로 빠르게 조회 중...")
    
    def sync_filter():
        dishes_to_add = set()
        for dish in standard_dishes:
            try:
                # 1. 음식명 -> 클러스터 ID 매핑
                cluster_id = data_loader.FOOD_TO_CLUSTER_MAP.get(dish)
                if cluster_id is None:
                    # 클러스터 매핑이 없는 경우, 2단계에서 판단하도록 통과
                    dishes_to_add.add(dish)
                    continue
                
                # 2. 클러스터 ID -> 대표 음식 매핑
                representative_food = data_loader.CLUSTER_TO_FOOD_MAP.get(cluster_id)
                if representative_food is None:
                    dishes_to_add.add(dish)
                    continue
                
                # 3. 사전 계산된 건강 판단 결과 조회
                results = data_loader.HEALTH_JUDGMENT_DB.get(
                    where={
                        "$and": [
                            {"disease": {"$eq": disease}}, 
                            {"food_name": {"$eq": representative_food}}
                        ]
                    },
                    limit=1
                )
                
                # 4. 적합 판정을 받은 음식만 통과
                if results and results['metadatas'] and results['metadatas'][0].get("is_suitable"):
                    dishes_to_add.add(dish)
                    
            except Exception as e:
                print(f"[1단계 필터링 오류] '{dish}' 조회 중 오류 발생: {e}")
                # 오류 발생 시 안전하게 통과시켜서 2단계에서 판단
                dishes_to_add.add(dish)
        
        return dishes_to_add
    
    # ChromaDB 조회는 동기 함수이므로 to_thread로 감싸서 비동기 컨텍스트에서 실행
    pre_approved_dishes = await asyncio.to_thread(sync_filter)
    
    print(f"-> 사전 판단 데이터 조회 완료: {len(pre_approved_dishes)}개 메뉴 통과")
    return pre_approved_dishes

# ========================================
# 2-1단계: 동적 수치 기반 필터링 (신규 구현)
# ========================================
async def _filter_by_dynamic_nutrition(
    candidate_dishes: Set[str], 
    disease: str, 
    db: AsyncSession
) -> Set[str]:
    """질병에 대한 영양 가이드라인을 동적으로 찾아 SQL 쿼리로 필터링합니다."""
    print("\n--- 2-1단계: 동적 수치 기반 필터링 시작 ---")
    
    # 1. 영양 기준 검색 (ChromaDB)
    guideline_text = await _get_nutritional_guidelines_from_db(disease)
    if not guideline_text:
        print("-> 영양 가이드라인을 찾지 못함. 이 단계를 건너뜁니다.")
        return candidate_dishes

    # 2. 필터링 조건 추출 (LLM)
    sql_conditions = await _extract_sql_conditions_from_guidelines(guideline_text)
    if not sql_conditions:
        print("-> 가이드라인에서 SQL 조건을 추출하지 못함. 이 단계를 건너뜁니다.")
        return candidate_dishes

    # 3. MySQL 조회
    sql_filtered_dishes = await _filter_menus_by_nutrition_in_sql(sql_conditions, db)
    if not sql_filtered_dishes:
        print("-> 영양 기준을 만족하는 메뉴를 SQL DB에서 찾지 못함.")
        return set()

    # 4. 후보군과 교집합 계산
    final_candidates = candidate_dishes.intersection(sql_filtered_dishes)
    print(f"-> 1단계 후보({len(candidate_dishes)})와 SQL 필터링 결과({len(sql_filtered_dishes)})의 교집합: {len(final_candidates)}개")
    return final_candidates

async def _get_nutritional_guidelines_from_db(disease: str) -> str:
    """ChromaDB에서 질병 관련 영양 가이드라인 텍스트를 검색합니다."""
    print(f"-> ChromaDB에서 '{disease}' 영양 가이드라인 검색...")
    try:
        def search_db():
            # 'chroma_db_food'를 가정. 실제로는 data_loader에 정의된 VectorDB를 사용해야 합니다.
            # 여기서는 HEALTH_JUDGMENT_DB를 예시로 사용합니다.
            docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(
                f"{disease} 환자의 식단 영양성분 기준", k=1
            )
            return docs[0].page_content if docs else ""
        
        guideline = await asyncio.to_thread(search_db)
        if guideline:
            print("-> 가이드라인 발견.")
        return guideline
    except Exception as e:
        print(f"[DB 검색 오류] 영양 가이드라인 검색 실패: {e}")
        return ""

async def _extract_sql_conditions_from_guidelines(guideline_text: str) -> Dict[str, Any]:
    """LLM을 이용해 텍스트에서 SQL 쿼리용 JSON 조건을 추출합니다."""
    print("-> LLM으로 가이드라인 텍스트를 SQL 조건으로 변환 중...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = f"""
    당신은 영양학 가이드라인을 분석하여 MySQL의 `food_nutrition` 테이블을 쿼리할 수 있는 JSON 조건으로 변환하는 전문가입니다.

    [테이블 컬럼 정보]
    - `energy_kcal`: 에너지 (kcal)
    - `carbs_g`: 탄수화물 (g)
    - `protein_g`: 단백질 (g)
    - `fat_g`: 지방 (g)
    - `sodium_mg`: 나트륨 (mg)

    [변환 규칙]
    - "OO 이내/이하" -> "max_COLUMN_NAME": value
    - "OO 이상" -> "min_COLUMN_NAME": value
    - JSON 형식으로만 응답해야 합니다. 예: {{"max_carbs_g": 60, "max_sodium_mg": 2000}}

    [영양 가이드라인 텍스트]
    {guideline_text}
    """
    try:
        response = await llm.ainvoke(prompt)
        conditions = json.loads(response.content)
        print(f"-> 변환된 SQL 조건: {conditions}")
        return conditions
    except Exception as e:
        print(f"[LLM 변환 오류] SQL 조건 추출 실패: {e}")
        return {}

async def _filter_menus_by_nutrition_in_sql(conditions: Dict[str, Any], db: AsyncSession) -> Set[str]:
    """추출된 조건으로 MySQL의 food_nutrition 테이블을 조회하여 적합한 음식 목록을 반환합니다."""
    print(f"-> MySQL `food_nutrition` 테이블 조회 시작 (조건: {conditions})")
    where_clauses = []
    params = {}
    
    for key, value in conditions.items():
        if key.startswith("max_"):
            column = key.replace("max_", "")
            where_clauses.append(f"`{column}` <= :{key}")
            params[key] = value
        elif key.startswith("min_"):
            column = key.replace("min_", "")
            where_clauses.append(f"`{column}` >= :{key}")
            params[key] = value

    if not where_clauses:
        return set()

    query_str = f"SELECT DISTINCT `food_name` FROM `food_nutrition` WHERE {' AND '.join(where_clauses)}"
    query = text(query_str)
    
    result = await db.execute(query, params)
    return {row[0] for row in result.fetchall()}

# ========================================
# 2-2단계: RAG 자가 교정 필터링 (요구사항에 맞게 강화)
# ========================================
async def _filter_by_rag_self_correction(
    standard_dishes: Set[str], 
    disease: str, 
    dietary_restrictions: str
) -> Set[str]:
    """
    매우 엄격한 영양사 역할을 하는 LLM이 RAG와 자가 교정을 통해 최종 메뉴를 검증합니다.
    """
    print("\n--- 2-2단계: RAG 자가 교정 필터링 시작 ---")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    
    # 1. 심층 정보 수집 (기존 _retrieve_health_knowledge 활용)
    retrieved_knowledge = await _retrieve_health_knowledge(disease, dietary_restrictions)

    # 2. 최종 재검증 (자가 교정 프롬프트 적용)
    prompt_template = """
    당신은 안전을 최우선으로 생각하는 매우 엄격한 수석 영양사입니다. 
    아래 [심층 건강 정보]를 바탕으로, 각 음식이 사용자의 [질병]에 정말로 적합한지 최종 판단을 내려주세요.

    [가장 중요한 지시사항]
    - 조금이라도 애매하거나 잠재적 위험이 있다면 반드시 '부적합(is_suitable: false)'으로 판정해야 합니다.
    - 안전이 100% 확실한 음식만 '적합(is_suitable: true)'으로 승인하세요.

    [심층 건강 정보]: {retrieved_knowledge}
    [최종 후보 음식 목록]: {standard_dishes}
    [사용자 정보]
    - 질병: {disease}
    - 추가 식단 제약: {dietary_restrictions}

    각 음식에 대해 아래 JSON 형식으로만 응답해주세요. 설명은 절대 추가하지 마세요.
    {{"음식명1": {{"is_suitable": boolean, "reason": "안전을 최우선으로 고려한 판단 이유"}}, "음식명2": ...}}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | JsonOutputParser()
    
    print(f"-> 엄격한 최종 검증 LLM에 요청 (대상 메뉴 {len(standard_dishes)}개)...")
    
    try:
        result = await chain.ainvoke({
            "retrieved_knowledge": retrieved_knowledge,
            "standard_dishes": json.dumps(list(standard_dishes), ensure_ascii=False),
            "disease": disease,
            "dietary_restrictions": dietary_restrictions or "없음"
        })
        
        suitable_dishes = set()
        print("\n--- 최종 RAG 자가 교정 결과 분석 ---")
        for dish, details in result.items():
            status = "✅ 최종 승인" if details.get("is_suitable") else "❌ 최종 반려"
            print(f"- {dish}: {status} (사유: {details.get('reason', '없음')})")
            if details.get("is_suitable"): 
                suitable_dishes.add(dish)
        
        return suitable_dishes
        
    except (OutputParserException, json.JSONDecodeError) as e:
        print(f"[LLM 최종 검증 오류] {e}. 안전을 위해 빈 목록을 반환합니다.")
        return set()

# ========================================
# 건강 정보 검색 헬퍼 함수
# ========================================

async def _retrieve_health_knowledge(disease: str, dietary_restrictions: str) -> str:
    """질병과 식단 제약에 대한 건강 정보를 검색"""
    retrieved_knowledge = ""
    
    if disease in DISEASE_KEYWORD_MAP:
        print(f"-> '{disease}'은(는) 로컬 검색 대상입니다. 로컬 DB에서 정보를 검색합니다.")
        def search_local_db():
            try:
                query = f"{disease} 환자에게 추천하는 {dietary_restrictions or ''} 식단"
                docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(query, k=3)
                return "\n\n".join([doc.page_content for doc in docs]) if docs else ""
            except Exception as e:
                print(f"   - ChromaDB 검색 중 오류: {e}")
                return ""
        
        retrieved_knowledge = await asyncio.to_thread(search_local_db)
    else:
        print(f"-> '{disease}'은(는) 로컬 DB에 없는 질병입니다. 웹에서 최신 정보를 검색합니다.")
        try:
            search_tool = GoogleSearchAPIWrapper()
            web_query = f"{disease} 식단 가이드라인 {dietary_restrictions or ''}"
            search_results = await asyncio.to_thread(search_tool.run, web_query)
            if search_results and "No good Google Search Result" not in search_results:
                llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
                summary_prompt = f"다음 웹 검색 결과를 바탕으로 '{web_query}'에 대한 핵심 식단 지침을 요약해줘:\n\n{search_results}"
                retrieved_knowledge = (await llm.ainvoke(summary_prompt)).content
        except Exception as e:
            print(f"   - 웹 검색 중 오류 발생: {e}")

    if not retrieved_knowledge.strip():
        retrieved_knowledge = "전문적인 식단 지침을 찾을 수 없었습니다."
        print("-> 관련 건강 정보를 찾는 데 실패했습니다.")

    return retrieved_knowledge

# ========================================
# 리뷰 필터링
# ========================================

async def filter_restaurants_by_review(restaurants: List[Dict], other_requests: str) -> List[Dict]:
    """리뷰 기반으로 레스토랑 목록을 필터링하고 재정렬"""
    print("\n--- 리뷰 기반 필터링 시작 ---")
    if not other_requests or other_requests.strip().lower() in ['없음', '없어요']:
        print("-> 추가 요청사항 없음. 리뷰 필터링을 건너뜁니다.")
        return restaurants

    print(f"-> 요청사항 '{other_requests}'으로 리뷰 기반 필터링을 진행합니다...")

    try:
        # 사용자 요청과 유사한 리뷰 검색
        def search_reviews():
            return data_loader.REVIEW_DB.similarity_search_with_score(other_requests, k=30)
        
        retrieved_reviews = await asyncio.to_thread(search_reviews)

        if not retrieved_reviews:
            print("-> 요청사항과 유사한 리뷰를 찾지 못했습니다. 기존 후보군을 그대로 반환합니다.")
            return restaurants

        # 리뷰 점수를 레스토랑별로 집계
        restaurant_scores = defaultdict(float)
        restaurant_counts = defaultdict(int)

        # 현재 추천 후보군에 오른 레스토랑 이름 목록
        candidate_restaurant_names = {f"{r['name']} {r.get('branch_name', '')}".strip() for r in restaurants}

        for doc, score in retrieved_reviews:
            restaurant_full_name = doc.metadata.get("restaurant_full_name")
            if restaurant_full_name and restaurant_full_name in candidate_restaurant_names:
                restaurant_scores[restaurant_full_name] += score  # 점수가 낮을수록 좋음
                restaurant_counts[restaurant_full_name] += 1
        
        if not restaurant_scores:
            print("-> 후보군에 오른 레스토랑과 관련된 리뷰를 찾지 못했습니다.")
            return restaurants

        # 평균 점수 계산 후 정렬 (점수가 낮을수록 순위가 높음)
        ranked_restaurant_names = sorted(
            restaurant_scores.keys(),
            key=lambda name: restaurant_scores[name] / restaurant_counts[name]
        )
        
        # 최종 레스토랑 목록을 순위에 맞게 재정렬
        restaurants_dict = {f"{r['name']} {r.get('branch_name', '')}".strip(): r for r in restaurants}
        
        final_restaurants = []
        for name in ranked_restaurant_names:
            if name in restaurants_dict:
                final_restaurants.append(restaurants_dict[name])

        print(f"-> 리뷰 필터링 완료. {len(final_restaurants)}개의 음식점 순서를 재정렬했습니다.")
        return final_restaurants[:10]  # 상위 10개만 반환

    except Exception as e:
        print(f"[리뷰 필터링 오류] {e}")
        return restaurants[:10]