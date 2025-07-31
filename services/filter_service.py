# services/filter_service.py
import asyncio
import json
import pandas as pd
from typing import Set, List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.vectorstores import Chroma
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.chains import LLMChain
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from . import data_loader  # 사전 로드된 DB 및 데이터 사용
from core.config import settings

DISEASE_KEYWORD_MAP = {
    "위식도 역류질환": ["위식도 역류질환", "GERD", "식도염", "식도역류증"], "위염 및 소화성 궤양": ["위염", "소화성 궤양", "위궤양"],
    "염증성 장질환": ["염증성 장질환", "IBD", "만성궤양성 대장염", "크론병"], "과민성 대장 증후군": ["과민성 대장 증후군", "IBS"],
    "변비 및 게실 질환": ["변비", "게실 질환", "게실염"], "간질환": ["간질환", "간염", "간경변", "B형 간염", "원발성 간암", "간암"],
    "담낭질환": ["담낭질환", "담석증", "담낭염"], "췌장염": ["췌장염", "Pancreatitis"],
    "고혈압 및 심혈관 질환": ["고혈압", "혈압", "심혈관", "심장", "동맥경화", "심부전", "심근경색증", "협심증", "관상동맥질환", "심장판막질환"],
    "고지혈증": ["고지혈증", "Hyperlipidemia", "콜레스테롤", "피가 탁함"], "뇌졸중": ["뇌졸중", "Stroke", "뇌경색", "뇌출혈", "중풍"],
    "당뇨병": ["당뇨병", "당뇨", "Diabetes", "혈당"], "통풍": ["통풍", "Gout"], "갑상선 질환": ["갑상선", "갑상선 기능 항진증", "갑상선 기능 저하증", "갑상선 결절"],
    "만성 신장질환": ["신장질환", "CKD", "신장", "콩팥", "신부전", "신증후군", "사구체콩팥염", "신장병"],
    "요로계 질환": ["요로계", "신결석", "요로결석", "요로감염"], "셀리악병": ["셀리악병", "Celiac Disease", "실리악 스푸루", "글루텐"],
    "유당 불내증": ["유당 불내증", "Lactose Intolerance"], "삼킴곤란": ["삼킴곤란", "연하곤란", "Dysphagia"],
    "빈혈": ["빈혈", "Anemia", "겸상 적혈구 빈혈증"], "암": ["암", "Cancer", "폐암", "백혈병", "유방암", "대장암", "림프종", "악성 종양"],
    "골격계 질환": ["골격계", "뼈", "골다공증", "골연화증", "구루병"], "알레르기": ["알레르기", "두드러기", "천식"],
    "급성 감염성 질환": ["감염", "감기", "폐렴", "코로나", "장염", "기관지염", "수두", "설사"],
}

async def filter_menus_by_health_rag_with_self_correction(
    db: AsyncSession,
    session_id: str, 
    standard_dishes: Set[str], 
    disease: Optional[str], 
    dietary_restrictions: Optional[str]
) -> Set[str]:
    """
    [재검증 강화됨] 초기 판단을 비판적으로 재검토하여 신뢰도를 높인 RAG 필터링 함수.
    """
    print("\n--- 3단계: 건강 정보 기반 메뉴(RAG) 필터링 (재검증 강화 + 동적 수치 필터링) 시작 ---")
    print(f"시작. 대상 메뉴: {len(standard_dishes)}개, 질병: {disease}")

    if not disease or disease.strip().lower() in ['없음', '없어요']:
        print("-> 질병 정보 없음. 모든 메뉴를 적합으로 간주합니다.")
        return standard_dishes

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    standard_disease = disease

    # 동적 수치 기반 필터링
    print("\n--- [NEW] 동적 수치 기반 사전 필터링 시작 ---")
    try:
        # 1. 로컬 ChromaDB에서 정확한 영양 기준 정보 검색
        retrieved_nutrition_standards = await get_precise_nutrition_standards_from_local_db(standard_disease, dietary_restrictions)
        
        if retrieved_nutrition_standards and retrieved_nutrition_standards.strip():
            print("-> 로컬 DB에서 정확한 영양 기준을 찾았습니다. 동적 필터링을 시작합니다.")
            
            # 2. 영양 기준에서 수치 조건 추출
            nutrition_criteria = await extract_nutrition_criteria_from_standards_for_menu_filtering(retrieved_nutrition_standards, standard_disease)
            
            if nutrition_criteria:
                print(f"-> 추출된 영양 조건: {nutrition_criteria}")
                
                # 3. 동적 조건으로 적합한 음식들을 MySQL에서 검색
                suitable_foods_from_mysql = await search_suitable_foods_with_dynamic_criteria(db, nutrition_criteria)
                
                if not suitable_foods_from_mysql.empty:
                    # 4. MySQL 결과와 현재 메뉴 목록을 교집합으로 필터링
                    mysql_food_names = set(suitable_foods_from_mysql['name'].str.strip().str.lower())
                    
                    # 부분 매칭도 고려 (음식명이 포함되어 있는 경우)
                    pre_filtered_dishes = set()
                    for dish in standard_dishes:
                        dish_lower = dish.strip().lower()
                        # 정확히 일치하거나 MySQL 음식명에 현재 메뉴가 포함되어 있는 경우
                        if dish_lower in mysql_food_names:
                            pre_filtered_dishes.add(dish)
                        else:
                            # 부분 매칭 확인
                            for mysql_food in mysql_food_names:
                                if dish_lower in mysql_food or mysql_food in dish_lower:
                                    pre_filtered_dishes.add(dish)
                                    break
                    
                    if pre_filtered_dishes:
                        print(f"-> 동적 수치 필터링 완료: {len(pre_filtered_dishes)}개 메뉴가 영양 기준을 통과했습니다.")
                        print(f"   통과한 메뉴: {list(pre_filtered_dishes)[:10]}...")  # 처음 10개만 출력
                        
                        # 동적 필터링을 통과한 메뉴들로 범위를 좁혀서 LLM 검증 진행
                        standard_dishes = pre_filtered_dishes
                        print("동적 필터링 통과")
                    else:
                        print("-> 동적 수치 필터링 결과, 영양 기준에 맞는 메뉴가 없습니다.")
                        print("영양 기준에 맞는 메뉴 없음")
                        # 빈 결과가 나와도 기존 LLM 로직으로 넘어가서 한번 더 확인
                else:
                    print("-> MySQL에서 영양 기준에 맞는 음식을 찾지 못했습니다.")
            else:
                print("-> 영양 기준에서 구체적인 수치를 추출하지 못했습니다.")
        else:
            print("-> 로컬 DB에서 정확한 영양 기준을 찾지 못했습니다. 기존 검색 로직을 사용합니다.")
            
    except Exception as e:
        print(f"[동적 수치 필터링 오류] {e}")
        print("-> 오류가 발생했지만 기존 LLM 검증 로직을 계속 진행합니다.")
    
    print("\n--- LLM 기반 검증 단계 시작 ---")
    
    # --- 2. 정보 검색 (Retrieve) ---
    retrieved_knowledge = ""
    if standard_disease in DISEASE_KEYWORD_MAP:
        print(f"-> '{standard_disease}'은(는) 로컬 검색 대상입니다. 로컬 DB에서 정보를 검색합니다.")
        def search_local_db():
            all_docs = []
            for db_instance in [data_loader.HEALTH_JUDGMENT_DB]:
                try:
                    query = f"{standard_disease} 환자에게 추천하는 {dietary_restrictions or ''} 식단"
                    docs = [doc.page_content for doc in db_instance.similarity_search(query, k=2)]
                    if docs:
                        all_docs.extend(docs)
                except Exception as e:
                    print(f"   - ChromaDB 검색 중 오류: {e}")
            return all_docs
        
        all_docs = await asyncio.to_thread(search_local_db)
        if all_docs:
            retrieved_knowledge = "\n\n".join(all_docs)
    else:
        print(f"-> '{standard_disease}'은(는) 로컬 DB에 없는 질병입니다. 웹에서 최신 정보를 검색합니다.")
        try:
            search_tool = GoogleSearchAPIWrapper()
            web_query = f"{standard_disease} 식단 가이드라인 {dietary_restrictions or ''}"
            search_results = await asyncio.to_thread(search_tool.run, web_query)
            if search_results and "No good Google Search Result" not in search_results:
                summary_prompt = f"다음 웹 검색 결과를 바탕으로 '{web_query}'에 대한 핵심 식단 지침을 요약해줘:\n\n{search_results}"
                retrieved_knowledge = (await llm.ainvoke(summary_prompt)).content
        except Exception as e:
            print(f"   - 웹 검색 중 오류 발생: {e}")

    if not retrieved_knowledge.strip():
        retrieved_knowledge = "전문적인 식단 지침을 찾을 수 없었습니다."
        print("-> 최종적으로 관련 건강 정보를 찾는 데 실패했습니다.")

    # --- 3. LLM 추론 (Generate) with Enhanced Self-Correction ---
    
    # [1단계] LLM을 통한 초기 판단 (빠른 필터링 역할)
    prompt_initial_template = """
    아래 건강 정보를 바탕으로 각 음식이 사용자의 질병에 적합한지 판단해주세요.

    [건강 정보]: {retrieved_knowledge}
    [음식 목록]: {standard_dishes}

    각 음식에 대해 다음 JSON 형식으로 응답해주세요:
    {{"음식명1": {{"is_suitable": boolean, "reason": "이유"}}, "음식명2": ...}}

    중요: 질병 관리에 도움이 되거나 해롭지 않은 음식만 is_suitable: true로 판단하세요.
    """
    initial_chain = ChatPromptTemplate.from_template(prompt_initial_template) | llm | JsonOutputParser()
    
    print("-> LLM에 초기 판단 요청...")
    print("="*50)
    print("[디버깅] LLM에 전달되는 정보 확인")
    print(f"  - 건강 정보 (retrieved_knowledge):\n{retrieved_knowledge}")
    print(f"  - 판단 대상 메뉴 (standard_dishes):\n{list(standard_dishes)}")
    print("="*50)
    
    try:
        initial_judgment = await initial_chain.ainvoke({
            "retrieved_knowledge": retrieved_knowledge,
            "standard_dishes": json.dumps(list(standard_dishes), ensure_ascii=False)
        })
        
        print("-" * 50)
        print("[디버깅] LLM의 1차 판단 전체 결과")
        print(json.dumps(initial_judgment, indent=4, ensure_ascii=False))
        print("-" * 50)
        
    except (OutputParserException, json.JSONDecodeError) as e:
        print(f"[LLM 초기 판단 오류] {e}. 자기 교정 없이 모든 메뉴를 통과시킵니다.")
        return standard_dishes
    
    # [2단계] 초기 판단 결과에서 '적합(suitable)' 판정을 받은 메뉴만 선별
    suitable_dishes_for_correction = {}
    
    # Case 1: LLM이 리스트 형태로 결과를 반환한 경우
    if initial_judgment and isinstance(initial_judgment, list):
        for item in initial_judgment:
            if isinstance(item, dict) and item.get("is_suitable") and "food" in item:
                food_name = item["food"]
                # 코드가 기대하는 {음식명: {상세정보}} 형태로 재구성
                suitable_dishes_for_correction[food_name] = {
                    "is_suitable": item["is_suitable"],
                    "reason": item.get("reason", "이유 없음")
                }
                
    # Case 2: LLM이 기존처럼 딕셔너리 형태로 결과를 반환한 경우
    elif initial_judgment and isinstance(initial_judgment, dict):
        for dish, details in initial_judgment.items():
            if isinstance(details, dict) and details.get("is_suitable"):
                suitable_dishes_for_correction[dish] = details
    
    if not suitable_dishes_for_correction:
        print("-> 초기 판단에서 적합한 메뉴를 찾지 못했습니다. 최종 필터링을 종료합니다.")
        return set()
    
    print(f"-> 1차 필터링 통과 메뉴: {list(suitable_dishes_for_correction.keys())}")
    
    # [3단계] '적합' 메뉴에 대해서만 "비판적 재검증" 요청
    prompt_correct_template = """
    아래는 1차로 적합하다고 판단된 음식들입니다. 이제 더 엄격한 기준으로 재검증해주세요.

    [건강 정보]: {retrieved_knowledge}
    [1차 통과 음식들]: {initial_judgment}

    각 음식을 다시 한번 비판적으로 검토하여 다음 JSON 형식으로 최종 판단해주세요:
    {{"음식명1": {{"is_suitable": boolean, "reason": "재검증 후 최종 이유"}}, "음식명2": ...}}

    재검증 기준:
    1. 질병 악화 가능성이 조금이라도 있는 음식은 제외
    2. 영양학적으로 확실히 도움이 되는 음식만 포함
    3. 애매한 경우는 안전을 위해 제외
    """
    final_chain = ChatPromptTemplate.from_template(prompt_correct_template) | llm | JsonOutputParser()

    print("-> LLM에 최종 판단(재검증) 요청...")
    try:
        final_judgment = await final_chain.ainvoke({
            "retrieved_knowledge": retrieved_knowledge,
            "initial_judgment": json.dumps(suitable_dishes_for_correction, ensure_ascii=False),
        })

        results = final_judgment
        suitable_dishes = set()
        print("\n--- 최종 RAG 필터링 결과 분석 ---")
        for dish, details in results.items():
            status = "✅ 최종 적합" if details.get("is_suitable") else "❌ 최종 부적합 (재검증 후 변경)"
            print(f"- {dish}: {status} (사유: {details.get('reason', '없음')})")
            if details.get("is_suitable"): 
                suitable_dishes.add(dish)
        
        print(f"✅ RAG 필터링(재검증 완료) 완료. {len(suitable_dishes)}개 메뉴 통과.")
        return suitable_dishes
        
    except (OutputParserException, json.JSONDecodeError, Exception) as e:
        print(f"[LLM 최종 판단 오류] {e}. 재검증 전 1차 통과 메뉴를 반환합니다.")
        return set(suitable_dishes_for_correction.keys())
    
async def get_precise_nutrition_standards_from_local_db(disease: str, dietary_restrictions: Optional[str]) -> str:
    """
    로컬 ChromaDB(특히 chroma_db_pdf)에서 질병별 정확한 영양 기준 정보를 검색합니다.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
        
        # 검색 쿼리 생성
        search_query_prompt = PromptTemplate(
            input_variables=["disease"],
            template="""질병명 "{disease}"를 기반으로 정확한 영양 기준 정보를 찾기 위한 검색 쿼리를 생성하세요.

다음을 포함하세요:
- 질병명 한국어 표현
- 질병명 영어 표현
- 정확한 섭취량, 수치, mg, g, kcal 등 구체적인 영양 기준 키워드
- 권장 섭취량, 식이요법, 영양 기준, 영양소 권장량, 식이지침, 나트륨 제한, 콜레스테롤 제한 등 키워드

질병명: {disease}
생성할 검색 쿼리:"""
        )
        
        query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
        search_query = (await query_chain.arun({"disease": disease})).strip()
        
        print(f"-> 생성된 검색 쿼리: {search_query}")
        
        # ChromaDB에서 검색 (health_judgment_db 사용)
        def search_db():
            try:
                docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(search_query, k=5)
                if docs:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    print(f"-> health_judgment_db에서 {len(docs)}개 문서 발견")
                    return context
            except Exception as e:
                print(f"   - health_judgment_db 검색 중 오류: {e}")
            return ""
        
        pdf_context = await asyncio.to_thread(search_db)
        
        if pdf_context:
            # LLM으로 영양 기준 추출
            extraction_prompt = PromptTemplate(
                template="""아래 문서에서 {disease} 환자의 구체적인 영양 섭취 기준 수치를 추출하세요.
특히 mg, g, kcal 등 정확한 수치가 포함된 내용만 추출해주세요.

문서 내용:
{context}

추출 규칙:
1. 구체적인 수치(숫자 + 단위)가 포함된 영양 기준만 추출
2. 나트륨, 콜레스테롤, 칼로리, 단백질, 지방, 탄수화물 등의 구체적 제한/권장 수치
3. "하루 2300mg 이하", "500mg/끼니" 같은 정확한 표현
4. 애매한 표현("적당히", "많이", "조금") 제외
5. 문서에 직접 명시된 내용만 사용

{disease} 환자의 정확한 영양 섭취 기준:""",
                input_variables=["context", "disease"]
            )
            
            extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
            extracted_standards = await extraction_chain.arun({"context": pdf_context, "disease": disease})
            
            if extracted_standards and len(extracted_standards.strip()) > 50:  # 의미있는 정보인지 확인
                print("-> DB에서 정확한 영양 기준을 성공적으로 추출했습니다.")
                return extracted_standards.strip()
        
        print("-> DB에서 정확한 수치를 찾지 못했습니다.")
        return ""
        
    except Exception as e:
        print(f"[정확한 영양 기준 검색 오류] {e}")
        return ""


async def extract_nutrition_criteria_from_standards_for_menu_filtering(nutrition_standards: str, disease: str) -> dict:
    """
    메뉴 필터링을 위한 영양 기준 수치 추출
    """
    try:
        # 하루 기준을 끼니별로 변환
        converted_standards = await convert_daily_to_meal_values(nutrition_standards)
        print(f"-> 끼니별 변환 완료: {converted_standards[:200]}...")
        
        # 사용 가능한 영양소 컬럼들 조회
        available_columns = await get_available_nutrition_columns()
        
        # 동적으로 영양소 리스트 생성
        nutrition_list = ""
        for col_name, kor_name in available_columns.items():
            unit = "mg" if "_mg" in col_name else ("ug" if "_ug" in col_name else ("kcal" if "kcal" in col_name else "g"))
            nutrition_list += f"- {col_name}: {kor_name} ({unit} 단위)\n"
        
        extract_criteria_prompt = PromptTemplate(
            input_variables=["nutrition_standards", "disease", "nutrition_list"],
            template="""아래 {disease} 환자의 영양 기준 정보에서 구체적인 수치(숫자와 단위)가 포함된 영양소 조건들을 모두 찾아서 JSON으로 추출하세요.

영양 기준 정보:
{nutrition_standards}

사용 가능한 영양소들:
{nutrition_list}

위 영양소들 중에서 영양 기준 정보에 언급된 모든 영양소의 수치 조건을 찾아서 JSON 형태로 출력하세요.

각 영양소는 다음과 같이 처리:
- 제한, 최대, 초과하지 않도록, 이하 등이 언급된 경우: max_영양소명
- 권장, 최소, 이상, 충분히 등이 언급된 경우: min_영양소명

예시:
{{"max_sodium_mg": 500, "min_protein_g": 20, "max_cholesterol_mg": 100, "min_calcium_mg": 300}}

중요: 영양 기준 정보에 실제로 언급된 영양소만 포함하고, 언급되지 않은 영양소는 포함하지 마세요.

JSON 형태로만 출력:"""
        )
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
        extract_chain = LLMChain(llm=llm, prompt=extract_criteria_prompt)
        result = (await extract_chain.arun({
            "nutrition_standards": converted_standards,
            "disease": disease,
            "nutrition_list": nutrition_list
        })).strip()
        
        print(f"-> 추출된 영양 조건 (raw): {result}")
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group().strip()
            try:
                criteria = json.loads(json_str)
                clean_criteria = {}
                for key, value in criteria.items():
                    if isinstance(value, (int, float)) and value > 0:
                        clean_criteria[key] = value
                    elif isinstance(value, str) and value.replace('.', '').isdigit():
                        clean_criteria[key] = float(value)
                
                print(f"-> 정제된 영양 조건: {clean_criteria}")
                return clean_criteria
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                return {}
        return {}
        
    except Exception as e:
        print(f"영양 조건 추출 오류: {e}")
        return {}


async def search_suitable_foods_with_dynamic_criteria(db: AsyncSession, criteria: dict) -> pd.DataFrame:
    """
    동적 영양 조건으로 MySQL에서 적합한 음식들을 검색
    """
    try:
        print(f"-> MySQL에서 동적 조건으로 음식 검색: {criteria}")
        
        # WHERE 절 조건 생성
        where_conditions = []
        params = {}
        
        for key, value in criteria.items():
            if key.startswith('max_'):
                column = key.replace('max_', '')
                where_conditions.append(f"{column} <= :{key}")
                params[key] = value
            elif key.startswith('min_'):
                column = key.replace('min_', '')
                where_conditions.append(f"{column} >= :{key}")
                params[key] = value
        
        if not where_conditions:
            return pd.DataFrame(columns=['name'])
        
        # 동적 쿼리 생성
        query_str = f"""
        SELECT name, {', '.join([key.replace('max_', '').replace('min_', '') for key in criteria.keys()])}
        FROM food_nutrition 
        WHERE {' AND '.join(where_conditions)}
        LIMIT 100
        """
        
        query = text(query_str)
        result = await db.execute(query, params)
        
        # DataFrame으로 변환
        rows = result.fetchall()
        if rows:
            columns = ['name'] + [key.replace('max_', '').replace('min_', '') for key in criteria.keys()]
            df = pd.DataFrame(rows, columns=columns)
            print(f"-> {len(df)}개의 적합한 음식을 찾았습니다.")
            return df
        else:
            print("-> 조건에 맞는 음식이 없습니다.")
            return pd.DataFrame(columns=['name'])
            
    except Exception as e:
        print(f"동적 조건 음식 검색 오류: {e}")
        return pd.DataFrame(columns=['name'])


async def convert_daily_to_meal_values(nutrition_standards: str) -> str:
    """하루 기준 수치를 끼니별(1/3)로 변환"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    
    prompt = f"""
    다음 영양 기준에서 "하루" 또는 "일일" 기준의 수치들을 끼니별(1끼니 = 하루의 1/3) 기준으로 변환해주세요.

    원본 기준: {nutrition_standards}

    변환 규칙:
    - "하루 X mg" → "끼니당 약 X ÷ 3 mg"
    - "일일 X g" → "끼니당 약 X ÷ 3 g"
    - 이미 끼니별 기준인 것은 그대로 유지

    변환된 기준:
    """
    
    result = await llm.ainvoke(prompt)
    return result.content


async def get_available_nutrition_columns() -> dict:
    """사용 가능한 영양소 컬럼 정보 반환 (예시)"""
    return {
        "sodium_mg": "나트륨",
        "cholesterol_mg": "콜레스테롤", 
        "protein_g": "단백질",
        "fat_g": "지방",
        "carb_g": "탄수화물",
        "kcal": "칼로리",
        "calcium_mg": "칼슘",
        "iron_mg": "철분"
    }


# 기존 함수들은 새로운 함수를 호출하도록 수정
async def filter_menus_by_health(standard_dishes: Set[str], disease: str, dietary_restrictions: str) -> Set[str]:
    """RAG와 LLM을 사용하여 건강/식단 제약 기반으로 메뉴를 필터링 (강화된 버전 사용)"""
    # 세션 ID는 임시로 생성 (실제로는 상위에서 전달받아야 함)
    import uuid
    session_id = str(uuid.uuid4())
    
    # DB 세션은 임시로 None 전달 (실제로는 상위에서 전달받아야 함) 
    # 이 부분은 chat_orchestrator에서 DB 세션을 전달하도록 수정 필요
    return await filter_menus_by_health_rag_with_self_correction(
        db=None,  # 이 부분은 수정 필요
        session_id=session_id,
        standard_dishes=standard_dishes,
        disease=disease,
        dietary_restrictions=dietary_restrictions
    )

async def filter_restaurants_by_review(restaurants: List[Dict], other_requests: str) -> List[Dict]:
    """리뷰 기반으로 음식점 목록을 필터링하고 재정렬"""
    if not other_requests or other_requests.lower() in ['없음', '없어요']:
        return restaurants

    # ChromaDB 검색은 동기 함수이므로 to_thread로 감싸 비동기 컨텍스트에서 안전하게 실행
    def search_reviews():
        return data_loader.REVIEW_DB.similarity_search_with_score(other_requests, k=30)
    
    retrieved_reviews = await asyncio.to_thread(search_reviews)

    if not retrieved_reviews:
        return restaurants

    # 점수 계산 (이 부분은 CPU bound이므로 그냥 둬도 무방)
    scores = {f"{r['name']} {r.get('branch_name', '')}".strip(): 0.0 for r in restaurants}
    for doc, score in retrieved_reviews:
        name = doc.metadata.get("restaurant_full_name")
        if name in scores:
            scores[name] += score  # score가 거리(distance)이므로 낮을수록 좋음

    # 점수가 낮은 순(유사도가 높은 순)으로 정렬
    ranked_names = sorted(scores.keys(), key=lambda name: scores[name])
    
    # 정렬된 이름 순서대로 최종 음식점 목록 생성
    restaurants_dict = {f"{r['name']} {r.get('branch_name', '')}".strip(): r for r in restaurants}
    final_list = [restaurants_dict[name] for name in ranked_names if name in restaurants_dict]

    return final_list