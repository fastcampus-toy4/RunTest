import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError


from models.schemas import CrawledInfo, FullVisitabilityAnalysis


def get_llm():
    from main import app
    return app.state.llm


async def get_final_recommendations_with_crawling(final_candidates: List[Dict], user_time: str) -> List[Dict]:
    """
    최종 후보 레스토랑 목록을 크롤링하여 실시간 정보를 확인하고,
    영업 중인 곳만 필터링하여 반환합니다.
    """
    open_restaurants = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        tasks = [_get_restaurant_info_task(browser, f"{r['name']} {r.get('branch_name', '')}".strip()) for r in final_candidates]
        crawled_results = await asyncio.gather(*tasks)
        await browser.close()

        visitable_check_tasks, successful_crawls = [], []
        for data in crawled_results:
            if data.get('crawling_success'):
                visitable_check_tasks.append(_is_restaurant_open_llm_async(data, user_time))
                successful_crawls.append(data)
        
        if visitable_check_tasks:
            visitable_results = await asyncio.gather(*visitable_check_tasks)
            for i, is_visitable in enumerate(visitable_results):
                if is_visitable:
                    open_restaurants.append(successful_crawls[i])
                    
    return open_restaurants


async def _get_restaurant_info_task(browser: Browser, restaurant_full_name: str) -> dict:
    """단일 레스토랑의 정보를 네이버 지도에서 크롤링합니다."""
    result_data = {"restaurant_full_name": restaurant_full_name, "crawling_success": False}
    page = await browser.new_page()
    try:
        await page.goto("https://map.naver.com/v5/search", wait_until="load", timeout=20000)
        await page.locator("input.input_search").wait_for(state="visible", timeout=10000)
        await page.locator("input.input_search").fill(restaurant_full_name)
        await page.locator("input.input_search").press("Enter")

        search_iframe = page.frame_locator("#searchIframe")
        first_result_selector = 'a.place_bluelink'
        try:
            await search_iframe.locator(first_result_selector).first.wait_for(state="visible", timeout=10000)
            await search_iframe.locator(first_result_selector).first.click()
        except PlaywrightTimeoutError:
            pass # 단일 결과로 바로 넘어가는 경우

        entry_iframe = page.frame_locator("#entryIframe")
        info_div_selector = "div.PIbes"
        await entry_iframe.locator(info_div_selector).wait_for(state="visible", timeout=10000)
        all_text = await entry_iframe.locator(info_div_selector).inner_text()

        if not all_text or not all_text.strip():
            raise ValueError("크롤링된 텍스트가 비어 있습니다.")

        processed_info = await _process_with_llm_crawler_async(all_text, restaurant_full_name)
        if processed_info:
            result_data.update(processed_info.model_dump())
            result_data["crawling_success"] = True
        else:
            result_data["crawling_error"] = "LLM 데이터 정형화 실패"
            
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        result_data["crawling_error"] = error_message
        print(f"크롤링 오류 '{restaurant_full_name}': {error_message}")
    finally:
        await page.close()
    return result_data


async def _process_with_llm_crawler_async(raw_text: str, name: str) -> Optional[CrawledInfo]:
    """LLM을 사용하여 크롤링된 텍스트에서 정형화된 정보를 추출합니다."""
    parser = JsonOutputParser(pydantic_object=CrawledInfo)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract restaurant info from text into JSON. Follow the format.\n{format_instructions}"),
        ("user", "Extract info for '{name}' from the following text. Rules: 1. Exclude current status like 'open' from hours. 2. Use 'holiday_info' for closing days. 3. Use null for missing values.\n\nText:\n---\n{raw_text}\n---")
    ])
    chain = prompt | get_llm() | parser
    try:
        return await chain.ainvoke({
            "format_instructions": parser.get_format_instructions(),
            "name": name, "raw_text": raw_text
        })
    except (ValidationError, OutputParserException) as e:
        print(f"LLM 파싱 오류 '{name}': {e}")
        return None


async def _is_restaurant_open_llm_async(crawled_data: Dict, user_time: str) -> bool:
    """LLM을 사용하여 크롤링된 영업시간 정보와 사용자 시간을 비교해 방문 가능 여부를 판단합니다."""
    today_weekday = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][datetime.today().weekday()]
    prompt = f"영업시간: {crawled_data.get('operating_hours', '정보 없음')}, 휴무일: {crawled_data.get('holiday_info', '정보 없음')}, 방문 시간: {user_time}, 오늘 요일: {today_weekday}. 방문 가능한지 JSON으로 분석해주세요."
    
    structured_llm = get_llm().with_structured_output(FullVisitabilityAnalysis)
    try:
        response = await structured_llm.ainvoke(prompt)
        return response.final_conclusion.is_visitable
    except Exception as e:
        print(f"영업 여부 판단 오류 '{crawled_data['restaurant_full_name']}': {e}")
        return False