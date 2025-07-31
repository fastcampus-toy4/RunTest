# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

# --- API 요청/응답 모델 ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None # 대화의 연속성을 위한 세션 ID

class FinalRecommendation(BaseModel):
    restaurant_full_name: str
    address: Optional[str] = None
    phone_number: Optional[str] = None
    operating_hours: Optional[str] = None
    holiday_info: Optional[str] = None
    reason: Optional[str] = "실시간 정보 확인 결과, 현재 방문 가능합니다."

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_final: bool = False # 추천이 최종적으로 완료되었는지 여부
    recommendations: Optional[List[FinalRecommendation]] = None


# --- 내부 로직용 Pydantic 모델 ---

class UserRequestInfo(BaseModel):
    """사용자의 전체 요청 정보를 구조화하는 모델"""
    location: Optional[str] = Field(None, description="사용자가 언급한 지역 또는 장소")
    time: Optional[str] = Field(None, description="사용자가 언급한 시간")
    dietary_restrictions: Optional[str] = Field(None, description="식단 제약 정보 (없으면 '없음')")
    disease: Optional[str] = Field(None, description="사용자가 언급한 질병 정보 (없으면 '없음')")
    amenities: List[str] = Field(default=[], description="요청한 편의시설 목록")
    other_requests: Optional[str] = Field(None, description="그 외 기타 요청사항 요약")
    
    def is_ready(self) -> bool:
        """추천 프로세스를 시작하기에 충분한 정보가 수집되었는지 확인"""
        return all([self.location, self.time, self.dietary_restrictions, self.disease])

class CrawledInfo(BaseModel):
    """크롤링된 텍스트에서 추출한 정보 모델"""
    address: Optional[str] = Field(None, description="추출된 주소")
    operating_hours: Optional[str] = Field(None, description="추출된 영업시간")
    phone_number: Optional[str] = Field(None, description="추출된 전화번호")
    holiday_info: Optional[str] = Field(None, description="추출된 휴무일 정보")

class StartChatRequest(BaseModel):
    user_id: Optional[str] = None

class StartChatResponse(BaseModel):
    state: dict
    bot_response: str

class ChatMessageRequest(BaseModel):
    state: dict
    user_input: str

class HistorySummary(BaseModel):
    session_id: str
    content: str
    created_at: datetime

class VisitabilityConclusion(BaseModel):
    is_visitable: bool = Field(..., description="현재 시간에 해당 식당이 방문 가능한지 여부")
    reason: Optional[str] = Field(None, description="그렇게 판단한 이유 요약")

class FullVisitabilityAnalysis(BaseModel):
    restaurant_full_name: str = Field(..., description="식당의 전체 이름")
    operating_hours: Optional[str] = Field(None, description="영업시간")
    holiday_info: Optional[str] = Field(None, description="휴무일")
    final_conclusion: VisitabilityConclusion

