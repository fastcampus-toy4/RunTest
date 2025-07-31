# services/nlu_service.py
from langchain_openai import ChatOpenAI
from models.schemas import UserIntent
from core.config import settings

async def extract_user_intent(message: str) -> UserIntent:
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(UserIntent)
    prompt = f"Extract information from the following user message: '{message}'"
    try:
        return await structured_llm.ainvoke(prompt)
    except Exception:
        return UserIntent()
