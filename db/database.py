# db/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from core.config import settings

# 비동기 DB 엔진 생성
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)

# 비동기 세션 생성기
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# API 엔드포인트에서 의존성 주입으로 사용할 함수
async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session