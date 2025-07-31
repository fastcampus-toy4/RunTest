# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "AI Food Recommendation API"
    API_V1_STR: str = "/api/v1"

    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

    # Database
    DB_USER: str = os.getenv("MYSQL_USER")
    DB_PASSWORD: str = os.getenv("MYSQL_PASSWORD")
    DB_HOST: str = os.getenv("MYSQL_HOST")
    DB_PORT: str = os.getenv("MYSQL_PORT")
    DB_NAME: str = os.getenv("MYSQL_DB_NAME")
    DATABASE_URL = (
        f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
    # Data Paths
    FOOD_CLUSTERS_PATH: str = os.getenv("FOOD_CLUSTERS_PATH")
    REPRESENTATIVE_FOODS_PATH: str = os.getenv("REPRESENTATIVE_FOODS_PATH")
    HEALTH_JUDGMENT_DB_PATH: str = os.getenv("HEALTH_JUDGMENT_DB_PATH")
    REVIEW_DB_PATH: str = os.getenv("REVIEW_DB_PATH")


settings = Settings()