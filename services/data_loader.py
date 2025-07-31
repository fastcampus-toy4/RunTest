# services/data_loader.py
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from core.config import settings
import sys

# 이 모듈이 처음 임포트될 때 단 한 번만 실행됩니다.
print("사전 계산된 클러스터 및 건강 판단 데이터를 로딩합니다...")

try:
    # 1. 클러스터 및 대표 음식 데이터 로드
    df_clusters = pd.read_csv(settings.FOOD_CLUSTERS_PATH)
    FOOD_TO_CLUSTER_MAP = pd.Series(df_clusters.cluster_id.values, index=df_clusters.food_name).to_dict()

    df_representatives = pd.read_csv(settings.REPRESENTATIVE_FOODS_PATH)
    CLUSTER_TO_FOOD_MAP = pd.Series(df_representatives.representative_food.values, index=df_representatives.index).to_dict()

    # 2. 임베딩 함수 및 벡터 DB 준비
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
    
    HEALTH_JUDGMENT_DB = Chroma(
        persist_directory=settings.HEALTH_JUDGMENT_DB_PATH,
        embedding_function=embedding_function
    )
    REVIEW_DB = Chroma(
        persist_directory=settings.REVIEW_DB_PATH,
        embedding_function=embedding_function
    )
    print("✅ 데이터 로딩 완료.")

except FileNotFoundError as e:
    print(f"[치명적 오류] 사전 계산 데이터 파일을 찾을 수 없습니다: {e}", file=sys.stderr)
    print("-> 'build_food_clusters.py'와 'batch_health_judge.py'를 먼저 실행하여 데이터 파일을 생성해야 합니다.", file=sys.stderr)
    # 서버가 시작되지 않도록 프로그램을 종료합니다.
    sys.exit(1)
except Exception as e:
    print(f"[치명적 오류] 데이터 로딩 중 예상치 못한 오류 발생: {e}", file=sys.stderr)
    sys.exit(1)