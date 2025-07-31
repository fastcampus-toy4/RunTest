# services/db_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Set

async def get_restaurants_from_db(db: AsyncSession, location_gu: str, amenities: List[str]) -> List[Dict]:
    """
    위치(구)와 편의시설 조건으로 DB에서 1차 후보 음식점 목록을 조회합니다.
    SQLAlchemy의 매개변수 바인딩을 사용하여 SQL 인젝션을 안전하게 방지합니다.
    """
    where_clauses = ["region_name LIKE :location_gu"]
    params = {"location_gu": f"%{location_gu}%"}

    amenity_map = {"주차": "has_parking", "와이파이": "has_wifi", "놀이방": "has_kids_zone"}
    for i, amenity in enumerate(amenities):
        if amenity in amenity_map:
            param_name = f"amenity_{i}"
            where_clauses.append(f"{amenity_map[amenity]} = :{param_name}")
            params[param_name] = 'Y'

    query_str = f"SELECT name, branch_name FROM restaurant WHERE {' AND '.join(where_clauses)} LIMIT 100"
    query = text(query_str)
    
    result = await db.execute(query, params)
    return [dict(row) for row in result.mappings()]

async def get_normalized_menus_for_restaurants(db: AsyncSession, restaurants: List[Dict]) -> Dict[str, Set[str]]:
    """선별된 음식점 목록에 대해 정규화된 메뉴를 조회합니다."""
    if not restaurants:
        return {}

    # 레스토랑 이름과 지점명으로 IN 절을 만들기 위한 준비
    # (name, branch_name) 쌍으로 조회하는 것이 더 정확하지만, 여기서는 name을 기준으로 단순화
    restaurant_names = list(set(r['name'] for r in restaurants))
    
    query = text("""
        SELECT restaurant_name, branch_name, normalized_name 
        FROM restaurant_menu 
        WHERE restaurant_name = ANY(:names) AND normalized_name IS NOT NULL
    """)
    params = {"names": restaurant_names}
    
    result = await db.execute(query, params)
    
    menus_by_restaurant = {}
    for row in result.mappings():
        full_name = f"{row['restaurant_name']} {row.get('branch_name', '')}".strip()
        if full_name not in menus_by_restaurant:
            menus_by_restaurant[full_name] = set()
        menus_by_restaurant[full_name].add(row['normalized_name'])
        
    return menus_by_restaurant