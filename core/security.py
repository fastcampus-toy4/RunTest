# core/security.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from .config import settings

# Spring Boot에서 토큰을 발급하는 엔드포인트 경로를 정확히 기재해야 합니다.
# 여기서는 형식적인 경로를 사용합니다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login") 

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub") # Spring Security JWT에서 'subject'를 username으로 사용했다면
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"username": username}