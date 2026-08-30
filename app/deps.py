import os
from fastapi import Depends, HTTPException, status, Request
from app.db import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_auth(request: Request) -> str:
    if os.getenv("AUTH_DISABLED", "1") == "1":
        return "dev"
    
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user
