import os
import secrets
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(req: LoginRequest, request: Request):
    app_user = os.getenv("APP_USERNAME", "admin")
    app_pass = os.getenv("APP_PASSWORD", "secret")
    
    # Use secrets.compare_digest for constant-time comparison
    if secrets.compare_digest(req.username, app_user) and secrets.compare_digest(req.password, app_pass):
        request.session["user"] = req.username
        return {"user": req.username}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@router.get("/me")
def me(request: Request):
    if os.getenv("AUTH_DISABLED", "1") == "1":
        return {"user": "dev"}
    
    user = request.session.get("user")
    if user:
        return {"user": user}
        
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
