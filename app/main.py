import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    from app.db import Base, engine
    Base.metadata.create_all(bind=engine)
    try:
        from app.services import rag
        rag.warm()
    except Exception as e:
        logging.warning("rag.warm failed: %s", e)
        
    try:
        from app.services import ml_model
        logging.info("ml health: %s", ml_model.health())
    except Exception as e:
        logging.error("ml_model not loaded: %s", e)
        
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="Churn Risk Agent", version="1.0", lifespan=lifespan)
    
    origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )
    
    from starlette.middleware.sessions import SessionMiddleware
    app.add_middleware(
        SessionMiddleware, 
        secret_key=os.getenv("SECRET_KEY", "dev-secret-key"), 
        https_only=False, 
        same_site="lax", 
        max_age=60*60*8
    )
    
    from app.routers import customers, analysis, health, auth
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(customers.router)
    app.include_router(analysis.router)
    
    fe = os.getenv("FRONTEND_DIR", "frontend")
    if os.path.isdir(fe):
        app.mount("/", StaticFiles(directory=fe, html=True), name="spa")
        
    return app

app = create_app()
