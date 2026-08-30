from fastapi import APIRouter
from app import schemas

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health", response_model=schemas.HealthResponse)
def health():
    comps = {}
    try:
        from app.services import ml_model
        comps["ml"] = ml_model.health()
    except Exception as e:
        comps["ml"] = {"loaded": False, "error": str(e)}
    try:
        from app.services import rag
        comps["rag"] = rag.backend_info() | {"snippets": rag.corpus_size()}
    except Exception as e:
        comps["rag"] = {"error": str(e)}
    from app.services import llm
    comps["llm"] = {"enabled": llm.LLM_ENABLED, "model": llm.GEMINI_MODEL}
    ok = comps["ml"].get("loaded", False)
    return {"status": "ok" if ok else "degraded", "components": comps}

@router.get("/meta")
def get_meta():
    from app.db import SessionLocal
    from app.models import Meta
    with SessionLocal() as session:
        m = session.query(Meta).first()
        if m:
            return {
                "reference_date": m.reference_date,
                "n_customers": m.n_customers,
                "n_demo_customers": m.n_demo_customers,
                "demo_customer_ids": m.demo_customer_ids,
                "overall_churn_rate": m.overall_churn_rate
            }
        return {}
