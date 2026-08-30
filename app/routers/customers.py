from fastapi import APIRouter, Depends, HTTPException, Query
from app.deps import get_db, require_auth
from app.services import orchestrator
from app.services import signals as signals_mod
from app.services.signals import CustomerNotFound
from app import schemas

router = APIRouter(prefix="/api", tags=["customers"], dependencies=[Depends(require_auth)])

@router.get("/customers", response_model=list[schemas.DashboardRow])
def list_customers(demo_only: bool = Query(True), db=Depends(get_db)):
    return orchestrator.list_dashboard_rows(db, demo_only=demo_only)

@router.get("/customers/{cid}", response_model=schemas.CustomerDetail)
def get_customer(cid: int, db=Depends(get_db)):
    try:
        sig = signals_mod.get_signals(db, cid)
    except CustomerNotFound:
        raise HTTPException(404, f"customer {cid} not found")
        
    last = orchestrator.get_last_analysis(db, cid)
    
    identity = {k: sig[k] for k in (
        "customer_id", "name", "plan_tier", "plan_tier_ord", "tenure_days", 
        "arr", "monthly_charges", "renewal_date", "days_to_renewal", 
        "num_devices", "age"
    ) if k in sig}
    
    scored = {k: sig[k] for k in (
        "usage_level", "usage_trend_pct", "last_login_days", "logins_last_30", 
        "active_days_last_30", "payment_failures", "support_contacts", 
        "open_tickets", "complaint_count", "avg_sentiment", 
        "last_feedback_sentiment", "engagement_score"
    ) if k in sig}
    
    return {
        "customer": identity, 
        "signals": scored,
        "usage_series": sig.get("usage_series", []),
        "feedback_list": sig.get("feedback_list", []),
        "last_analysis": last
    }
