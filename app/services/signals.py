from datetime import date, datetime
from statistics import mean
from dateutil.parser import parse

from app.models import Customer, UsageDaily, Feedback, Meta
from app.db import SessionLocal

class CustomerNotFound(Exception):
    pass

TREND_RECENT_DAYS = 7
TREND_BASELINE_DAYS = 30
FEEDBACK_WINDOW_DAYS = 90

def get_reference_date(session):
    meta_row = session.get(Meta, "reference_date")
    if meta_row and meta_row.value:
        return parse(meta_row.value).date()
    return date.today()

def _trend(rows):
    if len(rows) < 14:
        return 0.0
    
    recent_rows = rows[-TREND_RECENT_DAYS:]
    base_rows = rows[-(TREND_RECENT_DAYS + TREND_BASELINE_DAYS):-TREND_RECENT_DAYS]
    
    if not recent_rows or not base_rows:
        return 0.0
        
    recent = mean(r.active_minutes or 0 for r in recent_rows)
    base = mean(r.active_minutes or 0 for r in base_rows)
    
    if base <= 0:
        return 0.0
        
    trend = (recent - base) / base
    return max(-1.0, min(3.0, trend))

def _within(d_str, ref_date, days):
    d = parse(d_str).date()
    return (ref_date - d).days <= days

def _days_between(ref_date, target_str):
    if not target_str:
        return 0
    target = parse(target_str).date()
    return (target - ref_date).days

def get_signals(session, customer_id):
    c = session.get(Customer, customer_id)
    if c is None:
        raise CustomerNotFound(customer_id)
        
    ref_date = get_reference_date(session)

    # --- usage aggregates ---
    rows = session.query(UsageDaily).filter_by(customer_id=customer_id).order_by(UsageDaily.date).all()
    usage_trend_pct = _trend(rows)
    logins_last_30 = sum((r.logins or 0) for r in rows[-30:])
    active_days_30 = sum(1 for r in rows[-30:] if (r.active_minutes or 0) > 0)
    
    usage_level = c.usage_level
    if usage_level is None:
        usage_level = float(sum((r.active_minutes or 0) for r in rows[-30:]))

    # --- feedback aggregates (last 90d vs REFERENCE_DATE) ---
    fb = session.query(Feedback).filter_by(customer_id=customer_id).order_by(Feedback.date).all()
    fb90 = [f for f in fb if _within(f.date, ref_date, FEEDBACK_WINDOW_DAYS)]
    
    avg_sentiment = mean([f.sentiment for f in fb90 if f.sentiment is not None]) if [f.sentiment for f in fb90 if f.sentiment is not None] else None
    complaint_count = sum(1 for f in fb90 if f.is_complaint)
    open_tickets = sum(1 for f in fb90 if f.channel == "support" and f.is_complaint)
    last_feedback_sentiment = fb90[-1].sentiment if fb90 else None

    # --- renewal / tenure ---
    days_to_renewal = _days_between(ref_date, c.renewal_date)

    return {
        "customer_id": c.id,
        "name": c.name,
        "plan_tier": c.plan_tier,
        "plan_tier_ord": c.plan_tier_ord,
        "tenure_days": c.tenure_days,
        "arr": c.arr,
        "monthly_charges": c.monthly_charges,
        "renewal_date": str(c.renewal_date) if c.renewal_date else None,
        "days_to_renewal": days_to_renewal,
        "num_devices": c.num_devices,
        "age": c.age,
        "usage_level": usage_level,
        "usage_trend_pct": usage_trend_pct,
        "last_login_days": c.last_login_days,
        "logins_last_30": logins_last_30,
        "active_days_last_30": active_days_30,
        "payment_failures": c.payment_failures,
        "support_contacts": c.support_contacts,
        "open_tickets": open_tickets,
        "complaint_count": complaint_count,
        "avg_sentiment": avg_sentiment,
        "last_feedback_sentiment": last_feedback_sentiment,
        "engagement_score": c.engagement_score,
        "usage_series": [{"date": str(r.date), "active_minutes": r.active_minutes} for r in rows],
        "feedback_list": [{"date": str(f.date), "channel": f.channel, "text": f.text, "sentiment": f.sentiment, "is_complaint": bool(f.is_complaint)} for f in fb],
    }

def get_signals_by_id(customer_id):
    with SessionLocal() as session:
        return get_signals(session, customer_id)
