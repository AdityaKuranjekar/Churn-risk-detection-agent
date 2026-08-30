import os
import json
import logging
from datetime import datetime

from app.db import SessionLocal
from app.models import Analysis, Customer
from app.services import signals as signals_mod
from app.services import scoring, action_catalog, rag, llm, ml_model

CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.50"))

def _fallback_prob(sig):
    # Extremely naive fallback if ML is completely broken
    return 0.50

def _empty_snippet():
    return {"id": "", "title": "", "text": "", "score": 0.0, "retrieval_method": "none", "low_confidence": True}

def _ml_row(sig):
    # ml_model.build_feature_dict is expected to take a dict with these keys
    return sig

def _assemble(sig, ml, breakdown, base_action, snippet, llm_out):
    return {
        "summary": llm_out.get("summary", ""),
        "top_reasons": llm_out.get("top_reasons", []),
        "recommended_action": llm_out.get("recommended_action", ""),
        "priority": llm_out.get("priority", "P3"),
        "draft_message": llm_out.get("draft_message", ""),
        "playbook_citation": llm_out.get("playbook_citation", ""),
        "generated_by": llm_out.get("_generated_by", "unknown"),
        
        "customer": {
            "id": sig["customer_id"],
            "name": sig["name"],
            "plan_tier": sig["plan_tier"],
            "arr": sig["arr"],
            "renewal_date": sig["renewal_date"],
            "days_to_renewal": sig["days_to_renewal"]
        },
        "ml": {
            "churn_probability": round(float(ml["churn_probability"]), 4),
            "band": ml["band"],
            "top_features": ml["top_features"],
            "model_version": ml.get("model_version", "unknown")
        },
        "risk_breakdown": breakdown,
        "base_action": base_action,
        "playbook": {
            "id": snippet.get("id"),
            "title": snippet.get("title"),
            "text": snippet.get("text"),
            "score": snippet.get("score"),
            "method": snippet.get("retrieval_method"),
            "low_confidence": snippet.get("low_confidence")
        },
        "escalate": base_action["priority"] == "P0",
        "method_agreement": scoring.agreement(breakdown["health_score"], float(ml["churn_probability"])),
        "signals_snapshot": {k: sig.get(k) for k in (
            "usage_trend_pct", "last_login_days", "logins_last_30",
            "payment_failures", "support_contacts", "open_tickets",
            "complaint_count", "avg_sentiment", "engagement_score"
        )},
        "status": "new",
    }

def _low_risk_result(sig, ml, breakdown, base_action):
    ml_prob = float(ml["churn_probability"])
    
    positives = breakdown.get("positives", [])
    top_reasons = [p["label"] + ": " + p["detail"] for p in positives][:3]
    if not top_reasons:
        top_reasons = ["No significant risk signals detected."]
        
    return {
        "summary": f"{sig['name']} ({sig['plan_tier']}) is {breakdown['risk_band']} - health {breakdown['health_score']}/100, ML churn {ml_prob:.0%}. Routine monitoring.",
        "top_reasons": top_reasons,
        "recommended_action": base_action["action_label"],
        "priority": base_action["priority"],
        "draft_message": "",
        "playbook_citation": "",
        "generated_by": "low_risk_shortcut",
        
        "customer": {
            "id": sig["customer_id"],
            "name": sig["name"],
            "plan_tier": sig["plan_tier"],
            "arr": sig["arr"],
            "renewal_date": sig["renewal_date"],
            "days_to_renewal": sig["days_to_renewal"]
        },
        "ml": {
            "churn_probability": round(ml_prob, 4),
            "band": ml["band"],
            "top_features": ml["top_features"],
            "model_version": ml.get("model_version", "unknown")
        },
        "risk_breakdown": breakdown,
        "base_action": base_action,
        "playbook": _empty_snippet(),
        "escalate": False,
        "method_agreement": scoring.agreement(breakdown["health_score"], ml_prob),
        "signals_snapshot": {k: sig.get(k) for k in (
            "usage_trend_pct", "last_login_days", "logins_last_30",
            "payment_failures", "support_contacts", "open_tickets",
            "complaint_count", "avg_sentiment", "engagement_score"
        )},
        "status": "new",
    }

def _save(session, customer_id, result):
    row = Analysis(
        customer_id=customer_id,
        churn_probability=result["ml"]["churn_probability"],
        health_score=result["risk_breakdown"]["health_score"],
        risk_band=result["risk_breakdown"]["risk_band"],
        priority=result["priority"],
        recommended_action=result["recommended_action"],
        escalate=result["escalate"],
        playbook_id=result["playbook"].get("id"),
        generated_by=result["generated_by"],
        result_json=json.dumps(result, default=str),
        status="new",
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row

def get_last_analysis(session, customer_id):
    row = session.query(Analysis).filter_by(customer_id=customer_id).order_by(Analysis.created_at.desc()).first()
    if row:
        res = json.loads(row.result_json)
        res.update({"analysis_id": row.id, "created_at": row.created_at.isoformat(), "status": row.status})
        return res
    return None

def force_llm_override(sig):
    return False

def analyze_customer(session, customer_id: int, force: bool = False) -> dict:
    if not force:
        last = get_last_analysis(session, customer_id)
        if last:
            # Here we could check freshness if needed, but for now we just return it
            return last

    sig = signals_mod.get_signals(session, customer_id)
    
    try:
        ml = ml_model.predict_from_customer(_ml_row(sig))
    except Exception as e:
        logging.exception("ml failed")
        ml = {
            "churn_probability": _fallback_prob(sig),
            "band": "unknown",
            "top_features": [],
            "model_version": "unavailable"
        }
        
    churn_prob = float(ml["churn_probability"])
    sig["ml_top_features"] = ml["top_features"]
    
    breakdown = scoring.risk_breakdown(sig)
    base_action = action_catalog.pick_base_action(sig, churn_prob)
    
    if churn_prob < CHURN_THRESHOLD and breakdown["risk_band"] in ("healthy", "watch") and not force_llm_override(sig):
        result = _low_risk_result(sig, ml, breakdown, base_action)
    else:
        try:
            query = rag.build_query(sig, base_action)
            snippet = rag.retrieve_best(query)
        except Exception as e:
            logging.exception("rag failed")
            snippet = _empty_snippet()
            
        try:
            llm_out = llm.analyze(sig, churn_prob, breakdown, snippet, base_action)
        except Exception as e:
            logging.exception("llm failed")
            # llm handles its own fallback, but if it throws entirely, we construct a fallback here.
            # Since the instruction says `llm.analyze` returns fallback dict on any failure, it shouldn't throw.
            # Just in case:
            llm_out = {
                "summary": "Error calling LLM",
                "top_reasons": [],
                "recommended_action": base_action["action_label"],
                "priority": base_action["priority"],
                "draft_message": "",
                "playbook_citation": "",
                "_generated_by": "fallback"
            }

        result = _assemble(sig, ml, breakdown, base_action, snippet, llm_out)
        
    row = _save(session, customer_id, result)
    result["analysis_id"] = row.id
    result["created_at"] = row.created_at.isoformat()
    return result

def approve_analysis(session, analysis_id, message, status):
    row = session.get(Analysis, analysis_id)
    if not row:
        return None
    row.status = status
    row.approved_message = message
    row.approved_at = datetime.utcnow()
    session.commit()
    return {
        "analysis_id": row.id,
        "status": row.status,
        "approved_message": row.approved_message
    }

def list_dashboard_rows(session, demo_only=True):
    results = []
    query = session.query(Customer)
    if demo_only:
        query = query.filter(Customer.is_demo == 1)
    
    customers = query.all()
    for c in customers:
        sig = signals_mod.get_signals(session, c.id)
        breakdown = scoring.risk_breakdown(sig)
        
        top_reason = ""
        contributors = breakdown.get("contributors", [])
        if contributors:
            top_reason = f"{contributors[0]['label']} · {contributors[0]['detail']}"

        # check last analysis
        last = session.query(Analysis).filter_by(customer_id=c.id).order_by(Analysis.created_at.desc()).first()
        if last:
            results.append({
                "customer_id": c.id,
                "name": c.name,
                "churn_probability": last.churn_probability,
                "health_score": last.health_score,
                "risk_band": last.risk_band,
                "priority": last.priority,
                "recommended_action": last.recommended_action,
                "plan_tier": c.plan_tier,
                "arr": c.arr,
                "top_reason": top_reason,
                "days_to_renewal": sig.get("days_to_renewal"),
                "renewal_date": sig.get("renewal_date"),
                "escalate": last.priority == "P0"
            })
        else:
            try:
                ml = ml_model.predict_from_customer(_ml_row(sig))
            except Exception:
                ml = {"churn_probability": _fallback_prob(sig)}
                
            base_action = action_catalog.pick_base_action(sig, float(ml["churn_probability"]))
            
            results.append({
                "customer_id": c.id,
                "name": c.name,
                "churn_probability": float(ml["churn_probability"]),
                "health_score": breakdown["health_score"],
                "risk_band": breakdown["risk_band"],
                "priority": base_action["priority"],
                "recommended_action": base_action["action_label"],
                "plan_tier": c.plan_tier,
                "arr": c.arr,
                "top_reason": top_reason,
                "days_to_renewal": sig.get("days_to_renewal"),
                "renewal_date": sig.get("renewal_date"),
                "escalate": base_action["priority"] == "P0"
            })
            
    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return results[:40]
