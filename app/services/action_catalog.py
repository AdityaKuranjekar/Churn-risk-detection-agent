from collections import defaultdict
from app.services.scoring import risk_breakdown

HIGH_ARR = 300.0

CATALOG = [
    {
        "id": "billing_fix_outreach",
        "label": "Billing fix outreach + retention offer",
        "priority": "P0",
        "channel": "call",
        "sla_hours": 24,
        "predicate": lambda s, p: s.get("payment_failures", 0) >= 2,
        "matched_rule": "payment_failures>=2",
        "query_template": "{plan_tier} customer with {payment_failures} payment failures in 90 days, needs billing resolution before any content or engagement outreach",
        "fallback": ["csm_personal_checkin"],
    },
    {
        "id": "live_support_callback",
        "label": "Escalate to live support callback",
        "priority": "P0",
        "channel": "call",
        "sla_hours": 24,
        "predicate": lambda s, p: s.get("support_contacts", 0) >= 3 or s.get("open_tickets", 0) >= 2,
        "matched_rule": "support_contacts>=3 or open_tickets>=2",
        "query_template": "{plan_tier} customer with high support contact volume and unresolved tickets, negative sentiment",
        "fallback": ["csm_personal_checkin"],
    },
    {
        "id": "win_back_high_value",
        "label": "Senior CSM win-back (high ARR, high risk)",
        "priority": "P0",
        "channel": "csm_touch",
        "sla_hours": 48,
        "predicate": lambda s, p: p >= 0.80 and s.get("arr", 0) >= HIGH_ARR,
        "matched_rule": "churn_prob>=0.80 and arr>=HIGH_ARR",
        "query_template": "High-value {plan_tier} customer with ARR {arr} at high risk of churn, needs senior CSM intervention",
        "fallback": ["csm_personal_checkin"],
    },
    {
        "id": "renewal_downgrade_offer",
        "label": "Offer downgrade before renewal",
        "priority": "P1",
        "channel": "email",
        "sla_hours": 72,
        "predicate": lambda s, p: s.get("days_to_renewal", 999) <= 45 and s.get("engagement_score", 100) < 45,
        "matched_rule": "days_to_renewal<=45 and engagement<45",
        "query_template": "{plan_tier} customer approaching renewal with low engagement, potential downgrade candidate",
        "fallback": ["csm_personal_checkin", "reengagement_content_email"],
    },
    {
        "id": "reengagement_content_email",
        "label": "Curated content / value re-engagement email",
        "priority": "P2",
        "channel": "email",
        "sla_hours": 120,
        "predicate": lambda s, p: s.get("usage_trend_pct", 0) <= -0.20 and s.get("complaint_count", 0) == 0,
        "matched_rule": "usage_trend<=-20% and no complaints",
        "query_template": "{plan_tier} customer with declining usage but no complaints, send value re-engagement content",
        "fallback": ["feature_adoption_nudge"],
    },
    {
        "id": "feature_adoption_nudge",
        "label": "Feature adoption nudge (in-app + email)",
        "priority": "P2",
        "channel": "in_app",
        "sla_hours": 120,
        "predicate": lambda s, p: s.get("engagement_score", 100) < 55 and s.get("last_login_days", 0) < 14,
        "matched_rule": "engagement<55 and recent_login",
        "query_template": "{plan_tier} customer with moderate engagement but active logins, nudge for new feature adoption",
        "fallback": ["reengagement_content_email"],
    },
    {
        "id": "csm_personal_checkin",
        "label": "Personal CSM check-in",
        "priority": "P1",
        "channel": "csm_touch",
        "sla_hours": 72,
        "predicate": lambda s, p: p >= 0.50,
        "matched_rule": "churn_prob>=0.50",
        "query_template": "At-risk {plan_tier} customer requires a standard CSM check-in",
        "fallback": ["reengagement_content_email"],
    },
    {
        "id": "monitor",
        "label": "No action - monitor",
        "priority": "P3",
        "channel": "none",
        "sla_hours": 0,
        "predicate": lambda s, p: True,
        "matched_rule": "default",
        "query_template": "healthy {plan_tier} customer, routine monitoring",
        "fallback": [],
    },
]

def _materialize(rule, signals, churn_prob):
    safe_signals = defaultdict(str, signals)
    if "plan_tier" not in safe_signals or not safe_signals["plan_tier"]:
        safe_signals["plan_tier"] = "Standard"
        
    query = rule["query_template"].format_map(safe_signals)
    
    # Add top contributor context
    try:
        breakdown = risk_breakdown(signals)
        if breakdown["contributors"]:
            top_reason = breakdown["contributors"][0]["label"].lower()
            query = f"{query}. Primary risk factor: {top_reason}."
    except Exception:
        pass
        
    return {
        "action": rule["id"],
        "action_label": rule["label"],
        "priority": rule["priority"],
        "rationale": rule.get("rationale", rule["query_template"].format_map(safe_signals)), # Optional specific rationale
        "matched_rule": rule["matched_rule"],
        "channel": rule["channel"],
        "playbook_query": query,
        "sla_hours": rule["sla_hours"],
        "fallback_actions": rule["fallback"],
    }

def pick_base_action(signals, churn_prob):
    """
    Priority Semantics:
    P0: Act within 24h, human touch, likely escalation flag.
    P1: Act within 72h, CSM owns.
    P2: Automated/low-touch, within a week.
    P3: Monitor only, no task created.
    """
    for rule in CATALOG:
        try:
            if rule["predicate"](signals, churn_prob):
                return _materialize(rule, signals, churn_prob)
        except Exception:
            continue
            
    return _materialize(CATALOG[-1], signals, churn_prob)

if __name__ == "__main__":
    healthy = {
        "usage_trend_pct": 0.20,
        "last_login_days": 1,
        "engagement_score": 85,
        "support_contacts": 0,
        "open_tickets": 0,
        "avg_sentiment": 0.6,
        "payment_failures": 0,
        "tenure_days": 1200,
        "days_to_renewal": 200,
        "plan_tier": "Premium",
        "arr": 1500
    }
    
    mid = {
        "usage_trend_pct": -0.10,
        "last_login_days": 10,
        "engagement_score": 45,
        "support_contacts": 1,
        "open_tickets": 0,
        "avg_sentiment": -0.1,
        "payment_failures": 0,
        "tenure_days": 400,
        "days_to_renewal": 30,
        "plan_tier": "Standard",
        "arr": 100
    }
    
    critical = {
        "usage_trend_pct": -0.50,
        "last_login_days": 35,
        "engagement_score": 15,
        "support_contacts": 4,
        "open_tickets": 1,
        "avg_sentiment": -0.8,
        "payment_failures": 2,
        "tenure_days": 150,
        "days_to_renewal": 20,
        "plan_tier": "Premium",
        "arr": 500
    }

    print("--- Healthy (0.10) ---")
    print(pick_base_action(healthy, 0.10))
    
    print("\n--- Mid (0.60) ---")
    print(pick_base_action(mid, 0.60))
    
    print("\n--- Critical (0.95) ---")
    print(pick_base_action(critical, 0.95))
