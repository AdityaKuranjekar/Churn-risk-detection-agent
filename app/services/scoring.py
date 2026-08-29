import datetime

# -------------------------------------------------------------------------------
# CONSTANTS BLOCK
# -------------------------------------------------------------------------------
BASELINE = 100
FACTOR_WEIGHTS = {
    "usage":          25,
    "engagement":     15,
    "support":        20,
    "sentiment":      15,
    "payments":       20,
    "tenure_renewal": 10,
    "login_recency":  15,
}

BANDS = [(80, "healthy"), (60, "watch"), (40, "at_risk"), (0, "critical")]

USAGE_DROP_HIGH = -0.40
USAGE_DROP_MED  = -0.20
INACTIVE_HIGH_DAYS = 30
INACTIVE_MED_DAYS  = 14
SUPPORT_HIGH = 3
SUPPORT_MED  = 1
SENTIMENT_BAD = -0.30
RENEWAL_SOON_DAYS = 45

LABELS = {
    "usage": "Usage trend",
    "engagement": "Engagement score",
    "support": "Support volume",
    "sentiment": "Feedback sentiment",
    "payments": "Payment history",
    "tenure_renewal": "Tenure & Renewal risk",
    "login_recency": "Login recency",
}

# -------------------------------------------------------------------------------
# SCORING FACTORS
# -------------------------------------------------------------------------------
def score_usage(signals):
    t = signals.get("usage_trend_pct", 0.0)
    w = FACTOR_WEIGHTS["usage"]
    if t <= USAGE_DROP_HIGH:
        frac = 1.0
    elif t <= USAGE_DROP_MED:
        frac = 0.6
    elif t < 0:
        frac = 0.3
    elif t > 0.15:
        return (+4, "Usage growing", "positive")
    else:
        frac = 0.0
    
    impact = -round(w * frac, 1)
    detail = f"Usage {t:+.0%} vs baseline"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_login_recency(signals):
    d = signals.get("last_login_days", 0)
    w = FACTOR_WEIGHTS["login_recency"]
    if d >= INACTIVE_HIGH_DAYS:
        frac = 1.0
    elif d >= INACTIVE_MED_DAYS:
        frac = 0.6
    elif d >= 7:
        frac = 0.25
    else:
        frac = 0.0
        
    impact = -round(w * frac, 1)
    detail = f"{d} days since last login"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_engagement(signals):
    e = signals.get("engagement_score", 50)
    w = FACTOR_WEIGHTS["engagement"]
    frac = max(0.0, min(1.0, (40 - e) / 40)) if e < 40 else 0.0
    if e >= 75:
        return (+3, "Strong engagement", "positive")
        
    impact = -round(w * frac, 1)
    detail = f"Engagement score: {e}"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_support(signals):
    c = signals.get("support_contacts", 0)
    o = signals.get("open_tickets", 0)
    w = FACTOR_WEIGHTS["support"]
    
    if c >= SUPPORT_HIGH:
        frac = 0.8
    elif c >= SUPPORT_MED:
        frac = 0.4
    else:
        frac = 0.0
        
    if o > 0:
        frac = min(1.0, frac + 0.3)
        
    impact = -round(w * frac, 1)
    detail = f"{c} support contacts / {o} open in 90d"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_sentiment(signals):
    s = signals.get("avg_sentiment")
    w = FACTOR_WEIGHTS["sentiment"]
    
    if s is None:
        return (0.0, "No recent feedback", "low")
        
    if s <= SENTIMENT_BAD:
        frac = min(1.0, abs(s) / 0.7)
    elif s < 0:
        frac = 0.3
    elif s > 0.4:
        return (+4, "Positive feedback", "positive")
    else:
        frac = 0.0
        
    impact = -round(w * frac, 1)
    detail = f"Average sentiment: {s:.2f}"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_payments(signals):
    p = signals.get("payment_failures", 0)
    w = FACTOR_WEIGHTS["payments"]
    
    frac = 0.0 if p == 0 else 0.6 if p == 1 else 1.0
    impact = -round(w * frac, 1)
    detail = f"{p} payment failure(s) in 90d"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

def score_tenure_renewal(signals):
    dtr = signals.get("days_to_renewal", 999)
    ten = signals.get("tenure_days", 0)
    w = FACTOR_WEIGHTS["tenure_renewal"]
    
    near = dtr <= RENEWAL_SOON_DAYS
    frac = 0.0
    if near and ten < 180:
        frac = 1.0
    elif near:
        frac = 0.5
        
    if ten > 900 and not near:
        return (+3, "Long-tenured, loyal", "positive")
        
    impact = -round(w * frac, 1)
    detail = f"Tenure: {ten}d, Days to renewal: {dtr}"
    sev = "high" if frac >= 0.8 else "medium" if frac >= 0.4 else "low"
    return (impact, detail, sev)

# -------------------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------------------
def build_summary(health, contribs):
    if not contribs:
        return f"Health {health}/100 - Strong across all indicators."
    
    top = [c["label"].lower() for c in contribs[:2]]
    if len(top) == 1:
        reasons = top[0]
    else:
        reasons = f"{top[0]} and {top[1]}"
        
    return f"Health {health}/100 - driven by {reasons}."

def risk_breakdown(signals):
    factors = {
        "usage": score_usage,
        "login_recency": score_login_recency,
        "engagement": score_engagement,
        "support": score_support,
        "sentiment": score_sentiment,
        "payments": score_payments,
        "tenure_renewal": score_tenure_renewal,
    }
    
    contribs = []
    positives = []
    total = BASELINE
    
    for key, fn in factors.items():
        impact, detail, sev = fn(signals)
        total += impact
        row = {
            "factor": key,
            "label": LABELS[key],
            "impact": impact,
            "detail": detail,
            "severity": sev
        }
        if impact > 0:
            positives.append(row)
        elif impact < 0:
            contribs.append(row)
            
    health = int(max(0, min(100, round(total))))
    band = "critical"
    for cut, name in BANDS:
        if health >= cut:
            band = name
            break
            
    contribs.sort(key=lambda r: r["impact"]) # Most negative first
    
    summary_line = build_summary(health, contribs)
    
    return {
        "health_score": health,
        "risk_band": band,
        "contributors": contribs,
        "positives": positives,
        "summary_line": summary_line,
        "as_of": datetime.date.today().isoformat()
    }

def agreement(health_score, churn_prob):
    implied = (100 - health_score) / 100
    gap = abs(implied - churn_prob)
    agree = gap <= 0.25
    return {
        "agree": agree,
        "gap": round(gap, 2),
        "note": "" if agree else "Rule-based and ML risk disagree - review signals manually"
    }

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
        "days_to_renewal": 200
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
        "days_to_renewal": 60
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
        "days_to_renewal": 20
    }

    for name, s in [("Healthy", healthy), ("Mid", mid), ("Critical", critical)]:
        print(f"--- {name} ---")
        res = risk_breakdown(s)
        import json
        print(json.dumps(res, indent=2))
        print()
