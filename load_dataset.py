import os
import sys
import json
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -----------------------------------------------------------------------------
# PHASE 2 — THE COLUMN MAP
# -----------------------------------------------------------------------------
RAW = "data/customer_churn_raw.csv"
DB  = "churn_agent.db"
SEED = 42
USAGE_HISTORY_DAYS = 30
DEMO_SUBSET_SIZE = 40


CANDIDATES = {
    "ext_id":          ["customer_id", "customerID", "id", "user_id"],
    "churn":           ["churn", "churned", "is_churn", "exited", "target", "attrition", "churn_flag", "churn_label", "Churn Value"],
    "plan_tier":       ["plan_type", "subscription_type", "plan", "tier", "contract", "subscription_plan"],
    "tenure":          ["tenure", "tenure_months", "tenure_days", "months_subscribed", "account_age", "customer_tenure", "Tenure Months"],
    "monthly_charges": ["monthly_charges", "monthlycharges", "monthly_bill", "monthly_fee", "price", "subscription_fee", "mrr", "Monthly Charges"],
    "total_charges":   ["total_charges", "totalcharges", "lifetime_value", "total_spent", "Total Charges"],
    "last_login":      ["last_login", "last_active", "last_activity_date", "days_since_last_login", "last_seen", "weeks_since_last_login"],
    "usage":           ["usage_frequency", "avg_watch_time", "watch_hours", "hours_watched", "sessions_per_week", "avg_session_length", "usage_minutes", "number_of_logins", "login_frequency", "monthly_usage", "activity_score"],
    "payment_failures":["payment_failures", "failed_payments", "num_failed_payments", "billing_issues", "payment_delay", "late_payments"],
    "support_contacts":["support_tickets", "num_support_tickets", "customer_service_calls", "complaints", "support_calls", "num_complaints", "tickets_raised", "Customer Service Calls"],
    "engagement":      ["engagement_score", "engagement", "activity_level", "satisfaction_score", "nps", "csat", "Satisfaction Score"],
    "num_devices":     ["num_devices", "number_of_devices", "devices", "device_count"],
    "age":             ["age", "customer_age", "Age"],
    "gender":          ["gender", "sex", "Gender"],
    "region":          ["region", "country", "state", "location", "geography", "State"],
    "signup_date":     ["signup_date", "join_date", "registration_date", "start_date", "subscription_start"],
}

def resolve(df, key):
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in CANDIDATES.get(key, []):
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    return None

# -----------------------------------------------------------------------------
# PHASE 1 — INSPECT
# -----------------------------------------------------------------------------
def inspect():
    df = pd.read_csv(RAW)
    print("shape:", df.shape)
    print("\ncolumns:", list(df.columns))
    print("\ndtypes:\n", df.dtypes)
    print("\nnull counts:\n", df.isna().sum())
    print("\nhead:\n", df.head(10).to_string())
    for c in df.columns:
        if df[c].dtype == "object" or df[c].nunique() < 15:
            print(f"\n{c}  unique({df[c].nunique()}): {df[c].unique()[:15]}")
    print("\nnumeric describe:\n", df.describe().T.to_string())

# -----------------------------------------------------------------------------
# PHASE 3 — LOAD + CLEAN
# -----------------------------------------------------------------------------
def build_customers(df, resolved, reference_date):
    df = df.drop_duplicates().copy()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    
    n = len(df)
    cust = pd.DataFrame()

    # ext_id
    ext_col = resolved["ext_id"]
    cust["ext_id"] = df[ext_col].astype(str) if ext_col else df.index.astype(str)

    # churn_label
    churn_col = resolved["churn"]
    if churn_col is None:
        raise ValueError("Could not resolve 'churn' column.")
    
    def map_churn(val):
        if pd.isna(val): return None
        v = str(val).lower().strip()
        if v in ["yes", "y", "true", "1", "churned", "exited"]: return 1
        if v in ["no", "n", "false", "0", "active", "retained"]: return 0
        try:
            return int(float(v))
        except:
            raise ValueError(f"Unmappable churn value: {val}")
            
    cust["churn_label"] = df[churn_col].apply(map_churn)
    if not cust["churn_label"].isin([0, 1]).all():
        raise ValueError("Unmapped churn values exist.")

    # plan_tier
    plan_col = resolved["plan_tier"]
    if plan_col:
        def map_plan(val):
            if pd.isna(val): return "Standard"
            v = str(val).lower()
            if "basic" in v or "month" in v or "short" in v: return "Basic"
            if "premium" in v or "pro" in v or "plus" in v or "two" in v: return "Premium"
            return "Standard"
        cust["plan_tier"] = df[plan_col].apply(map_plan)
    else:
        # derive from monthly charges if missing
        mc_col = resolved["monthly_charges"]
        if mc_col:
            terciles = pd.qcut(df[mc_col], 3, labels=["Basic", "Standard", "Premium"])
            cust["plan_tier"] = terciles
        else:
            cust["plan_tier"] = "Standard"
            
    cust["plan_tier_ord"] = cust["plan_tier"].map({"Basic": 0, "Standard": 1, "Premium": 2})

    # tenure_days
    tenure_col = resolved["tenure"]
    if tenure_col:
        t_vals = df[tenure_col].fillna(df[tenure_col].median())
        cname = tenure_col.lower()
        if "day" in cname:
            t_days = t_vals
        elif "month" in cname or t_vals.max() < 120:
            t_days = t_vals * 30
        elif "year" in cname or t_vals.max() < 15:
            t_days = t_vals * 365
        else:
            t_days = t_vals
        cust["tenure_days"] = t_days.clip(lower=1, upper=4000).astype(int)
    else:
        cust["tenure_days"] = 365

    # monthly_charges & arr
    mc_col = resolved["monthly_charges"]
    if mc_col:
        cust["monthly_charges"] = df[mc_col].fillna(df[mc_col].median()).astype(float)
    else:
        prices = {"Basic": 9.99, "Standard": 15.49, "Premium": 19.99}
        cust["monthly_charges"] = cust["plan_tier"].map(prices)
    cust["arr"] = cust["monthly_charges"] * 12

    # signup_date and renewal_date
    su_col = resolved["signup_date"]
    if su_col:
        cust["signup_date"] = pd.to_datetime(df[su_col]).dt.strftime("%Y-%m-%d")
        # Compute exact tenure days to avoid mismatch
        actual_signup = pd.to_datetime(df[su_col])
        cust["tenure_days"] = (reference_date - actual_signup).dt.days.clip(lower=1)
    else:
        cust["signup_date"] = (reference_date - pd.to_timedelta(cust["tenure_days"], unit='D')).dt.strftime("%Y-%m-%d")

    def calc_renewal(row):
        period = 30 if row["plan_tier"] == "Basic" else 365
        t = row["tenure_days"]
        periods = np.floor(t / period) + 1
        days_to_add = periods * period
        su_date = pd.to_datetime(row["signup_date"])
        return (su_date + pd.Timedelta(days=days_to_add)).strftime("%Y-%m-%d")
        
    cust["renewal_date"] = cust.apply(calc_renewal, axis=1)

    # Simple numerics
    cust["num_devices"] = df[resolved["num_devices"]].fillna(1).clip(1, 10).astype(int) if resolved["num_devices"] else 1
    cust["age"] = df[resolved["age"]].astype("Int64") if resolved["age"] else pd.NA
    cust["gender"] = df[resolved["gender"]] if resolved["gender"] else None
    cust["region"] = df[resolved["region"]] if resolved["region"] else None
    cust["payment_failures"] = df[resolved["payment_failures"]].fillna(0).clip(0, 20).astype(int) if resolved["payment_failures"] else 0
    cust["support_contacts"] = df[resolved["support_contacts"]].fillna(0).clip(0, 50).astype(int) if resolved["support_contacts"] else 0

    # usage_level
    usage_col = resolved["usage"]
    rng = np.random.default_rng(SEED)
    
    eng_col = resolved["engagement"]
    if eng_col:
        e_vals = df[eng_col].fillna(df[eng_col].median())
        cust["engagement_score"] = 100 * (e_vals - e_vals.min()) / (e_vals.max() - e_vals.min() + 1e-9)
    else:
        cust["engagement_score"] = rng.uniform(0, 100, size=n)

    if usage_col:
        cust["usage_level"] = df[usage_col].fillna(df[usage_col].median()).astype(float)
    else:
        z_eng = (cust["engagement_score"] - cust["engagement_score"].mean()) / cust["engagement_score"].std()
        cust["usage_level"] = 300 * (1 / (1 + np.exp(-z_eng))) + rng.normal(0, 20, size=n)
        cust["usage_level"] = cust["usage_level"].clip(lower=0)

    if not eng_col: # adjust engagement proxy if we generated it randomly before but we have usage
        u_pct = cust["usage_level"].rank(pct=True)
        cust["engagement_score"] = (u_pct * 100 * 0.7) + (rng.uniform(0, 100, size=n) * 0.3)

    # last_login_days
    ll_col = resolved["last_login"]
    if ll_col:
        if "week" in ll_col.lower():
            cust["last_login_days"] = (df[ll_col] * 7).fillna(0)
        elif df[ll_col].dtype == "object": # likely a date
            try:
                cust["last_login_days"] = (reference_date - pd.to_datetime(df[ll_col])).dt.days
            except:
                cust["last_login_days"] = df[ll_col].fillna(0)
        else:
            cust["last_login_days"] = df[ll_col].fillna(0)
    else:
        base = rng.gamma(shape=2, scale=4, size=n)
        shift = cust["churn_label"] * 2.0
        cust["last_login_days"] = (base * (1 + shift)).clip(0, 365).astype(int)

    # ID/Names
    FIRST = ["Alex","Sam","Jordan","Riya","Neha","Arjun","Maya","Dev","Priya","Rahul","Zoe","Ken","Ana","Leo","Ivy","Omar", "Elena", "Liam"]
    LAST  = ["Sharma","Patel","Khan","Rao","Iyer","Nair","Bose","Das","Mehta","Singh","Roy","Jain","Kapoor","Reddy", "Smith", "Johnson"]
    
    names = []
    emails = []
    for i in range(n):
        f = rng.choice(FIRST)
        l = rng.choice(LAST)
        names.append(f"{f} {l} {cust['ext_id'].iloc[i][-4:] if len(cust['ext_id'].iloc[i]) >=4 else i}")
        emails.append(f"{f.lower()}.{l.lower()}{i}@example.com")
        
    cust["name"] = names
    cust["email"] = emails
    
    cust["id"] = range(1, n + 1)
    
    # Reorder columns
    cols = ["id", "ext_id", "name", "email", "plan_tier", "plan_tier_ord", "tenure_days", "signup_date", 
            "renewal_date", "num_devices", "monthly_charges", "arr", "age", "gender", "region", 
            "payment_failures", "support_contacts", "engagement_score", "last_login_days", 
            "usage_level", "churn_label"]
    cust = cust[cols]
    
    return cust

# -----------------------------------------------------------------------------
# PHASE 4 — DEMO SUBSET
# -----------------------------------------------------------------------------
def mark_demo_subset(cust, demo_size):
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    risk = (z(cust["last_login_days"]) * 0.3 + 
            z(cust["payment_failures"]) * 0.25 + 
            z(cust["support_contacts"]) * 0.2 - 
            z(cust["engagement_score"]) * 0.15 - 
            z(cust["usage_level"]) * 0.10)
            
    quartiles = pd.qcut(risk, 4, labels=["healthy", "watch", "at_risk", "critical"])
    cust["_bucket"] = quartiles
    
    demo_indices = []
    per_bucket = demo_size // 4
    
    rng = np.random.default_rng(SEED)
    
    for bucket in ["healthy", "watch", "at_risk", "critical"]:
        b_df = cust[cust["_bucket"] == bucket]
        
        c0 = b_df[b_df["churn_label"] == 0]
        c1 = b_df[b_df["churn_label"] == 1]
        
        # Try to stratify
        pick = []
        if len(c0) > 0 and len(c1) > 0:
            half = per_bucket // 2
            p0 = rng.choice(c0.index, min(half, len(c0)), replace=False).tolist()
            p1 = rng.choice(c1.index, min(per_bucket - len(p0), len(c1)), replace=False).tolist()
            pick = p0 + p1
        else:
            pick = rng.choice(b_df.index, min(per_bucket, len(b_df)), replace=False).tolist()
            
        demo_indices.extend(pick)
        
    # pad if short
    if len(demo_indices) < demo_size:
        rem = set(cust.index) - set(demo_indices)
        demo_indices.extend(rng.choice(list(rem), min(demo_size - len(demo_indices), len(rem)), replace=False).tolist())
        
    cust["is_demo"] = 0
    cust.loc[demo_indices, "is_demo"] = 1
    
    bucket_map = cust["_bucket"].to_dict()
    cust = cust.drop(columns=["_bucket"])
    return cust, bucket_map

# -----------------------------------------------------------------------------
# PHASE 5 — SYNTHESIZE usage_daily
# -----------------------------------------------------------------------------
def build_usage_daily(cust, bucket_map, reference_date):
    rows = []
    for _, row in cust.iterrows():
        cid = row["id"]
        cint = int(cid)
        rng = np.random.default_rng(SEED + cint)
        
        mean_daily = max(row["usage_level"] / 30.0, 1.0)
        
        bucket = bucket_map[row.name]
        if row["churn_label"] == 1 or bucket in ("critical", "at_risk"):
            slope = rng.uniform(-0.06, -0.03)
        elif bucket == "watch":
            slope = rng.uniform(-0.02, 0.0)
        else:
            slope = rng.uniform(-0.005, 0.01)
            
        ll_days = row["last_login_days"]
        
        for t in range(USAGE_HISTORY_DAYS):
            days_ago = USAGE_HISTORY_DAYS - 1 - t
            factor = 1 + slope * days_ago
            
            active_minutes = max(0, mean_daily * factor * rng.normal(1, 0.25))
            
            if days_ago < ll_days:
                active_minutes = 0
                
            sessions = 0 if active_minutes == 0 else max(1, round(active_minutes/35 + rng.normal(0,0.5)))
            logins = 0 if sessions == 0 else max(1, round(sessions * rng.uniform(0.6, 1.0)))
            feature_events = 0 if sessions == 0 else int(sessions * rng.integers(2, 9))
            
            d = (reference_date - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            rows.append({
                "customer_id": cid,
                "date": d,
                "active_minutes": float(active_minutes),
                "logins": int(logins),
                "sessions": int(sessions),
                "feature_events": int(feature_events)
            })
            
    usage = pd.DataFrame(rows)
    usage["id"] = range(1, len(usage) + 1)
    return usage[["id", "customer_id", "date", "active_minutes", "logins", "sessions", "feature_events"]]

# -----------------------------------------------------------------------------
# PHASE 6 — SYNTHESIZE feedback
# -----------------------------------------------------------------------------
SUPPORT_TEMPLATES = [
    "Contacted support about {topic}; issue still not resolved after {days} days.",
    "Third time reporting {topic}. Considering cancelling my {plan} plan.",
    "Payment failed again and I was charged twice. Very frustrating.",
    "The {topic} feature is completely broken on my end.",
    "Waiting for a response on my {topic} ticket.",
    "Horrible experience with {topic}. Fix this."
]
NPS_POS = [
    "Great catalog, works well on all my devices.",
    "Really enjoying the {plan} plan so far.",
    "Support was very helpful.",
    "Love the app, use it every day."
]
NPS_NEG = [
    "Used to love it, but I barely open the app now. Not sure it's worth {price}/mo.",
    "Too many bugs lately.",
    "Price keeps going up but the quality isn't.",
    "Very disappointed with the recent update."
]

def build_feedback(cust, bucket_map, reference_date):
    rows = []
    
    for _, row in cust.iterrows():
        cid = row["id"]
        cint = int(cid)
        rng = np.random.default_rng(SEED + cint + 99)
        
        bucket = bucket_map[row.name]
        
        n_support = min(row["support_contacts"], 3)
        pf = row["payment_failures"]
        
        for k in range(n_support):
            is_complaint = 1
            sentiment = rng.uniform(-0.9, -0.4)
            topic = "billing" if pf > 0 and rng.random() < 0.6 else rng.choice(["playback issues", "app crashes", "content missing"])
            days = rng.integers(2, 10)
            text = rng.choice(SUPPORT_TEMPLATES).format(topic=topic, days=days, plan=row["plan_tier"])
            d = (reference_date - pd.Timedelta(days=rng.integers(1, 45))).strftime("%Y-%m-%d")
            
            rows.append({
                "customer_id": cid, "date": d, "channel": "support", 
                "text": text, "sentiment": sentiment, "is_complaint": is_complaint
            })
            
        if pf > 0 and rng.random() < 0.8:
            rows.append({
                "customer_id": cid, 
                "date": (reference_date - pd.Timedelta(days=rng.integers(1, 15))).strftime("%Y-%m-%d"),
                "channel": "support", "text": "Payment declined unexpectedly.",
                "sentiment": rng.uniform(-0.8, -0.6), "is_complaint": 1
            })
            
        if rng.random() < 0.5:
            if bucket == "healthy":
                sentiment = rng.uniform(0.3, 0.9)
                is_c = 0
                text = rng.choice(NPS_POS).format(plan=row["plan_tier"])
            elif bucket == "watch":
                sentiment = rng.uniform(-0.2, 0.4)
                is_c = 0
                if sentiment > 0: text = rng.choice(NPS_POS).format(plan=row["plan_tier"])
                else: text = rng.choice(NPS_NEG).format(price=row["monthly_charges"])
            else:
                sentiment = rng.uniform(-0.8, 0.0)
                is_c = 1 if sentiment < -0.3 else 0
                text = rng.choice(NPS_NEG).format(price=row["monthly_charges"])
                
            d = (reference_date - pd.Timedelta(days=rng.integers(1, 45))).strftime("%Y-%m-%d")
            rows.append({
                "customer_id": cid, "date": d, "channel": rng.choice(["nps", "review"]),
                "text": text, "sentiment": sentiment, "is_complaint": is_c
            })
            
    if len(rows) > 0:
        fb = pd.DataFrame(rows)
        fb["id"] = range(1, len(fb) + 1)
        return fb[["id", "customer_id", "date", "channel", "text", "sentiment", "is_complaint"]]
    else:
        return pd.DataFrame(columns=["id", "customer_id", "date", "channel", "text", "sentiment", "is_complaint"])

# -----------------------------------------------------------------------------
# PHASE 7 — WRITE TO SQLITE
# -----------------------------------------------------------------------------
def write_sqlite(customers, usage, feedback, resolved_map, reference_date):
    eng = create_engine(f"sqlite:///{DB}")
    
    with eng.begin() as cx:
        for t in ["usage_daily", "feedback", "analyses", "customers", "meta"]:
            cx.exec_driver_sql(f"DROP TABLE IF EXISTS {t}")
            
    customers.to_sql("customers", eng, index=False)
    usage.to_sql("usage_daily", eng, index=False)
    feedback.to_sql("feedback", eng, index=False)
    
    # Create empty analyses table
    with eng.begin() as cx:
        cx.exec_driver_sql("""
        CREATE TABLE analyses (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            analysis_date TEXT,
            risk_score REAL,
            drivers JSON,
            recommendation TEXT
        )
        """)
        
    meta = pd.DataFrame([
        {"key": "built_at",       "value": datetime.now(timezone.utc).isoformat()},
        {"key": "reference_date", "value": reference_date.strftime("%Y-%m-%d")},
        {"key": "n_customers",    "value": str(len(customers))},
        {"key": "n_demo",         "value": str(int(customers["is_demo"].sum()))},
        {"key": "churn_rate",     "value": f"{customers['churn_label'].mean():.3f}"},
        {"key": "column_map",     "value": json.dumps(resolved_map)},
        {"key": "seed",           "value": str(SEED)},
    ])
    meta.to_sql("meta", eng, index=False)
    
    with eng.begin() as cx:
        cx.exec_driver_sql("CREATE INDEX ix_usage_customer ON usage_daily(customer_id);")
        cx.exec_driver_sql("CREATE INDEX ix_feedback_customer ON feedback(customer_id);")
        cx.exec_driver_sql("CREATE INDEX ix_customers_demo ON customers(is_demo);")

# -----------------------------------------------------------------------------
# PHASE 8 — VALIDATE
# -----------------------------------------------------------------------------
def validate():
    eng = create_engine(f"sqlite:///{DB}")
    cust = pd.read_sql("SELECT * FROM customers", eng)
    usage = pd.read_sql("SELECT * FROM usage_daily", eng)
    fb = pd.read_sql("SELECT * FROM feedback", eng)
    meta = pd.read_sql("SELECT * FROM meta", eng).set_index("key")["value"].to_dict()
    
    ref_date = pd.to_datetime(meta["reference_date"])
    
    errors = []
    if len(cust) != cust["ext_id"].nunique():
        errors.append("Customer count does not match unique ext_id count.")
    if not cust["churn_label"].isin([0, 1]).all():
        errors.append("churn_label contains values other than 0 and 1.")
        
    usage_counts = usage.groupby("customer_id").size()
    if not usage_counts.isin([0, 30]).all():
        errors.append("Not all customers have 0 or 30 usage records.")
    
    demo_custs = cust[cust["is_demo"] == 1]["id"]
    for dc in demo_custs:
        if dc not in usage_counts or usage_counts[dc] != 30:
            errors.append(f"Demo customer {dc} does not have 30 usage records.")
            
    if (usage["active_minutes"] < 0).any():
        errors.append("Negative active_minutes found.")
    if usage.isna().any().any():
        errors.append("NaN found in usage_daily.")
        
    if len(fb) > 0 and (fb["sentiment"] < -1).any() or (fb["sentiment"] > 1).any():
        errors.append("Sentiment out of bounds [-1, 1].")
        
    print("\n--- VALIDATION REPORT ---")
    print("Plan Tier Distribution:")
    print(cust["plan_tier"].value_counts(normalize=True))
    
    print(f"\nChurn Rate: {cust['churn_label'].mean():.3f}")
    
    ren_dates = pd.to_datetime(cust["renewal_date"])
    if (ren_dates <= ref_date).any():
        errors.append("Some renewal dates are in the past.")
        
    if len(errors) > 0:
        print("\nERRORS FOUND:")
        for e in errors: print(f" - {e}")
        sys.exit(1)
        
    print("\nSample Data:")
    try:
        h_id = cust[cust["is_demo"] == 1].sort_values("last_login_days").iloc[0]["id"]
        c_id = cust[(cust["is_demo"] == 1) & (cust["churn_label"] == 1)].sort_values("last_login_days", ascending=False).iloc[0]["id"]
        
        for name, cid in [("Healthy", h_id), ("Critical", c_id)]:
            print(f"\n[{name} Customer {cid}]")
            cu = usage[usage["customer_id"] == cid].sort_values("date")
            print(f"Usage: first 5 days mean = {cu['active_minutes'].head(5).mean():.1f}, last 5 days mean = {cu['active_minutes'].tail(5).mean():.1f}")
            cfb = fb[fb["customer_id"] == cid]
            if len(cfb) > 0:
                print("Feedback:")
                for _, r in cfb.iterrows():
                    print(f"  - {r['date']} ({r['sentiment']:.2f}): {r['text']}")
    except Exception as e:
        print("Could not print sample:", e)

    print("\nREADY FOR STEP 2")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    
    load_dotenv()
    demo_size = int(os.environ.get("DEMO_SUBSET_SIZE", DEMO_SUBSET_SIZE))
    
    if args.inspect:
        inspect()
        return
        
    if args.validate:
        validate()
        return
        
    if args.rebuild and os.path.exists(DB):
        os.remove(DB)
        
    print("Loading CSV...")
    df = pd.read_csv(RAW)
    resolved = {k: resolve(df, k) for k in CANDIDATES}
    
    reference_date = pd.to_datetime("2026-08-30")
    
    print("Building customers...")
    customers = build_customers(df, resolved, reference_date)
    customers, bucket_map = mark_demo_subset(customers, demo_size)
    
    print("Synthesizing usage...")
    usage = build_usage_daily(customers, bucket_map, reference_date)
    
    print("Synthesizing feedback...")
    feedback = build_feedback(customers, bucket_map, reference_date)
    
    print("Writing to SQLite...")
    write_sqlite(customers, usage, feedback, resolved, reference_date)
    
    print("Validating...")
    validate()

if __name__ == "__main__":
    main()
