import pandas as pd
import numpy as np

BASE_COLS = [
    "tenure_days", "monthly_charges", "arr", "plan_tier_ord", "num_devices",
    "age", "payment_failures", "support_contacts", "engagement_score",
    "last_login_days", "usage_level"
]

def derive_features(df_or_dict, preprocess_stats=None):
    """
    Takes a DataFrame or a dict and returns a DataFrame with derived features.
    If preprocess_stats is provided, it uses medians for imputation.
    """
    if isinstance(df_or_dict, dict):
        # Apply medians if provided
        if preprocess_stats and "medians" in preprocess_stats:
            r = {c: df_or_dict.get(c, preprocess_stats["medians"].get(c)) for c in BASE_COLS}
        else:
            r = {c: df_or_dict.get(c) for c in BASE_COLS}
        # Include raw region/gender if present in dict
        if "region" in df_or_dict: r["region"] = df_or_dict["region"]
        if "gender" in df_or_dict: r["gender"] = df_or_dict["gender"]
        df = pd.DataFrame([r])
        for c in BASE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df = df_or_dict.copy()

    # Derived Features
    df["usage_per_tenure"] = df["usage_level"] / (df["tenure_days"] + 1)
    df["charges_per_device"] = df["monthly_charges"] / df["num_devices"].replace(0, 1) # Avoid div by zero just in case
    df["arr_to_usage"] = df["arr"] / (df["usage_level"] + 1)
    
    # tenure_bucket (ordinal based on predefined cuts)
    df["tenure_bucket"] = pd.cut(df["tenure_days"], bins=[0, 90, 365, 730, 4000], labels=[0, 1, 2, 3], right=False)
    df["tenure_bucket"] = df["tenure_bucket"].astype(float)
    
    # Boolean flags as int
    df["is_month_to_month"] = (df["plan_tier_ord"] == 0).astype(int)
    df["high_support_flag"] = (df["support_contacts"] >= 2).astype(int)
    df["payment_risk_flag"] = (df["payment_failures"] >= 1).astype(int)
    df["recent_inactivity_flag"] = (df["last_login_days"] >= 21).astype(int)
    
    median_engagement = preprocess_stats["median_engagement"] if preprocess_stats and "median_engagement" in preprocess_stats else df["engagement_score"].median()
    if pd.isna(median_engagement):
        median_engagement = 50.0
    df["low_engagement_flag"] = (df["engagement_score"] < median_engagement).astype(int)
    
    return df
