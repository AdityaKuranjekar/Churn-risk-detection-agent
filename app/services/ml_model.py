import joblib
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Mapping
from ml.features import derive_features, BASE_COLS

ML_DIR = Path(os.getenv("ML_DIR", "ml"))
MODEL_PATH = ML_DIR / "churn_model.pkl"
COLS_PATH = ML_DIR / "feature_columns.pkl"
PRE_PATH = ML_DIR / "preprocess.pkl"
CARD_PATH = ML_DIR / "model_card.json"

class ModelNotLoaded(RuntimeError): ...

def _load():
    if not MODEL_PATH.exists():
        raise ModelNotLoaded(f"{MODEL_PATH} missing - run `py ml/train_model.py` first")
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
    pre = joblib.load(PRE_PATH)
    card = json.loads(CARD_PATH.read_text()) if CARD_PATH.exists() else {}
    return model, cols, pre, card

_MODEL, _COLS, _PRE, _CARD = _load()
_MEDIANS = _PRE.get("medians", {})
_THRESH = float(_CARD.get("threshold_used", os.getenv("CHURN_THRESHOLD", 0.5)))
_VERSION = _CARD.get("trained_at", "unknown")

try:
    import shap
    _EXPL = shap.TreeExplainer(_MODEL)
    _HAS_SHAP = True
except ImportError:
    _EXPL = None
    _HAS_SHAP = False

def _num(v, default):
    try:
        if v is None or pd.isna(v) or v == '':
            return default
        return float(v)
    except (ValueError, TypeError):
        return default

def build_feature_dict(row) -> dict:
    g = (lambda k, d=None: row.get(k, d)) if isinstance(row, Mapping) else (lambda k, d=None: getattr(row, k, d))
    
    plan_tier_val = g("plan_tier_ord")
    if plan_tier_val is None or pd.isna(plan_tier_val):
        plan_tier_val = {"Basic": 0, "Standard": 1, "Premium": 2}.get(g("plan_tier"), 1)
    
    fd = {
        "tenure_days": _num(g("tenure_days"), 0),
        "monthly_charges": _num(g("monthly_charges"), 0.0),
        "arr": _num(g("arr"), _num(g("monthly_charges"), 0) * 12),
        "plan_tier_ord": int(_num(plan_tier_val, 1)),
        "num_devices": int(_num(g("num_devices"), 1)),
        "age": g("age"),
        "payment_failures": int(_num(g("payment_failures"), 0)),
        "support_contacts": int(_num(g("support_contacts"), 0)),
        "engagement_score": _num(g("engagement_score"), None),
        "last_login_days": int(_num(g("last_login_days"), 0)),
        "usage_level": _num(g("usage_level"), 0.0),
    }
    
    # Ensure all BASE_COLS are covered
    for col in BASE_COLS:
        if col not in fd:
            fd[col] = _num(g(col), 0.0)
            
    return fd

def _vectorize(feature_dict: dict) -> pd.DataFrame:
    df = derive_features(feature_dict)
    for c in _COLS:
        if c not in df.columns:
            df[c] = _MEDIANS.get(c, 0.0)
    df = df[_COLS]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(pd.Series(_MEDIANS)).fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df

def predict(feature_dict: dict) -> float:
    X = _vectorize(feature_dict)
    p = float(_MODEL.predict_proba(X)[0, 1])
    return min(1.0, max(0.0, p))

def predict_many(feature_dicts: list[dict]) -> list[float]:
    if not feature_dicts: return []
    dfs = [_vectorize(fd) for fd in feature_dicts]
    X = pd.concat(dfs, ignore_index=True)
    probs = _MODEL.predict_proba(X)[:, 1]
    return [min(1.0, max(0.0, float(p))) for p in probs]

def band(prob: float) -> str:
    if prob >= 0.80: return "critical"
    if prob >= _THRESH: return "at_risk"
    if prob >= 0.30: return "watch"
    return "healthy"

def explain(feature_dict: dict, top_n=5) -> list[dict]:
    X = _vectorize(feature_dict)
    
    if _HAS_SHAP and _EXPL:
        sv = _EXPL.shap_values(X)[0]
        pairs = sorted(zip(_COLS, X.iloc[0].tolist(), sv),
                       key=lambda t: abs(t[2]), reverse=True)[:top_n]
        method = "shap"
    else:
        importances = _MODEL.feature_importances_
        row_vals = X.iloc[0].tolist()
        medians = [_MEDIANS.get(c, 0.0) for c in _COLS]
        
        # Approximate contribution: importance * (val - median)
        contribs = [imp * (val - med) if not pd.isna(val) else 0.0 for imp, val, med in zip(importances, row_vals, medians)]
        pairs = sorted(zip(_COLS, row_vals, contribs),
                       key=lambda t: abs(t[2]), reverse=True)[:top_n]
        method = "importance_deviation"
        
    return [
        {"feature": f, "value": float(v) if not pd.isna(v) else 0.0, "contribution": float(c) if not pd.isna(c) else 0.0, "method": method}
        for f, v, c in pairs
    ]

def predict_from_customer(customer_row) -> dict:
    fd = build_feature_dict(customer_row)
    p = predict(fd)
    return {
        "churn_probability": round(p, 4),
        "band": band(p),
        "threshold_used": _THRESH,
        "top_features": explain(fd, top_n=5),
        "model_version": _VERSION,
    }

def health() -> dict:
    return {
        "loaded": True,
        "n_features": len(_COLS),
        "model_version": _VERSION,
        "threshold": _THRESH,
        "test_auc": _CARD.get("test_auc"),
        "test_pr_auc": _CARD.get("test_pr_auc")
    }
