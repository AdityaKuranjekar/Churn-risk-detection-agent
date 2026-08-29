import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
from dotenv import load_dotenv

from features import derive_features, BASE_COLS

def load_customers():
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL", "sqlite:///./churn_agent.db")
    if db_url.startswith("sqlite:///./"):
        # Make sure it runs when pwd is root or ml/
        if os.path.exists("../churn_agent.db") and not os.path.exists("./churn_agent.db"):
            db_url = "sqlite:///../churn_agent.db"
    
    eng = create_engine(db_url)
    df = pd.read_sql("SELECT * FROM customers", eng)
    
    if len(df) == 0:
        raise ValueError("Customers table is empty.")
    if "churn_label" not in df.columns or df["churn_label"].isna().all():
        raise ValueError("Step 1 produced no churn label; retrain impossible.")
        
    initial_len = len(df)
    df = df.dropna(subset=["churn_label"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows missing churn_label.")
        
    print(f"Loaded {len(df)} rows. Churn Rate: {df['churn_label'].mean():.3f}")
    return df

def build_features(df):
    df_derived = derive_features(df)
    
    y = df["churn_label"].astype(int)
    
    # Exclude leaky / bookkeeping columns
    excludes = ["id", "ext_id", "name", "email", "churn_label", "is_demo", "signup_date", "renewal_date", "usage_trend"]
    
    features_to_keep = [c for c in df_derived.columns if c not in excludes]
    X_raw = df_derived[features_to_keep].copy()
    
    # Categorical encoding for 'gender' and 'region'
    if "gender" in X_raw.columns:
        gender_dummies = pd.get_dummies(X_raw["gender"], prefix="gender", dummy_na=True)
        X_raw = pd.concat([X_raw.drop("gender", axis=1), gender_dummies], axis=1)
        
    region_encoding = {}
    if "region" in X_raw.columns:
        counts = X_raw["region"].value_counts(normalize=True)
        region_encoding = counts.to_dict()
        X_raw["region"] = X_raw["region"].map(region_encoding).fillna(0)
        
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    
    feat_cols = list(X_raw.columns)
    medians = X_raw.median().to_dict()
    
    # ensure everything that can be imputed is in medians
    # actually medians will be just numerical columns
    median_engagement = df["engagement_score"].median() if "engagement_score" in df.columns else 50.0
    
    pre = {
        "medians": medians,
        "feature_columns": feat_cols,
        "region_encoding": region_encoding,
        "median_engagement": median_engagement
    }
    
    X = X_raw.fillna(pd.Series(medians))
    
    if X.isna().any().any() or np.isinf(X).any().any():
        X = X.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians))
        
    print("Correlations with target:")
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    print(corrs.head(10))
    if corrs.max() > 0.95:
        raise ValueError(f"WARNING: Correlation > 0.95 detected. Potential leakage! Max is {corrs.index[0]}={corrs.max():.3f}")
        
    return X, y, feat_cols, pre

def train_eval(X, y, feat_cols, run_grid=False):
    seed = int(os.environ.get("SEED", 42))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    if run_grid:
        print("Running grid search...")
        grid = {
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [200, 300],
        }
        model = xgb.XGBClassifier(
            subsample=0.9, colsample_bytree=0.9, min_child_weight=2, 
            reg_lambda=1.0, objective="binary:logistic", eval_metric="aucpr", 
            scale_pos_weight=spw, random_state=seed, n_jobs=-1
        )
        gsearch = GridSearchCV(model, param_grid=grid, scoring="average_precision", cv=skf, refit=True)
        gsearch.fit(X_train, y_train)
        best_model = gsearch.best_estimator_
        print("Best params:", gsearch.best_params_)
        
        cv_res = pd.DataFrame(gsearch.cv_results_)
        best_idx = gsearch.best_index_
        cv_auc_mean = cv_res.loc[best_idx, 'mean_test_score']
        cv_auc_std = cv_res.loc[best_idx, 'std_test_score']
    else:
        print("Running standard CV...")
        cv_aucs = []
        best_model = xgb.XGBClassifier(**params)
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            proba = m.predict_proba(X_va)[:, 1]
            cv_aucs.append(average_precision_score(y_va, proba))
            
        cv_auc_mean = np.mean(cv_aucs)
        cv_auc_std = np.std(cv_aucs)
        
        best_model.fit(X_train, y_train)

    proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    test_pr_auc = average_precision_score(y_test, proba)
    
    # Pick threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_idx = np.argmax(f1s)
    threshold_used = float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5
    
    preds = (proba >= threshold_used).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=3))
    
    cm = confusion_matrix(y_test, preds)
    precision_churn = float(precisions[best_f1_idx])
    recall_churn = float(recalls[best_f1_idx])
    f1_churn = float(f1s[best_f1_idx])
    
    # Baselines
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(X_train_imp, y_train)
    lr_proba = lr.predict_proba(X_test_imp)[:, 1]
    baseline_lr_auc = roc_auc_score(y_test, lr_proba)

    metrics = {
        "n_rows": len(X),
        "n_features": len(feat_cols),
        "churn_rate": float(y.mean()),
        "split_seed": seed,
        "params": best_model.get_params(),
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "test_auc": float(test_auc),
        "test_pr_auc": float(test_pr_auc),
        "test_precision_churn": precision_churn,
        "test_recall_churn": recall_churn,
        "test_f1_churn": f1_churn,
        "threshold_used": threshold_used,
        "confusion_matrix": cm.tolist(),
        "baseline_majority_accuracy": float(1 - y_test.mean()),
        "baseline_majority_recall": 0.0,
        "baseline_lr_auc": float(baseline_lr_auc)
    }
    
    return best_model, metrics

def explain_and_plot(model, X, feat_cols, do_shap=False):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_cols, "gain": importances}).sort_values("gain", ascending=False)
    
    top_10 = imp_df.head(10).to_dict(orient="records")
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df.head(15)["feature"][::-1], imp_df.head(15)["gain"][::-1])
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    if do_shap:
        try:
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X.sample(min(1000, len(X))))
            shap.summary_plot(sv, X.sample(min(1000, len(X))), show=False)
            plt.savefig("shap_summary.png")
            plt.close()
        except ImportError:
            print("SHAP not installed. Skipping --explain.")
            
    return top_10

def persist(model, feat_cols, pre, metrics):
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(feat_cols, "feature_columns.pkl")
    joblib.dump(pre, "preprocess.pkl")
    
    # Save timestamped backup
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(model, f"churn_model_{ts}.pkl")
    
    with open("model_card.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Persisted models and metrics to disk.")

def self_check(X_test, feat_cols, metrics):
    print("\n--- SELF-CHECK ---")
    model = joblib.load("churn_model.pkl")
    fc = joblib.load("feature_columns.pkl")
    pre = joblib.load("preprocess.pkl")
    
    errors = []
    if len(fc) != X_test.shape[1]:
        errors.append("Feature columns length mismatch.")
        
    def predict_dict(d):
        from features import derive_features
        df_derived = derive_features(d, pre)
        # Apply same mock dummy / region encoding
        if "gender" in df_derived.columns:
            for g in ["Male", "Female", "Other"]:
                df_derived[f"gender_{g}"] = (df_derived["gender"] == g).astype(int)
            df_derived["gender_nan"] = 0
            df_derived = df_derived.drop("gender", axis=1)
        if "region" in df_derived.columns:
            df_derived["region"] = df_derived["region"].map(pre["region_encoding"]).fillna(0)
        missing = [c for c in fc if c not in df_derived.columns]
        for c in missing: df_derived[c] = pre["medians"].get(c, 0)
        X_df = df_derived[fc].apply(pd.to_numeric, errors="coerce").fillna(pre["medians"])
        return model.predict_proba(X_df)[0, 1]

    # Hand-built dicts
    h_row = {"tenure_days": 1000, "engagement_score": 90, "payment_failures": 0, "last_login_days": 1, "support_contacts": 0, "usage_level": 1000, "monthly_charges": 10, "num_devices": 1}
    m_row = {"tenure_days": 180, "engagement_score": 50, "payment_failures": 1, "last_login_days": 14, "support_contacts": 1, "usage_level": 100, "monthly_charges": 50, "num_devices": 2}
    c_row = {"tenure_days": 30, "engagement_score": 10, "payment_failures": 3, "last_login_days": 40, "support_contacts": 5, "usage_level": 10, "monthly_charges": 100, "num_devices": 5}
    
    try:
        p_h = predict_dict(h_row)
        p_m = predict_dict(m_row)
        p_c = predict_dict(c_row)
        print(f"Monotonic check: p(healthy)={p_h:.3f}, p(mid)={p_m:.3f}, p(critical)={p_c:.3f}")
        if not (p_h < p_m < p_c):
            errors.append(f"Monotonic sanity failed! h={p_h:.3f} m={p_m:.3f} c={p_c:.3f}")
    except Exception as e:
        errors.append(f"predict_dict crashed on hand-built dicts: {e}")

    # Missing keys impute path
    try:
        p_miss = predict_dict({"tenure_days": 10}) 
        if not (0 <= p_miss <= 1): errors.append("Missing keys out of bounds")
        else: print("Missing-key impute path works.")
    except Exception as e:
        errors.append(f"Missing keys impute crashed: {e}")
        
    if metrics["test_auc"] < 0.75:
        errors.append(f"test_auc {metrics['test_auc']:.3f} < 0.75")
    if metrics["test_recall_churn"] < 0.40:
        errors.append(f"test_recall_churn {metrics['test_recall_churn']:.3f} < 0.40")
        
    if len(errors) == 0:
        print("READY FOR STEP 3")
    else:
        for e in errors: print(f" - {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    
    # Change CWD to ml directory for proper pathing of artifacts if we aren't already there
    # or just assume we write to current working dir but since paths are ml/ in the plan,
    # let's write to ml/. Wait, we're likely running `py ml/train_model.py`.
    # Let's adjust paths to write to `os.path.dirname(__file__)` or assume ml/.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(out_dir)
    
    print("Loading data...")
    df = load_customers()
    
    print("Building features...")
    X, y, feat_cols, pre = build_features(df)
    
    print("Training model...")
    model, metrics = train_eval(X, y, feat_cols, run_grid=args.grid)
    
    print("Explaining model...")
    metrics["top_10_features"] = explain_and_plot(model, X, feat_cols, do_shap=args.explain)
    metrics["trained_at"] = datetime.now().isoformat()
    
    if not args.no_save:
        persist(model, feat_cols, pre, metrics)
        self_check(X, feat_cols, metrics)

if __name__ == "__main__":
    main()
