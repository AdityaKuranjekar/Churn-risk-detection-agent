# Churn Risk Agent — End-to-End Implementation Plan

**Problem:** Customer Success teams manage hundreds of accounts and usually discover a
customer is unhappy only *after* they decide to cancel. This project builds an AI agent
that analyzes usage, support interactions, feedback, and account activity to:

1. Identify customers at risk of churn (trained ML model + explainable rules)
2. Explain the warning signals (deterministic risk breakdown)
3. Recommend personalized retention actions (RAG-grounded LLM), with a CSM approve/edit flow

**Build stance:** One Python process, one `uvicorn` command, nothing external to debug live.
The orchestration layer is a plain Python module (`orchestrator.py`) that mirrors what would
be an n8n workflow in production — kept in-process for hackathon reliability.

Dataset: https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset

---

## 1. Final Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        FRONTEND (SPA)                       │
│              frontend/index.html (Tailwind CDN)             │
└───────────────────────────┬────────────────────────────────┘
                            │ REST (fetch)
┌───────────────────────────▼────────────────────────────────┐
│                        FASTAPI APP (app/main.py)           │
│                                                            │
│  routers/customers.py  → GET  /customers                   │
│                          GET  /customers/{id}              │
│  routers/analysis.py   → POST /customers/{id}/analyze      │
│                          POST /customers/{id}/approve      │
│                                                            │
│  services/                                                 │
│    scoring.py        — deterministic health/risk breakdown │
│    ml_model.py       — loads trained XGBoost, predict()    │
│    rag.py            — embeds + retrieves playbook chunks  │
│    action_catalog.py — pick_base_action(signals, prob)     │
│    llm.py            — one structured LLM call             │
│    orchestrator.py   — THE PIPELINE (replaces n8n)         │
│                                                            │
│  db.py / models.py  — SQLAlchemy + SQLite                  │
└────────────────────────────────────────────────────────────┘

Offline (built once, before serving):
  load_dataset.py  → cleans Kaggle CSV → SQLite tables
  ml/train_model.py → churn_model.pkl + feature_columns.pkl
  rag build step    → playbook_embeddings.pkl
```

### The pipeline (orchestrator.analyze_customer)

```python
def analyze_customer(customer_id: int) -> dict:
    signals     = get_signals(customer_id)                       # from DB
    prob        = ml_model.predict(signals)                      # trained XGBoost
    breakdown   = scoring.risk_breakdown(signals)                # explainable factors
    base_action = action_catalog.pick_base_action(signals, prob) # (action, priority)

    if prob < THRESHOLD:                                         # e.g. 0.50
        result = low_risk_result(prob, breakdown, base_action)
        save_analysis(customer_id, result)
        return result

    playbook = rag.retrieve(query_from(signals, base_action), top_k=1)
    result   = llm.analyze(signals, prob, breakdown, playbook, base_action)
    result["escalate"] = (result["priority"] == "P0")
    save_analysis(customer_id, result)
    return result
```

---

## 2. File Structure

```
churn-agent/
├── data/
│   ├── customer_churn_raw.csv            # uploaded Kaggle file
│   ├── playbook.json              # 8–12 retention playbook snippets
│   └── playbook_embeddings.pkl    # generated, cached
├── ml/
│   ├── train_model.py
│   ├── churn_model.pkl            # generated
│   └── feature_columns.pkl       # generated
├── app/
│   ├── main.py                   # FastAPI app, mounts routers, CORS, static frontend
│   ├── config.py                 # thresholds, paths, model/env keys
│   ├── db.py                     # engine, SessionLocal, get_db()
│   ├── models.py                 # SQLAlchemy: Customer, UsageDaily, Feedback, Analysis
│   ├── schemas.py                # Pydantic request/response
│   ├── routers/
│   │   ├── customers.py
│   │   └── analysis.py
│   └── services/
│       ├── scoring.py
│       ├── ml_model.py
│       ├── rag.py
│       ├── action_catalog.py
│       ├── llm.py
│       └── orchestrator.py
├── load_dataset.py               # cleans CSV → SQLite
├── requirements.txt
├── .env                          # ANTHROPIC_API_KEY / OPENAI_API_KEY
└── frontend/
    └── index.html                # single-file SPA
```

### requirements.txt

```
fastapi
uvicorn[standard]
sqlalchemy
pydantic
pandas
numpy
scikit-learn
xgboost
joblib
python-dotenv
google-generativeai   # Gemini LLM
httpx                 # calls Ollama embeddings endpoint
```

### Prerequisites / one-time setup

| Thing | Command / action |
|---|---|
| Python 3.10+ | `python --version` |
| Dataset CSV | download from Kaggle → `data/customer_churn_raw.csv` (see `data/README.md`) |
| Gemini API key | paste into `.env` → `GEMINI_API_KEY` (key from https://aistudio.google.com/app/apikey) |
| Ollama installed | https://ollama.com |
| Embedding model pulled | `ollama pull nomic-embed-text` |
| Ollama running | `ollama serve` (usually auto-starts; listens on `http://localhost:11434`) |
| Deps | `pip install -r requirements.txt` |

`.env` is already created at the project root with all keys/placeholders.

---

## 3. Data Model (SQLite via SQLAlchemy)

| Table        | Key columns |
|--------------|-------------|
| `customers`  | `id`, `name`, `email`, `plan_tier` (Basic/Standard/Premium), `tenure_days`, `num_devices`, `num_profiles`, `arr`, `renewal_date`, `churn_label` (0/1, training only), `signup_date` |
| `usage_daily`| `id`, `customer_id` (FK), `date`, `active_minutes`, `logins`, `sessions`, `feature_events` |
| `feedback`   | `id`, `customer_id` (FK), `date`, `channel` (support/nps/review), `text`, `sentiment` (-1..1), `is_complaint` (bool) |
| `analyses`   | `id`, `customer_id` (FK), `created_at`, `churn_probability`, `health_score`, `risk_breakdown` (JSON), `summary`, `top_reasons` (JSON), `recommended_action`, `priority`, `draft_message`, `playbook_id`, `status` (new/approved/edited/dismissed), `approved_message`, `escalate` (bool) |

> The Kaggle dataset is a **flat per-customer table**. `load_dataset.py` maps its real
> columns into `customers`, and **synthesizes** `usage_daily` / `feedback` rows from the
> flat features (e.g. spread total product usage across a declining 30-day trend, generate a
> complaint row when a support/complaint flag is set). Inspect the actual CSV headers on
> first load and adjust the mapping — do not hardcode column names blindly.

---

## 4. Step-by-Step Build Order

### Step 1 — Data ingestion · `load_dataset.py` · 45–60 min

1. `pd.read_csv("data/customer_churn_raw.csv")`, print `df.columns`, `df.head()`, `df.describe()`,
   `df.isna().sum()`. Confirm the churn label column name and its encoding.
2. Clean: drop/impute nulls, strip whitespace, normalize categorical values.
3. Encode categoricals: `plan_tier` ordinal (Basic=0, Standard=1, Premium=2); device type
   one-hot or count.
4. Synthesize where genuinely absent:
   - `renewal_date` = `signup_date` + tenure rounded to plan period (2-line pandas)
   - `arr` = plan price × 12
   - `usage_daily`: 30 rows/customer; if a watch-time trend feature exists use it, else
     model a mild decline for churn=1 and flat for churn=0 around the customer's mean
   - `feedback`: emit 1–3 rows when a complaint/support flag is present; sentiment derived
     from the flag / any rating column
5. Build a **demo subset of ~40 customers** stratified by `churn_label` × engagement
   percentile so the dashboard spans healthy → critical. Keep the **full cleaned dataset**
   for training.
6. Write everything to `churn_agent.db` (SQLite). Tables: `customers`, `usage_daily`,
   `feedback`. Idempotent: drop & recreate on rerun.

**Done when:** `SELECT count(*) FROM customers` and a spot-check of one customer's usage +
feedback rows look sane.

### Step 2 — Train ML model · `ml/train_model.py` · 30–45 min

Feature engineering (per customer, from the flat dataset):

| Feature | Source |
|---|---|
| `tenure_days` | direct |
| `usage_level` (current) | direct (engagement/usage column) |
| `usage_trend` | slope of last-N usage, or `usage_level / tenure_days` proxy |
| `days_since_last_login` | direct or derived from last-login date |
| `plan_tier` | ordinal encoded |
| `num_devices`, `num_profiles` | direct |
| `payment_failures` | direct if present, else 0 |
| `support_contacts` / `is_complaint` | direct if present |
| `engagement_score` | direct if present |

```python
import xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pos = y_train.sum(); neg = len(y_train) - pos
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=neg / max(pos, 1), eval_metric="auc",
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "ml/churn_model.pkl")
joblib.dump(list(X.columns), "ml/feature_columns.pkl")   # exact order for inference
```

If imbalance is severe and `scale_pos_weight` isn't enough, oversample the minority class
on the **training split only**.

**Done when:** `churn_model.pkl` + `feature_columns.pkl` exist and the printed
AUC/precision/recall are captured for slides (real evidence of "actual ML").

### Step 3 — Scoring + action catalog · 45 min

`services/scoring.py`:

```python
WEIGHTS = {"usage": 0.35, "support": 0.30, "sentiment": 0.20, "tenure_renewal": 0.15}

def risk_breakdown(signals: dict) -> dict:
    """Returns {health_score: 0-100, contributors: [{factor, impact, detail}]}"""
    contributors = []
    # usage: declining watch time / logins → negative impact
    # support: complaint volume, unresolved tickets → negative
    # sentiment: avg feedback sentiment < 0 → negative
    # tenure_renewal: near renewal + low engagement → negative
    # each contributor: impact is a signed number (points off 100)
    health = round(100 + sum(c["impact"] for c in contributors))
    health = max(0, min(100, health))
    return {"health_score": health, "contributors": sorted(
        contributors, key=lambda c: c["impact"])}
```

Keep it **fully deterministic and explainable** — this is the "why" panel. It runs
independently of the ML probability.

`services/action_catalog.py`:

```python
def pick_base_action(signals: dict, churn_prob: float) -> dict:
    """Rules table → constrained action + priority. Single source of truth."""
    # e.g.
    # 2+ payment failures in 30d           → ("billing_fix_outreach", "P0")
    # high complaint/support volume        → ("live_support_callback", "P0")
    # declining watch time, no complaints  → ("curated_content_email", "P2")
    # near renewal + low engagement        → ("offer_downgrade_option", "P1")
    # healthy                              → ("monitor", "P3")
    return {"action": ..., "priority": ...}
```

The LLM must not invent actions — it is *constrained* to elaborate on `base_action`.

### Step 4 — RAG · `services/rag.py` · 45–60 min

1. Author **8–12 short retention playbook snippets** in `data/playbook.json`
   (`[{id, title, text}]`). Fictional but specific, e.g.:
   - Declining watch time, no complaints → curated content recommendation email for their
     top genre.
   - 2+ payment failures in 30 days → billing-fix outreach with a one-month discount,
     *before* any content intervention.
   - Approaching renewal with reduced engagement → present a downgrade option before
     cancellation to retain partial revenue.
   - High complaint/support volume → escalate to a live support callback, not an automated
     email.
2. Embed all chunks once at startup **via Ollama** (`nomic-embed-text`, 768-dim); **cache
   vectors to `data/playbook_embeddings.pkl`** (skip re-embedding on restart).
3. `retrieve(query_text, top_k=1)` — cosine similarity over an in-memory numpy array. No
   vector DB.

```python
import httpx, numpy as np, os

_OLLAMA = os.environ["OLLAMA_BASE_URL"]
_EMBED_MODEL = os.environ["OLLAMA_EMBED_MODEL"]

def embed(text: str) -> np.ndarray:
    r = httpx.post(f"{_OLLAMA}/api/embeddings",
                   json={"model": _EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=float)

def retrieve(query_text: str, top_k: int = 1) -> list[dict]:
    q = embed(query_text)
    sims = playbook_vectors @ q / (
        np.linalg.norm(playbook_vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    idx = np.argsort(sims)[::-1][:top_k]
    return [playbook_chunks[i] for i in idx]
```

Query is built from the dominant signal + `base_action`, e.g.
`"premium customer, declining watch time, no complaints, near renewal"`.

### Step 5 — ML serving wrapper · `services/ml_model.py` · 15 min

```python
import joblib, numpy as np

_model = joblib.load("ml/churn_model.pkl")
_cols  = joblib.load("ml/feature_columns.pkl")

def predict(feature_dict: dict) -> float:
    row = np.array([[feature_dict.get(c, 0) for c in _cols]], dtype=float)
    return float(_model.predict_proba(row)[0, 1])
```

Loaded once at import. `feature_dict` is produced from a customer's signals by the same
feature-engineering helper used in training (share the function).

### Step 6 — LLM call · `services/llm.py` · 20–30 min

```python
def analyze(signals, churn_prob, breakdown, playbook_snippet, base_action) -> dict:
    prompt = f"""
Customer signals: {json.dumps(signals)}
Churn probability (ML model): {churn_prob:.0%}
Risk breakdown (rule-based): {json.dumps(breakdown)}
Retention playbook guidance: {playbook_snippet['text']}
Required action: {base_action['action']} (priority: {base_action['priority']})

Return ONLY valid JSON:
{{"summary": str, "top_reasons": [str, ...max 3],
  "recommended_action": str, "priority": str, "draft_message": str}}

Rules:
- Ground every reason in the signals above; cite the playbook guidance in the summary
  (e.g. "per playbook: ...").
- recommended_action must be an elaboration of the Required action, not a new action.
- Personalize draft_message to this customer's specific situation and name.
"""
    raw = call_llm(prompt)                 # Gemini (see below)
    return safe_json_parse(raw, retry_once=True)
```

```python
import google.generativeai as genai, os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_gem = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

def call_llm(prompt: str) -> str:
    resp = _gem.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json",
                           "temperature": 0.4},
    )
    return resp.text
```

`response_mime_type: application/json` makes Gemini return strict JSON, so
`safe_json_parse` rarely needs its retry.

`safe_json_parse` strips code fences, `json.loads`, and on failure re-calls the LLM once
with a "your previous output was malformed JSON, return only JSON" message.

### Step 7 — Orchestrator · `services/orchestrator.py` · 20 min

The thinnest file — pure sequencing, per §1. `get_signals(customer_id)` pulls the customer
row + aggregates usage/feedback into the `signals` dict. `save_analysis` writes an
`analyses` row with `status="new"`. Low-risk path skips RAG + LLM entirely.

Demo line: *"This pipeline mirrors what we'd deploy as an n8n workflow in production for
scheduling and multi-channel triggers — for the hackathon we kept it in-process for
reliability."*

### Step 8 — FastAPI routers · 45–60 min

| Method & path | Behavior |
|---|---|
| `GET /customers` | List: `id`, `name`, `plan_tier`, `arr`, `health_score`, `churn_probability` (last analysis or on-the-fly), top contributor. **Sorted by risk desc.** |
| `GET /customers/{id}` | Full detail: signals, 30-day usage trend, feedback list, latest analysis. |
| `POST /customers/{id}/analyze` | `orchestrator.analyze_customer(id)` → store + return full result. |
| `POST /customers/{id}/approve` | Body `{message, status}` → save `approved_message`, set `status` to `approved`/`edited`/`dismissed`. |

`app/main.py`: create tables on startup, warm `ml_model` + `rag` (triggers embed/cache),
enable CORS, mount `frontend/` as static.

### Step 9 — Frontend · `frontend/index.html` · 90–120 min

Single file, Tailwind CDN, vanilla JS `fetch`. Two views:

1. **Dashboard** — table of accounts sorted by risk. Columns: name, plan, ARR,
   health score (colored bar), ML churn %, top warning signal, renewal date. Row → detail.
2. **Customer 360** — header (name, plan, ARR, renewal). Side-by-side:
   - **ML churn probability** (e.g. 87%) and **rule-based health score** (e.g. 39) with the
     contributor breakdown list ("usage −31, support −24, sentiment −17").
   - 30-day usage sparkline, recent feedback with sentiment.
   - **Analyze** button → shows summary (with the quoted `per playbook: ...` line),
     top 3 reasons, recommended action + priority badge, editable **draft message**.
   - **Approve / Edit / Dismiss** buttons → `POST /approve`.
   - P0 → visible **escalation** banner.

### Step 10 — End-to-end test + demo script · 45–60 min

- Run the full flow for 3–4 customers spanning risk buckets (critical / at-risk /
  healthy). Fix broken handoffs (signal dict keys, feature alignment, JSON parse).
- Verify the low-risk short-circuit returns fast and skips the LLM.
- Rehearse the 3-minute demo:
  *"ML model scored this account 87%. Our rules engine shows why — usage down 60%, two
  support complaints, sentiment negative. RAG pulled the matching retention playbook. The
  LLM drafted this outreach. The CSM reviews, edits, approves."*

### Step 11 — Auth + public deployment · 45–60 min

#### Auth — HTTP Basic (single shared demo account)

```python
# app/auth.py
import secrets, os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_basic = HTTPBasic()
_USER = os.environ["APP_USERNAME"]
_PASS = os.environ["APP_PASSWORD"]

def require_auth(cred: HTTPBasicCredentials = Depends(_basic)):
    ok = (secrets.compare_digest(cred.username, _USER)
          and secrets.compare_digest(cred.password, _PASS))
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            headers={"WWW-Authenticate": "Basic"})
    return cred.username
```

Apply globally: `app = FastAPI(dependencies=[Depends(require_auth)])`, and also guard the
static frontend mount. Browser shows the native username/password prompt; over HTTPS
(host-provided) this is fine for a gated demo. Credentials come from `APP_USERNAME` /
`APP_PASSWORD` env vars.

#### RAG embeddings in production

Ollama can't run on a free web host. Make `embed()` backend-switchable:

```python
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama")   # ollama | gemini
```

- **Local dev:** `ollama` + `nomic-embed-text` (per `.env`)
- **Deployed:** `gemini` → `genai.embed_content(model="models/text-embedding-004", ...)`,
  no infra, same API key

Playbook chunk vectors are precomputed and cached to `data/playbook_embeddings.pkl`
**per backend** (filename suffixed with the backend), committed to the repo so the deploy
doesn't re-embed the corpus — only the incoming query is embedded at request time.

#### Deploy to Render (free Web Service)

1. Push repo to GitHub (private is fine).
2. `render.yaml` at repo root:

```yaml
services:
  - type: web
    name: churn-agent
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: bash start.sh
    envVars:
      - key: GEMINI_API_KEY      # set in Render dashboard
        sync: false
      - key: APP_USERNAME
        sync: false
      - key: APP_PASSWORD
        sync: false
      - key: EMBED_BACKEND
        value: gemini
      - key: GEMINI_MODEL
        value: gemini-2.0-flash
```

3. `start.sh` — idempotent bootstrap (data + model rebuilt from repo files if missing,
   since Render's disk is ephemeral):

```bash
#!/usr/bin/env bash
set -e
[ -f churn_agent.db ]      || python load_dataset.py
[ -f ml/churn_model.pkl ]  || python ml/train_model.py
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

4. In the Render dashboard, paste the three secrets (`GEMINI_API_KEY`, `APP_USERNAME`,
   `APP_PASSWORD`). Deploy → you get `https://churn-agent.onrender.com`.

> Notes: free instance sleeps after 15 min idle (~30 s cold start). SQLite resets on
> redeploy — acceptable because all data is seeded from `data/customer_churn_raw.csv` + the
> committed `.pkl` files. `customer_churn_raw.csv` must be committed to the repo for the deploy
> to seed itself (it's a public dataset, so that's fine).

### Step 12 — LinkedIn post · 15 min

Short build writeup: problem, hybrid approach (deterministic + trained ML + RAG + LLM),
screenshot of the dashboard and the AUC/precision/recall output.

---

## 5. Time Budget

| Phase | Time |
|---|---|
| Data ingestion | 45–60 min |
| ML training | 30–45 min |
| Scoring + action catalog | 45 min |
| RAG | 45–60 min |
| ML serving + LLM call | 35–45 min |
| Orchestrator | 20 min |
| FastAPI routers | 45–60 min |
| Frontend | 90–120 min |
| Testing + demo prep | 45–60 min |
| Auth + deployment | 45–60 min |
| LinkedIn | 15 min |
| **Total** | **~8–9.5 hrs** |

---

## 6. Sequencing & Risk

- Build in order: **data → ML → scoring/actions → RAG → LLM → orchestrator → API →
  frontend**. Each layer is standalone-testable before the next.
- The orchestrator is glue around services that already work in isolation — integrate last,
  integrate fast.
- **If time runs short, cut RAG:** drop the `retrieve` step and the playbook line from the
  LLM prompt. Everything else stays intact and demoable.
- One process, one `uvicorn app.main:app --reload`. Nothing external to debug live.

---

## 7. "Two Numbers, Two Methods" Framing

Always show both, side by side:

```
Churn Probability (ML model):        87%
Health Score (rule-based, explainable): 39 / 100
Contributors: usage −31 · support −24 · sentiment −17 · renewal −4
```

The ML number is the *prediction*; the rule-based breakdown is the *explanation*. Hybrid
deterministic + ML + RAG + LLM is the engineering-maturity story.
