# 🔍 Churn Risk Detection Agent

> An AI-powered Customer Success Agent that detects at-risk customers, explains warning signals via two independent methods, retrieves matching playbooks, and drafts personalized retention outreach — all routed through a human-in-the-loop approval flow.

---

## ✨ Features

- **Dual-method risk scoring** — XGBoost ML model (SHAP-explained) + deterministic health score run independently; their agreement is surfaced to the CSM
- **Retrieval-Augmented Generation (RAG)** — cosine-similarity playbook retrieval with lexical fallback
- **LLM-drafted outreach** — Gemini `gemini-2.5-flash` in JSON mode produces summaries, top reasons, recommended actions, and ready-to-send draft messages
- **Action catalog** — rule-based action picker (P0–P3 priority) maps risk signals → playbook queries
- **Full-stack SPA** — vanilla JS frontend served as a static file from the same FastAPI process
- **Hackathon-ready UI** — dark-mode professional dashboard with per-customer drill-down

---

## 🏗️ Architecture

```
frontend/index.html         (vanilla JS SPA, same-origin)
        │  REST /api/*
        ▼
FastAPI  app/main.py        (lifespan: init_db → rag.warm → ml health; CORS; StaticFiles)
        │
app/routers/
  customers.py              list / get / trigger analysis
  analysis.py               fetch latest analysis results
  health.py                 /api/health liveness probe
  auth.py                   session-based auth (SessionMiddleware)
        │
app/services/
  signals.py                get_signals(session, id) → aggregated usage/feedback dict
  ml_model.py               predict_from_customer(row) → {churn_probability, band, top_features (SHAP), model_version}
  scoring.py                risk_breakdown(signals) → {health_score 0-100, contributors[], positives[], risk_band}
                            agreement(health, churn_prob) → do the two methods agree?
  action_catalog.py         pick_base_action(signals, churn_prob) → {action, label, priority P0-P3, channel, playbook_query}
  rag.py                    retrieve_best(query) → playbook snippet (cosine similarity, lexical fallback)
  embeddings.py             embed_one / embed_many — backend: ollama (local) | gemini (deploy)
  llm.py                    analyze(...) → {summary, top_reasons ≤3, recommended_action, priority, draft_message, playbook_citation}
  prompts.py                SYSTEM_INSTRUCTION + build_user_prompt
  orchestrator.py           analyze_customer(id, force) — full pipeline entrypoint
        │
app/db.py                   SQLAlchemy engine, SessionLocal, Base
app/models.py               Customer, UsageDaily, Feedback, Analysis
app/schemas.py              Pydantic v2 request/response models
app/deps.py                 get_db, require_auth
```

**Offline artifacts (built once, committed):**

```
load_dataset.py             Kaggle CSV → SQLite churn_agent.db
ml/features.py              Single source of feature derivation (dict + DataFrame)
ml/train_model.py           XGBoost training → churn_model.pkl, feature_columns.pkl,
                            preprocess.pkl, model_card.json
```

---

## 🤖 ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (`binary:logistic`) |
| Training rows | 10,000 |
| Features | 25 engineered features |
| Dataset churn rate | 10.21% |
| Test AUC (ROC) | **0.881** |
| Test PR-AUC | 0.634 |
| Precision / Recall (churn class) | 56.8% / 57.4% |
| Decision threshold | 0.698 |
| Trained at | 2026-08-30 |

**Top 5 features by gain:**

| Feature | Gain |
|---|---|
| `recent_inactivity_flag` | 0.359 |
| `last_login_days` | 0.195 |
| `payment_failures` | 0.052 |
| `tenure_days` | 0.041 |
| `usage_per_tenure` | 0.034 |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Use `py -3.12` on Windows |
| [Ollama](https://ollama.ai) | Running locally for RAG embeddings — pull `nomic-embed-text` |
| Google Gemini API key | For LLM analysis (`gemini-2.5-flash`) |

### 1. Clone & install

```bash
git clone https://github.com/AdityaKuranjekar/Churn-risk-detection-agent.git
cd Churn-risk-detection-agent
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your-secret-key
CORS_ORIGINS=*
FRONTEND_DIR=frontend
EMBED_BACKEND=ollama          # or "gemini" for cloud deploy
OLLAMA_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Seed the database

```bash
python load_dataset.py
```

This ingests the Kaggle telco-churn CSV into `churn_agent.db` (SQLite).

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the SPA is served from `frontend/index.html`.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/customers` | List all customers with current risk bands |
| `GET` | `/api/customers/{id}` | Get full customer profile + latest analysis |
| `POST` | `/api/customers/{id}/analyze` | Trigger full pipeline for one customer |
| `GET` | `/api/analysis/{id}` | Fetch stored analysis result |

---

## 🗂️ Project Structure

```
.
├── app/
│   ├── main.py               # FastAPI app factory + lifespan
│   ├── db.py                 # SQLAlchemy setup
│   ├── models.py             # ORM models
│   ├── schemas.py            # Pydantic schemas
│   ├── deps.py               # Dependency injections
│   ├── routers/              # API route handlers
│   └── services/             # Business logic & AI services
├── ml/
│   ├── train_model.py        # XGBoost training script
│   ├── features.py           # Feature engineering (shared train + serve)
│   ├── churn_model.pkl       # Trained model artifact
│   ├── feature_columns.pkl   # Feature column order
│   ├── preprocess.pkl        # Preprocessing pipeline
│   └── model_card.json       # Model metrics & metadata
├── frontend/
│   └── index.html            # Vanilla JS SPA dashboard
├── data/                     # Raw data files
├── docs/                     # Additional documentation
├── tests/                    # Test suite
├── load_dataset.py           # Data ingestion script
├── requirements.txt
└── .env                      # Local environment config (not committed)
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## ☁️ Deployment

Target: **Hugging Face Spaces** (Docker SDK, free CPU basic, 16 GB RAM)

Key considerations:
- Set `EMBEDDING_BACKEND=gemini` — Ollama is not available in HF Spaces
- Set all secrets via HF Spaces environment variables (never commit `.env`)
- The `FRONTEND_DIR` static mount serves the SPA from the same process — no separate CDN needed
- SQLite `churn_agent.db` is bundled; for production swap to Postgres via `DATABASE_URL`

---

## 📖 Documentation

- [HANDOFF.md](HANDOFF.md) — Full project context and architecture handoff document
- [implementation.md](implementation.md) — Implementation notes and decisions
- [Automation Roadmap (n8n Workflow Blueprint)](docs/n8n_workflow.md)
- Step-by-step build logs: `step1_data_ingestion.txt` → `step11_auth_and_hosting.txt`

---

## 🙏 Credits

Built as an AI hackathon project. Uses:
- [XGBoost](https://xgboost.readthedocs.io/) for churn prediction
- [Google Gemini](https://ai.google.dev/) for LLM analysis & cloud embeddings
- [Ollama](https://ollama.ai/) + `nomic-embed-text` for local embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
- [SHAP](https://shap.readthedocs.io/) for model explainability
