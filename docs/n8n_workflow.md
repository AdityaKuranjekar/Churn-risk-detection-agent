# Roadmap: n8n Automation Blueprint

This document outlines the end-to-end implementation plan for transitioning the Churn Risk Agent into a fully automated, event-driven Customer Success system orchestrated by n8n.

Everything in this blueprint runs today as an in-process Python pipeline (`orchestrator.py`). n8n adds scheduling, event triggers, retries, integrations (Slack / email / CRM), and a visual operations canvas — without moving business logic out of the backend.

## 1. Architecture

```mermaid
flowchart TD
    React[React SPA] --> Fast[FastAPI\napplication APIs · business rules · auth · persistence]
    Fast --> PG[PostgreSQL + pgvector\nsource of truth]
    Fast --> XGB[XGBoost model\nchurn P + SHAP]
    
    subgraph n8n[n8n orchestration]
        Fast --> N8N[scheduling · triggers · AI workflow · integrations]
        N8N --> LLM1[LLM\nreasoning & personalize]
        N8N --> RAG[RAG playbook\nretrieval]
        N8N --> Auto[Automation\nSlack · email]
    end
    
    N8N --> Rec[Recommendations]
    Rec --> Fast
```

## 2. Daily Workflow

```mermaid
flowchart TD
    S1[1 Schedule Trigger\nCron · every day 06:00] --> S2[2 Fetch usage\nHTTP Request → /internal/usage]
    S2 --> S3[3 Fetch support tickets\nHTTP Request → /internal/support]
    S3 --> S4[4 Fetch latest feedback\nHTTP Request → /internal/feedback]
    S4 --> S5[5 Aggregate signals\nFunction · 30/90-day windows]
    S5 --> S6[6 Health score\nHTTP → /risk/breakdown]
    S6 --> S7[7 Churn probability\nHTTP → /ml/predict]
    S7 --> S8{8 IF risk gate\nchurn ≥ 0.50 OR band ∈ at-risk/critical}
    S8 -- FALSE --> Log[Log: monitor]
    S8 -- TRUE --> S9[9 LLM · analyze why\nAI node · grounded in signals]
    S9 --> S10[10 Retrieve playbook RAG\nVector search · top-1 of 10 playbooks]
    S10 --> S11[11 LLM · next-best-action\nAI node · action constrained by rules]
    S11 --> S12[12 Generate CSM message\nAI node · personalized draft]
    S12 --> S13[13 Store recommendation\nHTTP → /analyses]
    S13 --> S14[14 Notify CSM\nSlack / Email · with Approve link]
```

## 3. Event-driven

```mermaid
flowchart TD
    E1[Webhook Trigger\nCustomer feedback / NPS submitted] --> E2[Sentiment analysis\nAI node]
    E2 --> E3{IF sentiment negative?}
    E3 -- NO --> End[End · store only]
    E3 -- YES --> E4[Look up account risk\nHTTP → /customers/id]
    E4 --> E5{IF churn ≥ 0.70?}
    E5 -- NO --> Q[Queue for daily run]
    E5 -- YES --> E6[Escalate to CSM\nSlack DM · P0]
    E6 --> E7[Draft recommended response\nAI node]
```

## 4. Agent Brain

```mermaid
flowchart TD
    N8N[n8n] --> S[Signal agent]
    N8N --> R[Risk agent]
    N8N --> I[Insight agent]
    
    S --> A[Action agent]
    R --> A
    I --> A
    
    A --> C[Communication agent]
```

## 5. Who does what

| Component | Responsibility |
| :--- | :--- |
| **React** | User experience |
| **FastAPI** | Application APIs, business rules, persistence, auth |
| **PostgreSQL** | Source of truth (+ pgvector for playbook embeddings) |
| **XGBoost** | Churn probability & feature attribution (SHAP) |
| **n8n** | Orchestration, schedules, event triggers, AI workflow, external integrations, retries, observability |
| **LLM (Gemini)** | Reasoning, root-cause explanation, message personalization |

**n8n anti-patterns we avoid:** Not doing: health-score math across 15 nodes · React calling n8n for every click · 10 LLM calls where 2 suffice.

## 6. Live vs roadmap

| LIVE IN THIS BUILD | WITH n8n (production) |
| :--- | :--- |
| `orchestrator.analyze_customer()` | Same steps as a visual n8n workflow |
| runs on the Analyze click | Scheduled nightly for all accounts |
| Rule engine + XGBoost + RAG + LLM | unchanged (called via HTTP nodes) |
| SQLite | PostgreSQL + pgvector |
| In-memory cosine over 10 playbooks | pgvector similarity search |
| CSM approves in the UI | + Slack/email notify with deep link |
| Manual re-analyze | Event triggers (negative feedback, payment failure, usage cliff) |
| — | Retries, dead-letter, run history, outcome logging & feedback loop |
