# FairGov AI — Scholarship Eligibility ML + RAG System

> **Academic prototype** for the *Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojana* (DTE Maharashtra / MahaDBT). All applicant data is **synthetic**. This system must **never** be used to approve or deny real scholarship applications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Decision Routing Workflow](#decision-routing-workflow)
- [Repository Layout](#repository-layout)
- [System Architecture](#system-architecture)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Data Sent to Gemini](#data-sent-to-gemini)
- [Adding Official Policy Documents](#adding-official-policy-documents)
- [Running Tests](#running-tests)
- [Eligibility Rules Modeled](#eligibility-rules-modeled)
- [Academic Limitations & Caveats](#academic-limitations--caveats)
- [Official Reference](#official-reference)

---

## Project Overview

FairGov AI is a **research-grade** ML + RAG pipeline that demonstrates:

1. **Synthetic data generation** — 5,000 applicant records derived from policy rules.
2. **XGBoost classifier** — trained and evaluated on synthetic data.
3. **Deterministic rule engine** — independent policy-condition checks.
4. **3-path decision routing** — routes applications based on ML × Rule agreement (`POST /evaluate`).
5. **SHAP + RAG rejection explanation** — tells rejected students *why* and *what to fix*.
6. **Multilingual Gemini responses** — translates user-facing messages (`message`, `recommended_action`) and generates grounded policy explanations in `English`, `Marathi`, or `Hindi`, with automatic fallback to English if no API key is configured.
7. **TF-IDF RAG** — retrieves curated policy excerpts and local official PDF/TXT documents.
8. **FastAPI service** — exposes all the above as REST endpoints.

---

## Decision Routing Workflow

This is the **core business logic** of FairGov AI. Every application from the frontend hits `POST /evaluate`, passes through two independent checks, and is routed to one of three outcomes:

```
              ┌─────────────────────────────────┐
              │   Frontend: Application Form    │
              └─────────────┬───────────────────┘
                            │
                ┌───────────▼────────────┐
                │  Task 1: Dual Check    │
                │                        │
                │  ① XGBoost Classifier  │  → True / False
                │  ② Rule Engine         │  → True / False
                └───────────┬────────────┘
                            │
          ┌─────────────────┼──────────────────────┐
          │                 │                      │
    Both TRUE         MISMATCH               Both FALSE
  (ML=✓, Rule=✓)  (ML≠Rule either way)    (ML=✗, Rule=✗)
          │                 │                      │
          ▼                 ▼                      ▼
  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
  │ Route A:     │  │ Route C:         │  │ Task 2 (Route B):       │
  │ Forwarded to │  │ Forwarded to Gov.│  │                         │
  │ Gov. Employee│  │ Employee for     │  │ ① SHAP: which fields    │
  │ for final    │  │ manual review &  │  │    caused rejection     │
  │ approval &   │  │ verification     │  │                         │
  │ processing   │  │                  │  │ ② RAG: fetch policy     │
  │              │  │ (Message in      │  │    clauses for failed   │
  │ (Message in  │  │  student's       │  │    rules & SHAP factors │
  │  student's   │  │  language via    │  │                         │
  │  language via│  │  Gemini)         │  │ ③ Gemini: ground policy │
  │  Gemini)     │  └──────────────────┘  │    explanation +        │
  └──────────────┘                        │    translate student    │
                                          │    messages             │
                                          └───────────┬─────────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────────┐
                                          │ Student receives (in    │
                                          │ chosen language):       │
                                          │ • Rejection message     │
                                          │ • Failed rules & SHAP   │
                                          │ • Policy references     │
                                          │ • Recommended actions   │
                                          └─────────────────────────┘
```

### Why this design matters

| Outcome | Route | Who handles it | What happens |
|---------|-------|---------------|--------------|
| Both ✅ Eligible | `route_a` | Government employee | Student receives a friendly confirmation (in their chosen language); application is flagged for officer approval |
| Both ❌ Rejected | `route_b` | Automated system | SHAP + RAG + Gemini generates a cited rejection explanation & next steps in the student's language so they can self-correct and reapply |
| ⚠️ Mismatch | `route_c` | Government employee | Ambiguous case is flagged for manual officer review; student receives a non-technical verification notice in their language |

**Key benefits:**
- Government employees only handle clear approvals and ambiguous edge cases.
- Clear rejections are fully automated with actionable, policy-cited student guidance.
- All user-facing messages (`message`, `recommended_action`, and `policy_explanation`) support `English`, `Marathi`, and `Hindi` dynamically via Gemini.

---

## Repository Layout

```
FairGov---AI/
├── README.md                        ← this file (top-level overview)
├── AGENT.md                         ← codebase briefing & guardrails for AI agents
├── .gitignore
└── project/
    ├── README.md                    ← concise local run instructions
    ├── main.py                      ← FastAPI app entrypoint (auto-loads .env)
    ├── requirements.txt             ← pinned Python dependencies
    ├── .env.example                 ← environment template
    ├── .gitignore
    ├── data/
    │   └── synthetic_applicants.csv (generated at runtime)
    ├── models/
    │   └── xgboost_dte.joblib       (generated at runtime)
    ├── policies/
    │   ├── dte_rules.json           ← policy conditions metadata (JSON)
    │   ├── policy_evidence.json     ← curated RAG source excerpts
    │   └── official_documents/      ← drop .pdf / .txt GR files here
    ├── reports/
    │   ├── dataset_validation.json  (generated at runtime)
    │   └── model_evaluation.json    (generated at runtime)
    ├── src/
    │   ├── generate.py              ← synthetic dataset generator
    │   ├── validate.py              ← data quality & label checks
    │   ├── train.py                 ← XGBoost train + evaluate
    │   ├── service.py               ← predict, SHAP, 3-path evaluate router & Gemini translation
    │   ├── rules.py                 ← deterministic rule engine
    │   ├── rag.py                   ← lightweight TF-IDF retriever
    │   ├── rag_full.py              ← full RAG: retrieve + Gemini grounded answer
    │   └── pipeline.py              ← orchestrates generate → validate → train
    ├── research/
    │   ├── eda.py                   ← exploratory data analysis helpers
    │   ├── tuning.py                ← hyperparameter search experiments
    │   ├── interpret.py             ← SHAP / interpretation utilities
    │   ├── validation_extended.py   ← boundary case and extended checks
    │   ├── retrieval.py             ← research retrieval wrappers
    │   └── audit.py                 ← ML vs rule-engine audit + group metrics
    └── tests/
        ├── test_pipeline.py         ← rule boundary + retrieval tests
        └── test_full_rag.py         ← full RAG integration tests
```

---

## System Architecture

```
  Frontend
     │  Application Form (POST /evaluate)
     ▼
  FastAPI (main.py) ── auto-loads project/.env via python-dotenv
     │
     ├── src/service.py ──── evaluate() ──► XGBoost + Rule Engine + SHAP
     │                                             │
     │                          ┌──────────────────┼──────────────────┐
     │                       Route A            Route C            Route B
     │                      (Both ✅)         (Mismatch ⚠️)       (Both ❌)
     │                          │                  │                  │
     │                     _translate()       _translate()    SHAP → RAG Query
     │                          │                  │                  │
     │                          ▼                  ▼                  ▼
     │                     Gov. Officer       Gov. Officer     src/rag_full.py
     │                   (Final Approval)   (Manual Review)   (TF-IDF + Gemini)
     │                                                                │
     │                                                        + _translate()
     │                                                                │
     └────────────────────────────────────────────────────────► Student Guidance
```

**Module responsibilities:**

| Module | Role in workflow |
|--------|-----------------|
| `src/generate.py` | One-time: create synthetic training data |
| `src/validate.py` | One-time: data quality checks |
| `src/train.py` | One-time: train & save XGBoost model |
| `src/rules.py` | Per-request: deterministic eligibility check |
| `src/service.py` | Per-request: ML prediction, SHAP explanation, 3-path routing (`evaluate`), & Gemini message translation (`_translate`) |
| `src/rag_full.py` | Per-rejection / Q&A: TF-IDF policy retrieval + Gemini grounded generation |

---

## Quickstart

### Prerequisites

- Python 3.10+ recommended
- (Optional) A valid [Gemini API key](https://aistudio.google.com/app/apikey) for grounded RAG answers and multilingual UI translations

### 1 — Clone and enter the project directory

```bash
git clone https://github.com/Prashantpowar45/FairGov---AI.git
cd FairGov---AI/project
```

### 2 — Create a virtual environment and install dependencies

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3 — (Optional) Configure Gemini API key

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=... (leave empty to use retrieval-only + English fallback)
```

### 4 — Run the pipeline (generate data → validate → train model)

```bash
python -m src.pipeline
```

This creates:
- `data/synthetic_applicants.csv`
- `models/xgboost_dte.joblib`
- `reports/dataset_validation.json`
- `reports/model_evaluation.json`

### 5 — Start the API server

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + scope info |
| `GET` | `/validation` | Synthetic dataset quality report |
| `GET` | `/metrics` | XGBoost holdout evaluation metrics |
| `POST` | **`/evaluate`** | **Main workflow endpoint** — runs 3-path decision router (Route A / B / C), SHAP + RAG explanation on rejection, and Gemini multilingual translation |
| `POST` | `/predict` | Debug/dev: ML prediction + rule-engine check + mismatch flag |
| `POST` | `/explain` | Debug/dev: same as `/predict` + raw SHAP feature contributions |
| `GET` | `/rag/sources` | All indexed policy sources with provenance |
| `POST` | `/rag/retrieve` | TF-IDF cosine-similarity policy retrieval |
| `POST` | `/rag/ask` | Standalone RAG Q&A with optional Gemini grounded answer |

### Main workflow schema (`POST /evaluate`)

Inherits all 12 student fields plus `language` (`"English"`, `"Marathi"`, or `"Hindi"`) and `top_k` (`1`–`10`):

```json
{
  "annual_income": 950000,
  "maharashtra_domicile": 1,
  "indian_national": 1,
  "cap_admission": 1,
  "bonafide": 1,
  "professional_course": 1,
  "private_or_deemed_university": 0,
  "other_scholarship": 0,
  "family_beneficiaries": 1,
  "attendance_pct": 40,
  "first_semester": 0,
  "gap_years": 0,
  "language": "Marathi",
  "top_k": 3
}
```

**Three possible response routes:**
- **`route_a`** (`decision: "eligible"`, `forward_to_employee: true`): Both XGBoost and rules passed → sends translated confirmation `message`.
- **`route_b`** (`decision: "rejected"`, `forward_to_employee: false`): Both rejected → returns `failed_rules`, `shap_top_factors`, `policy_explanation`, `policy_hits`, `answer_mode`, `recommended_action`, and `message` (in the requested language).
- **`route_c`** (`decision: "escalated"`, `forward_to_employee: true`): ML and rules disagree (`mismatch: true`) → sends friendly verification `message` in the requested language and flags for manual officer review.

### Student input schema (`POST /predict` and `POST /explain`)

```json
{
  "annual_income": 600000,
  "maharashtra_domicile": 1,
  "indian_national": 1,
  "cap_admission": 1,
  "bonafide": 1,
  "professional_course": 1,
  "private_or_deemed_university": 0,
  "other_scholarship": 0,
  "family_beneficiaries": 1,
  "attendance_pct": 75.0,
  "first_semester": 0,
  "gap_years": 0
}
```

### RAG question schema (`POST /rag/retrieve` and `POST /rag/ask`)

```json
{
  "question": "What is the annual family income limit for the DTE scholarship?",
  "top_k": 3,
  "language": "English"
}
```

`language` accepts `"English"`, `"Marathi"`, or `"Hindi"`.

---

## Configuration

The FastAPI server **auto-loads `project/.env` on startup** via `python-dotenv` (`override=False`). Copy `.env.example` to `.env`, add your key, and restart the server — no manual shell exports needed.

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
uvicorn main:app --reload
```

```bash
# project/.env
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite   # or gemini-2.5-flash, etc.
```

> Shell environment variables always take precedence over `.env` values (`override=False`). Never commit `.env` to Git (it is already listed in `.gitignore`).

**Without `GEMINI_API_KEY` (graceful fallback):**
- `/rag/ask` returns `answer_mode: retrieval_only` with matched policy excerpts — it never hallucinates an answer.
- `/evaluate` Route B returns `answer_mode: retrieval_only` and English fallback for `message` / `recommended_action`.
- `/evaluate` Route A & Route C return English fallback for `message`.

---

## Data Sent to Gemini

For transparency and privacy, **no student form values or PII are sent to Gemini**. Only the following leave your machine when `GEMINI_API_KEY` is configured:

1. **System instructions** (hardcoded policy-grounding prompt in `src/rag_full.py`).
2. **Question / policy query** — either the user's question (`/rag/ask`) or a short summary constructed from failed rule names and SHAP feature names in Route B (e.g., `"Family income <= INR 800,000, attendance_pct"` — never raw income figures or applicant identity).
3. **Retrieved policy excerpts** — top-`k` chunks from `policies/policy_evidence.json` and `policies/official_documents/`.
4. **Fixed UI strings** — the short template confirmation/rejection/escalation sentences in `src/service.py` when `language != "English"`.

---

## Adding Official Policy Documents

1. Download the relevant government GR / policy PDFs from the [official MahaDBT scheme page](https://mahadbt.maharashtra.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A7F4D327BDEB7125DE0BA4AE1C51180C272281017EBEF6F7C).
2. Verify the scheme name, year, and authenticity.
3. Place **only official public policy documents** (`.pdf` or `.txt`) in `policies/official_documents/` — never place applicant records there.
4. Restart the API server — documents are auto-ingested and chunked on startup.

> **Note:** Scanned PDFs without selectable text require external OCR before placement. No automatic OCR is performed.

---

## Running Tests

Run from `project/` using `python -m pytest` (so `project/` is on Python's import path):

```bash
cd project
python -m pytest -q
```

Tests cover:
- **Boundary rule checks** — income limit, attendance exception, gap year limit (`tests/test_pipeline.py`).
- **TF-IDF retrieval** — confirms the index returns results for known policy queries (`tests/test_pipeline.py`).
- **Full RAG integration** — end-to-end chunking, retrieval, `retrieval_only` fallback, and FastAPI RAG endpoints (`tests/test_full_rag.py`).

---

## Eligibility Rules Modeled

These are the policy-derived conditions implemented in `src/rules.py` and `policies/dte_rules.json`:

| Rule | Condition |
|------|-----------|
| Indian nationality | Must be an Indian national |
| Maharashtra domicile | Must hold Maharashtra domicile certificate |
| Family annual income | ≤ INR 8,00,000 |
| CAP admission | Must be admitted through the CAP process |
| Bonafide student | Must be a bonafide student of the institution |
| Course type | Must be enrolled in a professional/technical course |
| University type | Must **not** be a private or deemed university |
| Other scholarships | Must **not** hold any other scholarship simultaneously |
| Family beneficiaries | At most **2** family members receiving the scheme |
| Attendance | ≥ 50% attendance, **or** first-semester exemption applies |
| Study gap | Fewer than **2** gap years |

> ⚠️ These are **simplified** rule implementations. Many edge cases, category-specific exceptions, and document verification requirements are not modeled.

---

## Academic Limitations & Caveats

- **Synthetic data only** — labels and features are generated by the same rule code; high accuracy on this data does not imply real-world performance.
- **Incomplete rules** — many scheme-specific GR exceptions are not implemented; passing rule checks is not an official eligibility determination.
- **TF-IDF retrieval** — not a neural/embedding-based RAG. Adding a transformer embedder + FAISS/pgvector would make it a full semantic RAG system.
- **No access control** — do not submit real student PII to this system or to Gemini.
- **RAG citations** — `policy_evidence.json` contains curated source-linked summaries, not the full official GR documents; always verify against the official source before use.
- **No legal validity** — never use this prototype to approve or deny actual scholarship applications.

---

## Official Reference

Scheme details: [MahaDBT — Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojana](https://mahadbt.maharashtra.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A7F4D327BDEB7125DE0BA4AE1C51180C272281017EBEF6F7C)
