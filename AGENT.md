# AGENT.md — FairGov AI: Codebase Briefing for AI Agents

> **Read this first.** This file gives you everything you need to work on this codebase without re-exploring files you've already conceptually understood. Skip any section whose task you are not doing.

---

## 1. What This Project Is

**FairGov AI** is an academic/research-grade ML + RAG system for the *Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojana* scholarship scheme run by DTE Maharashtra (MahaDBT).

It is **NOT** a production government system. All applicant data is **synthetic**. It must never be used to approve or reject real scholarship applications.

**Core components:**
1. **Synthetic dataset generator** — 5,000 policy-derived records
2. **XGBoost binary classifier** — predicts scholarship eligibility
3. **Deterministic rule engine** — independent policy-condition checks
4. **3-path decision router** — `evaluate()` / `POST /evaluate` routing based on ML × Rule agreement
5. **SHAP explainability** — per-prediction feature contributions
6. **TF-IDF RAG** — retrieves curated policy excerpts & local official PDF/TXT documents
7. **Gemini integration** — grounded policy explanations (`rag_full.answer`) + dynamic translation of all user-facing route messages (`service._translate`) in `English` / `Marathi` / `Hindi`
8. **FastAPI REST service** — exposes everything above (auto-loads `project/.env` via `python-dotenv`)

---

## 2. Core Business Logic — Decision Routing Workflow

> **Read this before touching any code.** The system is NOT just "predict and return." It has a 3-path decision router. Every change to `service.py`, `rules.py`, `rag_full.py`, or `main.py` must preserve this contract.

### The Three Paths

```
Frontend Application Form (POST /evaluate)
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Task 1: Dual Eligibility Check         │
 │                                         │
 │  ① XGBoost  → ml_prediction (bool)      │
 │  ② rules.check() → preliminary_eligible │
 └──────────────────┬──────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
  Both TRUE    MISMATCH     Both FALSE
  (ML=T,       (ML ≠ Rule   (ML=F,
   Rule=T)      either way)  Rule=F)
       │            │            │
       ▼            ▼            ▼
 ┌──────────┐ ┌──────────┐ ┌─────────────────────────────────┐
 │ Route A: │ │ Route C: │ │ Task 2 (Route B):               │
 │Forward to│ │Forward to│ │                                 │
 │Gov.      │ │Gov.      │ │ ① SHAP values → which features  │
 │Officer   │ │Officer   │ │   most pushed prediction down   │
 │(approval │ │(manual   │ │                                 │
 │ process) │ │ review)  │ │ ② RAG retrieve → policy docs    │
 │          │ │          │ │   for failed rules + SHAP fields│
 │_translate│ │_translate│ │                                 │
 │(message) │ │(message) │ │ ③ Gemini → grounded policy      │
 └──────────┘ └──────────┘ │   explanation + _translate()    │
                           │   for message/action            │
                           └───────────────┬─────────────────┘
                                           │
                                           ▼
                                 Student receives (in chosen language):
                                 • Rejection message
                                 • Which rule(s) failed
                                 • Which SHAP feature(s) drove rejection
                                 • Relevant policy clause (cited)
                                 • Recommended action to reapply
```

### Decision Routing Rules (Implement Exactly This)

| `ml_prediction` | `preliminary_eligible` | `mismatch` | Action |
|-----------------|------------------------|------------|--------|
| `True` | `True` | `False` | → **Route A** (`route_a`): Forward to government employee for approval; translate `message` |
| `False` | `False` | `False` | → **Route B** (`route_b`): Run SHAP + RAG + Gemini → send cited explanation & translated messages to student |
| `True` | `False` | `True` | → **Route C** (`route_c`): Forward to government employee for manual review; translate non-technical verification `message` |
| `False` | `True` | `True` | → **Route C** (`route_c`): Forward to government employee for manual review; translate non-technical verification `message` |

### What Is Already Built vs. What Needs Wiring

| Component | Status | Location |
|-----------|--------|----------|
| XGBoost prediction | ✅ Built | `src/service.py → predict()` |
| Rule engine check | ✅ Built | `src/rules.py → check()` |
| Mismatch detection | ✅ Built | `service.predict()` returns `mismatch` flag |
| SHAP feature contributions | ✅ Built | `service.predict(explain=True)` returns `shap` dict |
| Policy retrieval (TF-IDF) | ✅ Built | `src/rag_full.py → retrieve()` |
| Gemini grounded answer | ✅ Built | `src/rag_full.py → answer()` |
| **Decision router** | ✅ **Built** | `src/service.py → evaluate()` + `POST /evaluate` in `main.py` |
| **SHAP → RAG bridge** | ✅ **Built** | Inside `evaluate()`: top-3 negative SHAP features + `failed_rules` → query → `rag_full.answer()` |
| **Student explanation payload** | ✅ **Built** | Route B response: `failed_rules`, `shap_top_factors`, `policy_hits`, `policy_explanation`, `recommended_action`, `message` |
| **Multilingual messages (all routes)** | ✅ **Built** | `_translate()` + `_MESSAGES` in `src/service.py` — Gemini translates `message` + `recommended_action` per request |
| **`.env` auto-loading** | ✅ **Built** | `main.py` loads `project/.env` via `python-dotenv` (`override=False`) |
| **Gov. employee queue** | ⚙️ **Not yet wired** | Route A & C return `forward_to_employee: true` — actual notification/queue/DB persistence is out of scope of prototype |

### How Route B works (SHAP → RAG — implemented in `src/service.py → evaluate()`)

```python
# 1. predict() with explain=True gives ML + rules + SHAP in one call
raw = predict(student, explain=True)

# 2. Filter to only negatively contributing features, sort worst-first, take top 3
top_negative = sorted(
    ((f, v) for f, v in raw['shap']['feature_contributions'].items() if v < 0),
    key=lambda x: x[1],
)[:3]

# 3. Combine failed rules + SHAP features into a focused RAG query
failed_rules = raw['rule_engine']['failed_rules']
query_terms = failed_rules + [f for f, _ in top_negative]
query = (
    'Eligibility requirements and conditions for: ' + ', '.join(query_terms)
    if query_terms
    else 'DTE Maharashtra scholarship eligibility criteria and disqualifying conditions'
)

# 4. rag_full.answer() → TF-IDF retrieval + optional Gemini grounding
explanation = rag_answer(query, top_k, language)

# 5. Translate template user-facing strings via Gemini (English fallback if key absent)
t = _translate(
    {'message': _MESSAGES['route_b_message'], 'recommended_action': _MESSAGES['route_b_action']},
    language,
)
```

---

## 3. Repository Layout (Authoritative)

> Decision routing is implemented in `src/service.py → evaluate()` and exposed via `POST /evaluate` in `main.py`. `/predict` and `/explain` remain available as raw debug/dev endpoints.

```
FairGov---AI/
├── README.md                        ← user-facing project README
├── AGENT.md                         ← this file
├── .gitignore                       ← ignores venvs, .env, local notebooks/, generated artifacts
└── project/                         ← main Python project (run commands from here)
    ├── README.md                    ← concise project run instructions
    ├── main.py                      ← FastAPI app (auto-loads .env; all endpoints here)
    ├── requirements.txt             ← pinned dependencies (from tested venv)
    ├── .env.example                 ← env var template
    ├── .gitignore
    ├── data/
    │   └── synthetic_applicants.csv (generated at runtime; not in git)
    ├── models/
    │   └── xgboost_dte.joblib       (generated at runtime; not in git)
    ├── policies/
    │   ├── dte_rules.json           ← policy conditions metadata
    │   ├── policy_evidence.json     ← RAG knowledge base (curated excerpts)
    │   └── official_documents/      ← user-supplied .pdf/.txt policy files
    ├── reports/
    │   ├── dataset_validation.json  (generated at runtime)
    │   └── model_evaluation.json    (generated at runtime)
    ├── src/                         ← core application modules
    │   ├── __init__.py
    │   ├── rules.py                 ← ROOT path, FEATURES list, check()
    │   ├── generate.py              ← make_dataset() → CSV
    │   ├── validate.py              ← validate() → JSON report + assertions
    │   ├── train.py                 ← train() → saves model + JSON metrics
    │   ├── service.py               ← _MESSAGES, _translate(), predict(), evaluate()
    │   ├── rag.py                   ← lightweight retrieve() (used by test_pipeline.py)
    │   ├── rag_full.py              ← load_documents(), index(), retrieve(), answer()
    │   └── pipeline.py              ← __main__: generate → validate → train
    ├── research/                    ← experimental / EDA modules (not imported by API)
    │   ├── __init__.py
    │   ├── eda.py                   ← overview()
    │   ├── tuning.py                ← tune()
    │   ├── interpret.py             ← explain()
    │   ├── validation_extended.py   ← independently_specified_cases(), evaluate_dataset()
    │   ├── retrieval.py             ← query(), grounded_answer()
    │   └── audit.py                 ← audit_cases(), group_metric_demo()
    └── tests/
        ├── test_pipeline.py         ← boundary rule tests + retrieval smoke test
        └── test_full_rag.py         ← full RAG integration tests
```

---

## 4. Key Symbols & Their Locations

Do **not** re-read files just to locate these — they are verified:

| Symbol | File & Line | Purpose |
|--------|-------------|---------|
| `ROOT` | `src/rules.py:3` | `Path` to `project/` directory — used by all modules |
| `FEATURES` | `src/rules.py:5` | Ordered list of 12 ML feature names (exact column order matters) |
| `check(record)` | `src/rules.py:7` | Deterministic rule engine → `{preliminary_eligible, checks, failed_rules, warning}` |
| `make_dataset(n, seed)` | `src/generate.py:5` | Generates synthetic CSV using `check()` for labels |
| `validate()` | `src/validate.py:5` | Reads CSV, re-checks labels, writes `dataset_validation.json`, asserts |
| `train()` | `src/train.py:9` | Reads CSV → XGBoost → saves joblib + `model_evaluation.json` |
| `_MESSAGES` | `src/service.py:9` | English source strings for Route A/B/C `message` & `recommended_action` |
| `_translate(texts, language)` | `src/service.py:29` | Translates dict of UI strings via Gemini; English fallback if key missing/error |
| `predict(student, explain)` | `src/service.py:79` | Loads model, runs ML + rules, optionally adds SHAP |
| `evaluate(student, language, top_k)` | `src/service.py:117` | Full 3-path decision router → Route A / B / C response dict |
| `Student` | `main.py:18` | Pydantic model for the 12 applicant features |
| `EvaluationRequest` | `main.py:33` | Pydantic model — inherits `Student` + `language` + `top_k` |
| `Question` | `main.py:46` | Pydantic model — `question`, `top_k`, `language` |
| `evaluate_application(req)` | `main.py:78` | `POST /evaluate` handler |
| `load_documents()` | `src/rag_full.py:34` | Loads `policy_evidence.json` + any `.pdf`/`.txt` in `official_documents/` |
| `index()` | `src/rag_full.py:61` | Cached TF-IDF vectorizer + matrix over all documents |
| `retrieve(question, top_k)` | `src/rag_full.py:74` | Cosine similarity retrieval → ranked hits |
| `answer(question, top_k, language)` | `src/rag_full.py:85` | `retrieve()` + optional Gemini generation |
| `audit_cases(model, inputs)` | `research/audit.py:4` | ML vs rules comparison DataFrame |
| `group_metric_demo(df, model, group)` | `research/audit.py:13` | Group fairness metrics (synthetic only) |

---

## 5. Feature Schema (Critical — Never Change Column Order)

The 12 features in `FEATURES` must appear in this exact order in any DataFrame passed to the model:

| # | Feature | Type | Notes |
|---|---------|------|-------|
| 1 | `annual_income` | int | Family annual income in INR |
| 2 | `maharashtra_domicile` | 0/1 | Maharashtra domicile certificate |
| 3 | `indian_national` | 0/1 | Indian nationality |
| 4 | `cap_admission` | 0/1 | Admitted via CAP process |
| 5 | `bonafide` | 0/1 | Bonafide student |
| 6 | `professional_course` | 0/1 | Technical/professional course |
| 7 | `private_or_deemed_university` | 0/1 | 1 = ineligible |
| 8 | `other_scholarship` | 0/1 | 1 = ineligible |
| 9 | `family_beneficiaries` | int | Count; > 2 = ineligible |
| 10 | `attendance_pct` | float | 0–100; < 50 fails unless `first_semester=1` |
| 11 | `first_semester` | 0/1 | Exempts attendance check |
| 12 | `gap_years` | int | ≥ 2 = ineligible |

---

## 6. API Endpoints (Defined in `main.py`)

> **Current state:** All endpoints + multilingual translation are implemented and verified. `POST /evaluate` is the main workflow endpoint; `/predict` and `/explain` remain available for debugging/dev use.

### All endpoints

| Endpoint | Method | Handler | Requires pipeline? |
|----------|--------|---------|-------------------|
| `/` | GET | inline | No |
| `/validation` | GET | reads `reports/dataset_validation.json` | Yes |
| `/metrics` | GET | reads `reports/model_evaluation.json` | Yes |
| **`/evaluate`** | **POST** | **`service.evaluate(student, language, top_k)`** | **Yes — main workflow endpoint** |
| `/predict` | POST | `service.predict(student)` | Yes (debug/dev) |
| `/explain` | POST | `service.predict(student, explain=True)` | Yes (debug/dev) |
| `/rag/sources` | GET | `rag_full.load_documents()` | No |
| `/rag/retrieve` | POST | `rag_full.retrieve(q, top_k)` | No |
| `/rag/ask` | POST | `rag_full.answer(q, top_k, language)` | No (Gemini key optional) |

### `POST /evaluate` — request schema (`EvaluationRequest` in `main.py`)

All 12 `Student` fields (see Section 5) plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `language` | string | `"English"` | Language for all user-facing messages across Route A/B/C and Route B `policy_explanation` (`English` / `Marathi` / `Hindi`) |
| `top_k` | int (1–10) | `4` | Policy docs to retrieve for Route B |

### `POST /evaluate` — response schemas

```json
// Route A — Both eligible → forward to government employee
{
  "route": "route_a",
  "decision": "eligible",
  "forward_to_employee": true,
  "ml_prediction": true,
  "ml_probability_eligible": 0.97798,
  "rule_engine": { "preliminary_eligible": true, "checks": {...}, "failed_rules": [] },
  "mismatch": false,
  "disclaimer": "Demonstration prediction only; not an official scholarship decision.",
  "message": "Your application looks good! It has been sent to a government officer for final approval and processing."
}

// Route B — Both rejected → explanation sent to student
{
  "route": "route_b",
  "decision": "rejected",
  "forward_to_employee": false,
  "failed_rules": ["Family income <= INR 800,000"],
  "shap_top_factors": [{"feature": "annual_income", "contribution": -4.91371}],
  "policy_explanation": "Based on the provided evidence... [DTE-P2]",
  "policy_hits": [{"source_id": "DTE-P2", "title": "...", "score": 0.1625, ...}],
  "answer_mode": "gemini_grounded",
  "recommended_action": "Please review the reasons listed above, make the necessary corrections, and resubmit your application once the conditions are met.",
  "message": "Unfortunately your application does not meet the eligibility criteria at this time. Please read the explanation below to understand what needs to be corrected before reapplying."
}

// Route C — Mismatch → escalate to government employee
{
  "route": "route_c",
  "decision": "escalated",
  "forward_to_employee": true,
  "mismatch": true,
  "ml_prediction": false,
  "rule_engine": { "preliminary_eligible": true, ... },
  "message": "Your application requires additional verification. It has been forwarded to a government officer who will review it and reach out to you shortly."
}
```

**Pydantic models in `main.py`:**
- `Student` — 12 features with field-level constraints
- `EvaluationRequest(Student)` — adds `language` + `top_k`; used by `/evaluate`
- `Question` — validates `question` (3–2000 chars), `top_k` (1–10), `language`

---

## 7. Data Flow Diagrams

### Pipeline (one-time setup)

```
python -m src.pipeline
    │
    ├─ src/generate.py → make_dataset(5000)
    │      └─ uses src/rules.check() to label each row
    │      └─ writes data/synthetic_applicants.csv
    │
    ├─ src/validate.py → validate()
    │      └─ re-labels every row and asserts 0 mismatches
    │      └─ writes reports/dataset_validation.json
    │
    └─ src/train.py → train()
           └─ 80/20 stratified split → XGBClassifier
           └─ writes models/xgboost_dte.joblib
           └─ writes reports/model_evaluation.json
```

### `POST /evaluate` — full decision router

```
POST /evaluate (EvaluationRequest JSON)
    │
    └─ main.py → service.evaluate(student_dict, language, top_k)
           │
           ├─ predict(student, explain=True)               [src/service.py]
           │      ├─ XGBoost → ml_prediction, ml_probability_eligible
           │      ├─ rules.check() → preliminary_eligible, failed_rules
           │      ├─ SHAP TreeExplainer → feature_contributions dict
           │      └─ mismatch = (ml_prediction != preliminary_eligible)
           │
           ├─ ml=True  AND rule=True  AND mismatch=False  ──► ROUTE A
           │      ├─ _translate({'message': _MESSAGES['route_a_message']}, language)
           │      └─ return {route: "route_a", decision: "eligible",
           │                 forward_to_employee: true, message}
           │
           ├─ mismatch=True  ────────────────────────────────► ROUTE C
           │      ├─ _translate({'message': _MESSAGES['route_c_message']}, language)
           │      └─ return {route: "route_c", decision: "escalated",
           │                 forward_to_employee: true, message}
           │
           └─ ml=False AND rule=False AND mismatch=False  ──► ROUTE B
                  ├─ filter shap contributions where value < 0
                  ├─ sort ascending → top 3 most negative features
                  ├─ query = "Eligibility requirements and conditions for: " + failed_rules + SHAP features
                  ├─ rag_full.answer(query, top_k, language)   [src/rag_full.py]
                  │      ├─ TF-IDF cosine retrieve → ranked policy hits
                  │      └─ Gemini generate (if GEMINI_API_KEY set) → cited explanation
                  ├─ _translate({'message', 'recommended_action'}, language)
                  └─ return {route: "route_b", decision: "rejected",
                             failed_rules, shap_top_factors, policy_explanation,
                             policy_hits, answer_mode, recommended_action, message}
```

### Legacy / Debug endpoints

```
POST /predict  →  service.predict(student)        [no SHAP]
POST /explain  →  service.predict(student, True)  [with SHAP]
POST /rag/ask  →  rag_full.answer(question, ...)  [standalone RAG]
```

---

## 8. Environment Variables

> **Auto-loaded via `python-dotenv`:** `main.py` calls `load_dotenv(project/.env, override=False)` at startup.
> Create `project/.env` from `project/.env.example` and fill in your key — the server picks it up automatically.
> Shell variables always win over `.env` values (`override=False`). Never commit `.env` to Git (already in `.gitignore`).

| Variable | Required | Default | Effect |
|----------|----------|---------|--------|
| `GEMINI_API_KEY` | No | `""` | Empty → `/rag/ask` and Route B use `retrieval_only`; `_translate()` returns English fallback. No hallucination in any mode. |
| `GEMINI_MODEL` | No | `.env.example`: `gemini-3.5-flash-lite` / code fallback: `gemini-2.5-flash` | Gemini model called by `rag_full.answer()` and `service._translate()` |

**Setup workflow (one-time):**
```bash
cp .env.example .env      # create your local env file
# edit .env and paste your GEMINI_API_KEY
uvicorn main:app --reload # server loads .env automatically on startup
```

**Data sent to Gemini:** Only system prompt + query string (failed-rule & SHAP feature names in Route B, or user question in `/rag/ask`) + retrieved policy snippets + `_MESSAGES` template strings. **Never** raw student form values or PII.

---

## 9. Policy Config (`policies/dte_rules.json`)

The JSON file is metadata/documentation — it is **not** evaluated at runtime by `rules.py`. The actual rule logic is hardcoded in `rules.check()`. If policy thresholds change, update **both** `dte_rules.json` and `rules.check()`.

Key thresholds (as of `checked_date: 2026-10-29`):
- Income cap: **INR 8,00,000**
- Family beneficiaries max: **2**
- Attendance minimum: **50%** (waived for `first_semester=1`)
- Gap years: **< 2** (i.e., 0 or 1 allowed)

---

## 10. RAG Knowledge Base

- **`policy_evidence.json`** — curated short excerpts linked to official sources. Each item has `id`, `title`, `url`, `text`, `provenance`.
- **`policies/official_documents/`** — drop `.pdf` or `.txt` files here; they're auto-chunked (800-char chunks, 140-char overlap) and added to the TF-IDF index on startup.
- The TF-IDF index is **cached** (`@lru_cache(maxsize=1)`). Call `rag_full.reset_index()` after adding new documents to the folder at runtime.

---

## 11. Clone, Setup, Running & Testing

> **Important for fresh clones:** `data/`, `models/`, `reports/`, and `.env` are gitignored. After cloning, you **must** run `python -m src.pipeline` once before calling `/evaluate`, `/predict`, `/explain`, `/validation`, or `/metrics` (otherwise those endpoints return `503 Run python -m src.pipeline`).

### Fresh clone & setup (Linux / macOS)

```bash
# 1. Clone repo and enter project directory
git clone https://github.com/Prashantpowar45/FairGov---AI.git
cd FairGov---AI/project

# 2. Create virtual environment & install exact tested dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment (optional — leave GEMINI_API_KEY empty for retrieval-only + English fallback)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=... (main.py auto-loads project/.env on startup)

# 4. Generate synthetic dataset, validate, and train XGBoost model (required once)
python -m src.pipeline

# 5. Run test suite (NOTE: use python -m pytest, NOT bare pytest)
python -m pytest -q

# 6. Start FastAPI server (http://127.0.0.1:8000/docs)
uvicorn main:app --reload
```

### Fresh clone & setup (Windows PowerShell)

```powershell
git clone https://github.com/Prashantpowar45/FairGov---AI.git
cd FairGov---AI\project

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Edit .env to add GEMINI_API_KEY (optional)

python -m src.pipeline
python -m pytest -q
uvicorn main:app --reload
```

### Quick smoke check once server is running

```bash
curl -s http://127.0.0.1:8000/
curl -s -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"annual_income":950000,"maharashtra_domicile":1,"indian_national":1,"cap_admission":1,"bonafide":1,"professional_course":1,"private_or_deemed_university":0,"other_scholarship":0,"family_beneficiaries":1,"attendance_pct":40,"first_semester":0,"gap_years":0,"language":"English","top_k":3}'
```

**Tests:**
- `tests/test_pipeline.py` — `test_rules_and_boundaries()`: boundary checks (`annual_income=800001`, `attendance_pct=49` with/without `first_semester`). `test_retrieve()`: smoke test on `rag.retrieve()`.
- `tests/test_full_rag.py` — full RAG integration (`_chunks`, `retrieve`, `answer` `retrieval_only` fallback, and FastAPI RAG endpoints).
- Note: `POST /evaluate` currently has no dedicated unit/integration tests in `tests/` (only verified manually via `curl`).

---

## 12. Key Constraints & Known Gotchas

- **Do not change `FEATURES` order** — the serialized XGBoost model expects columns in this exact order.
- **`ROOT` is `project/`** — all relative paths (`data/`, `models/`, `policies/`, `reports/`) resolve from there.
- **Run tests with `python -m pytest -q` (from `project/`)** — running bare `pytest -q` omits `project/` from `sys.path` and fails with `ModuleNotFoundError: No module named 'main'` / `'src'`.
- **`.env` auto-loading only runs in `main.py`** — if you run `src/rag_full.py` or `src/service.py` standalone outside `main.py`, export `GEMINI_API_KEY` in the shell or call `load_dotenv()` explicitly.
- **Gemini API call count per `/evaluate` request** — Route A & C make 0 calls for English or 1 call (`_translate`) for Marathi/Hindi; Route B makes 1 call (`rag_full.answer`) for English or 2 calls (`rag_full.answer` + `_translate`) for Marathi/Hindi.
- **Labels are rule-derived** — `validate()` re-runs `check()` on every row. If you change rules, regenerate the dataset or labels will mismatch.
- **Two RAG modules exist** — `src/rag.py` (lightweight, used by `test_pipeline.py`) and `src/rag_full.py` (full, used by API). They are not identical.
- **`index()` is cached** — adding documents at runtime requires calling `rag_full.reset_index()`.
- **Synthetic accuracy is not real accuracy** — the rule engine and ML share the same feature space; high agreement is expected and does not validate real-world generalizability.

---

## 13. What NOT to Do

- ❌ Do not send real student PII or raw form values to Gemini (never place applicant data in `policies/official_documents/` either).
- ❌ Do not commit `.env` or `GEMINI_API_KEY` to any file tracked by git.
- ❌ Do not use this system to make or inform actual scholarship decisions.
- ❌ Do not treat synthetic test metrics as proof of real-world model performance.
- ❌ Do not change column order in `FEATURES` or DataFrames without retraining the model.

---

## 14. Common Agent Tasks & Where to Start

### Workflow / Routing tasks

| Task | Status | Start here |
|------|--------|-----------|
| `/evaluate` decision router | ✅ Done | `src/service.py → evaluate()` + `main.py → evaluate_application()` |
| SHAP → RAG bridge (Route B) | ✅ Done | `src/service.py:178–206` |
| Multilingual message translation | ✅ Done | `src/service.py → _MESSAGES` (`:9`) + `_translate()` (`:29`) |
| **Gov. employee notification (Route A & C)** | ⚙️ To do | Add email/queue/webhook after `forward_to_employee: true` branches in `evaluate()` |
| **Add automated tests for `/evaluate`** | ⚙️ To do | Add `tests/test_evaluate.py` testing Route A, B, C & `_translate()` fallback via `TestClient(app)` |
| Change message wording (approval/rejection/escalation) | Edit as needed | `src/service.py` `_MESSAGES` dict (`:9–26`) — Gemini translates whatever strings are there |
| Change translation language options | Edit as needed | `main.py` (`EvaluationRequest.language` & `Question.language` regex pattern) |
| Change how many SHAP features feed RAG query | Edit as needed | `src/service.py:186` — change `[:3]` slice on `top_negative` |
| Change RAG query phrasing | Edit as needed | `src/service.py:191` — edit `query` string construction |

### Model / Rules / Infrastructure tasks

| Task | Start here |
|------|-----------|
| Add a new eligibility rule | `src/rules.py → check()` + update `policies/dte_rules.json` + regenerate dataset (`python -m src.pipeline`) |
| Add a new API endpoint | `main.py` + add handler in appropriate `src/` module |
| Change XGBoost hyperparameters | `src/train.py → XGBClassifier(...)` |
| Add more policy evidence to RAG | Append to `policies/policy_evidence.json` or add `.pdf`/`.txt` to `policies/official_documents/` |
| Change Gemini model or RAG prompt | `.env` (`GEMINI_MODEL`) and `src/rag_full.py → answer()` (`:98–105`) |
| Add a new test | `tests/test_pipeline.py` (rule/retrieve) or `tests/test_full_rag.py` (RAG); run via `python -m pytest -q` |
| Understand a prediction result | Read `service.predict()` (`src/service.py:79`) and `service.evaluate()` (`src/service.py:117`) |
| Debug a mismatch (ML ≠ rules) | Use `research/audit.py → audit_cases()` |
| Add SHAP for a new feature | Update `FEATURES` in `src/rules.py` → retrain (`python -m src.pipeline`) → `service.py` handles SHAP automatically |

---

*Last updated: 2026-11-09 (synced with pinned requirements.txt, multilingual Gemini routing, dotenv auto-loading, and GitHub repo layout) | Scheme: Maharashtra DTE RCSMSSY | Status: Academic prototype — all core endpoints live*
