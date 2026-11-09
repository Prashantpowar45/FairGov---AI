# FairGov AI — Mega ML + Grounded RAG module

**Use case:** Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Shishyavrutti Yojana, DTE Maharashtra (MahaDBT).

## Pipeline
1. Generate **5,000 synthetic** applicant records (`src/generate.py`). These are not government applicant records.
2. Validate missing data, duplicates and labels (`src/validate.py`). Note that rule-label validation uses the same code as data generation; separate manual boundary tests provide limited independent checks.
3. Train/test XGBoost (`src/train.py`), export metrics/model.
4. Generate SHAP feature contributions (`src/service.py`).
5. Independently verify modeled policy conditions (`src/rules.py`), compare against ML prediction.
6. RAG: ingest curated official-source-linked excerpts and local official policy PDFs/TXT; chunk PDF text, create TF-IDF retrieval index, retrieve with cosine similarity; optionally use **Gemini API** to generate grounded, cited explanations (`src/rag_full.py`).

## Start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
python -m pytest -q
uvicorn main:app --reload
```
Open **http://127.0.0.1:8000/docs**.

## Gemini API (optional, needed for generative answers)
Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The FastAPI server (`main.py`) automatically loads `project/.env` on startup via `python-dotenv` (`override=False`). Never commit `.env` or API keys.

```powershell
# Or override via shell environment:
$env:GEMINI_API_KEY="YOUR_PERSONAL_API_KEY"
$env:GEMINI_MODEL="gemini-3.5-flash-lite"
uvicorn main:app --reload
```
Without the key, `/rag/ask` and `/evaluate` (Route B) **do not invent an LLM answer**; they return retrieved evidence with `answer_mode=retrieval_only` and English fallback UI messages. Model/API access and quotas require your valid Gemini account.

## Endpoints
- `GET /validation` — synthetic dataset quality checks
- `GET /metrics` — model holdout metrics (synthetic only)
- `POST /evaluate` — main 3-path decision router (XGBoost + rule engine → approval forwarding, SHAP+RAG+Gemini rejection explanation, or escalation) with multilingual Gemini translation
- `POST /predict`, `POST /explain` — debug classifier/rule audit + SHAP
- `GET /rag/sources` — indexed sources and provenance
- `POST /rag/retrieve` — relevant policy excerpts
- `POST /rag/ask` — optional Gemini grounded answers with evidence IDs

Example request:
```json
{"question":"What is the annual family income limit for the DTE scholarship?", "language":"Marathi", "top_k":3}
```

## Add actual official documents
Download the relevant government GR/policy PDFs from the official source, verify scheme/year and place them in `policies/official_documents/`. Restart API. Files with no selectable text require authorized OCR/extraction externally; no OCR is automatically performed. The included `policy_evidence.json` is a collection of short **curated source-linked summaries**, not the full original government policy documents. RAG citations should be reviewed before high-impact use.

## Academic limitations
- Labels are synthetic and rules are **incomplete**; this is not legal scholarship eligibility validation.
- High synthetic accuracy **does not prove real accuracy**. No individual-level actual government approvals/rejections were used for training.
- Rule-engine and ML use matching features, so expected agreement is high; mismatch evaluation requires separately constructed challenge sets.
- Default retrieval uses TF-IDF, not pretrained neural embeddings/vector DB. It is a functioning retrieve-then-generate RAG pipeline **only when Gemini is configured**; to claim neural embedding-based RAG add a transformer embedder and FAISS/pgvector.
- No access control, personal-data protection or officer sign-off is included in this module. Don't submit real student PII to Gemini.
- Never use this prototype to approve/deny actual scholarships.

Official reference: https://mahadbt.maharashtra.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A7F4D327BDEB7125DE0BA4AE1C51180C272281017EBEF6F7C
