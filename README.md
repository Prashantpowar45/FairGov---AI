# FairGov AI — Maharashtra DTE Scholarship Decision Audit

Pipeline: synthetic applicant dataset → dataset validation → XGBoost prediction → SHAP explanation → policy evidence retrieval → independent policy rule checker → mismatch detection → FastAPI.

**Important:** The 5,000 applicant training rows are SYNTHETIC. They are not real scholarship decisions. Policy retrieval uses TF-IDF over curated source-linked summaries (not yet full LLM RAG). The rule engine models a subset of scheme requirements, and its result is not an official scholarship decision.

Official sources: [MahaDBT DTE scholarship](https://mahadbt.maharashtra.gov.in/SchemeData/SchemeData?str=E9DDFA703C38E51A7F4D327BDEB7125DE0BA4AE1C51180C272281017EBEF6F7C) · [DTE](https://dte.maharashtra.gov.in/rajarshi-chhatrapati-shahu-maharaj/)

## Local setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
python -m pytest -q
uvicorn main:app --reload
```
Swagger: http://127.0.0.1:8000/docs

## API routes
`GET /validation`, `GET /metrics`, `POST /predict`, `POST /explain`, `POST /rag/retrieve`.

Full project archive was generated separately in this conversation. Code upload is in progress; this README alone does not constitute a runnable project.
