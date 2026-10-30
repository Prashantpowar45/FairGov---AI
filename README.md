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

## Google Colab ML Evaluation

Open [`notebooks/FairGov_DTE_ML_Colab_Evaluation.ipynb`](notebooks/FairGov_DTE_ML_Colab_Evaluation.ipynb) in Google Colab and run cells in order. Includes 5,000-record synthetic generation, validation, XGBoost evaluation, SHAP waterfall, and manually expected boundary-case rule/ML comparisons. The Colab notebook clones the public repository; this requires internet access. It does not use real scholarship applicant labels. A separate self-contained notebook can be obtained from the project conversation.
