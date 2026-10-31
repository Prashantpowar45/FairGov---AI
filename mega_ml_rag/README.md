# FairGov AI — Full RAG extension

Use the existing top-level source for 5,000 synthetic dataset records, validation, XGBoost, SHAP and independent rule checking.

This folder adds optional Gemini grounded policy-answer generation. Dataset is **synthetic**, not real government applicant data.

## Install and run from repository root

```powershell
pip install -r mega_ml_rag/requirements.txt
python -m src.pipeline
uvicorn mega_ml_rag.main:app --reload
```

Open http://127.0.0.1:8000/docs.

For Gemini answers set `GEMINI_API_KEY` in your shell. Without a key, `/rag/ask` returns evidence only.

To add full official policy documents, create `policies/official_documents` at the repository root and place verified PDFs/TXT there. Restart the server to rebuild the retrieval index.

**Limitations:** This uses TF-IDF vector retrieval plus optional Gemini generation, not a neural embedding vector DB. RAG snippets are curated evidence summaries until actual official PDFs are added. No claim of real-world ML model accuracy or real government approvals. Never send personal applicants' data to the LLM.
