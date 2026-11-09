import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file).
# Variables already set in the shell take precedence (override=False).
load_dotenv(Path(__file__).resolve().parent / '.env', override=False)

from src.service import predict, evaluate
from src.rag_full import answer, retrieve, load_documents
from src.rules import ROOT

app = FastAPI(title='FairGov AI — DTE ML + Grounded RAG', version='2.0')


class Student(BaseModel):
    annual_income: int = Field(ge=0)
    maharashtra_domicile: int = Field(ge=0, le=1)
    indian_national: int = Field(ge=0, le=1)
    cap_admission: int = Field(ge=0, le=1)
    bonafide: int = Field(ge=0, le=1)
    professional_course: int = Field(ge=0, le=1)
    private_or_deemed_university: int = Field(ge=0, le=1)
    other_scholarship: int = Field(ge=0, le=1)
    family_beneficiaries: int = Field(ge=0)
    attendance_pct: float = Field(ge=0, le=100)
    first_semester: int = Field(ge=0, le=1)
    gap_years: int = Field(ge=0)


class EvaluationRequest(Student):
    """Student fields + evaluation options for the decision-router endpoint."""
    language: str = Field(
        default='English',
        pattern='^(English|Marathi|Hindi)$',
        description='Language for policy explanation in the rejection response (Route B).',
    )
    top_k: int = Field(
        default=4, ge=1, le=10,
        description='Number of policy documents to retrieve for Route B explanation.',
    )


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    language: str = Field(default='English', pattern='^(English|Marathi|Hindi)$')


@app.get('/')
def home():
    return {
        'status': 'running',
        'docs': '/docs',
        'scheme': 'Maharashtra DTE Rajarshi scholarship',
        'scope': 'academic prototype',
    }


@app.get('/validation')
def validation():
    path = ROOT / 'reports/dataset_validation.json'
    if not path.exists():
        raise HTTPException(503, 'Run python -m src.pipeline')
    return json.loads(path.read_text())


@app.get('/metrics')
def metrics():
    path = ROOT / 'reports/model_evaluation.json'
    if not path.exists():
        raise HTTPException(503, 'Run python -m src.pipeline')
    return json.loads(path.read_text())


@app.post('/evaluate')
def evaluate_application(req: EvaluationRequest):
    """Main workflow endpoint — implements the 3-path decision router.

    Route A (both eligible)  → forward_to_employee: true
    Route B (both rejected)  → SHAP + RAG explanation sent to student
    Route C (mismatch)       → forward_to_employee: true for manual review
    """
    if not (ROOT / 'models/xgboost_dte.joblib').exists():
        raise HTTPException(503, 'Run python -m src.pipeline')
    data = req.model_dump()
    lang = data.pop('language')
    top_k = data.pop('top_k')
    return evaluate(data, language=lang, top_k=top_k)


@app.post('/predict')
def prediction(student: Student):
    if not (ROOT / 'models/xgboost_dte.joblib').exists():
        raise HTTPException(503, 'Run python -m src.pipeline')
    return predict(student.model_dump())


@app.post('/explain')
def explanation(student: Student):
    if not (ROOT / 'models/xgboost_dte.joblib').exists():
        raise HTTPException(503, 'Run python -m src.pipeline')
    return predict(student.model_dump(), True)


@app.get('/rag/sources')
def sources():
    return [
        {'source_id': d['source_id'], 'title': d['title'],
         'url': d['url'], 'provenance': d['provenance']}
        for d in load_documents()
    ]


@app.post('/rag/retrieve')
def rag_retrieve(q: Question):
    return retrieve(q.question, q.top_k)


@app.post('/rag/ask')
def rag_ask(q: Question):
    return answer(q.question, q.top_k, q.language)
