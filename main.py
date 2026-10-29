from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.service import predict
from src.rag import retrieve
from src.rules import ROOT
import json

app = FastAPI(title="FairGov AI — DTE ML / Explainability / Policy Retrieval", version="1.0")

class Student(BaseModel):
    annual_income: int = Field(ge=0)
    maharashtra_domicile: int = Field(ge=0,le=1)
    indian_national: int = Field(ge=0,le=1)
    cap_admission: int = Field(ge=0,le=1)
    bonafide: int = Field(ge=0,le=1)
    professional_course: int = Field(ge=0,le=1)
    private_or_deemed_university: int = Field(ge=0,le=1)
    other_scholarship: int = Field(ge=0,le=1)
    family_beneficiaries: int = Field(ge=0)
    attendance_pct: float = Field(ge=0,le=100)
    first_semester: int = Field(ge=0,le=1)
    gap_years: int = Field(ge=0)

class Question(BaseModel):
    question: str = Field(min_length=3)

@app.get("/")
def home():
    return {"status":"running","scheme":"MahaDBT DTE Rajarshi Shahu Maharaj","docs":"/docs"}

@app.get("/metrics")
def metrics():
    p = ROOT / "reports/model_evaluation.json"
    if not p.exists(): raise HTTPException(503,"Run python -m src.pipeline first")
    return json.loads(p.read_text())

@app.get("/validation")
def validation():
    p = ROOT / "reports/dataset_validation.json"
    if not p.exists(): raise HTTPException(503,"Run python -m src.pipeline first")
    return json.loads(p.read_text())

@app.post("/predict")
def prediction(x:Student):
    try: return predict(x.model_dump())
    except FileNotFoundError: raise HTTPException(503,"Run python -m src.pipeline first")

@app.post("/explain")
def explain(x:Student):
    try: return predict(x.model_dump(),True)
    except FileNotFoundError: raise HTTPException(503,"Run python -m src.pipeline first")

@app.post("/rag/retrieve")
def rag(q:Question):
    return retrieve(q.question)
