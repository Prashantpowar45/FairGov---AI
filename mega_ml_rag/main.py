from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.rules import ROOT
from src.service import predict
from mega_ml_rag.src.rag_full import retrieve,answer
import json

app=FastAPI(title="FairGov AI Mega ML + RAG")
class Student(BaseModel):
    annual_income:int=Field(ge=0)
    maharashtra_domicile:int=Field(ge=0,le=1)
    indian_national:int=Field(ge=0,le=1)
    cap_admission:int=Field(ge=0,le=1)
    bonafide:int=Field(ge=0,le=1)
    professional_course:int=Field(ge=0,le=1)
    private_or_deemed_university:int=Field(ge=0,le=1)
    other_scholarship:int=Field(ge=0,le=1)
    family_beneficiaries:int=Field(ge=0)
    attendance_pct:float=Field(ge=0,le=100)
    first_semester:int=Field(ge=0,le=1)
    gap_years:int=Field(ge=0)
class Question(BaseModel):
    question:str=Field(min_length=3,max_length=2000)
    top_k:int=Field(default=4,ge=1,le=10)
    language:str="English"
@app.get("/")
def home():return {"status":"running","scope":"academic proof-of-concept"}
@app.get("/validation")
def validation():return json.loads((ROOT/"reports/dataset_validation.json").read_text())
@app.get("/metrics")
def metrics():return json.loads((ROOT/"reports/model_evaluation.json").read_text())
@app.post("/predict")
def prediction(x:Student):return predict(x.model_dump())
@app.post("/explain")
def explain(x:Student):return predict(x.model_dump(),True)
@app.post("/rag/retrieve")
def get_relevant(q:Question):return retrieve(q.question,q.top_k)
@app.post("/rag/ask")
def grounded(q:Question):return answer(q.question,q.top_k,q.language)
