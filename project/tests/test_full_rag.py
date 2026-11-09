from fastapi.testclient import TestClient
from main import app
from src.rag_full import retrieve,answer,_chunks

def test_chunks_and_retrieval():
    assert _chunks('Hello there') == ['Hello there']
    assert retrieve('annual income eight lakh scholarship')['hits']

def test_no_key_returns_retrieval(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    r=answer('scholarship income limit')
    assert r['answer_mode']=='retrieval_only'
    assert r['hits']

def test_endpoints():
    c=TestClient(app)
    assert c.get('/').status_code==200
    assert c.post('/rag/retrieve',json={'question':'CAP admission scholarship'}).status_code==200
    assert c.post('/rag/ask',json={'question':'family income limit'}).status_code==200
    assert c.get('/rag/sources').status_code==200
