import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .rules import ROOT

DOCS = json.loads((ROOT / "policies/policy_evidence.json").read_text(encoding="utf-8"))
VECTORIZER = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
MATRIX = VECTORIZER.fit_transform([d["title"] + " " + d["text"] for d in DOCS])

def retrieve(question, top_k=3):
    q = VECTORIZER.transform([question])
    sim = cosine_similarity(q,MATRIX).ravel()
    idx = sim.argsort()[::-1][:top_k]
    return {
        "query": question,
        "retrieval_type": "TF-IDF evidence retrieval (not full generative RAG)",
        "results": [{"score": round(float(sim[i]),4), **DOCS[i]} for i in idx if sim[i]>0],
        "warning": "Policy evidence excerpts are brief verified-source summaries; inspect official GRs and effective date before any legal conclusion."
    }
