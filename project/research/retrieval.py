from src.rag_full import retrieve,answer,load_documents as documents,_chunks as chunks

def query(question='What is the annual family income limit?'):
    return retrieve(question,top_k=4)

def grounded_answer(question,language='English'):
    return answer(question,top_k=4,language=language)
