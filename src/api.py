import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import (
    load_data, create_documents,
    build_vectorstore, load_vectorstore,
    build_rag_chain, nl_to_sql, run_sql
)

load_dotenv()

# ── Initialize on module load ─────────────────────────────────────
print("Loading data...")
sales, reps, mkt = load_data()
docs = create_documents(sales, reps, mkt)

if not os.path.exists("data/faiss_index"):
    vectorstore = build_vectorstore(docs)
else:
    vectorstore = load_vectorstore()

chain = build_rag_chain(vectorstore)
print("PharmaIQ API ready")

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="PharmaIQ API", version="1.0")

# ── Request/Response models ───────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str

class SQLResponse(BaseModel):
    question: str
    sql: str
    results: list

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "PharmaIQ Sales Intelligence API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QuestionRequest):
    try:
        answer = chain.invoke(request.question)
        return RAGResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sql", response_model=SQLResponse)
async def sql_endpoint(request: QuestionRequest):
    try:
        sql = nl_to_sql(request.question)
        result_df = run_sql(sql)
        results = result_df.to_dict(orient='records')
        return SQLResponse(
            question=request.question,
            sql=sql,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))