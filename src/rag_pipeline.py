import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))

# ── Step 1: Load data from Supabase ──────────────────────────────
def load_data():
    sales = pd.read_sql("SELECT * FROM sales_performance", engine)
    reps  = pd.read_sql("SELECT * FROM reps", engine)
    mkt   = pd.read_sql("SELECT * FROM market_share", engine)
    return sales, reps, mkt

# ── Step 2: Convert data to documents for RAG ────────────────────
def create_documents(sales, reps, mkt):
    docs = []
    
    # Rep performance summaries
    rep_summary = (sales.groupby('rep_id')
                   .agg(avg_attainment=('attainment_pct', 'mean'),
                        total_sales=('actual_sales', 'sum'),
                        total_visits=('total_visits', 'sum'))
                   .reset_index())
    rep_summary = rep_summary.merge(reps[['rep_id','name','region','territory','drug_promoted']], on='rep_id')
    
    for _, row in rep_summary.iterrows():
        status = "above quota" if row['avg_attainment'] >= 100 else "below quota"
        doc = Document(
            page_content=f"""
            Sales Rep: {row['name']} (ID: {row['rep_id']})
            Region: {row['region']}, Territory: {row['territory']}
            Drug Promoted: {row['drug_promoted']}
            Average Attainment: {row['avg_attainment']:.1f}% ({status})
            Total Sales: ₹{row['total_sales']:,.0f}
            Total Doctor Visits: {row['total_visits']}
            """,
            metadata={'type': 'rep_performance', 'rep_id': row['rep_id']}
        )
        docs.append(doc)
    
    # Regional summaries
    region_summary = (sales.groupby('region')
                      .agg(total_sales=('actual_sales', 'sum'),
                           avg_attainment=('attainment_pct', 'mean'))
                      .reset_index())
    
    for _, row in region_summary.iterrows():
        doc = Document(
            page_content=f"""
            Region: {row['region']}
            Total Sales: ₹{row['total_sales']:,.0f}
            Average Attainment: {row['avg_attainment']:.1f}%
            """,
            metadata={'type': 'region_summary'}
        )
        docs.append(doc)
    
    # Drug summaries
    drug_summary = (sales.groupby('drug')
                    .agg(total_sales=('actual_sales', 'sum'),
                         avg_attainment=('attainment_pct', 'mean'))
                    .reset_index())
    
    for _, row in drug_summary.iterrows():
        doc = Document(
            page_content=f"""
            Drug: {row['drug']}
            Total Sales: ₹{row['total_sales']:,.0f}
            Average Attainment: {row['avg_attainment']:.1f}%
            """,
            metadata={'type': 'drug_summary'}
        )
        docs.append(doc)
    
    # Market share summaries
    mkt_summary = (mkt.groupby(['drug', 'region'])
                   .agg(avg_market_share=('market_share_pct', 'mean'),
                        avg_prescriptions=('total_prescriptions', 'mean'))
                   .reset_index())
    
    for _, row in mkt_summary.iterrows():
        doc = Document(
            page_content=f"""
            Drug: {row['drug']} in {row['region']} region
            Average Market Share: {row['avg_market_share']:.1f}%
            Average Monthly Prescriptions: {row['avg_prescriptions']:.0f}
            """,
            metadata={'type': 'market_share'}
        )
        docs.append(doc)
    
    return docs

# ── Step 3: Build FAISS vector store ────────────────────────────
def build_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv('GEMINI_API_KEY')
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("data/faiss_index")
    print(f"Vector store built with {len(docs)} documents")
    return vectorstore

# ── Step 4: RAG Chain ────────────────────────────────────────────
def build_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0.1
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = PromptTemplate.from_template("""
    You are a pharma sales analytics assistant for PharmaIQ.
    Answer questions about sales rep performance, regional analytics, 
    drug performance and market share using the provided context.
    Always mention specific numbers and names when available.
    If you don't know, say so clearly.
    
    Context: {context}
    Question: {question}
    Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Step 5: NL-to-SQL ────────────────────────────────────────────
def nl_to_sql(question: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0
    )
    
    schema = """
    Tables:
    - reps(rep_id, name, email, region, territory, drug_promoted, hire_date, manager)
    - sales_performance(id, rep_id, month, quota, actual_sales, attainment_pct, drug, region, territory, new_doctors_reached, total_visits)
    - market_share(id, month, region, drug, market_share_pct, total_prescriptions, competitor_share_pct)
    - doctor_visits(id, rep_id, doctor_id, doctor_name, specialty, visit_date, drug_detailed, follow_up, prescription_generated)
    """
    
    prompt = f"""You are a SQL expert. Convert this natural language question to a PostgreSQL query.
    
    {schema}
    
    Rules:
    - Return ONLY the SQL query, nothing else
    - No markdown, no backticks, no explanation
    - Use proper PostgreSQL syntax
    - Always LIMIT to 20 rows unless asked for all
    
    Question: {question}
    SQL:"""
    
    response = llm.invoke(prompt)
    sql = response.content.strip()
    # clean up any markdown if model adds it
    sql = sql.replace('```sql', '').replace('```', '').strip()
    return sql

def run_sql(sql: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(sql), conn)
        return result
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

# ── Main: test everything ─────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    sales, reps, mkt = load_data()
    
    print("Creating documents...")
    docs = create_documents(sales, reps, mkt)
    
    print("Building vector store...")
    vectorstore = build_vectorstore(docs)
    
    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)
    
    # Test RAG
    print("\n--- RAG TEST ---")
    questions = [
        "Which region has the highest sales?",
        "Who are the top performing sales reps?",
        "Which drug has the lowest market share?"
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}")
    # Test NL-to-SQL
    print("\n--- NL-TO-SQL TEST ---")
    nl_questions = [
        "Show me all reps with average attainment below 80%",
        "Which territory has the highest total sales?",
        "Show me top 5 reps by total visits"
    ]
    for q in nl_questions:
        print(f"\nQ: {q}")
        sql = nl_to_sql(q)
        print(f"SQL: {sql}")
        result = run_sql(sql)
        print(result.head())