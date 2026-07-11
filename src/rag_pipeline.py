import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ── Embeddings (local, free, no API needed) ───────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# ── Step 1: Load data ─────────────────────────────────────────────
def load_data():
    sales = pd.read_sql("SELECT * FROM sales_performance", engine)
    reps  = pd.read_sql("SELECT * FROM reps", engine)
    mkt   = pd.read_sql("SELECT * FROM market_share", engine)
    return sales, reps, mkt

# ── Step 2: Create documents ──────────────────────────────────────
def create_documents(sales, reps, mkt):
    docs = []

    rep_summary = (sales.groupby('rep_id')
                   .agg(avg_attainment=('attainment_pct', 'mean'),
                        total_sales=('actual_sales', 'sum'),
                        total_visits=('total_visits', 'sum'))
                   .reset_index())
    rep_summary = rep_summary.merge(
        reps[['rep_id','name','region','territory','drug_promoted']],
        on='rep_id')

    for _, row in rep_summary.iterrows():
        status = "above quota" if row['avg_attainment'] >= 100 else "below quota"
        docs.append(Document(page_content=f"""
        Sales Rep: {row['name']} (ID: {row['rep_id']})
        Region: {row['region']}, Territory: {row['territory']}
        Drug Promoted: {row['drug_promoted']}
        Average Attainment: {row['avg_attainment']:.1f}% ({status})
        Total Sales: Rs {row['total_sales']:,.0f}
        Total Doctor Visits: {row['total_visits']}
        """))

    region_summary = (sales.groupby('region')
                      .agg(total_sales=('actual_sales', 'sum'),
                           avg_attainment=('attainment_pct', 'mean'))
                      .reset_index())

    for _, row in region_summary.iterrows():
        docs.append(Document(page_content=f"""
        Region: {row['region']}
        Total Sales: Rs {row['total_sales']:,.0f}
        Average Attainment: {row['avg_attainment']:.1f}%
        """))

    drug_summary = (sales.groupby('drug')
                    .agg(total_sales=('actual_sales', 'sum'),
                         avg_attainment=('attainment_pct', 'mean'))
                    .reset_index())

    for _, row in drug_summary.iterrows():
        docs.append(Document(page_content=f"""
        Drug: {row['drug']}
        Total Sales: Rs {row['total_sales']:,.0f}
        Average Attainment: {row['avg_attainment']:.1f}%
        """))

    mkt_summary = (mkt.groupby(['drug','region'])
                   .agg(avg_market_share=('market_share_pct', 'mean'),
                        avg_prescriptions=('total_prescriptions', 'mean'))
                   .reset_index())

    for _, row in mkt_summary.iterrows():
        docs.append(Document(page_content=f"""
        Drug: {row['drug']} in {row['region']} region
        Average Market Share: {row['avg_market_share']:.1f}%
        Average Monthly Prescriptions: {row['avg_prescriptions']:.0f}
        """))

    return docs

# ── Step 3: Build FAISS vectorstore ──────────────────────────────
def build_vectorstore(docs):
    print("Building embeddings...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("data/faiss_index")
    print(f"Vector store built with {len(docs)} documents")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ── Step 4: RAG Chain using Groq ─────────────────────────────────
def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.1
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate.from_template("""
    You are a pharma sales analytics assistant for PharmaIQ.
    Answer questions about sales rep performance, regional analytics,
    drug performance and market share using ONLY the provided context.
    Always mention specific numbers and names when available.
    If the answer is not in the context, say "I don't have enough data."

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

# ── Step 5: NL-to-SQL using Groq ─────────────────────────────────
def nl_to_sql(question: str) -> str:
    schema = """
    Tables and their EXACT column names:

    reps(rep_id, name, email, region, territory, drug_promoted, hire_date, manager)

    sales_performance(id, rep_id, month, quota, actual_sales, attainment_pct, 
                  drug, region, territory, new_doctors_reached, total_visits)
    NOTE: in sales_performance, the drug column is called 'drug' NOT 'drug_promoted'

    market_share(id, month, region, drug, market_share_pct, 
             total_prescriptions, competitor_share_pct)

    doctor_visits(id, rep_id, doctor_id, doctor_name, specialty, 
              visit_date, drug_detailed, follow_up, prescription_generated)

    IMPORTANT RULES:
    - Use 'drug' column from sales_performance for drug-related queries on sales
    - Use 'drug_promoted' from reps only for rep profile queries
    - Always use exact column names as listed above
    """

    prompt = f"""You are a SQL expert. Convert this question to a PostgreSQL query.

{schema}

Rules:
- Return ONLY the SQL query, nothing else
- No markdown, no backticks, no explanation
- Use proper PostgreSQL syntax
- Always LIMIT to 20 rows unless asked for all

Question: {question}
SQL:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.replace('```sql', '').replace('```', '').strip()
    return sql

def run_sql(sql: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(sql), conn)
        return result
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data from Supabase...")
    sales, reps, mkt = load_data()

    print("Creating documents...")
    docs = create_documents(sales, reps, mkt)

    if not os.path.exists("data/faiss_index"):
        print("Building vector store...")
        vectorstore = build_vectorstore(docs)
    else:
        print("Loading existing vector store...")
        vectorstore = load_vectorstore()

    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)

    print("\n" + "="*50)
    print("PharmaIQ Sales Intelligence Assistant")
    print("="*50)
    print("Commands: 'rag' for AI answer, 'sql' for data query, 'quit' to exit")
    print("="*50 + "\n")

    while True:
        mode = input("Mode (rag/sql): ").strip().lower()

        if mode == 'quit':
            print("Goodbye!")
            break

        question = input("Your question: ").strip()

        if mode == 'rag':
            print("\nThinking...")
            answer = chain.invoke(question)
            print(f"\nAnswer: {answer}\n")

        elif mode == 'sql':
            print("\nGenerating SQL...")
            sql = nl_to_sql(question)
            print(f"SQL: {sql}")
            result = run_sql(sql)
            print(f"\nResult:")
            print(result.to_string())
            print()

        else:
            print("Invalid mode. Type 'rag', 'sql', or 'quit'\n")