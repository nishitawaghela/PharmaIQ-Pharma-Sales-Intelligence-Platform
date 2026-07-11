import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaIQ",
    page_icon="💊",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────
st.title("💊 PharmaIQ — Pharma Sales Intelligence Platform")
st.markdown("AI-powered analytics for pharma sales rep performance, drug market share, and regional insights.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("PharmaIQ")
    st.markdown("**Mode**")
    mode = st.radio("Select query mode:", ["🤖 RAG — Ask AI", "🗄️ NL-to-SQL — Query Data"])
    st.divider()
    st.markdown("**Sample Questions**")
    if "RAG" in mode:
        st.markdown("""
        - Which region has the lowest attainment?
        - Which drug is performing best?
        - Who are the top performing reps?
        - Which region needs more support?
        """)
    else:
        st.markdown("""
        - Show average attainment by region
        - Show total sales by drug
        - Show top 10 reps by visits
        - Show all reps in South region
        """)

# ── Main interface ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input(
        "Ask a question about pharma sales:",
        placeholder="e.g. Which region has the lowest average attainment?"
    )

    submit = st.button("🔍 Analyse", type="primary")

if submit and question:
    if "RAG" in mode:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/rag",
                    json={"question": question}
                )
                data = response.json()
                st.subheader("🤖 AI Insight")
                st.success(data['answer'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        with st.spinner("Generating SQL and querying database..."):
            try:
                response = requests.post(
                    f"{API_URL}/sql",
                    json={"question": question}
                )
                data = response.json()

                st.subheader("📊 Query Results")

                with st.expander("Generated SQL"):
                    st.code(data['sql'], language='sql')

                if data['results']:
                    df = pd.DataFrame(data['results'])
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} rows returned")
                else:
                    st.warning("No results found")

            except Exception as e:
                st.error(f"Error: {str(e)}")

elif submit and not question:
    st.warning("Please enter a question")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("PharmaIQ | Built with FastAPI + Groq LLM + FAISS + Streamlit | Data: Supabase PostgreSQL")