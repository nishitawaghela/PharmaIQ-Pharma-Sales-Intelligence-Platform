import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaIQ",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styling ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #0F1226 0%, #171B3A 45%, #1F2547 100%);
        color: #EDEBFF;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero banner */
    .pharmaiq-hero {
        background: linear-gradient(120deg, #7C3AED 0%, #DB2777 55%, #F59E0B 100%);
        border-radius: 24px;
        padding: 2.6rem 2.8rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 20px 50px -20px rgba(124, 58, 237, 0.55);
        position: relative;
        overflow: hidden;
    }
    .pharmaiq-hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.25), transparent 70%);
        border-radius: 50%;
    }
    .pharmaiq-hero h1 {
        font-family: 'Sora', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        color: white;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .pharmaiq-hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
        max-width: 640px;
    }
    .pharmaiq-badges {
        margin-top: 1.1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .pharmaiq-badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Cards */
    .pharmaiq-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(6px);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12142B 0%, #1A1D3D 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #E4E1FF !important;
    }

    .stTextInput input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    .stTextInput input::placeholder { color: rgba(237,235,255,0.45) !important; }

    div.stButton > button {
        background: linear-gradient(120deg, #7C3AED, #DB2777);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 10px 25px -10px rgba(219, 39, 119, 0.6);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px -10px rgba(219, 39, 119, 0.75);
    }

    /* Radio (mode selector) */
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }

    h2, h3 { font-family: 'Sora', sans-serif; color: #F5F3FF; }

    .stAlert { border-radius: 14px !important; }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    footer-caption {
        text-align: center;
        color: rgba(237,235,255,0.5);
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="pharmaiq-hero">
    <h1>💊 PharmaIQ</h1>
    <p>AI-powered analytics for pharma sales rep performance, drug market share, and regional insights — ask in plain English, get answers instantly.</p>
    <div class="pharmaiq-badges">
        <span class="pharmaiq-badge">⚡ Groq LLM</span>
        <span class="pharmaiq-badge">🔎 FAISS Retrieval</span>
        <span class="pharmaiq-badge">🗄️ Live SQL</span>
        <span class="pharmaiq-badge">📊 Supabase Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 PharmaIQ")
    st.markdown("**Choose your mode**")
    mode = st.radio(
        "Select query mode:",
        ["🤖 RAG — Ask AI", "🗄️ NL-to-SQL — Query Data"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**💡 Try asking**")
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
    st.divider()
    st.caption("Switch modes anytime — RAG gives narrative insight, SQL mode gives raw query results.")

# ── Main interface ────────────────────────────────────────────────
st.markdown('<div class="pharmaiq-card">', unsafe_allow_html=True)

question = st.text_input(
    "Ask a question about pharma sales:",
    placeholder="e.g. Which region has the lowest average attainment?"
)
submit = st.button("🔍 Analyse", type="primary")

st.markdown('</div>', unsafe_allow_html=True)
st.write("")

if submit and question:
    if "RAG" in mode:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/rag",
                    json={"question": question}
                )
                data = response.json()
                st.markdown("### 🤖 AI Insight")
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

                st.markdown("### 📊 Query Results")

                with st.expander("Generated SQL"):
                    st.code(data['sql'], language='sql')

                if data['results']:
                    df = pd.DataFrame(data['results'])
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"✅ {len(df)} rows returned")
                else:
                    st.warning("No results found")

            except Exception as e:
                st.error(f"Error: {str(e)}")

elif submit and not question:
    st.warning("Please enter a question")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p class="footer-caption">PharmaIQ · Built with FastAPI + Groq LLM + FAISS + Streamlit · Data: Supabase PostgreSQL</p>',
    unsafe_allow_html=True
)