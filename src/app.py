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
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Source Sans 3', sans-serif;
    }

    /* App background — soft neutral off-white */
    .stApp {
        background: #F7F8F5;
        color: #1F2A24;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Hero banner */
    .pharmaiq-hero {
        background: #FFFFFF;
        border: 1px solid #E3E8DF;
        border-left: 6px solid #2F7A50;
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.6rem;
    }
    .pharmaiq-hero h1 {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        color: #1B4332;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.3px;
    }
    .pharmaiq-hero p {
        color: #4A5A50;
        font-size: 1.02rem;
        margin: 0;
        max-width: 640px;
    }
    .pharmaiq-badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .pharmaiq-badge {
        background: #EAF3EC;
        border: 1px solid #CDE3D3;
        color: #2F7A50;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    /* Card */
    .pharmaiq-card {
        background: #FFFFFF;
        border: 1px solid #E3E8DF;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
    }

    section[data-testid="stSidebar"] {
        background: #FBFCF9;
        border-right: 1px solid #E3E8DF;
    }
    section[data-testid="stSidebar"] * {
        color: #2A352F !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #1B4332 !important;
        font-family: 'Manrope', sans-serif;
    }

    .stTextInput input {
        background: #FFFFFF !important;
        border: 1px solid #D3DCD0 !important;
        border-radius: 10px !important;
        color: #1F2A24 !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
    }
    .stTextInput input::placeholder { color: #8A968E !important; }
    .stTextInput input:focus {
        border-color: #2F7A50 !important;
        box-shadow: 0 0 0 2px rgba(47,122,80,0.15) !important;
    }

    div.stButton > button {
        background: #2F7A50;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    div.stButton > button:hover {
        background: #26603F;
        transform: translateY(-1px);
    }

    div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #E3E8DF;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }

    h2, h3 { font-family: 'Manrope', sans-serif; color: #1B4332; }

    .stAlert { border-radius: 12px !important; }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E3E8DF;
        border-radius: 10px;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #E3E8DF; }

    .footer-caption {
        text-align: center;
        color: #8A968E;
        font-size: 0.82rem;
    }

    hr { border-color: #E3E8DF !important; }
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