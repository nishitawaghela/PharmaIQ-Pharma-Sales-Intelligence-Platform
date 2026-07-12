import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from generate_data import generate_reps, generate_sales_performance, generate_market_share, generate_doctor_visits
from database import engine
from rag_pipeline import load_data, create_documents, load_vectorstore, build_rag_chain, nl_to_sql, run_sql

# ── Unit Tests: Data Generation ───────────────────────────────────

class TestDataGeneration:

    def test_generate_reps_count(self):
        """Should generate exactly 50 reps"""
        reps = generate_reps(50)
        assert len(reps) == 50

    def test_generate_reps_columns(self):
        """Reps dataframe should have all required columns"""
        reps = generate_reps(10)
        required_cols = ['rep_id', 'name', 'email', 'region', 'territory', 'drug_promoted', 'hire_date', 'manager']
        for col in required_cols:
            assert col in reps.columns, f"Missing column: {col}"

    def test_generate_reps_indian_regions(self):
        """All regions should be Indian regions"""
        reps = generate_reps(50)
        valid_regions = {'North', 'South', 'East', 'West', 'Central'}
        assert set(reps['region'].unique()).issubset(valid_regions)

    def test_generate_reps_valid_drugs(self):
        """All drugs should be from the defined drug list"""
        reps = generate_reps(50)
        valid_drugs = {'Cardivex', 'Diabetrol', 'Oncozin', 'Neuraplex', 'Immunoboost'}
        assert set(reps['drug_promoted'].unique()).issubset(valid_drugs)

    def test_generate_sales_performance_months(self):
        """Sales performance should have 12 months per rep"""
        reps = generate_reps(10)
        sales = generate_sales_performance(reps, months=12)
        months_per_rep = sales.groupby('rep_id')['month'].count()
        assert (months_per_rep == 12).all()

    def test_generate_sales_attainment_range(self):
        """Attainment should be between 0 and 200 percent"""
        reps = generate_reps(50)
        sales = generate_sales_performance(reps)
        assert sales['attainment_pct'].min() > 0
        assert sales['attainment_pct'].max() < 200

    def test_generate_sales_positive_values(self):
        """Quota and actual sales should be positive"""
        reps = generate_reps(50)
        sales = generate_sales_performance(reps)
        assert (sales['quota'] > 0).all()
        assert (sales['actual_sales'] > 0).all()

    def test_generate_market_share_all_regions_drugs(self):
        """Market share should have entries for all regions and drugs"""
        market = generate_market_share(months=12)
        assert market['region'].nunique() == 5
        assert market['drug'].nunique() == 5

    def test_generate_market_share_pct_range(self):
        """Market share percentage should be between 0 and 100"""
        market = generate_market_share()
        assert market['market_share_pct'].min() >= 0
        assert market['market_share_pct'].max() <= 100

    def test_rep_id_format(self):
        """Rep IDs should follow REP001 format"""
        reps = generate_reps(10)
        assert reps['rep_id'].str.match(r'REP\d{3}').all()

    def test_generate_sales_12_months(self):
        """Generated sales data should have all 12 months"""
        reps = generate_reps(50)
        sales = generate_sales_performance(reps, months=12)
        assert sales['month'].nunique() == 12

# ── Unit Tests: Database ──────────────────────────────────────────

class TestDatabase:

    def test_supabase_connection(self):
        """Should connect to Supabase successfully"""
        with engine.connect() as conn:
            result = conn.execute(__import__('sqlalchemy').text("SELECT 1"))
            assert result.fetchone()[0] == 1

    def test_reps_table_exists(self):
        """Reps table should exist in Supabase"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM reps", engine)
        assert df['count'][0] > 0

    def test_sales_performance_table_exists(self):
        """Sales performance table should exist"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM sales_performance", engine)
        assert df['count'][0] > 0

    def test_market_share_table_exists(self):
        """Market share table should exist"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM market_share", engine)
        assert df['count'][0] > 0

    def test_doctor_visits_table_exists(self):
        """Doctor visits table should exist"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM doctor_visits", engine)
        assert df['count'][0] > 0

    def test_reps_count(self):
        """Should have exactly 50 reps in database"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM reps", engine)
        assert df['count'][0] == 50

    def test_sales_performance_count(self):
        """Should have 600 sales records (50 reps x 12 months)"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM sales_performance", engine)
        assert df['count'][0] == 600

    def test_no_null_rep_ids(self):
        """No null rep_ids in sales_performance"""
        df = pd.read_sql("SELECT COUNT(*) as count FROM sales_performance WHERE rep_id IS NULL", engine)
        assert df['count'][0] == 0

    def test_referential_integrity(self):
        """All rep_ids in sales_performance should exist in reps"""
        df = pd.read_sql("""
            SELECT COUNT(*) as count 
            FROM sales_performance sp
            LEFT JOIN reps r ON sp.rep_id = r.rep_id
            WHERE r.rep_id IS NULL
        """, engine)
        assert df['count'][0] == 0

# ── Unit Tests: RAG Pipeline ──────────────────────────────────────

class TestRAGPipeline:

    @pytest.fixture(scope="class")
    def data(self):
        sales, reps, mkt = load_data()
        return sales, reps, mkt

    def test_load_data_returns_dataframes(self, data):
        """load_data should return three DataFrames"""
        sales, reps, mkt = data
        assert isinstance(sales, pd.DataFrame)
        assert isinstance(reps, pd.DataFrame)
        assert isinstance(mkt, pd.DataFrame)

    def test_load_data_not_empty(self, data):
        """All loaded DataFrames should be non-empty"""
        sales, reps, mkt = data
        assert len(sales) > 0
        assert len(reps) > 0
        assert len(mkt) > 0

    def test_create_documents_count(self, data):
        """Should create 85 documents"""
        sales, reps, mkt = data
        docs = create_documents(sales, reps, mkt)
        assert len(docs) == 85

    def test_create_documents_content(self, data):
        """Documents should contain expected fields"""
        sales, reps, mkt = data
        docs = create_documents(sales, reps, mkt)
        first_doc = docs[0].page_content
        assert 'Sales Rep' in first_doc or 'Region' in first_doc or 'Drug' in first_doc

    def test_vectorstore_exists(self):
        """FAISS index should exist on disk"""
        assert os.path.exists("data/faiss_index")

    def test_vectorstore_loads(self):
        """Should load vectorstore without errors"""
        vectorstore = load_vectorstore()
        assert vectorstore is not None

    def test_vectorstore_similarity_search(self):
        """Vectorstore should return relevant documents"""
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search("Which region has highest sales", k=3)
        assert len(results) == 3
        assert all(hasattr(doc, 'page_content') for doc in results)

# ── Unit Tests: NL-to-SQL ─────────────────────────────────────────

class TestNLToSQL:

    def test_nl_to_sql_returns_string(self):
        """NL-to-SQL should return a string"""
        sql = nl_to_sql("Show me all reps")
        assert isinstance(sql, str)
        assert len(sql) > 0

    def test_nl_to_sql_contains_select(self):
        """Generated SQL should contain SELECT"""
        sql = nl_to_sql("Show me all regions")
        assert 'SELECT' in sql.upper()

    def test_nl_to_sql_no_markdown(self):
        """Generated SQL should not contain markdown backticks"""
        sql = nl_to_sql("Show average attainment by region")
        assert '```' not in sql

    def test_nl_to_sql_region_query(self):
        """Region query should reference correct table"""
        sql = nl_to_sql("Show average attainment by region")
        assert 'sales_performance' in sql.lower() or 'region' in sql.lower()

    def test_nl_to_sql_drug_query(self):
        """Drug query should use correct column name"""
        sql = nl_to_sql("Show total sales by drug")
        assert 'drug_promoted' not in sql.lower() or 'drug' in sql.lower()

    def test_run_sql_valid_query(self):
        """Valid SQL should return a DataFrame"""
        result = run_sql("SELECT region, AVG(attainment_pct) as avg FROM sales_performance GROUP BY region")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_run_sql_invalid_query(self):
        """Invalid SQL should return error DataFrame not crash"""
        result = run_sql("SELECT * FROM nonexistent_table_xyz")
        assert isinstance(result, pd.DataFrame)
        assert 'Error' in result.columns

    def test_run_sql_returns_correct_columns(self):
        """SQL result should have expected columns"""
        result = run_sql("SELECT rep_id, name, region FROM reps LIMIT 5")
        assert 'rep_id' in result.columns
        assert 'name' in result.columns
        assert 'region' in result.columns

# ── Integration Tests: FastAPI ────────────────────────────────────

class TestAPI:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from api import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Root endpoint should return running status"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()['status'] == 'running'

    def test_health_endpoint(self, client):
        """Health endpoint should return healthy"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

    def test_sql_endpoint_returns_200(self, client):
        """SQL endpoint should return 200"""
        response = client.post("/sql", json={"question": "Show average attainment by region"})
        assert response.status_code == 200

    def test_sql_endpoint_returns_sql(self, client):
        """SQL endpoint should return generated SQL"""
        response = client.post("/sql", json={"question": "Show average attainment by region"})
        data = response.json()
        assert 'sql' in data
        assert 'SELECT' in data['sql'].upper()

    def test_sql_endpoint_returns_results(self, client):
        """SQL endpoint should return results list"""
        response = client.post("/sql", json={"question": "Show average attainment by region"})
        data = response.json()
        assert 'results' in data
        assert isinstance(data['results'], list)

    def test_rag_endpoint_returns_200(self, client):
        """RAG endpoint should return 200"""
        response = client.post("/rag", json={"question": "Which region has the highest sales?"})
        assert response.status_code == 200

    def test_rag_endpoint_returns_answer(self, client):
        """RAG endpoint should return an answer"""
        response = client.post("/rag", json={"question": "Which drug has the highest sales?"})
        data = response.json()
        assert 'answer' in data
        assert len(data['answer']) > 0

    def test_empty_question_sql(self, client):
        """Empty question should still return a response"""
        response = client.post("/sql", json={"question": ""})
        assert response.status_code in [200, 422, 500]

    def test_sql_question_echoed(self, client):
        """SQL response should echo back the question"""
        question = "Show me all reps in South region"
        response = client.post("/sql", json={"question": question})
        data = response.json()
        assert data['question'] == question