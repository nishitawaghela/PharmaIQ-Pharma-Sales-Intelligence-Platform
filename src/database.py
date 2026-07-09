import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))

def create_tables():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reps (
                rep_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                email VARCHAR,
                region VARCHAR,
                territory VARCHAR,
                drug_promoted VARCHAR,
                hire_date DATE,
                manager VARCHAR
            );
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sales_performance (
                id SERIAL PRIMARY KEY,
                rep_id VARCHAR,
                month VARCHAR,
                quota FLOAT,
                actual_sales FLOAT,
                attainment_pct FLOAT,
                drug VARCHAR,
                region VARCHAR,
                territory VARCHAR,
                new_doctors_reached INT,
                total_visits INT
            );
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_share (
                id SERIAL PRIMARY KEY,
                month VARCHAR,
                region VARCHAR,
                drug VARCHAR,
                market_share_pct FLOAT,
                total_prescriptions INT,
                competitor_share_pct FLOAT
            );
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS doctor_visits (
                id SERIAL PRIMARY KEY,
                rep_id VARCHAR,
                doctor_id VARCHAR,
                doctor_name VARCHAR,
                specialty VARCHAR,
                visit_date VARCHAR,
                drug_detailed VARCHAR,
                follow_up BOOLEAN,
                prescription_generated BOOLEAN
            );
        """))
        conn.commit()
        print("Tables created successfully")

def load_data():
    reps = pd.read_csv('data/reps.csv')
    sales = pd.read_csv('data/sales_performance.csv')
    market = pd.read_csv('data/market_share.csv')
    visits = pd.read_csv('data/doctor_visits.csv')
    
    reps.to_sql('reps', engine, if_exists='replace', index=False)
    sales.to_sql('sales_performance', engine, if_exists='replace', index=False)
    market.to_sql('market_share', engine, if_exists='replace', index=False)
    visits.to_sql('doctor_visits', engine, if_exists='replace', index=False)
    
    print(f"Loaded {len(reps)} reps")
    print(f"Loaded {len(sales)} sales records")
    print(f"Loaded {len(market)} market share records")
    print(f"Loaded {len(visits)} doctor visit records")

if __name__ == '__main__':
    create_tables()
    load_data()