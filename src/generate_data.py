import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

# Constants
REGIONS = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
DRUGS = ['Cardivex', 'Diabetrol', 'Oncozin', 'Neuraplex', 'Immunoboost']
SPECIALTIES = ['Cardiologist', 'Endocrinologist', 'Oncologist', 'Neurologist', 'Immunologist']
DRUG_SPECIALTY_MAP = {
    'Cardivex': 'Cardiologist',
    'Diabetrol': 'Endocrinologist',
    'Oncozin': 'Oncologist',
    'Neuraplex': 'Neurologist',
    'Immunoboost': 'Immunologist'
}

def generate_reps(n=50):
    reps = []
    for i in range(n):
        drug = random.choice(DRUGS)
        region = random.choice(REGIONS)
        reps.append({
            'rep_id': f'REP{i+1:03d}',
            'name': fake.name(),
            'email': fake.email(),
            'region': region,
            'territory': f'{region}-{random.randint(1,5)}',
            'drug_promoted': drug,
            'hire_date': fake.date_between(start_date='-5y', end_date='-6m'),
            'manager': fake.name()
        })
    return pd.DataFrame(reps)

def generate_sales_performance(reps_df, months=12):
    records = []
    start_date = datetime(2024, 1, 1)
    
    for _, rep in reps_df.iterrows():
        # some reps are consistently high, some low, some average
        performance_tier = random.choice(['high', 'medium', 'low'])
        base_quota = random.randint(80000, 150000)
        
        for month in range(months):
            date = start_date + timedelta(days=30*month)
            
            if performance_tier == 'high':
                attainment = random.uniform(0.95, 1.30)
            elif performance_tier == 'medium':
                attainment = random.uniform(0.75, 1.05)
            else:
                attainment = random.uniform(0.40, 0.80)
            
            quota = base_quota * random.uniform(0.9, 1.1)
            actual_sales = quota * attainment
            
            records.append({
                'rep_id': rep['rep_id'],
                'month': date.strftime('%Y-%m'),
                'quota': round(quota, 2),
                'actual_sales': round(actual_sales, 2),
                'attainment_pct': round(attainment * 100, 2),
                'drug': rep['drug_promoted'],
                'region': rep['region'],
                'territory': rep['territory'],
                'new_doctors_reached': random.randint(5, 30),
                'total_visits': random.randint(20, 80)
            })
    return pd.DataFrame(records)

def generate_market_share(months=12):
    records = []
    start_date = datetime(2024, 1, 1)
    
    for month in range(months):
        date = start_date + timedelta(days=30*month)
        for region in REGIONS:
            for drug in DRUGS:
                records.append({
                    'month': date.strftime('%Y-%m'),
                    'region': region,
                    'drug': drug,
                    'market_share_pct': round(random.uniform(10, 35), 2),
                    'total_prescriptions': random.randint(500, 5000),
                    'competitor_share_pct': round(random.uniform(20, 50), 2)
                })
    return pd.DataFrame(records)

def generate_doctor_visits(reps_df, months=12):
    records = []
    start_date = datetime(2024, 1, 1)
    
    doctors = [{
        'doctor_id': f'DOC{i:04d}',
        'name': fake.name(),
        'specialty': random.choice(SPECIALTIES),
        'region': random.choice(REGIONS),
        'territory': f'{random.choice(REGIONS)}-{random.randint(1,5)}'
    } for i in range(200)]
    
    for _, rep in reps_df.iterrows():
        target_specialty = DRUG_SPECIALTY_MAP[rep['drug_promoted']]
        relevant_doctors = [d for d in doctors if d['specialty'] == target_specialty]
        
        for month in range(months):
            date = start_date + timedelta(days=30*month)
            num_visits = random.randint(15, 60)
            visited = random.sample(relevant_doctors, min(num_visits, len(relevant_doctors)))
            
            for doc in visited:
                records.append({
                    'rep_id': rep['rep_id'],
                    'doctor_id': doc['doctor_id'],
                    'doctor_name': doc['name'],
                    'specialty': doc['specialty'],
                    'visit_date': date.strftime('%Y-%m'),
                    'drug_detailed': rep['drug_promoted'],
                    'follow_up': random.choice([True, False]),
                    'prescription_generated': random.choice([True, False])
                })
    
    return pd.DataFrame(records)

if __name__ == '__main__':
    print("Generating data...")
    reps = generate_reps(50)
    sales = generate_sales_performance(reps)
    market = generate_market_share()
    visits = generate_doctor_visits(reps)
    
    reps.to_csv('data/reps.csv', index=False)
    sales.to_csv('data/sales_performance.csv', index=False)
    market.to_csv('data/market_share.csv', index=False)
    visits.to_csv('data/doctor_visits.csv', index=False)
    
    print(f"Reps: {len(reps)} rows")
    print(f"Sales Performance: {len(sales)} rows")
    print(f"Market Share: {len(market)} rows")
    print(f"Doctor Visits: {len(visits)} rows")
    print("Data saved to /data folder")