"""
Deal Sourcing Bot & Database Manager

Description:
1. Scrapes (simulates) target acquisition companies.
2. Filters them based on Private Equity criteria (Revenue > $2M).
3. Saves the valid leads into a SQLite relational database for persistence.
"""

import sqlite3
import pandas as pd
import time
import random
from datetime import datetime

DB_NAME = "deal_flow.db"
MIN_REVENUE_THRESHOLD = 2000000  
def simulate_scraping_run():
    """
    Simulates scraping a business-for-sale listing site.
    In a live environment, this would use BeautifulSoup/Selenium.
    """
    print("Initialize Scraper: Connecting to sources...")
    time.sleep(1)
    
    # Mock Data
    raw_leads = [
        {"company": "Midwest Logistics LLC", "industry": "Transportation", "revenue": 5500000, "location": "Ohio", "contact": "owner@midwestlog.com"},
        {"company": "Main St. Bakery", "industry": "Food & Bev", "revenue": 450000, "location": "Vermont", "contact": "n/a"}, # Too small
        {"company": "Apex HVAC Services", "industry": "Construction", "revenue": 3200000, "location": "Texas", "contact": "sales@apexhvac.com"},
        {"company": "TechStart Inc", "industry": "SaaS", "revenue": 1500000, "location": "Remote", "contact": "info@techstart.io"}, # Too small
        {"company": "Precision Machining Co.", "industry": "Manufacturing", "revenue": 8100000, "location": "Michigan", "contact": "bizdev@precisionmfg.com"},
    ]
    
    df = pd.DataFrame(raw_leads)
    print(f"Scraping Complete. {len(df)} raw leads found.")
    return df

def filter_leads(df):
    """
    Applies investment criteria logic.
    """
    print(f"\nFiltering for Revenue > ${MIN_REVENUE_THRESHOLD:,.0f}...")
    qualified_df = df[df['revenue'] >= MIN_REVENUE_THRESHOLD].copy()
    qualified_df['scraped_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    qualified_df['status'] = 'New'
    
    print(f"Filter Complete. {len(qualified_df)} qualified targets identified.")
    return qualified_df

def save_to_database(df):
    """
    Persists the data to a local SQLite database.
    """
    print(f"\nConnecting to Database: {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS targets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT,
        industry TEXT,
        revenue INTEGER,
        location TEXT,
        contact TEXT,
        status TEXT,
        scraped_at TEXT
    );
    """
    cursor.execute(create_table_query)
    df.to_sql('targets', conn, if_exists='append', index=False)
    print("Data successfully committed to SQL.")
    
    # VERIFICATION 
    print("--- VERIFYING DB CONTENTS ---")
    verify_df = pd.read_sql("SELECT * FROM targets ORDER BY revenue DESC", conn)
    print(verify_df)
    conn.close()

if __name__ == "__main__":
    raw_data = simulate_scraping_run()
    clean_leads = filter_leads(raw_data)
    save_to_database(clean_leads)
    print("\nPipeline Finished Successfully.")