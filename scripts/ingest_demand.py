import sqlite3
import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
from ingest_energy import save_to_sqlite, get_esios_data

load_dotenv()

# --- CONFIGURATION ---
INDICATOR_ID = "1293"  # Electricity Demand Indicator
GEO_ID = os.getenv("GEO_ID")              # "8741" for peninsula
TIMEZONE = os.getenv("TIMEZONE")          # "Europe/Madrid"

# --- PATH MANAGEMENT ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "energy_market.db")

# Ensure 'data' folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# Historical data retrieval
if __name__ == "__main__":
    # --- ETL EXECUTION ---
    START = os.getenv("START_DATE")
    END = os.getenv("END_DATE")
    
    print(f"--- Starting Price ETL ({START} to {END}) ---")

    print(f"\nDownloading Energy Demand (ID {INDICATOR_ID})...")
    months = pd.date_range(start=START, end=END, freq='MS')
    all_energy_dfs = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_end > pd.to_datetime(END):
            month_end = pd.to_datetime(END)
            
        s_date = month_start.strftime("%Y-%m-%d")
        e_date = month_end.strftime("%Y-%m-%d")
        
        df_chunk = get_esios_data(s_date, e_date, indicator_id=INDICATOR_ID)
        if not df_chunk.empty:
            all_energy_dfs.append(df_chunk)
        
        time.sleep(1.0) 

    # SAVE
    print("\nSaving to Database...")
    
    if all_energy_dfs:
        df_demand_total = pd.concat(all_energy_dfs, ignore_index=True)
        df_demand_total = df_demand_total.drop_duplicates(subset='timestamp').sort_values('timestamp')
        df_demand_total.columns = ['timestamp', 'demand_value']
        save_to_sqlite(df_demand_total, "energy_demand", replace=True)
    else:
        print("CRITICAL: No energy data retrieved.")
    
    print("\n--- ESIOS Data saved ---")