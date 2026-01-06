import sqlite3
import requests
import requests_cache
import pandas as pd
import os
import time
from retry_requests import retry
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# --- CONFIGURATION ---
INDICATOR_ID = "600"  # Electricity Market Price Indicator
GEO_ID = "8741"         # Peninsula
TIMEZONE = "Europe/Madrid"

# Location: Granada
LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))

# --- PATH MANAGEMENT ---
# Robust way to find the 'data' folder relative to this script
# If script is in /Project/src/etl.py, we want /Project/data/energy_market.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "energy_market.db")

# Ensure 'data' folder exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_esios_data(start_date, end_date, indicator_id="600", val_name='value'):
    """
    Generic function to fetch data from ESIOS API.
    Args:
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        indicator_id (str): '600' for Price, '1293' for Demand, etc.
    """
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise ValueError("CRITICAL ERROR: Token not found in .env file")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/vnd.esios-api-v2+json",
        "x-api-key": token,
        "User-Agent": "Granada-Project/1.0"
    }
    
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    
    params = {
        "start_date": f"{start_date}T00:00:00",
        "end_date": f"{end_date}T23:59:00"
    }
    
    # LOGIC: Some indicators need GeoID, some break with it.
    # ID 600 (Price) breaks if we force GeoID 8741 in query params (returns 403).
    # ID 1293 (Demand) often works better with GeoID to get Peninsula specifically.
    if indicator_id == "1293":
        params["geo_ids[]"] = GEO_ID

    print(f"   >> Requesting ESIOS (ID {indicator_id}): {start_date} -> {end_date}...", end=" ")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        values_list = data['indicator']['values']
        df = pd.DataFrame(values_list)
        
        if df.empty:
            print("CRITICAL: Empty response.")
            return pd.DataFrame()

        # Clean and rename
        df_clean = df[['datetime', 'value']].copy()
        # We rename 'value' to a generic 'value' or specific depending on usage
        # But to keep your pipeline simple, let's return 'value' and rename outside
        df_clean.columns = ['timestamp', val_name]
        
        # Timezone Conversion
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
        df_clean['timestamp'] = df_clean['timestamp'].dt.tz_convert(TIMEZONE)
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        print(f"OK ({len(df_clean)} rows)")
        return df_clean

    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

def save_to_sqlite(df, table_name, replace=False):
    """Saves dataframe to the DB in /data folder."""
    if df.empty: return
    if replace: method='replace'
    else: method='append'
    print(f"Saving to: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table_name, conn, if_exists=method, index=False)
        print(f"Saved {len(df)} rows to table '{table_name}'")
    except Exception as e:
        print(f"SQL Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # --- ETL EXECUTION ---
    START = os.getenv("START_DATE")
    END = os.getenv("END_DATE")
    
    print(f"--- Starting ETL Process ({START} to {END}) ---")

    print("\n[1/2] Downloading Energy Prices...")
    months = pd.date_range(start=START, end=END, freq='MS')
    all_energy_dfs = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_end > pd.to_datetime(END):
            month_end = pd.to_datetime(END)
            
        s_date = month_start.strftime("%Y-%m-%d")
        e_date = month_end.strftime("%Y-%m-%d")
        
        df_chunk = get_esios_data(s_date, e_date, val_name='price')
        if not df_chunk.empty:
            all_energy_dfs.append(df_chunk)
        
        time.sleep(1.0) 

    # SAVE
    print("\nSaving to Database...")
    
    if all_energy_dfs:
        df_prices_total = pd.concat(all_energy_dfs, ignore_index=True)
        df_prices_total = df_prices_total.drop_duplicates(subset='timestamp').sort_values('timestamp')
        df_prices_total['timestamp'] = pd.to_datetime(df_prices_total['timestamp'], utc=True).dt.tz_convert("Europe/Madrid")
        df_prices_total = df_prices_total.resample('h', on='timestamp')['price'].mean().reset_index()
        save_to_sqlite(df_prices_total, "energy_prices", replace=True)
    else:
        print("CRITICAL: No energy data retrieved.")
    
    print("\n--- ESIOS Data saved ---")