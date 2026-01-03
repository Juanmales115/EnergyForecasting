import openmeteo_requests
import sqlite3
import requests
import requests_cache
import pandas as pd
import os
import time
from retry_requests import retry
from dotenv import load_dotenv
from src.ingest_energy import save_to_sqlite, TIMEZONE, DB_PATH

load_dotenv()
# Location: Granada
LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))

def get_energy_weather(start_date, end_date):
    """Fetches hourly weather data from Open-Meteo API."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover_low", "rain"],
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
    }
    
    print(f"\nRequesting Open-Meteo ({start_date} -> {end_date})...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    
    hourly_data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["wind_speed_100m"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["solar_radiation"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data['cloud_cover'] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data['rain'] = hourly.Variables(4).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    # 1. Normalización crítica: Redondear a la hora para que coincida con ESIOS
    df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE).dt.floor('h')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Success: {len(df)} weather rows.")
    return df


def save_weather_safe(df, table_name="weather_data"):
    """Borra el rango de fechas existente y guarda el nuevo para evitar duplicados."""
    if df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 2. Preparar el rango para SQLite (formato string YYYY-MM-DD HH:MM:SS)
    start_range = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_range = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # 3. Borrar solo lo que vamos a sobreescribir
        cursor.execute(
            f"DELETE FROM {table_name} WHERE timestamp >= ? AND timestamp <= ?",
            (start_range, end_range)
        )
        
        # 4. Guardar los nuevos datos (convertidos a string para SQLite)
        df_to_save = df.copy()
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_sql(table_name, conn, if_exists='append', index=False)
        
        conn.commit()
        print(f"[*] {table_name}: Rango {start_range} a {end_range} actualizado.")
    except Exception as e:
        conn.rollback()
        print(f"[!] Error guardando clima: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    load_dotenv()
    START = os.getenv("START_DATE")
    END = os.getenv("END_DATE")
    
    df_weather = get_energy_weather(START, END)

    if not df_weather.empty:
        # Usamos la nueva función segura
        save_weather_safe(df_weather)
    else:
        print("CRITICAL: No weather data retrieved")