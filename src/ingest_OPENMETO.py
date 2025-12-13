import openmeteo_requests
import sqlite3
import requests
import requests_cache
import pandas as pd
import os
import time
from retry_requests import retry
from dotenv import load_dotenv
from ingest_energy import save_to_sqlite, TIMEZONE

load_dotenv()
# Location: Granada
LATITUDE = float(os.getenv("LATITUDE"))
LONGITUDE = float(os.getenv("LONGITUDE"))

def get_energy_weather(start_date, end_date):
    """Fetches hourly weather data from Open-Meteo API."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
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
    df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Success: {len(df)} weather rows.")
    return df


if __name__ == "__main__":
    START = os.getenv("START_DATE")
    END = os.getenv("END_DATE")
    print("\nDownloading Weather Data...")
    df_weather = get_energy_weather(START, END)

    if not df_weather.empty:
        save_to_sqlite(df_weather, "weather_data")
        print("\n -- Open-Meteo data saved --")
    else:
        print("CRITICAL: No weather data retrieved")