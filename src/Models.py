import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import time
from datetime import datetime
from src.ingest_energy import get_esios_data, DB_PATH, TIMEZONE
from src.ingest_OPENMETO import get_energy_weather
from src.Models import Create_Custom_Features

def build_forecasting_pipeline(cols_m1, cat_cols):
    from sklearn.pipeline import Pipeline
    from src.Transformers import Create_Custom_Features, StackedDirect
    pipeline = Pipeline([
        ('features', Create_Custom_Features(time_col='timestamp')),
        ('model', StackedDirect(cols_m1=cols_m1, cat_cols=cat_cols))
    ])
    return pipeline

def get_local_data_or_fetch(table_name, start_date, end_date, indicator_id):
    conn = sqlite3.connect(DB_PATH)
    target_start_dt = pd.to_datetime(start_date, utc=True).tz_convert(TIMEZONE)
    target_end_dt = pd.to_datetime(end_date, utc=True).tz_convert(TIMEZONE)

    try:
        # 1. Intentar leer todo el rango solicitado de la DB
        query = f"SELECT * FROM {table_name} WHERE timestamp >= ? AND timestamp <= ?"
        df_local = pd.read_sql(query, conn, params=(start_date, end_date))
        if not df_local.empty:
            df_local['timestamp'] = pd.to_datetime(df_local['timestamp'], utc=True).tz_convert(TIMEZONE)
    except Exception:
        df_local = pd.DataFrame()
    finally:
        conn.close()

    # 2. Verificar si necesitamos pedir datos
    if not df_local.empty:
        max_local = df_local['timestamp'].max()
        # Comparamos timestamps completos, no solo .date()
        if max_local >= target_end_dt:
            return df_local.sort_values('timestamp')
        
        # El nuevo inicio es el máximo que tenemos + 1 unidad de frecuencia
        fetch_start = (max_local + pd.Timedelta(hours=1))
    else:
        fetch_start = target_start_dt

    # 3. Fetch de lo que falta (Delta o Todo)
    print(f"Fetching for {table_name}: {fetch_start.isoformat()} to {target_end_dt.isoformat()}")
    df_new = get_esios_data(fetch_start.isoformat(), target_end_dt.isoformat(), indicator_id=indicator_id)

    if not df_new.empty:
        # Estandarización de columnas
        if table_name == "energy_demand":
            df_new = df_new.rename(columns={'value': 'demand_value'})
        
        # Persistencia
        conn = sqlite3.connect(DB_PATH)
        df_new.to_sql(table_name, conn, if_exists='append', index=False)
        conn.close()
        print(f"Persisted {len(df_new)} new rows to {table_name}")

        # Unir y limpiar
        df_final = pd.concat([df_local, df_new]).drop_duplicates(subset='timestamp')
        return df_final.sort_values('timestamp')

    return df_local

def run_production_forecast(model_path='model_v1.joblib'):
    """
    Production entry point: Fetches data, predicts next 24h, and saves results.
    """
    # 1. Setup execution metadata
    today = pd.Timestamp.now(tz=TIMEZONE).normalize()
    execution_time = pd.Timestamp.now(tz=TIMEZONE)
    
    # We need 16 days of history to satisfy the 336h (14 days) lags + buffers
    start_history = (today - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
    target_date = today.isoformat()
    forecast_end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 2. Efficient Data Ingestion
    df_prices = get_local_data_or_fetch(table_name="energy_prices", start_date=start_history, end_date=target_date, indicator_id="600")
    df_demand = get_local_data_or_fetch(table_name="energy_demand", start_date=start_history, end_date=target_date, indicator_id="1293")
    
    # Weather is always fetched as it contains future forecast data
    df_weather = get_energy_weather(start_history, forecast_end)

    # 3. Preprocessing & Merging
    # Price column in DB is 'value', but Transformer expects 'price'
    df_prices = df_prices.rename(columns={'value': 'price'})
    
    # Process demand to hourly features
    df_demand_h = df_demand.resample('h', on='timestamp')['demand_value'].agg(['min', 'max', 'mean', 'std']).reset_index()
    df_demand_h.columns = ['timestamp', 'demand_hourly_min', 'demand_hourly_max', 'demand_hourly_mean', 'demand_hourly_std']

    data = df_weather.merge(df_prices, on='timestamp', how='left')
    data = data.merge(df_demand_h, on='timestamp', how='left')

    # 4. Inference
    model = joblib.load(model_path)
    # The model handles lag creation via Create_Custom_Features
    predictions = model.predict(data)
    
    # We take the latest prediction row (which represents the 24h forecast from 'today')
    latest_forecast = predictions.tail(1).copy()
    latest_forecast['execution_date'] = today
    latest_forecast['execution_timestamp'] = execution_time

    # 5. Save Forecast to SQLite (New Table)
    conn = sqlite3.connect(DB_PATH)
    latest_forecast.to_sql("Forecasting_prices", conn, if_exists='append', index=False)
    conn.close()
    
    print(f"Forecast for {today.date()} successfully saved to 'Forecasting_prices'.")
    return latest_forecast

def should_retrain(model_path):
    """
    Checks if the model file is older than 7 days (Weekly retraining).
    """
    if not os.path.exists(model_path):
        return True
    
    file_age_days = (time.time() - os.path.getmtime(model_path)) / (24 * 3600)
    return file_age_days > 7

def get_full_training_data():
    """
    Loads all available history from SQLite for retraining.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT * FROM weather_data 
    INNER JOIN energy_prices USING(timestamp)
    INNER JOIN energy_demand USING(timestamp)
    ORDER BY timestamp ASC;
    """
    data = pd.read_sql(query, conn)
    conn.close()
    
    # Pre-process for the pipeline
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True).dt.tz_convert(TIMEZONE)
    data = data.rename(columns={'value': 'price'})
    # Note: Ensure demand features match the Transformer expectations
    return data


def retrain_model(model_path='model_v1.joblib'):
    # 1. Cargar datos brutos
    data = get_full_training_data()
    X_raw = data.drop(columns=['price'])
    y_raw = data[['price']] # StackedDirect espera un DataFrame/Series para y

    # 2. DRY RUN: Generar las columnas para poder seleccionarlas
    # Instanciamos solo el transformador para "ver" qué columnas crea
    featurizer = Create_Custom_Features(time_col='timestamp')
    X_sample = featurizer.fit_transform(X_raw.iloc[:10]) # Solo 10 filas para ir rápido
    
    # 3. DEFINICIÓN DE COLUMNAS (Aquí van tus líneas)
    # Buscamos las columnas deterministas (const, trend, fourier...)
    deterministic_cols = [c for c in X_sample.columns if c in ['const', 'trend'] or 's(' in c or 'fourier' in c]
    
    # Buscamos los lags de precio y demanda para el Modelo 1 (Ridge)
    price_demand_lags = [c for c in X_sample.columns if 'price_lag' in c or 'demand_hourly' in c]
    cols_m1 = deterministic_cols + price_demand_lags
    
    cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns

    # 4. Construir el Pipeline con las columnas ya identificadas
    pipeline = build_forecasting_pipeline(cols_m1=cols_m1, cat_cols=cat_cols)

    # 5. Entrenar y guardar
    pipeline.fit(X_raw, y_raw)
    joblib.dump(pipeline, model_path)