import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import time
from datetime import datetime
from scripts.ingest_energy import get_esios_data, DB_PATH, TIMEZONE
from scripts.ingest_OPENMETEO import get_energy_weather
from scripts.Transformers import Create_Custom_Features
import plotly.express as px
from pathlib import Path

def build_forecasting_pipeline(cols_m1, cat_cols):
    from sklearn.pipeline import Pipeline
    from scripts.Transformers import Create_Custom_Features, StackedDirect
    pipeline = Pipeline([
        ('features', Create_Custom_Features(time_col='timestamp')),
        ('model', StackedDirect(cols_m1=cols_m1, cat_cols=cat_cols))
    ])
    return pipeline

def get_local_data_or_fetch(table_name, start_date, end_date, indicator_id):
    conn = sqlite3.connect(DB_PATH)
    
    # Date normalization
    target_start_dt = pd.to_datetime(start_date, utc=True).floor('h').tz_convert(TIMEZONE)
    target_end_dt = pd.to_datetime(end_date, utc=True).floor('h').tz_convert(TIMEZONE)

    sql_start = target_start_dt.strftime('%Y-%m-%d %H:%M:%S')
    sql_end = target_end_dt.strftime('%Y-%m-%d %H:%M:%S')

    try:
        query = f"SELECT * FROM {table_name} WHERE timestamp >= ? AND timestamp <= ?"
        df_local = pd.read_sql(query, conn, params=(sql_start, sql_end))
        
        if not df_local.empty:
            df_local['timestamp'] = pd.to_datetime(df_local['timestamp'], utc=True).dt.floor('h').dt.tz_convert(TIMEZONE)
            df_local = df_local.drop_duplicates(subset='timestamp')
    except Exception:
        df_local = pd.DataFrame()
    finally:
        conn.close()

    # STATUS LOG
    if not df_local.empty:
        max_local = df_local['timestamp'].max()
        print(f"[*] DB {table_name}: history found up to {max_local.strftime('%H:%M')}")
        
        if max_local >= target_end_dt:
            print(f"[OK] {table_name} up to date. No fetch required.")
            return df_local.sort_values('timestamp')
        
        fetch_start = max_local + pd.Timedelta(hours=1)
    else:
        print(f"[!] {table_name} not found in DB for this range.")
        fetch_start = target_start_dt

    if fetch_start >= target_end_dt:
        return df_local.sort_values('timestamp')

    # Data download
    print(f"[API] Fetching {table_name} from {fetch_start.strftime('%H:%M')}...")
    df_new = get_esios_data(fetch_start.isoformat(), target_end_dt.isoformat(), indicator_id=indicator_id)

    if not df_new.empty:
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True).dt.floor('h')
        
        # Name mapping
        val_col = 'price' if table_name == 'energy_prices' else 'demand_value'
        df_new = df_new.rename(columns={'value': val_col})
        
        # With this logic timestamp column is preserved, as it will be needed.
        df_new = df_new.groupby('timestamp')[val_col].mean().reset_index()
        
        conn = sqlite3.connect(DB_PATH)
        try:
            df_to_save = df_new.copy()
            df_to_save['timestamp'] = df_to_save['timestamp'].dt.tz_convert(TIMEZONE).astype(str)
            df_to_save.to_sql(table_name, conn, if_exists='append', index=False)
        finally:
            conn.close()

        df_new['timestamp'] = df_new['timestamp'].dt.tz_convert(TIMEZONE)
        return pd.concat([df_local, df_new]).drop_duplicates(subset='timestamp').sort_values('timestamp')

    return df_local

def run_production_forecast(model_path='model_v1.joblib'):
    """
    Production entry point: Fetches data, predicts next 24h, and saves results.
    """
    # Setup execution metadata
    today = pd.Timestamp.now(tz=TIMEZONE).normalize()
    execution_time = pd.Timestamp.now(tz=TIMEZONE)
    
    # We need 16 days of history to satisfy the 336h (14 days) lags + buffers
    start_history = (today - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
    target_date = today.isoformat()
    forecast_end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Data Ingestion
    df_prices = get_local_data_or_fetch(table_name="energy_prices", start_date=start_history, end_date=target_date, indicator_id="600")
    df_demand = get_local_data_or_fetch(table_name="energy_demand", start_date=start_history, end_date=target_date, indicator_id="1293")
    
    # Weather is always fetched as it contains future forecast data
    df_weather = get_energy_weather(start_history, forecast_end)

    # Preprocessing & Merging
    # Price column in DB is 'value', but Transformer expects 'price'
    df_prices = df_prices.rename(columns={'value': 'price'})
    
    # Process demand to hourly features
    df_demand_h = df_demand.resample('h', on='timestamp')['demand_value'].agg(['min', 'max', 'mean', 'std']).reset_index()
    df_demand_h.columns = ['timestamp', 'demand_hourly_min', 'demand_hourly_max', 'demand_hourly_mean', 'demand_hourly_std']

    data = df_weather.merge(df_prices, on='timestamp', how='left')
    data = data.merge(df_demand_h, on='timestamp', how='left')

    # Inference
    model = joblib.load(model_path)
    preds = model.predict(data) # Returns shape (1, 24)

    # Transform (1, 24) into a 24-row DataFrame
    forecast_steps = []
    for h in range(preds.shape[1]):
        forecast_steps.append({
            'execution_timestamp': execution_time,
            'forecast_timestamp': today + pd.Timedelta(hours=h + 1),
            'predicted_price': float(preds.iloc[-1, h]) # .iloc es la clave aquí
        })
    
    forecast_df = pd.DataFrame(forecast_steps)

    # Save
    conn = sqlite3.connect(DB_PATH)
    forecast_df.to_sql("Forecasting_prices", conn, if_exists='append', index=False)
    conn.close()
    
    print(f"Forecast for {today.date()} (24h) saved.")
    return forecast_df

def should_retrain(model_path='model_v1.joblib'):
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
    conn = sqlite3.connect("data/energy_market.db")
    query = """
    SELECT * FROM weather_data 
    INNER JOIN energy_prices USING(timestamp)
    ORDER BY timestamp ASC;
    """
    data = pd.read_sql(query, conn)
    # Robust datetime parsing: support mixed formats, coerce unparsable values and then convert timezone
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', utc=True, errors='coerce')
    if data['timestamp'].isna().any():
        raise ValueError("Found unparsable timestamps in `weather_data`/`energy_prices`. Check raw DB format.")
    data['timestamp'] = data['timestamp'].dt.tz_convert("Europe/Madrid")

    demand_data = pd.read_sql("SELECT * FROM energy_demand", conn)
    demand_data.columns = ['timestamp', 'demand_value'] #rename columns for clarity
    # Robust datetime parsing for demand: handle mixed formats and fail early with clear message
    demand_data['timestamp'] = pd.to_datetime(demand_data['timestamp'], format='mixed', utc=True, errors='coerce')
    if demand_data['timestamp'].isna().any():
        raise ValueError("Found unparsable timestamps in `energy_demand`. Check raw DB format.")
    demand_data['timestamp'] = demand_data['timestamp'].dt.tz_convert("Europe/Madrid")
    # Extract hourly features (by default data is in 5 or 10 minute intervals)
    hourly_features = demand_data.resample('h', on='timestamp')['demand_value'].agg([
            'min', 
            'max', 
            'mean', 
            'std', 
        ]).reset_index()
    daily_features = hourly_features.add_prefix('demand_hourly_')

    # Merge data to main dataframe
    data = data.merge(daily_features, left_on='timestamp', right_on='demand_hourly_timestamp', how='left').drop('demand_hourly_timestamp', axis=1)
        
    return data

def retrain_model(model_path='model_v1.joblib'):
    # Load all data for training
    data = get_full_training_data()
    
    # Create target column
    horizon = 24
    y_list = []
    for h in range(1, horizon + 1):
        y_list.append(data['price'].shift(-h).rename(f'price_h{h}'))
    
    y_raw = pd.concat(y_list, axis=1)
    
    # Input preparation
    X_raw = data.copy()
    X_raw = X_raw.iloc[:-horizon]
    y_raw = y_raw.iloc[:-horizon]
    
    # 3. DRY RUN and column definition
    featurizer = Create_Custom_Features(time_col='timestamp')
    X_sample = featurizer.fit_transform(X_raw.tail(1000)) # Needed to get features names
    
    deterministic_cols = [c for c in X_sample.columns if c in ['const', 'trend'] or 's(' in c or 'fourier' in c]
    cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns

    # Build and train the pipeline
    pipeline = build_forecasting_pipeline(cols_m1=deterministic_cols, cat_cols=cat_cols)
    pipeline.fit(X_raw, y_raw)
    
    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved with 24h horizon support.")

def get_model_importances(model_path='model_v1.joblib'):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file was not found at: {model_path}, consider retraining")

    model = joblib.load(model_path)
    fi_model = model.named_steps['model'].model_2 # Main model
    feature_importance = fi_model.get_feature_importance() # Catboost

    try:
        feature_names = fi_model.feature_names_
    except AttributeError:
        data = get_full_training_data()
        X_dummy = model.named_steps['features'].transform(data.head(1))
        feature_names = X_dummy.columns

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
     }).sort_values(by='importance', ascending=False)

    return df_importance

def save_importance_plot(folder='Reports/Dashboards', filename='feature_importance.html'):
    output_path = Path(folder)
    output_path.mkdir(parents=True, exist_ok=True)
    df = get_model_importances().head(20)
    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importances (Top 20)',
        labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
        template='plotly_white',
        color='importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800)
    
    full_file_path = output_path / filename
    fig.write_html(str(full_file_path))
    
    print(f"Dashboard saved successfully at: {full_file_path}")