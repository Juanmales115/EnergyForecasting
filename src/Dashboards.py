import pandas as pd
import numpy as np
import sqlite3
from utils import *
import os


# Load data
print('Loading data...')

conn = sqlite3.connect("data/energy_market.db")
query = """
SELECT * FROM weather_data 
INNER JOIN energy_prices USING(timestamp)
ORDER BY timestamp ASC;
"""
data = pd.read_sql(query, conn)
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True).dt.tz_convert("Europe/Madrid") # Set timezone to Madrid
start_date, end_date = data['timestamp'].min(), data['timestamp'].max()

# Add demand data, which is measured at higher frequency (5 or 10 minutes interval instead of hourly like prices and weather)
demand_data = pd.read_sql("SELECT * FROM energy_demand", conn, parse_dates=['timestamp'])
demand_data.columns = ['timestamp', 'demand_value'] #rename columns for clarity
demand_data['timestamp'] = pd.to_datetime(demand_data['timestamp'], utc=True).dt.tz_convert("Europe/Madrid")
# Extract hourly features (by default data is in 5 or 10 minute intervals)
daily_features = demand_data.resample('h', on='timestamp')['demand_value'].agg([ # Resample to hourly frequency
        'min', 
        'max', 
        'mean', 
        'std', 
    ]).reset_index()
daily_features = daily_features.add_prefix('demand_hourly_')

# Merge data to main dataframe
data = data.merge(daily_features, left_on='timestamp', right_on='demand_hourly_timestamp', how='left').drop('demand_hourly_timestamp', axis=1)
conn.close()
print('Data loaded successfully')

# Check for missing hours in the data
print('Checking for missing hours in the dataset...\n'
      '-'*50)
print(f'Start date:{start_date}, End date:{end_date}')
missing_hours = get_missing_hours(data)
if missing_hours.empty:
    print("All hours are present in the dataset")
else:
    print("Missing hours detected:")
    print(missing_hours)

# Duplicated and missing values
duplicated = data.duplicated()

if duplicated.sum() > 0:
    print('There are duplicated rows in the dataframe: dropping them...')
    old_len = data.shape[0]
    data = data.drop_duplicates()
    print(f'Old lenght: {old_len} --- New lenght: {data.shape[0]}')
else:
    print('No duplicated rows found in the dataset.')


na_number = data.isna().sum().sum()
if na_number > 0:
    print(f'Found {na_number} missing values. Dropping them...')
    data = data.dropna()
else:
    print('No NaN values found.')

# Add features
print('Adding date and holidays features...')
data = create_date_features(data)
data = add_national_holidays(data)
print('Adding net load feature...')
data = create_net_load_aproximation(data)
print('Adding tariff period feature...')
data = add_tariff_period(data)

# Plots
print('Generating dashboards...')
# Create Dashboards folder if does not exist
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_path = os.path.join(project_root, 'Reports', 'Dashboards')
os.makedirs(output_path, exist_ok=True)

print(f"Images will be saved in: {output_path}")

box_fig = plot_dual_axis_boxplot(data)
box_fig.write_image(os.path.join(output_path, "box_plot.png"), width=1280, height=720, scale=2)

hourly_fig = plot_line(data, x='hour', color='month')
hourly_fig.write_image(os.path.join(output_path, "hourly_profile.png"), width=1280, height=720, scale=2)

dow_fig = plot_line(data, x='dow', color='month')
dow_fig.write_image(os.path.join(output_path, "Week_profile.png"), width=1280, height=720, scale=2)

daily_fig = plot_line(data, x='day', color='year')
daily_fig.write_image(os.path.join(output_path, "Daily_profile.png"), width=1280, height=720, scale=2)

tariff_box = plot_box(data, x='tariff_period')
tariff_box.write_image(os.path.join(output_path, "tariff_distribution.png"), width=1280, height=720, scale=2)

candlestick_plot = plot_price_candlestick(data, year=2024)
candlestick_plot.update_layout(title_text="Prices variability of 2024")
candlestick_plot.write_image(os.path.join(output_path, "Candlestick_plot.png"), width=1280, height=720, scale=2)

print('='*50)
print('Dashboards generated.')
print('='*50)