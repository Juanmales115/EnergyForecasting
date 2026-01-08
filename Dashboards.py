import pandas as pd
import numpy as np
import sqlite3
from src.utils import *
import os
from src.Model_utils import get_full_training_data


# Load data
print('Loading data...')

data = get_full_training_data()
start_date, end_date = data['timestamp'].min(), data['timestamp'].max()
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
from pathlib import Path

project_root = Path(__file__).parent 
output_path = project_root / 'Reports' / 'Dashboards'

# Create in case folder does not exist
output_path.mkdir(parents=True, exist_ok=True)

print(f"Images will be saved in: {output_path.absolute()}")

print(f"Images will be saved in: {output_path}")

box_fig = plot_dual_axis_boxplot(data)
box_fig.write_html(os.path.join(output_path, "box_plot.html"))

hourly_fig = plot_line(data, x='hour', color='month')
hourly_fig.write_html(os.path.join(output_path, "hourly_profile.html"))

dow_fig = plot_line(data, x='dow', color='month')
dow_fig.write_html(os.path.join(output_path, "Week_profile.html"))

daily_fig = plot_line(data, x='day', color='year')
daily_fig.write_html(os.path.join(output_path, "Daily_profile.html"))

tariff_box = plot_box(data, x='tariff_period')
tariff_box.write_html(os.path.join(output_path, "tariff_distribution.html"))

candlestick_plot = plot_price_candlestick(data, year=2024)
candlestick_plot.update_layout(title_text="Prices variability of 2024")
candlestick_plot.write_html(os.path.join(output_path, "Candlestick_plot.html"))

print('='*50)
print('Dashboards generated.')
print('='*50)