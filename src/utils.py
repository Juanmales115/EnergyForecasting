import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import holidays



# Check functions
def get_missing_hours(df, start_date=None, end_date=None):
    """
    Check for missing hours in a dataset. Assumes the dataframe has a 'timestamp' column with a datetime type.
    Assumes data should be hourly frequency and in the "Europe/Madrid" timezone.
    Args:
        :param df: pandas Dataframe
        :param start_date: datetime, start of the expected range. If None, uses the minimum of df
        :param end_date: datetime, end of the expected range. If None, uses the maximum of df
    Returns:
        missing_dates: pandas DatetimeIndex of missing timestamps.
    """

    if start_date is None:
        start_date = df['timestamp'].min()
    if end_date is None:
        end_date = df['timestamp'].max()

    # Create expected range
    expected_range = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq='h', 
        tz="Europe/Madrid"
    )
    
    # Ensure timestamps are in the correct timezone and format
    actual_timestamps = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("Europe/Madrid")
    
    # Search for discrepancies
    missing_dates = expected_range.difference(actual_timestamps)
    
    return missing_dates

# Plot functions
def plot_price_candlestick(df, year=None, month=None, frequency='D'):
    """
    Create a Candlestick figure using raw hourly data to calculate real OHLC.
    
    Args:
        :param df: The original dataframe containing the 'timestamp' and 'price' data.
        :param year: Year to filter (optional).
        :param month: Month to filter (optional). <--- ADDED
        :param frequency: Resampling frequency ('D' for Daily candles, 'W' for Weekly).
    Return:
        Figure
    """
    # Filter Data on the raw dataframe
    df_working = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_working['timestamp']):
        df_working['timestamp'] = pd.to_datetime(df_working['timestamp'])

    # Apply filters if arguments are provided
    if year:
        df_working = df_working[df_working['timestamp'].dt.year == int(year)]

    if month:
        df_working = df_working[df_working['timestamp'].dt.month == int(month)]
        
    # Dynamic title suffix
    if month and year:
        title_suffix = f" ({month}/{year})"
    elif year:
        title_suffix = f" ({year})"
    else:
        title_suffix = " (All Time)"

    # Calculate OHLC and Holiday status from raw data
    agg_ops = {
        'price': ['first', 'max', 'min', 'last'],
    }
    
    # Check for holiday column and include max aggregation
    include_holiday_marker = 'is_national_holiday' in df_working.columns
    if include_holiday_marker:
        agg_ops['is_national_holiday'] = 'max'
        
    df_ohlc_full = df_working.set_index('timestamp').resample(frequency).agg(agg_ops)
    
    # Flatten column names and handle NaNs from resampling
    if include_holiday_marker:
        df_ohlc_full.columns = ['Open', 'High', 'Low', 'Close', 'is_holiday_day']
    else:
        df_ohlc_full.columns = ['Open', 'High', 'Low', 'Close']
        
    df_ohlc = df_ohlc_full.dropna()

    # Create Candlestick Figure
    fig = go.Figure(data=[go.Candlestick(
        x=df_ohlc.index,
        open=df_ohlc['Open'],
        high=df_ohlc['High'],
        low=df_ohlc['Low'],
        close=df_ohlc['Close'],
        name="Price"
    )])

    # Add Holiday Markers (Conditional Trace)
    if include_holiday_marker:
        df_holidays_marker = df_ohlc[df_ohlc['is_holiday_day'] == 1].reset_index()
        
        marker_y_position = df_holidays_marker['High'] * 1.01 

        fig.add_trace(go.Scatter(
            x=df_holidays_marker['timestamp'],
            y=marker_y_position,
            mode='markers',
            name='National Holiday',
            marker=dict(
                size=8,
                color='rgba(0, 150, 0, 0.8)', 
                symbol='circle'
            ),
            hovertext=df_holidays_marker['timestamp'].dt.strftime('%Y-%m-%d') + ' (Holiday)',
            hoverinfo='text',
        ))

    # Layout configuration
    fig.update_layout(
        title=f"Electricity Price Candles (Freq: {frequency}){title_suffix}",
        yaxis_title="Price (€/MWh)",
        xaxis_title="Date",
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )
    
    return fig

def plot_correlation(df, variable_x):
    """
    Create Scatterplot and tendency line between price and a feature given.
    Assumes that the dataframe given has a price column.

    Args:
        :param df: Dataframe
        :param variable_x: Feature to compare
    Return: Figure
    """
    fig = px.scatter(
        df, 
        x=variable_x, 
        y="price", 
        opacity=0.5, 
        title=f"Análisis de Correlación: {variable_x} vs Precio",
        trendline="ols", # Tendency line
        color="price", 
        color_continuous_scale="Plasma",
        height=600 # Altura fija para que no 'baile' al cambiar
    )

    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )
    return fig

def plot_dual_prices(df, variable_y, year=None, month=None):
    """
    Create a figure Price vs a secondary variable with optional filtering by year and month.
    Assumes that the dataframe given has price and timestamp columns.
    
    Args:
        :param df: Dataframe to plot.
        :param variable_y: Feature to plot in the secondary axis.
        :param year: Year to plot, if None uses all dataframe data.
        :param month: Months to plot, if None uses all dataframe data, (Only used if a year is given).
    Return: Figure
    
    """
    # Create a copy to avoid modifying the original dataframe
    df_to_plot = df.copy()
    
    # Ensure timestamp is in datetime format
    df_to_plot['timestamp'] = pd.to_datetime(df_to_plot['timestamp'])

    # Apply filters if arguments are provided
    if year:
        df_to_plot = df_to_plot[df_to_plot['timestamp'].dt.year == int(year)]
        if month:
            df_to_plot = df_to_plot[df_to_plot['timestamp'].dt.month == int(month)]

    # Create a figure with secondary Y-axis
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Price
    fig_dual.add_trace(
        go.Scatter(
            x=df_to_plot['timestamp'], 
            y=df_to_plot['price'], 
            name="Price (€)", 
            line=dict(color="#2c64ff", dash='dot')
        ),
        secondary_y=False,
    )

    # Trace 2: Variable Y
    fig_dual.add_trace(
        go.Scatter(
            x=df_to_plot['timestamp'], 
            y=df_to_plot[variable_y], 
            name=variable_y, 
            line=dict(color="#b92929", dash='dot')
        ),
        secondary_y=True,
    )

    # Layout details with dynamic title
    title_suffix = f" ({month}/{year})" if month and year else f" ({year})" if year else ""
    
    fig_dual.update_layout(title_text=f"Market Dynamics: Price vs {variable_y}{title_suffix}")
    fig_dual.update_yaxes(title_text="Price (€/MWh)", secondary_y=False)
    fig_dual.update_yaxes(title_text=variable_y, secondary_y=True)

    fig_dual.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )

    return fig_dual

def plot_dual_axis_boxplot(df, col_primary='price', col_secondary='net_load_proxy', col_x='dow'):
    """
    Plots a dual-axis boxplot comparing Price vs a secondary variable (e.g., Net Load), 
    grouped by Day of Week.

    Args:
        :param df: Dataframe containing the data.
        :param col_primary: Column name for the primary Y-axis (Left).
        :param col_secondary: Column name for the secondary Y-axis (Right).
        :param col_x: Column name for the x-axis grouping (Day of Week).

    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Primary Axis (Price)
    fig.add_trace(
        go.Box(
            x=df[col_x],
            y=df[col_primary],
            name=col_primary.capitalize(),
            marker_color="#c0392b",
            offsetgroup='A',  # Group A pushes box to the left
        ),
        secondary_y=False,
    )

    # Trace 2: Secondary Axis (Net Load)
    fig.add_trace(
        go.Box(
            x=df[col_x],
            y=df[col_secondary],
            name=col_secondary.replace('_', ' ').title(),
            marker_color="#2980b9",
            offsetgroup='B', # Group B pushes box to the right
        ),
        secondary_y=True,
    )

    # Layout Adjustments
    fig.update_layout(
        title_text=f"{col_primary.capitalize()} vs {col_secondary.replace('_', ' ').title()} Distribution by Day of Week",
        boxmode='group', # tells plotly to group boxes side-by-side
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            title="Day of Week"
        )
    )

    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=f"{col_primary.capitalize()} (€/MWh)", secondary_y=False)
    fig.update_yaxes(title_text=col_secondary.replace('_', ' ').title(), secondary_y=True)
    

    return fig

def plot_line(df, x, y='price', color=None):
    """
    Return figure of a plotly lineplot. Groupby x value and use mean to resample
    
    Args:
        :param df: Pandas Dataframe
        :param x: string: Column name for xaxis
        :param y: string, Default = 'Price', value for the y axis 
        :param color: (Optional) Separate value with distinct color (for example, distinct month)
    
    Return:
        plotly line figure
    """
    if color:
        df_grouped = df.groupby([color, x])[y].mean().reset_index()
    else:
        df_grouped = df.groupby([x])[y].mean().reset_index()

    fig = px.line(  
        df_grouped, 
        x=x, 
        y=y, 
        color=color,
        markers=True,
        title=f"Mean values of {y} by {x}"
    )
    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )

    return fig

def plot_bar(df, x, y='price'):
    """
    Return figure of a plotly barplot. Groupby x value and use mean to resample
    
    Args:
        :param df: Pandas Dataframe
        :param x: string: Column name for xaxis
        :param y: string, Default = 'Price', value for the y axis 
    
    Return:
        plotly bar figure
    """
    profile = df.groupby(x)[y].mean().reset_index()

    min_range = min(40, profile[y].min())
    color_range = [min_range, profile[y].max()]
    fig = px.bar(
        profile,
        x=x,
        y=y,
        title=f"Mean values of {y} by {x}",
        color=y,
        color_continuous_scale='Plasma',
        range_color=color_range,
    )

    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )

    return fig

def plot_box(df, x, y='price'):
    fig = px.box(
        df,
        x=x,
        y=y,
        title=f'Distribution of {y} values by {x}',
    )
    fig.update_xaxes(tickmode='linear', dtick=1)

    fig.update_layout(
        width=1280,
        height=720,
        font=dict(family="Arial") 
    )

    return fig

# Feature creation
def create_date_features(df):
    """
    Create data features from timestamp column such as day of the week (dow).
    
    Args:
        :param df: Dataframe. Will not be modified.
    Return:
        data: Dataframe with all features added.
    """
    data = df.copy()
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['dow'] = data['timestamp'].dt.dayofweek
    data['woy'] = data['timestamp'].dt.isocalendar().week
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['is_weekend'] = data['dow'].isin([5,6]).astype(int)

    # Add COVID-19 lockdown indicator (Spain)
    covid_start = "2020-03-14" #Also lockdown start
    lockdown_finish = "2020-06-21"
    covid_end = "2021-05-09"
    data['is_covid'] = data['timestamp'].between(covid_start, covid_end, inclusive='both').astype(int)
    # This is not needed as there is no data in this range, but to be consistent:
    data['is_covid_lockdown'] = data['timestamp'].between(covid_start, lockdown_finish, inclusive='both').astype(int)
    return data

def add_national_holidays(df, country_code='ES'):
    """
    Adds a binary feature 'is_national_holiday' using the holidays library.
    Automatically detects years present in the dataframe.
    """
    unique_years = df['timestamp'].dt.year.unique()
    es_holidays = holidays.country_holidays(country_code, years=unique_years)
    
    temp_dates = df['timestamp'].dt.date
    df['is_national_holiday'] = temp_dates.isin(es_holidays).astype(int)
    df['holiday_name'] = temp_dates.map(es_holidays)
    df['holiday_name'] = df['holiday_name'].fillna("Not Holiday")
    
    return df

def make_lags(ts, lags, shift_step=0, prefix = None):
    """
    Make lags feature of a given timestamp
    
    Args:
        :param ts: Datetime array
        :param lags: array -> create a lag feature for each value in the array
        :param shift_step: int, default = 0 -> Shift to past
        :param prefix: string (optional) -> Prefix added to column's names

    Return: 
        Pandas dataframe with the new features
    """
    if prefix is None: prefix = ts.name
    ts_shifted = ts.shift(shift_step)
    return pd.concat({
        f'{prefix}_lag_{i}_sh{shift_step}': ts_shifted.shift(i)
        for i in lags
    }, axis=1)

def make_diffs(ts, diffs, shift_step=0, prefix=None):
    """
    Calculates differences based on SHIFTED data to avoid leakage.
    Logic: (Value at t-shift) - (Value at t-shift-diff)
    
    Args:
    :param ts: Datetime array
    :param diffs: array -> create a .diff feature for each value in it.
    :param shift_step: int, default = 0 -> Shift to past
    :param prefix: string (optional) -> Prefix added to column's names

    Return: 
        Pandas dataframe with the new features
    """
    ts_safe = ts.shift(shift_step)
    
    return pd.concat({
        f'{prefix}_diff_{i}_shifted_{shift_step}': ts_safe.diff(i)
        for i in diffs
    }, axis=1)

def make_rollings(ts, windows, shift_step=1, prefix=None):
    """
    Calculates EWM/Rolling stats on SHIFTED data.
    
    Args:
    :param ts: Datetime array
    :param windows: array -> create features for each window in it.
    :param shift_step: int, default = 1 -> Shift to past
    :param prefix: string (optional) -> Prefix added to column's names

    Return: 
        Pandas dataframe with the new features
    """
    ts_shifted = ts.shift(shift_step)
    
    if prefix is None: prefix = ''
    features = {}

    for i in windows:
        features[f'{prefix}_exp_roll_{i}_mean_sh{shift_step}'] = ts_shifted.ewm(span=i, adjust=False).mean()
        features[f'{prefix}_roll_{i}_median_sh{shift_step}'] = ts_shifted.rolling(window=i).median()
        features[f'{prefix}_roll_{i}_std_sh{shift_step}'] = ts_shifted.rolling(window=i).std()

    return pd.concat(features, axis=1)

def add_tariff_period(df):
    """
    Adds tariff periods using vectorized operations (zero loops).
    Handles the structural break of June 1st, 2021 (2.0TD).

    Args:
        :param df: Pandas Dataframe -> Original will not be changed
    Return:
        Pandas Dataframe with added tariff period
    """
    df = df.copy() #copy to avoid changing the original
    
    ts = df['timestamp']
    hour = df['hour']
    month = df['month']

    is_weekend_or_holiday = (df['is_weekend'] == 1) | (df['is_national_holiday'] == 1)

    # Structure break June 1st
    DATE_CHANGE = pd.Timestamp("2021-06-01", tz=ts.dt.tz)
    mask_new_system = ts >= DATE_CHANGE
    mask_old_system = ~mask_new_system

    # ---------------------------------------------------------
    # 2.0TD - After June 2021
    # ---------------------------------------------------------

    cond_p3 = is_weekend_or_holiday | (hour < 8)

    cond_p2 = (
        ((hour >= 8) & (hour < 10)) | 
        ((hour >= 14) & (hour < 18)) | 
        ((hour >= 22) & (hour < 24))
    ) & (~is_weekend_or_holiday) # Exclude weekends and holidays

    df.loc[mask_new_system, 'tariff_period'] = np.select(
        [cond_p3[mask_new_system], cond_p2[mask_new_system]], 
        ['P3_Valle', 'P2_Llano'], 
        default='P1_Punta'
    )

    # ---------------------------------------------------------
    # 2.0 DHA - Before June 2021
    # ---------------------------------------------------------
    
    if mask_old_system.any(): # Only if old data 
        # seasons aproximation
        is_summer = (month > 3) & (month < 10)
        is_winter = ~is_summer
        
        punta_summer = is_summer & (hour >= 13) & (hour < 23)
        punta_winter = is_winter & (hour >= 12) & (hour < 22)

        is_old_punta = punta_summer | punta_winter
        
        df.loc[mask_old_system, 'tariff_period'] = np.where(
            is_old_punta[mask_old_system], 
            'Old_Punta', 
            'Old_Valle'
        )

    return df

def create_net_load_aproximation(df):
    """
    Create net load aproximation which must show an aproximation of the percentage of demand covered by renewable energies.
    Assumes that dataframe has solar_radiation and wind_speed_100m columns.
    
    Args:
        :param df: Pandas Dataframe
    Return:
        Pandas Dataframe with renewable_potential_proxy and net_load_proxy features added.
    """
    df = df.copy()
    df['renewable_potential_proxy'] = (df['solar_radiation'] / df['solar_radiation'].max()) + \
                                        (df['wind_speed_100m'] / df['wind_speed_100m'].max())

    df['net_load_proxy'] = df['demand_hourly_mean'] / (1 + df['renewable_potential_proxy'])
    return df

