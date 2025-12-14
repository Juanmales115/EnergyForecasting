import pandas as pd
import numpy as np

from utils import create_date_features, add_national_holidays, make_lags, make_diffs, make_rollings, add_tariff_period, create_net_load_aproximation

# --- Series Temporales y Estadística ---
import statsmodels.api as sm
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# --- Machine Learning (Core & Modelos) ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

# --- Preprocesamiento y Pipelines ---
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Validación y Métricas ---
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Optimización de Hiperparámetros ---
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution


class Create_Custom_Features(BaseEstimator, TransformerMixin):
    def __init__(self, X, y=None, time_col = 'timestamp', timezone='Europe/Madrid'):
        self.start_date, self.end_date = None, None
        self.time_col = time_col
        self.timezone = timezone
        self.horizon = None
        self.weather_horizon = 1
        self.price_horizon = 1

        self.weather_cols = ['temperature_2m', 'wind_speed_100m', 'solar_radiation']

        self.price_cols = ['demand_hourly_min', 'demand_hourly_max', 'demand_hourly_mean',
            'demand_hourly_std', 'price']

    def fit(self, X, y=None):
        if self.time_col != 'timestamp': # Ensure date is in a column named timestamp, which is required by the functions
            X['timestamp'] = X[self.time_col]
            X = X.drop(self.time_col, axis=1) # Drop old timestamp column

        self.start_date = X['timestamp'].min()
        self.end_date = X['timestamp'].max()
        return self

    def transform(self, X):
        X_t = X.copy()
        # Add features.
        X_t = create_date_features(X_t)
        X_t = add_national_holidays(X_t)
        X_t = add_tariff_period(X_t)
        X_t = create_net_load_aproximation(X_t)

        # Lags/Diffs/Rollings
        base_lags = [0, 1, 8, 16, 24, 48, 72, 168, 336] #168= 7*24, 336 = 14*24
        base_windows = [24, 168, 336] #4 weeks = 672 hours

        X_lags = []
        for col in self.weather_cols:
            X_lags.append(make_lags(X_t[col], lags=base_lags, shift_step=self.weather_horizon, prefix=col))
            X_lags.append(make_diffs(X_t[col], diffs=base_lags, shift_step=self.weather_horizon, prefix=col))
            X_lags.append(make_rollings(X_t[col], windows=base_windows, shift_step=self.weather_horizon, prefix=col))

        for col in self.price_cols:
            X_lags.append(make_lags(X_t[col], lags=base_lags, shift_step=self.price_horizon, prefix=col))
            X_lags.append(make_diffs(X_t[col], diffs=base_lags, shift_step=self.price_horizon, prefix=col))
            X_lags.append(make_rollings(X_t[col], windows=base_windows, shift_step=self.price_horizon, prefix=col))

        all_lags = pd.concat(X_lags, axis=1)
        X_t = pd.concat([X_t, all_lags], axis=1)
        # This process creates null values at the begining, but since it is catboost the model that will be used, there is no need to clean those values.

        return X_t