import pandas as pd
import numpy as np
import torch

from utils import create_date_features, add_national_holidays, make_lags, make_diffs, make_rollings, add_tariff_period, create_net_load_aproximation

# --- Series Temporales y Estadística ---
import statsmodels.api as sm
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# --- Machine Learning (Core & Modelos) ---
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


class Create_Custom_Features(BaseEstimator, TransformerMixin):
    def __init__(self, time_col = 'timestamp', timezone='Europe/Madrid'):
        self.start_date, self.end_date = None, None
        self.time_col = time_col
        self.timezone = timezone
        self.horizon = None
        self.weather_horizon = 1 # Known (forecast weather)
        self.price_horizon = 1 # No need to horizon as we will do it in the definition of y (price.shift(-h) | h = 1 -> 24)

        self.fourier = CalendarFourier(freq="D", order=3) # Estacionalidad diaria
        self.dp = None

        self.weather_cols = ['temperature_2m', 'wind_speed_100m', 'solar_radiation']

        self.price_cols = ['demand_hourly_min', 'demand_hourly_max', 'demand_hourly_mean',
            'demand_hourly_std', 'price']

    def fit(self, X, y=None):
        if self.time_col != 'timestamp': # Ensure date is in a column named timestamp, which is required by some functions
            X['timestamp'] = X[self.time_col]
            X = X.drop(self.time_col, axis=1) # Drop old timestamp column
        
        X['timestamp'] = pd.to_datetime(X['timestamp'], utc=True).dt.tz_convert(self.timezone)
        self.start_date = X['timestamp'].min()
        self.end_date = X['timestamp'].max()

        temp_df = X.set_index(self.time_col).resample('h').asfreq()
        
        self.dp = DeterministicProcess(
            index=temp_df.index,
            constant=True,
            order=1,            # Tendencia lineal
            seasonal=True,       # Estacionalidad dummy
            additional_terms=[self.fourier],
            drop=True
        )

        return self
    
    def get_dynamic_features(self, X_t):
        # Determine the point where training ended
        last_train_date = self.dp.last_date_
        
        # Split X_t into what is already known and what is new
        is_past = X_t[self.time_col] <= last_train_date
        is_future = X_t[self.time_col] > last_train_date
        
        features_parts = []

        # 1. In-sample
        if is_past.any():
            # Extraemos las fechas y sus índices originales
            past_data = X_t.loc[is_past, [self.time_col]]
            
            # Obtenemos las features usando las fechas como llave
            in_sample_full = self.dp.in_sample()
            is_features = in_sample_full.loc[past_data[self.time_col]]
            
            # REGLA DE ORO: Forzamos el índice original de X_t
            is_features.index = past_data.index
            features_parts.append(is_features)

        # 2. Out-of-sample
        if is_future.any():
            future_data = X_t.loc[is_future, [self.time_col]]
            steps = len(future_data)
            
            oos_features = self.dp.out_of_sample(steps=steps)
            
            # Forzamos el índice original de X_t (la parte futura)
            oos_features.index = future_data.index
            features_parts.append(oos_features)

        # Combinamos y ordenamos para que coincida con el orden de X_t
        return pd.concat(features_parts).sort_index()

    def transform(self, X):
        X_t = X.copy()
        X_t[self.time_col] = pd.to_datetime(X_t[self.time_col], utc=True).dt.tz_convert(self.timezone)

        # Deterministic process
        dp_features = self.get_dynamic_features(X_t)
        dp_features.index = X_t.index # Alineamos índices

        # Add features.
        X_t = create_date_features(X_t)
        X_t = add_national_holidays(X_t)
        X_t = add_tariff_period(X_t)
        X_t = create_net_load_aproximation(X_t)

        # Concat all
        X_t = pd.concat([X_t, dp_features], axis=1)

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
    

# Model

class StackedDirect(BaseEstimator, RegressorMixin):
    
    def __init__(self, cols_m1, cat_cols=None, model_1=None, model_2=None, prefix='M1_Pred', n_splits=5):
        self.cols_m1 = cols_m1
        self.cat_cols = cat_cols
        self.prefix = prefix
        self.y_columns = None
        self.n_splits = n_splits
        
        self.model_1 = model_1 if model_1 else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        
        if model_2:
            self.model_2 = model_2
        else:
            if cat_cols is None:
                raise ValueError("Configuration Error: 'cat_cols' must be provided if 'model_2' is not defined.")
            device = "GPU" if torch.cuda.is_available() else "CPU"

            self.model_2 = CatBoostRegressor(
                loss_function='MultiRMSE',
                task_type=device,
                devices='0',
                verbose=0,
                boosting_type='Plain',
                cat_features=cat_cols.tolist()
            )

    def fit(self, X, y):
        X_1 = X[self.cols_m1].copy()
        X_2 = X.copy()
        self.y_columns = y.columns
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        meta_features_list = []
        
        # Generate meta-features (Out-Of-Fold)
        for train_index, test_index in tscv.split(X_1):
            X_train_fold, X_test_fold = X_1.iloc[train_index], X_1.iloc[test_index]
            y_train_fold = y.iloc[train_index]
            
            self.model_1.fit(X_train_fold, y_train_fold)
            preds_fold = self.model_1.predict(X_test_fold)
            
            preds_df = pd.DataFrame(
                preds_fold,
                index=X_test_fold.index,
                columns=[f'{self.prefix}_{c}' for c in y.columns]
            )
            meta_features_list.append(preds_df)

        y_fit = pd.concat(meta_features_list)
        
        # Align data (remove initial training gap)
        X_2_trimmed = X_2.loc[y_fit.index]
        y_trimmed = y.loc[y_fit.index]
        
        X_2_augm = pd.concat([X_2_trimmed, y_fit], axis=1)
        
        # Final training
        self.model_2.fit(X_2_augm, y_trimmed)
        self.model_1.fit(X_1, y)
        
        return self
    
    def predict(self, X):
        X_1 = X[self.cols_m1].copy()
        X_2 = X.copy()
        
        y_pred_m1 = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, 
            columns=[f'{self.prefix}_{c}' for c in self.y_columns]
        )
        
        X_2_augm = pd.concat([X_2, y_pred_m1], axis=1)
        y_pred_m2_arr = self.model_2.predict(X_2_augm)
        
        return pd.DataFrame(
            y_pred_m2_arr,
            index=X_2_augm.index,
            columns=self.y_columns
        )