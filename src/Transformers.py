import pandas as pd
import numpy as np
import torch

from src.utils import create_date_features, add_national_holidays, make_lags, make_diffs, make_rollings, add_tariff_period, create_net_load_aproximation

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
        self.last_train_date_ = None
        self.weather_horizon = 1 # Known (forecast weather)
        self.price_horizon = 1 # No need to horizon as we will do it in the definition of y (price.shift(-h) | h = 1 -> 24)

        self.fourier = CalendarFourier(freq="D", order=3) # Estacionalidad diaria
        self.dp = None

        self.weather_cols = ['temperature_2m', 'wind_speed_100m', 'solar_radiation']

        self.price_cols = ['demand_hourly_min', 'demand_hourly_max', 'demand_hourly_mean',
            'demand_hourly_std', 'price']

    def fit(self, X, y=None):
        X_t = X.copy()
        
        # 1. REDONDEAR PRIMERO EN UTC (Crucial para evitar AmbiguousTimeError)
        # UTC no tiene cambios de hora, por lo que el redondeo es seguro.
        X_t[self.time_col] = pd.to_datetime(X_t[self.time_col], utc=True).dt.floor('h')
        
        # 2. Ahora convertimos a la zona horaria local
        X_t[self.time_col] = X_t[self.time_col].dt.tz_convert(self.timezone)
        
        # 3. Eliminar duplicados generados por el redondeo
        X_t = X_t.sort_values(self.time_col).drop_duplicates(subset=self.time_col)
        
        # 4. Crear el rango sintético para statsmodels (NAIVE)
        # Usamos el mínimo y máximo ya redondeados
        start_naive = X_t[self.time_col].min().tz_localize(None)
        end_naive = X_t[self.time_col].max().tz_localize(None)
        
        dp_index = pd.date_range(start=start_naive, end=end_naive, freq='h')

        self.dp = DeterministicProcess(
            index=dp_index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[self.fourier],
            drop=True
        )
        
        # Guardamos la última fecha como aware para las máscaras
        self.last_train_date_ = X_t[self.time_col].max()

        self.is_fitted_ = True

        return self
    
    def get_dynamic_features(self, X_t):
        last_train_date = self.last_train_date_
        
        # Máscaras usando objetos aware (ambos tienen zona horaria)
        is_past = X_t[self.time_col] <= last_train_date
        is_future = X_t[self.time_col] > last_train_date
        
        features_parts = []

        if is_past.any():
            past_data = X_t.loc[is_past, [self.time_col]]
            in_sample_full = self.dp.in_sample()
            
            # OJO AQUÍ: Para buscar en in_sample_full (que es naive), 
            # convertimos las fechas de búsqueda a naive temporalmente
            search_keys = past_data[self.time_col].dt.tz_localize(None)
            is_features = in_sample_full.loc[search_keys]
            
            # Restauramos el índice original de X_t
            is_features.index = past_data.index
            features_parts.append(is_features)

        if is_future.any():
            future_data = X_t.loc[is_future, [self.time_col]]
            steps = len(future_data)
            oos_features = self.dp.out_of_sample(steps=steps)
            
            # Restauramos el índice original de X_t
            oos_features.index = future_data.index
            features_parts.append(oos_features)

        return pd.concat(features_parts).sort_index()

    def transform(self, X):
        X_t = X.copy()
        X_t[self.time_col] = pd.to_datetime(X_t[self.time_col], utc=True).dt.floor('h')
        X_t[self.time_col] = X_t[self.time_col].dt.tz_convert(self.timezone)

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
        if 'timestamp' in X_1.columns:
            X_1 = X_1.drop(columns=['timestamp'])

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


        if 'timestamp' in X_2_augm.columns:
            X_2_augm = X_2_augm.drop(columns=['timestamp'])
        # Final training
        self.model_2.fit(X_2_augm, y_trimmed)
        self.model_1.fit(X_1, y)

        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        X_1 = X[self.cols_m1].copy()
        if 'timestamp' in X_1.columns:
            X_1 = X_1.drop(columns=['timestamp'])

        X_2 = X.copy()
        
        y_pred_m1 = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, 
            columns=[f'{self.prefix}_{c}' for c in self.y_columns]
        )
        
        X_2_augm = pd.concat([X_2, y_pred_m1], axis=1)
        if 'timestamp' in X_2_augm.columns:
            X_2_augm = X_2_augm.drop(columns=['timestamp'])

        y_pred_m2_arr = self.model_2.predict(X_2_augm)
        
        return pd.DataFrame(
            y_pred_m2_arr,
            index=X_2_augm.index,
            columns=self.y_columns
        )