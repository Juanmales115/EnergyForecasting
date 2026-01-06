import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate
# Corrected import: mean_absolute_error instead of mae_score
from src.Model_utils import get_full_training_data
import joblib

def run_time_series_training():
    # 1. Data Acquisition
    data = get_full_training_data()
    
    if data is None or data.empty:
        raise ValueError("The training dataset is empty or could not be loaded.")

    # 2. Data Preparation
    # Assuming 'target' is the last column if not specified
    X = data.copy()
    horizon = 24
    y_list = []
    for h in range(1, horizon + 1):
        # Desplazamos el precio hacia atrás para traer el futuro al presente
        y_list.append(data['price'].shift(-h).rename(f'price_h{h}'))
    
    y = pd.concat(y_list, axis=1)
    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]
    # 3. Time Series Cross-Validation Setup
    tscv = TimeSeriesSplit(n_splits=5)

    scoring_metrics = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }

    model = joblib.load("model_v1.joblib")

    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=tscv,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )
    avg_rmse = -np.mean(results['test_rmse'])
    avg_mae = -np.mean(results['test_mae'])


    print(f"Results across folds:")
    for i, (rmse, mae) in enumerate(zip(results['test_rmse'], results['test_mae'])):
        print(f"Fold {i + 1} | RMSE: {-rmse:.4f} | MAE: {-mae:.4f}")

    print(f"\nAverage Results:")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"MAE: {avg_mae:.4f}")

if __name__ == "__main__":
    run_time_series_training()