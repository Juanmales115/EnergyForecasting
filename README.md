# Energy Price Forecasting — Spanish Electricity Market

## Abstract
This project forecasts Spanish electricity market prices (PVPC) using meteorological forecasts, historical demand, and price data. It features a complete end-to-end pipeline including automated data ingestion, advanced feature engineering, stacking-based modeling, and automated reporting.

The core architecture uses a **Stacked Direct 24-hour forecasting** approach:
1. **Ridge Regression**: Captures linear trends and deterministic seasonal patterns (Fourier features).
2. **CatBoost**: Predicts the residuals (errors) of the Ridge model to capture non-linear market dynamics.

## Model Performance (Metrics)
The model was evaluated using **Time Series Cross-Validation** (5 folds). 

Results across folds:
Fold 1 | RMSE: 106.0976 | MAE: 70.6199
Fold 2 | RMSE: 65.6378 | MAE: 50.8361
Fold 3 | RMSE: 37.5519 | MAE: 28.2797
Fold 4 | RMSE: 34.2561 | MAE: 24.7475
Fold 5 | RMSE: 31.1272 | MAE: 23.3014

Average Results:
RMSE: 54.9341
MAE: 39.5569

*Note: Model stability significantly improved in recent folds due to the stabilization of the energy market post-2022 crisis and increased data availability. Nevertheless, the error is still excessive, mainly because meteorological data is not enough for preciseful insights*

## Project Structure
- `main.py`: Entry point for the production forecasting pipeline.
- `src/`: Core logic and utilities.
    - `Models.py`: Logic for retraining, production inference, and the `get_local_data_or_fetch` system.
    - `Transformers.py`: Custom Scikit-Learn classes for Feature Engineering and the Stacking Model.
    - `utils.py`: Helper functions for dashboarding and technical indicators.
- `scripts/`:
    - `ingest_energy.py` / `ingest_demand.py`: ETL scripts for ESIOS API.
    - `ingest_OPENMETO.py`: ETL for Open-Meteo weather data.
    - `Dashboards.py`: Automated generation of visual reports.
- `data/energy_market.db`: SQLite database storing weather, demand, price history, and model forecasts.

## Installation & Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file (see `.env.example`):
    - `ESIOS_TOKEN`: Your API key.
    - `LATITUDE` / `LONGITUDE`: Target location (e.g., Granada).
    - `START_DATE`: Historical start point.

## Sources
- **ESIOS (Red Eléctrica Española)**: Real-time price and demand data.
- **Open Meteo**: Historical and forecasted meteorological variables (Wind, Solar Radiation).