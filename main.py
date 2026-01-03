import pandas as pd
import numpy as np
from src.Models import run_production_forecast, should_retrain, retrain_model

if should_retrain():
    retrain_model()

run_production_forecast()