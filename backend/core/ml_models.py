# core/ml_models.py

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_model(model_name: str):
    """
    Returns an unfitted instance of a model based on its name.
    """
    if model_name == 'LightGBMClassifier':
        # Using parameters that are generally good for financial data
        return lgb.LGBMClassifier(objective='multiclass', n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42)
    elif model_name == 'XGBoostClassifier':
        return xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, random_state=42)
    # TODO: Add Regressor models here if needed later
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def get_scaler(scaler_name: str):
    """
    Returns an instance of a scaler based on its name.
    """
    if scaler_name == 'StandardScaler':
        return StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        return MinMaxScaler()
    else:
        return None