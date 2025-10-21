# core/ml_models.py

import lightgbm as lgb
import xgboost as xgb

# --- Scikit-learn Imports ---
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

UNSUPERVISED_MODELS = [
    'KMeans',
    'DBSCAN',
    'AgglomerativeClustering',
    'PCA',
    'IsolationForest',
    'OneClassSVM'
]

def get_model(model_name: str, params: dict = None):
    """
    Returns an unfitted instance of a model based on its name,
    optionally configured with the given parameters.
    """
    if params is None:
        params = {}
        
    # Convert the string 'none' from the UI into Python's None object.
    # This is a common requirement when interfacing with web frontends.
    if params.get('class_weight') == 'none':
        params['class_weight'] = None
    
    # --- Classification Models ---
    if model_name == 'LogisticRegression':
        # Default params for stability and reproducibility
        default_params = {'max_iter': 1000, 'random_state': 42}
        return LogisticRegression(**{**default_params, **params})
        
    elif model_name == 'RandomForestClassifier':
        default_params = {'n_estimators': 100, 'class_weight': 'none', 'random_state': 42, 'n_jobs': -1}
        return RandomForestClassifier(**{**default_params, **params})

    elif model_name == 'LightGBMClassifier':
        default_params = {'random_state': 42}
        return lgb.LGBMClassifier(**{**default_params, **params})

    elif model_name == 'XGBoostClassifier':
        default_params = {'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': 42}
        return xgb.XGBClassifier(**{**default_params, **params})

    elif model_name == 'SVC':
        default_params = {'probability': True, 'random_state': 42}
        return SVC(**{**default_params, **params})

    elif model_name == 'KNeighborsClassifier':
        default_params = {'n_jobs': -1}
        return KNeighborsClassifier(**{**default_params, **params})

    elif model_name == 'DecisionTreeClassifier':
        default_params = {'random_state': 42}
        return DecisionTreeClassifier(**{**default_params, **params})
        
    elif model_name == 'GaussianNB':
        return GaussianNB(**params)

    # --- Regression Models ---
    elif model_name == 'LinearRegression':
        default_params = {'n_jobs': -1}
        return LinearRegression(**{**default_params, **params})
        
    elif model_name == 'RandomForestRegressor':
        default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        return RandomForestRegressor(**{**default_params, **params})
        
    elif model_name == 'LightGBMRegressor':
        default_params = {'objective': 'regression', 'random_state': 42}
        return lgb.LGBMRegressor(**{**default_params, **params})

    elif model_name == 'XGBoostRegressor':
        default_params = {'objective': 'reg:squarederror', 'random_state': 42}
        return xgb.XGBRegressor(**{**default_params, **params})
        
    elif model_name == 'SVR':
        return SVR(**params)
        
    elif model_name == 'Ridge':
        default_params = {'random_state': 42}
        return Ridge(**{**default_params, **params})
        
    elif model_name == 'Lasso':
        default_params = {'random_state': 42}
        return Lasso(**{**default_params, **params})

    # --- Unsupervised Models ---
    elif model_name == 'KMeans':
        # n_init='auto' is the modern default to avoid FutureWarning
        default_params = {'n_init': 'auto', 'random_state': 42}
        return KMeans(**{**default_params, **params})
        
    elif model_name == 'DBSCAN':
        default_params = {'n_jobs': -1}
        return DBSCAN(**{**default_params, **params})
        
    elif model_name == 'AgglomerativeClustering':
        return AgglomerativeClustering(**params)
        
    elif model_name == 'PCA':
        default_params = {'random_state': 42}
        return PCA(**{**default_params, **params})
        
    elif model_name == 'IsolationForest':
        default_params = {'random_state': 42, 'n_jobs': -1}
        return IsolationForest(**{**default_params, **params})
        
    elif model_name == 'OneClassSVM':
        return OneClassSVM(**params)
        
    # --- Fallback ---
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