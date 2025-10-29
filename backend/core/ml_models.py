# core/ml_models.py

import lightgbm as lgb
import xgboost as xgb

# --- Scikit-learn Imports ---
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
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
    
    # Make a copy to avoid modifying the original dict during iteration
    params = params.copy()
    
    def clean_params(d):
        if not isinstance(d, dict):
            return d
        
        cleaned_dict = {}
        for key, value in d.items():
            
            # Handle the ambiguous integer '1' for max_samples/max_features
            if key in ['max_samples', 'max_features'] and value == 1:
                cleaned_dict[key] = 1.0 # Convert integer 1 to float 1.0
                continue # Move to the next item in the loop
            
            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower == 'true':
                    cleaned_dict[key] = True
                elif val_lower == 'false':
                    cleaned_dict[key] = False
                elif val_lower == 'none':
                    cleaned_dict[key] = None
                else:
                    cleaned_dict[key] = value # Keep other strings as is
            elif isinstance(value, dict):
                # Recurse for nested dictionaries and assign the cleaned result
                cleaned_dict[key] = clean_params(value)
            else:
                # Keep other types (numbers, lists, etc.) as is
                cleaned_dict[key] = value
        return cleaned_dict
    
    params = clean_params(params)
    
    # --- Classification Models ---
    if model_name == 'LogisticRegression':
        # Default params for stability and reproducibility
        default_params = {'max_iter': 1000, 'random_state': 42}
        return LogisticRegression(**{**default_params, **params})
        
    elif model_name == 'RandomForestClassifier':
        default_params = {'n_estimators': 100, 'class_weight': 'none', 'random_state': 42, 'n_jobs': -1}
        return RandomForestClassifier(**{**default_params, **params})
    
    elif model_name == 'BaggingClassifier':
        base_model_name = params.pop('baseModelName', None)
        base_model_params = params.pop('baseModelHyperparameters', {})
        
        # Extract the bagger's own parameters from the nested dictionary.
        # The .get() method returns an empty dict {} if 'baggingHyperparameters' isn't found,
        # which prevents errors.
        bagging_params = params.pop('baggingHyperparameters', {})
        
        if not base_model_name:
            raise ValueError("BaggingClassifier requires a 'baseModelName' parameter.")
        
        print(f"Instantiating base model '{base_model_name}' for BaggingClassifier...")
        
        # Recursively call get_model to create the base estimator
        base_estimator = get_model(base_model_name, base_model_params)
        
        default_params = {'random_state': 42, 'n_jobs': -1}
        
        # Now, we combine the defaults with the UNPACKED bagging_params.
        # The 'local_params' dictionary is now empty or contains other top-level
        # params if you ever add them, which is fine.
        final_bagging_params = {**default_params, **bagging_params, **params}
        
        print(final_bagging_params)
        
        # The remaining params in the dictionary are for the BaggingClassifier itself
        return BaggingClassifier(
            estimator=base_estimator, 
            **final_bagging_params
        )
    
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