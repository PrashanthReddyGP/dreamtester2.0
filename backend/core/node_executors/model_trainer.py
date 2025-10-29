# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

# ==============================================================================
# Third-party Library Imports (primarily for ML)
# ==============================================================================
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
# The base class that this executor inherits from.
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

# A utility function to get an ML model instance based on its name.
from core.ml_models import get_model 

# ==============================================================================
# Helper Function for Model Evaluation
# ==============================================================================

def _calculate_model_performance(model: Any, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str], threshold: float = 0.5) -> Tuple[Dict, Dict]:
    """
    Evaluates a trained model and computes a rich set of metrics and analysis.
    Dynamically handles classification vs. regression tasks.
    
    Args:
        model: The trained scikit-learn compatible model object.
        X_test: The test features.
        y_test: The true test labels.
        feature_names: The list of column names for the features.
    
    Returns:
        A tuple containing two dictionaries: (model_metrics, model_analysis).
        - model_metrics: A flat dictionary of key performance indicators (e.g., accuracy, R2).
        - model_analysis: A nested dictionary with detailed analysis (e.g., confusion matrix, feature importance).
    """
    model_metrics = {}
    model_analysis = {}
    
    y_pred = model.predict(X_test)
    
    # --- Determine if the task is classification or regression ---
    is_classification = y_test.dtype != 'float' and y_test.nunique() < 30
    
    # --- Generate Predictions (y_pred) using the appropriate method ---
    if is_classification and hasattr(model, "predict_proba") and y_test.nunique() == 2:
        print(f"Applying custom prediction threshold: {threshold}")
        # Get probabilities for the positive class (usually the second column)
        y_proba = model.predict_proba(X_test)[:, 1]
        # Apply threshold to get final predictions
        y_pred = (y_proba >= threshold).astype(int)
    else:
        # Fallback for regression, multi-class classification, or models without predict_proba
        y_pred = model.predict(X_test)
    
    if is_classification:
        print("Calculating classification metrics...")
        model_metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
        
        avg_method = 'binary' if y_test.nunique() == 2 else 'weighted'
        model_metrics['precision'] = round(precision_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
        model_metrics['recall'] = round(recall_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
        model_metrics['f1_score'] = round(f1_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
        
        # ROC AUC is calculated on probabilities, so it's not affected by the threshold
        # We calculate it here before it's lost
        if avg_method == 'binary' and 'y_proba' in locals():
            model_metrics['roc_auc'] = round(roc_auc_score(y_test, y_proba), 4)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cleaned_report = {
            str(key): {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else float(value)
            for key, value in report.items()
        }
        model_analysis['classification_report'] = cleaned_report
        
        class_labels = sorted(list(y_test.unique()))
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        model_analysis['confusion_matrix'] = {
            'labels': [str(label) for label in class_labels],
            'values': cm.tolist()
        }
        
    else: # Regression
        print("Calculating regression metrics...")
        model_metrics['mean_squared_error'] = round(mean_squared_error(y_test, y_pred), 4)
        model_metrics['r2_score'] = round(r2_score(y_test, y_pred), 4)
    
    importances = None # Initialize to None

    # Case 1: The model has a direct feature_importances_ attribute (e.g., RandomForest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        print("Found direct feature_importances_ attribute.")
        importances = model.feature_importances_

    # Case 2: The model is a BaggingClassifier, and its base estimators have importances (e.g., DecisionTree)
    elif isinstance(model, BaggingClassifier) and hasattr(model.estimators_[0], 'feature_importances_'):
        print("Calculating averaged feature importances for BaggingClassifier...")
        
        # Collect importances from all trees in the bag
        all_importances = np.array([
            estimator.feature_importances_ for estimator in model.estimators_
        ])
        
        # Average the importances across all estimators
        importances = np.mean(all_importances, axis=0)

    # If we successfully found or calculated importances, format them for the UI
    if importances is not None:
        feature_importance_data = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        model_analysis['feature_importance'] = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in feature_importance_data
        ]
    
    return model_metrics, model_analysis


# ==============================================================================
# The Main Executor Class
# ==============================================================================

class ModelTrainerExecutor(BaseNodeExecutor):
    """
    Executes the logic for a 'modelTrainer' node.
    
    This node takes a DataFrame with features and a 'label' column, trains a
    specified machine learning model, evaluates its performance, and stores both
    the trained model object and its performance metadata in the execution context.
    
    The node passes its input DataFrame through unmodified, allowing further
    pipeline steps to use the original data.
    """
    
    def execute(
        self, 
        node: Any,
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        """
        The core execution method for the model trainer.
        """
        print(f"Executing Model Trainer Node: {node.data.get('label', node.id)}")
        
        # --- 1. Semantic Validation ---
        # Provide more specific error messages about which handle is missing.
        if 'train' not in inputs:
            raise ValueError("ModelTrainer node is missing a connection to its 'train' input handle.")
        if 'test' not in inputs:
            raise ValueError(
                "ModelTrainer node is missing a connection to its 'test' input handle. "
                "The test data should connect directly from the DataValidation node's 'test' output."
            )
        
        # test_input_info = inputs['test']
        
        # if test_input_info['source_node_type'] != 'dataValidation':
        #     raise ValueError("ModelTrainer 'test' input must come from a DataValidation node.")
        # if test_input_info['source_handle'] != 'test':
        #     raise ValueError("ModelTrainer 'test' input must be connected to the 'test' output of a DataValidation node.")
        
        # --- 2. Get Data ---
        df_train = inputs['train']['data'].copy()
        df_test = inputs['test']['data'].copy()
        
        # Get feature names from the test set to ensure consistency
        feature_cols = [col for col in df_test.columns if col not in ['label', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        X_train = df_train[feature_cols]
        y_train = df_train['label']
        
        X_test = df_test[feature_cols]
        y_test = df_test['label']
        
        # --- 2. Configuration ---
        model_name = node.data.get('modelName')
        hyperparameters = node.data.get('hyperparameters', {})
        prediction_threshold = node.data.get('predictionThreshold', 0.5)
        
        if not model_name:
            raise ValueError("ModelTrainer node requires a 'modelName' in its configuration.")
        
        # --- 3. Model Training ---
        print(f"Training model: {model_name}...")
        model = get_model(model_name, hyperparameters)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # --- 4. Model Evaluation ---
        metrics, analysis = _calculate_model_performance(
            model, X_test, y_test, feature_cols, threshold=prediction_threshold
        )
        
        # --- 7. Storing Results in Context ---
        metadata_payload = {
            "model_metrics": metrics,
            "model_analysis": analysis,
            "model_info": {
                "Model Name": model_name,
                "Prediction Threshold": prediction_threshold,
                "Problem Type": "Classification" if (y_test.dtype != 'float' and y_test.nunique() < 30) else "Regression",
                "Train Samples": len(y_train),
                "Test Samples": len(y_test),
                "Total Features": len(feature_cols)
            }
        }
        
        context.node_metadata[node.id] = metadata_payload
        context.trained_models[node.id] = model
        print(f"Stored trained model and performance metadata for node {node.id}")
        
        # Return on the 'default' handle
        return {
            "default": df_test
        }