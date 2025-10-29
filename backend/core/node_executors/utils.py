# src/executors/utils.py

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

def resolve_parameters(data: Any, context_params: Dict[str, Any]) -> Any:
    """
    Recursively traverses a data structure (dict, list, string) and replaces
    `{{variable_name}}` placeholders with values from the context_params.
    """
    print("########", context_params)
    if isinstance(data, dict):
        return {k: resolve_parameters(v, context_params) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_parameters(item, context_params) for item in data]
    elif isinstance(data, str):
        # Check if the entire string is a placeholder.
        match = re.fullmatch(r"\{\{([a-zA-Z0-9_]+)\}\}", data)
        if match:
            placeholder = match.group(1)
            # If it's a full match and the variable exists in the context,
            # return the value from the context directly, preserving its type (e.g., int, float).
            if placeholder in context_params:
                return context_params[placeholder]
        
        # If it's not a full match (or the variable wasn't found),
        # perform a simple string substitution for any placeholders within the string.
        # This handles cases like "file_{{id}}.csv".
        def replace_func(m):
            key = m.group(1)
            return str(context_params.get(key, m.group(0)))
        
        return re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", replace_func, data)
    else:
        # Return numbers, booleans, etc. as-is
        return data

# ==============================================================================
# Helper Function for Neural Network Model Evaluation
# ==============================================================================

def _calculate_nn_performance(
    model_path: str, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    training_history: Dict[str, List[float]],
    threshold: float = 0.5
) -> Tuple[Dict, Dict]:
    """
    Evaluates a trained Keras model and computes metrics and analysis.
    This is specifically designed for binary classification NNs.
    
    Args:
        model_path: The file path to the saved Keras model (.h5 or saved_model format).
        X_test: The test features (as a NumPy array).
        y_test: The true test labels (as a NumPy array).
        training_history: The history object from model.fit(), containing loss and metrics per epoch.
        threshold: The probability threshold for converting probabilities to class labels.
    
    Returns:
        A tuple containing two dictionaries: (model_metrics, model_analysis).
    """
    model_metrics = {}
    model_analysis = {}
    
    # 1. Load the trained model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Generate Predictions
    # predict() returns probabilities. Flatten to a 1D array.
    y_proba = model.predict(X_test).flatten()
    # Apply threshold to get final binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # 3. Calculate Classification Metrics
    print("Calculating classification metrics for Neural Network...")
    model_metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
    model_metrics['precision'] = round(precision_score(y_test, y_pred, zero_division=0), 4)
    model_metrics['recall'] = round(recall_score(y_test, y_pred, zero_division=0), 4)
    model_metrics['f1_score'] = round(f1_score(y_test, y_pred, zero_division=0), 4)
    # ROC AUC is calculated on the raw probabilities
    model_metrics['roc_auc'] = round(roc_auc_score(y_test, y_proba), 4)
    
    # 4. Generate Detailed Analysis
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cleaned_report = {
        str(key): {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else float(value)
        for key, value in report.items()
    }
    model_analysis['classification_report'] = cleaned_report
    
    # Confusion Matrix
    class_labels = sorted(list(np.unique(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    model_analysis['confusion_matrix'] = {
        'labels': [str(label) for label in class_labels],
        'values': cm.tolist()
    }
    
    # Training History (for plotting loss curves on the frontend)
    epochs = range(1, len(training_history.get('loss', [])) + 1)
    history_data = [
        {
            'epoch': epoch,
            'loss': round(training_history.get('loss', [])[i], 4),
            'accuracy': round(training_history.get('accuracy', [])[i], 4),
            'val_loss': round(training_history.get('val_loss', [])[i], 4),
            'val_accuracy': round(training_history.get('val_accuracy', [])[i], 4),
        }
        for i, epoch in enumerate(epochs)
    ]
    model_analysis['training_history'] = history_data
    
    # Note: Feature importance is non-trivial for NNs and is omitted here.
    # Libraries like SHAP or LIME would be needed for a deeper implementation.
    
    return model_metrics, model_analysis