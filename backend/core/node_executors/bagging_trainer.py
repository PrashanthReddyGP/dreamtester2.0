from typing import Any
import numpy as np

from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
from core.ml_models import get_model
from .model_trainer import _calculate_model_performance 

class BaggingTrainerExecutor(BaseNodeExecutor):
    """
    Executes the logic for a 'baggingTrainer' node.
    
    This node trains a BaggingClassifier with a dynamically specified base model.
    It reuses the evaluation logic from the standard ModelTrainerExecutor.
    """
    
    def execute(
        self, 
        node: Any,
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Bagging Trainer Node: {node.data.get('label', node.id)}")
        
        # --- 1. Validation ---
        if 'train' not in inputs:
            raise ValueError("BaggingTrainer node is missing a connection to its 'train' input.")
        if 'test' not in inputs:
            raise ValueError("BaggingTrainer node is missing a connection to its 'test' input.")
        
        # --- 2. Get Data ---
        df_train = inputs['train']['data'].copy()
        df_test = inputs['test']['data'].copy()
        
        feature_cols = [col for col in df_test.columns if col not in ['label', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        X_train = df_train[feature_cols]
        y_train = df_train['label']
        X_test = df_test[feature_cols]
        y_test = df_test['label']
        
        # --- 3. Configuration ---
        # The hyperparameters are now a nested structure
        bagging_params = node.data.get('baggingHyperparameters', {})
        base_model_name = node.data.get('baseModelName')
        base_model_params = node.data.get('baseModelHyperparameters', {})
        
        prediction_threshold = node.data.get('predictionThreshold', 0.5)
        
        if not base_model_name:
            raise ValueError("BaggingTrainer node requires a 'baseModelName' in its configuration.")
        
        # --- 3. Configuration & Training ---
        model_name = 'BaggingClassifier'
        
        hyperparameters = {
            'baseModelName': base_model_name,
            'baseModelHyperparameters': base_model_params,
            'baggingHyperparameters': bagging_params
        }
        
        # --- 4. Model Training ---
        print(f"Training model: {model_name}...")
        model = get_model(model_name, hyperparameters)
        model.fit(X_train, y_train) 
        print("Model training complete.")
        
        # --- 5. Model Evaluation (Reusing the existing helper!) ---
        metrics, analysis = _calculate_model_performance(
            model, X_test, y_test, feature_cols, threshold=prediction_threshold
        )
        
        # --- 6. Storing Results in Context ---
        metadata_payload = {
            "model_metrics": metrics,
            "model_analysis": analysis,
            "model_info": {
                "Model Name": f"BaggingClassifier (Base: {base_model_name})",
                "Prediction Threshold": prediction_threshold,
                "Problem Type": "Classification",
                "Train Samples": len(y_train),
                "Test Samples": len(y_test),
                "Total Features": len(feature_cols)
            }
        }
        
        context.node_metadata[node.id] = metadata_payload
        context.trained_models[node.id] = model
        print(f"Stored trained bagging model and metadata for node {node.id}")
        
        return {
            "default": df_test
        }