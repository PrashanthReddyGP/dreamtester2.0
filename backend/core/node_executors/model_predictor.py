from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any

class ModelPredictorExecutor(BaseNodeExecutor):
    """
    Executor for 'modelPredictor'. Uses a pre-trained model to generate predictions.
    """
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Model Predictor Node: {node.data.get('label', node.id)}")
        
        # --- 1. Get Configuration ---
        # The ID of the ModelTrainer node that holds the model we want to use.
        trainer_node_id = node.data.get('trainerNodeId')
        
        if not trainer_node_id:
            raise ValueError("ModelPredictor node must be configured with a trainerNodeId.")
        
        # --- 2. Retrieve the Trained Model from Context ---
        model = context.trained_models.get(trainer_node_id)
        if model is None:
            raise ValueError(f"Model from trainer node '{trainer_node_id}' not found in context. Ensure the trainer has been executed.")
        
        # --- 3. Get Input Data ---
        # This node assumes the first input is the data to predict on.
        if not inputs:
            raise ValueError("ModelPredictor node requires an input DataFrame.")
        
        df_input = list(inputs.values())[0]['data'].copy()
        
        # --- 4. Prepare Features for Prediction ---
        # We need to use the exact same feature columns the model was trained on.
        # This information is available on the model object itself in scikit-learn >= 1.0
        if not hasattr(model, 'feature_names_in_'):
            raise RuntimeError("The trained model does not have 'feature_names_in_'. Cannot guarantee correct feature set.")
        
        feature_cols = model.feature_names_in_
        
        # Ensure all required columns are present
        missing_cols = set(feature_cols) - set(df_input.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required feature columns: {missing_cols}")
        
        X_predict = df_input[feature_cols]
        
        # --- 5. Generate and Append Predictions ---
        predictions = model.predict(X_predict)
        df_output = df_input.copy()
        df_output['prediction'] = predictions
        
        return {"default": df_output}