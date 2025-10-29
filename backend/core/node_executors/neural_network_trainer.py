# ==============================================================================
# Standard Library Imports
# ==============================================================================
import os
from typing import Dict, Any, Tuple, List

# ==============================================================================
# Third-party Library Imports
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
from .utils import _calculate_nn_performance # Import our new helper

# ==============================================================================
# The Main Executor Class
# ==============================================================================
class NeuralNetworkTrainerExecutor(BaseNodeExecutor):
    """
    Executes the logic for a 'neuralNetworkTrainer' node.
    
    This node handles the complete lifecycle for a Keras-based neural network:
    1. Dynamically builds a model architecture from node configuration.
    2. Compiles the model with a specified optimizer and loss function.
    3. Trains the model over multiple epochs with early stopping.
    4. Saves the trained model to disk instead of holding it in memory.
    5. Evaluates performance and stores detailed metrics, including training history.
    """
    
    def execute(
        self, 
        node: Any,
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Neural Network Trainer Node: {node.data.get('label', node.id)}")
        
        # --- 1. Get and Prepare Data ---
        df_train = inputs['train']['data'].copy()
        df_test = inputs['test']['data'].copy()
        
        df_train['label'] = df_train['label'].astype(int)
        df_test['label'] = df_test['label'].astype(int)
        
        # Define the feature set using the clean test data, just like in the ModelTrainerExecutor.
        # This ensures we ignore any columns that might have NaNs from SMOTE (like timestamp, etc.).
        non_feature_cols = ['label', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_test.columns if col not in non_feature_cols]
        
        # Now, select ONLY these clean feature columns from both dataframes.
        X_train = df_train[feature_cols]
        y_train = df_train['label']
        
        X_test = df_test[feature_cols]
        y_test = df_test['label']
        
        # --- Data Integrity Check ---
        # Keras is sensitive to NaNs. Let's explicitly check and handle them.
        if X_train.isnull().sum().sum() > 0:
            print("  - Warning: NaNs found in training features after selection. Filling with 0.")
            X_train = X_train.fillna(0) # Or use a more sophisticated imputer
        if X_test.isnull().sum().sum() > 0:
            print("  - Warning: NaNs found in test features after selection. Filling with 0.")
            X_test = X_test.fillna(0)
        
        # --- 2. Get Complex Configuration ---
        architecture_config = node.data.get('architecture', {})
        training_config = node.data.get('training', {})
        prediction_threshold = node.data.get('predictionThreshold', 0.5)
        
        # --- 3. Build Model Dynamically ---
        try:
            model = self._build_model(architecture_config, input_shape=(X_train.shape[1],))
        except Exception as e:
            raise ValueError(f"Failed to build model architecture: {e}")
        
        # --- 4. Compile Model ---
        model.compile(
            optimizer=training_config.get('optimizer', 'adam'),
            loss=training_config.get('loss', 'binary_crossentropy'),
            metrics=['accuracy']
        )
        print("Model architecture compiled successfully:")
        model.summary()
        
        # --- 5. Train Model ---
        print("Starting Neural Network training...")
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=training_config.get('earlyStoppingPatience', 10), 
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=training_config.get('epochs', 50),
            batch_size=training_config.get('batchSize', 32),
            validation_split=0.2, # Use part of training data for validation during training
            callbacks=[early_stopping],
            verbose=2 
        )
        print("Model training complete.")
        
        # --- 6. Save Model to Disk (Crucial for NNs) ---
        run_dir = os.path.join("pipeline_runs", context.run_id)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, f"{node.id}_model.keras")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Store the PATH to the model, not the model object itself
        context.trained_models[node.id] = model_path
        
        # --- 7. Evaluate Model using our specialized utility ---
        metrics, analysis = _calculate_nn_performance(
            model_path, X_test, y_test, history.history, threshold=prediction_threshold
        )
        
        # --- 8. Storing Results in Context ---
        context.node_metadata[node.id] = {
            "model_metrics": metrics,
            "model_analysis": analysis,
            "model_info": {
                "Model Type": "Neural Network (Keras)",
                "Prediction Threshold": prediction_threshold,
                "Optimizer": training_config.get('optimizer', 'adam'),
                "Loss Function": training_config.get('loss', 'binary_crossentropy'),
                "Epochs (Requested)": training_config.get('epochs', 50),
                "Epochs (Actual)": len(history.history['loss']),
                "Batch Size": training_config.get('batchSize', 32),
            }
        }
        print(f"Stored NN model performance metadata for node {node.id}")
        
        return {"default": df_test}

    def _build_model(self, config: Dict, input_shape: tuple) -> tf.keras.Model:
        """Dynamically constructs a Keras Sequential model from a configuration dict."""
        model = Sequential()
        layers_config = config.get('layers', [])
        
        if not layers_config:
            raise ValueError("Architecture config must contain a non-empty 'layers' list.")
            
        # Add the input layer based on the first layer's config
        model.add(tf.keras.Input(shape=input_shape))

        for i, layer_conf in enumerate(layers_config):
            layer_type = layer_conf.get('type')
            if layer_type == 'Dense':
                model.add(Dense(
                    units=int(layer_conf.get('units', 64)),
                    activation=layer_conf.get('activation', 'relu')
                ))
            elif layer_type == 'Dropout':
                model.add(Dropout(rate=float(layer_conf.get('rate', 0.5))))
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' at layer {i}.")
        
        # Add the final output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))
        return model