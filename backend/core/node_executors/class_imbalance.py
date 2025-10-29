import pandas as pd
from typing import Dict, Any

from imblearn.over_sampling import SMOTE
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

class ClassImbalanceExecutor(BaseNodeExecutor):
    """
    Takes the training data as input, applies SMOTE to the numeric feature
    columns, and returns the new, balanced training DataFrame with all

    original columns preserved.
    """
    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Class Imbalance Node: {node.data.get('label', node.id)}")

        # --- 1. Input Validation ---
        if len(inputs) != 1:
            raise ValueError("ClassImbalance node must have exactly one input.")
        
        input_info = list(inputs.values())[0]
        # if input_info['source_node_type'] != 'dataValidation':
        #     raise ValueError("ClassImbalance node input must come from a DataValidation node.")
        # if input_info['source_handle'] != 'train':
        #     raise ValueError("ClassImbalance node must be connected to the 'train' output of a DataValidation node.")
        
        df_train_input = input_info['data'].copy()

        if 'label' not in df_train_input.columns:
            raise ValueError("Input training data must contain a 'label' column.")

        # --- 2. Configuration & Data Prep (ROBUST METHOD) ---
        method = node.data.get('method', 'SMOTE')

        # Explicitly define and separate feature columns from non-feature columns.
        non_feature_cols_to_preserve = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]
        
        # Find which of the non-feature columns actually exist in the input.
        existing_non_features = [
            col for col in non_feature_cols_to_preserve if col in df_train_input.columns
        ]

        # Feature columns are everything else, excluding the label and existing non-features.
        feature_cols = [
            col for col in df_train_input.columns 
            if col not in existing_non_features and col != 'label'
        ]

        if not feature_cols:
            raise ValueError("No numeric feature columns found to apply SMOTE.")

        # Separate the DataFrame into three parts
        X_train_features = df_train_input[feature_cols]
        y_train = df_train_input['label']
        df_non_features = df_train_input[existing_non_features]

        original_distribution = y_train.value_counts().to_dict()
        print(f"Original training set distribution: {original_distribution}")

        # --- 3. Apply SMOTE to Numeric Features Only ---
        print("Applying SMOTE to the training set...")
        smote = SMOTE(random_state=42)
        X_resampled_features, y_resampled = smote.fit_resample(X_train_features, y_train)
        
        resampled_distribution = pd.Series(y_resampled).value_counts().to_dict()
        print(f"Resampled training set distribution: {resampled_distribution}")

        # --- 4. Reconstruct the Full DataFrame ---
        # Convert the resampled numpy arrays back to DataFrames
        df_resampled_features = pd.DataFrame(X_resampled_features, columns=feature_cols)
        
        num_original = len(df_train_input)
        num_synthetic = len(y_resampled) - num_original

        # Create a placeholder DataFrame for the non-feature columns of the new synthetic samples
        df_synthetic_non_features = pd.DataFrame(
            index=range(num_synthetic), 
            columns=existing_non_features
        )
        if 'timestamp' in df_synthetic_non_features.columns:
            df_synthetic_non_features['timestamp'] = pd.NaT

        # Combine the original non-features with the new synthetic placeholders
        df_full_non_features = pd.concat([
            df_non_features.reset_index(drop=True), 
            df_synthetic_non_features
        ], ignore_index=True)

        # Combine all parts back together
        df_output = pd.concat([
            df_full_non_features,
            df_resampled_features
        ], axis=1)
        df_output['label'] = y_resampled

        # --- 5. Store Metadata ---
        context.node_metadata[node.id] = {
            "resampling_info": {
                "Method": method,
                "Original Distribution": {str(k): int(v) for k, v in original_distribution.items()},
                "Distribution (After Resampling)": {str(k): int(v) for k, v in resampled_distribution.items()},
                "Original Training Rows": num_original,
                "Synthetic Rows Added": num_synthetic,
                "Total Training Rows After": len(df_output)
            }
        }
        
        # --- 6. Return the new, balanced training DataFrame ---
        return {
            "train": df_output
        }