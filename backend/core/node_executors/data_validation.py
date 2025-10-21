# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Dict, Any
import pandas as pd

# ==============================================================================
# Third-party Library Imports
# ==============================================================================
from sklearn.model_selection import train_test_split

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

# ==============================================================================
# The Main Executor Class
# ==============================================================================

class DataValidationExecutor(BaseNodeExecutor):
    """
    Executes the logic for the 'dataValidation' node.

    This node is responsible for splitting the dataset into training and testing sets.
    It takes a single DataFrame as input and does NOT return a DataFrame itself.
    Instead, it populates the ExecutionContext with the split data (`X_train`, `X_test`,
    `y_train`, `y_test`), which downstream nodes like ModelTrainer can then use.
    It also calculates and stores metadata about the split.
    """

    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Data Validation Node: {node.data.get('label', node.id)}")
        
        # --- 1. Input Validation ---
        if not inputs:
            raise ValueError("DataValidation node requires an input connection.")
        
        df_input = list(inputs.values())[0]['data'].copy()

        if 'label' not in df_input.columns:
            raise ValueError("Input data for DataValidation must contain a 'label' column.")

        # --- 2. Data Preparation ---
        df_labeled = df_input.dropna(subset=['label']).copy()
        
        non_feature_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_labeled.columns if col not in non_feature_cols and col != 'label']
        
        if not feature_cols:
            raise ValueError("No feature columns found for data validation.")

        X = df_labeled[feature_cols]
        y = df_labeled['label']

        # --- 3. Perform the Split ---
        method = node.data.get('validationMethod', 'train_test_split')
        split_info = {}

        if method == 'train_test_split':
            train_pct = node.data.get('trainSplit', 80) / 100.0
            
            df_train, df_test = train_test_split(
                df_labeled, train_size=train_pct, random_state=42, shuffle=False
            )
            
            split_info = {
                "method": "Train/Test Split",
                "train_test_split": f"{train_pct*100}%",
                "train_samples": len(df_train),
                "test_samples": len(df_test),
                "total_samples": len(df_labeled)
            }
        
        elif method == 'walk_forward':
            # Note: Walk-forward is more complex to implement as it generates multiple folds.
            # For now, we'll raise an error to indicate it's not fully supported in this new paradigm yet.
            raise NotImplementedError("Walk-Forward validation method is not yet supported with multi-output handles.")
        else:
            raise ValueError(f"Unknown validation method: {method}")
        
        # --- 5. Store Metadata ---
        label_counts = df_labeled['label'].value_counts()
        label_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}
        
        context.node_metadata[node.id] = {
            "validation_info": split_info,
            "label_distribution": label_distribution
        }
        
        # --- Return two separate DataFrames keyed by handle ID ---
        return {
            "train": df_train,
            "test": df_test
        }
