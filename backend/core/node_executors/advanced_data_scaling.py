# executors/advanced_data_scaling.py

# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Any
import pandas as pd
import numpy as np

# ==============================================================================
# Third-party Library Imports
# ==============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

# ==============================================================================
# The Main Executor Class
# ==============================================================================

class AdvancedDataScalingExecutor(BaseNodeExecutor):
    """
    Executes complex, multi-method scaling and optional PCA while correctly
    handling train/test data to prevent leakage.
    Fits all transformers ONLY on the training data and then transforms both sets.
    """

    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Advanced Data Scaling Node: {node.data.get('label', node.id)}")

        # --- 1. Input Validation ---
        if 'train_in' not in inputs:
            raise ValueError("Advanced Scaling Node requires a 'train_in' input.")
        
        train_df = inputs['train_in']['data'].copy()
        # Test data is optional for fitting, but required for output
        test_df = inputs.get('test_in', {}).get('data', pd.DataFrame()).copy()

        config = node.data

        # --- Discovery Phase Logic ---
        # The discovery run is now explicitly defined by the isConfigured flag
        is_discovery_run = not config.get('isConfigured', False)

        if is_discovery_run:
            print("  - Discovery run: Sending feature list to frontend.")
            non_feature_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
            all_features = [col for col in train_df.columns if col not in non_feature_cols]
            context.node_metadata[node.id] = {"feature_list": all_features}
            # The isConfigured flag will be set by the frontend after this run
            return {"train_out": train_df, "test_out": test_df if not test_df.empty else train_df.copy()}

        std_cols = config.get('standardFeatures', [])
        minmax_cols = config.get('minmaxFeatures', [])
        robust_cols = config.get('robustFeatures', [])

        # If no features are selected for scaling, just pass the data through
        if not (std_cols or minmax_cols or robust_cols):
            print("  - No features selected for scaling. Passing data through.")
            context.node_metadata[node.id] = {"message": "No features were selected for scaling."}
            return {"train_out": train_df, "test_out": test_df}

        # --- 2. Build the Preprocessor ---
        transformers = []
        # Ensure we only add transformers if columns are selected for them
        if std_cols:
            transformers.append(('standard_scaler', StandardScaler(), std_cols))
        if minmax_cols:
            transformers.append(('minmax_scaler', MinMaxScaler(), minmax_cols))
        if robust_cols:
            transformers.append(('robust_scaler', RobustScaler(), robust_cols))

        # The ColumnTransformer is the perfect tool for this.
        # 'remainder="passthrough"' ensures columns not selected for scaling are kept.
        preprocessor = ColumnTransformer(
            transformers=transformers, 
            remainder='passthrough', 
            verbose_feature_names_out=False
        )
        # Configure it to output a pandas DataFrame, which is much easier to work with
        preprocessor.set_output(transform="pandas")

        # --- 3. FIT ONLY ON TRAIN DATA ---
        print("  - Fitting preprocessor on training data...")
        preprocessor.fit(train_df)
        print("  - Preprocessor fitted successfully.")

        # --- 4. TRANSFORM BOTH DATASETS ---
        print("  - Transforming train and test data...")
        scaled_train_df = preprocessor.transform(train_df)
        
        scaled_test_df = pd.DataFrame()
        if not test_df.empty:
            scaled_test_df = preprocessor.transform(test_df)

        # --- (Optional) Add PCA/Correlation logic here ---
        # This would also need to be fit on train and transformed on both.
        # For simplicity, this is omitted for now but would follow the same pattern.

        # --- 5. Store the Fitted Object and Return ---
        # print("  - Storing fitted preprocessor in context.")
        # context.ml_objects[node.id + '_preprocessor'] = preprocessor
        
        context.node_metadata[node.id] = {
            "message": "Scaling applied successfully.",
            "scalers_used": {
                "Standard": std_cols,
                "MinMax": minmax_cols,
                "Robust": robust_cols
            }
        }

        return {
            "train_out": scaled_train_df,
            "test_out": scaled_test_df
        }