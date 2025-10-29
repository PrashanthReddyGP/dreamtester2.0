# executors/data_scaling.py

# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Any, Dict, List
import pandas as pd
import numpy as np

# ==============================================================================
# Third-party Library Imports
# ==============================================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

# ==============================================================================
# The Main Executor Class
# ==============================================================================

class DataScalingExecutor(BaseNodeExecutor):
    """
    Executes data scaling, correlation removal, and PCA based on node configuration.
    Returns the transformed DataFrame and places informational metadata into the context.
    """

    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Data Scaling Node: {node.data.get('label', node.id)}")

        # --- 1. Input Validation ---
        if not inputs:
            raise ValueError("DataScaling node requires an input connection.")
        
        df_input = list(inputs.values())[0]['data']
        df = df_input.copy()
        
        # This dictionary will hold all metadata for display
        metadata = {}

        # --- 2. Identify Feature Columns ---
        non_feature_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found for scaling.")
        
        print(f"  - Original feature columns: {len(feature_cols)}")
        
        # --- 3. Remove Highly Correlated Features ---
        if node.data.get('removeCorrelated', False):
            threshold = node.data.get('correlationThreshold', 0.9)
            corr_matrix = df[feature_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
                feature_cols = [col for col in feature_cols if col not in to_drop]
                # Add to metadata for display
                metadata['correlated_features_removed'] = to_drop
                print(f"  - Removed {len(to_drop)} correlated features: {to_drop}")

        # --- 4. Apply Scaler ---
        scaler_type = node.data.get('scaler', 'none')
        scaler_instance = None
        if scaler_type != 'none':
            if scaler_type == 'standard':
                scaler_instance = StandardScaler()
            elif scaler_type == 'min_max':
                scaler_instance = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler_instance = RobustScaler()
            
            if scaler_instance:
                df[feature_cols] = scaler_instance.fit_transform(df[feature_cols])
                
                context.ml_objects[f'{node.id}_scaler'] = scaler_instance
                # Add to metadata for display
                metadata['scaler_applied'] = scaler_type
                print(f"  - Applied {scaler_type} scaler.")

        # --- 5. Apply PCA ---
        if node.data.get('usePCA', False):
            n_components = node.data.get('pcaComponents', 5)
            n_components = min(n_components, len(feature_cols))

            if n_components > 0:
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(df[feature_cols])
                
                pca_df = pd.DataFrame(
                    data=principal_components, 
                    columns=[f'PC_{i+1}' for i in range(n_components)],
                    index=df.index
                )
                
                df = df.drop(columns=feature_cols)
                df = pd.concat([df, pca_df], axis=1)
                
                context.ml_objects[f'{node.id}_pca'] = pca
                # Add to metadata for display
                metadata['pca_applied'] = {
                    'n_components': n_components,
                    'explained_variance_ratio': [round(x, 4) for x in pca.explained_variance_ratio_.tolist()],
                    'cumulative_explained_variance': round(np.sum(pca.explained_variance_ratio_).item(), 4)
                }
                print(f"  - Applied PCA with {n_components} components.")

        # --- 6. Store all collected metadata in the context's side channel ---
        context.node_metadata[node.id] = metadata

        # --- 7. Return the transformed DataFrame on the data highway ---
        # The key 'default' corresponds to the single output handle.
        return {
            "default": df
        }