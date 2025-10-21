from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class DataScalingExecutor(BaseNodeExecutor):
    """Executor for 'dataScaling' node. Preprocesses features."""
    
    def execute(self, node: Any, inputs:  ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Data Scaling Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            raise ValueError("DataScaling node requires an input.")
        df_input = list(inputs.values())[0]['data'].copy()
        
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'label']
        feature_cols = [col for col in df_input.columns if col not in original_cols]
        
        if not feature_cols:
            print("Warning: No feature columns found to scale.")
            return df_input
        
        features_df = df_input[feature_cols]
        
        if node.data.get('removeCorrelated'):
            threshold = node.data.get('correlationThreshold', 0.9)
            corr_matrix = features_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            features_df = features_df.drop(columns=to_drop)
        
        scaler_type = node.data.get('scaler', 'none')
        if scaler_type != 'none':
            scaler = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler()}[scaler_type]
            scaled_features = scaler.fit_transform(features_df)
            features_df = pd.DataFrame(scaled_features, index=features_df.index, columns=features_df.columns)
        
        if node.data.get('usePCA'):
            n_components = node.data.get('pcaComponents', 5)
            n_components = min(n_components, len(features_df.columns), len(features_df))
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(features_df)
            pca_cols = [f'pca_{i+1}' for i in range(n_components)]
            features_df = pd.DataFrame(data=principal_components, columns=pca_cols, index=features_df.index)
        
        non_feature_df = df_input[[col for col in df_input.columns if col not in feature_cols]]
        df_output = pd.concat([non_feature_df, features_df], axis=1)
        
        return {"default": df_output}