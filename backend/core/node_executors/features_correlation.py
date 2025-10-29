import json
import pandas as pd
from typing import Dict, Any

from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

class FeaturesCorrelationExecutor(BaseNodeExecutor):
    """
    Executor for the 'features_correlation' node. Calculates the correlation matrix for numeric features in the DataFrame.
    This node does not modify the DataFrame; it just passes it through.
    The correlation matrix is returned in the 'info' dictionary.
    """
    
    def execute(
        self, 
        node: Any, 
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Features Correlation Node: {node.data.get('label', node.id)}")
        
        # Input Validation
        if not inputs:
            raise ValueError("ModelTrainer node requires an input connection.")
        
        input_node_id = list(inputs.keys())[0]
        df_input = inputs[input_node_id]['data'].copy()
        
        if df_input.empty:
            return df_input, {"error": "Input data is empty."}
        
        method = node.data.get('method', 'pearson')
        
        # Select only numeric columns for correlation calculation
        numeric_df = df_input.select_dtypes(include='number')
        
        if numeric_df.shape[1] < 2:
            return df_input, {"info": "Not enough numeric columns to calculate correlation."}
        
        # Calculate the correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Convert to a JSON-serializable format that's easy to parse on the frontend
        # 'split' orientation is great for this: { "columns": [...], "index": [...], "data": [[...]] }
        corr_data_json = corr_matrix.to_json(orient='split')
        
        # We parse it back here to avoid double-encoding JSON strings
        correlation_data = json.loads(corr_data_json)
        
        # The info dict will be sent to the frontend
        # The original DataFrame is passed through unmodified
        
        # Store Metadata
        context.node_metadata[node.id] = {
            "correlation_info": {
                "method": method,
                "correlation_data": correlation_data
            }
        }
        return {"default": df_input}