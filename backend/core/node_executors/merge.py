import pandas as pd
from typing import Dict, Any

from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

class MergeExecutor(BaseNodeExecutor):
    """
    Executor for the 'merge' node. Merges two DataFrames based on their timestamp index.
    """

    def execute(
        self, 
        node: Any, 
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Merge Node: {node.data.get('label', node.id)}")
        
        # --- 1. Validate Inputs ---
        if 'a' not in inputs or 'b' not in inputs:
            raise ValueError("Merge node requires inputs on both 'a' and 'b' handles.")
        
        # Inputs are now keyed by handle ID, making it explicit
        df_a = inputs['a']['data'].copy()
        df_b = inputs['b']['data'].copy()
        
        # --- 2. Get Configuration from the Node ---
        # The 'how' parameter determines the type of merge. Defaults to 'left'.
        # Options: 'left', 'right', 'outer', 'inner'
        merge_method = node.data.get('mergeMethod', 'left').lower()
        
        if merge_method not in ['left', 'right', 'outer', 'inner']:
            raise ValueError(f"Invalid merge method '{merge_method}'. Must be one of 'left', 'right', 'outer', 'inner'.")
        
        # Suffixes to add to overlapping column names (excluding the index)
        left_suffix = node.data.get('leftSuffix', '_left')
        right_suffix = node.data.get('rightSuffix', '_right')
        
        # --- 3. Perform the Merge ---
        # We assume the DataFrames are indexed by timestamp. If they have a 'timestamp'
        # column instead, you could set it as the index first.
        if 'timestamp' in df_a.columns: 
            df_a.set_index('timestamp', inplace=True)
        
        if 'timestamp' in df_b.columns: 
            df_b.set_index('timestamp', inplace=True)
        
        print(f"Performing '{merge_method}' merge on two DataFrames.")
        print(f"Left DF shape: {df_a.shape}, Right DF shape: {df_b.shape}")
        
        df_merged = pd.merge(
            df_a,
            df_b,
            left_index=True,
            right_index=True,
            how=merge_method,
            suffixes=(left_suffix, right_suffix)
        )
        
        print(f"Merged DF shape: {df_merged.shape}")
        
        # --- 4. Store Metadata (Optional but helpful) ---
        context.node_metadata[node.id] = {
            "merge_info": {
                "merge_method": merge_method,
                "left_shape": list(df_a.shape),
                "right_shape": list(df_b.shape),
                "output_shape": list(df_merged.shape),
                "common_columns_renamed": any(col.endswith(left_suffix) for col in df_merged.columns)
            }
        }
        
        # --- 5. Final Step: Convert Index to Column ---
        # Convert the timestamp index back into a regular 'timestamp' column
        # to ensure its output format is consistent with other nodes.
        df_merged.reset_index(inplace=True)
        
        return {"default": df_merged}