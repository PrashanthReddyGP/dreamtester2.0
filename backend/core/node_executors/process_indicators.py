from .base import BaseNodeExecutor, ExecutionContext
import pandas as pd
from typing import Dict, Any

class ProcessIndicatorsExecutor(BaseNodeExecutor):
    """Executor for 'processIndicators'. Merges features into a base DataFrame."""
    
    def execute(self, node: Any, inputs: Dict[str, pd.DataFrame], context: ExecutionContext) -> pd.DataFrame:
        print(f"Executing Process Indicators Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            raise ValueError("ProcessIndicators node requires at least one input.")
        
        selected_ids = [pid for pid, is_selected in node.data.get('selectedIndicators', {}).items() if is_selected]
        
        # Filter to only parents that are both connected and selected
        connected_and_selected_ids = list(set(inputs.keys()) & set(selected_ids))
        
        # Find the "base" dataframe (e.g., the direct output from dataSource)
        # This logic is complex and relies on knowing parent node types. A more robust
        # way is to have dedicated input handles on the node (e.g., 'base_df', 'feature_df').
        # For now, we assume the first input is the base.
        parent_ids = list(inputs.keys())
        base_df_parent_id = parent_ids[0]
        df_output = inputs[base_df_parent_id].copy()
        
        # Merge selected feature columns
        for feature_node_id in connected_and_selected_ids:
            if feature_node_id == base_df_parent_id:
                continue # Don't try to merge the base df with itself
            
            feature_df = inputs[feature_node_id]
            # Find the new column(s) the feature node added
            new_cols = feature_df.columns.difference(df_output.columns).tolist()
            
            # Ensure the index (timestamp) is available for merging
            merge_on_col = feature_df.index.name if feature_df.index.name else 'timestamp'
            if merge_on_col not in feature_df:
                feature_df_for_merge = feature_df.reset_index()
            else:
                feature_df_for_merge = feature_df
            
            df_output = pd.merge(df_output, feature_df_for_merge[new_cols + [merge_on_col]], on=merge_on_col, how='left')
        
        return df_output
    
    
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any

class ProcessIndicatorsExecutor(BaseNodeExecutor):
    """Executor for 'processIndicators'. Merges features into a base DataFrame."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Process Indicators Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            raise ValueError("ProcessIndicators node requires at least one input.")
        
        # Create a simple {node_id: DataFrame} dictionary that the original logic expects.
        parent_dataframes = {
            info['source_node_id']: info['data']
            for info in inputs.values()
        }
        
        # 1. Get the list of parent node IDs that are checked in the UI
        selected_ids = {pid for pid, is_selected in node.data.get('selectedIndicators', {}).items() if is_selected}
        
        # 2. Get all connected parent node IDs
        connected_parent_ids = list(parent_dataframes.keys())
        
        if not connected_parent_ids:
            raise ValueError("ProcessIndicators has connections but no valid parent data.")
        
        # --- 1. Map all incoming connections to a {node_id: DataFrame} dictionary ---
        parent_dataframes = {
            info['source_node_id']: info['data']
            for info in inputs.values()
        }
        
        # --- 2. Get the list of parent node IDs that are CHECKED in the UI ---
        selected_parent_ids = {pid for pid, is_selected in node.data.get('selectedIndicators', {}).items() if is_selected}
        
        # --- 3. Filter the connected DataFrames to only those that are selected ---
        dataframes_to_merge = [
            df for node_id, df in parent_dataframes.items() 
            if node_id in selected_parent_ids
        ]
        
        if not dataframes_to_merge:
            # If nothing is selected, we can return an empty DataFrame or the first input as a passthrough.
            # Let's return the first connected DataFrame to avoid breaking the pipeline.
            print("Warning: No indicators were selected in the Process Indicators node. Passing through the first input.")
            first_df = list(parent_dataframes.values())[0]
            return {"default": first_df}

        # --- 4. Iteratively merge the selected DataFrames ---
        
        # Start with the first selected DataFrame as our base
        df_output = dataframes_to_merge[0].copy()

        # Ensure the base is indexed by 'timestamp' for merging
        if 'timestamp' in df_output.columns:
            df_output = df_output.set_index('timestamp')

        # Loop through the REST of the selected DataFrames
        for i in range(1, len(dataframes_to_merge)):
            feature_df = dataframes_to_merge[i].copy()
            if 'timestamp' in feature_df.columns:
                feature_df = feature_df.set_index('timestamp')

            # Find the new column(s) this feature DataFrame adds
            # This is critical for avoiding duplicate columns
            new_cols = feature_df.columns.difference(df_output.columns).tolist()

            if not new_cols:
                # This can happen if a feature node is connected but doesn't add a unique column
                continue
            
            print(f"Merging columns: {new_cols}")

            df_output = df_output.merge(
                feature_df[new_cols],
                left_index=True,
                right_index=True,
                how='left'  # Use 'left' to preserve all rows from the base
            )
        
        # --- 5. Finalize output ---
        # Convert the timestamp index back to a column to match other nodes' output format
        df_output.reset_index(inplace=True)
        
        return {"default": df_output}