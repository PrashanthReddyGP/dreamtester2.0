from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
import numpy as np
from typing import Dict, Any

class ChartingExecutor(BaseNodeExecutor):
    """Executor for 'charting' node. Prepares data for frontend visualization."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Charting Node: {node.data.get('label', node.id)}")
        
        # --- 1. Semantic Validation ---
        if len(inputs) != 1:
            raise ValueError("ClassImbalance node must have exactly one input.")
        
        df_input = list(inputs.values())[0]['data'].copy()
        
        chart_config = node.data
        chart_type = chart_config.get('chartType')
        x_axis = chart_config.get('xAxis')
        y_axis = chart_config.get('yAxis')
        group_by = chart_config.get('groupBy')
        
        chart_data = []
        SAMPLE_LIMIT = 5000
        
        if chart_type == 'histogram' and x_axis and x_axis in df_input.columns:
            counts, bins = np.histogram(df_input[x_axis].dropna(), bins=20)
            chart_data = [
                {"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])}
                for i in range(len(counts))
            ]
        elif x_axis and y_axis and x_axis in df_input.columns and y_axis in df_input.columns:
            cols_to_keep = [x_axis, y_axis]
            if group_by and group_by in df_input.columns:
                cols_to_keep.append(group_by)
            
            plot_df = df_input[list(set(cols_to_keep))].dropna()
            
            if len(plot_df) > SAMPLE_LIMIT:
                plot_df = plot_df.sample(n=SAMPLE_LIMIT, random_state=42)
            
            chart_data = plot_df.to_dict(orient='records')
        
        message = f"Displaying {len(chart_data)} of {len(df_input)} total points." if len(chart_data) < len(df_input) else None
        
        context.node_metadata[node.id] = {
            "chart_config": chart_config,
            "chart_data": chart_data,
            "info_message": message
        }
        
        # This node is informational and does not modify the data
        return {"default": df_input}