from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any

from core.machinelearning import load_data_for_ml

class DataSourceExecutor(BaseNodeExecutor):
    """Executor for the 'dataSource' node. Loads initial data."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing DataSource Node: {node.data.get('label', node.id)}")
        
        # This node has no inputs, it's the start of a pipeline.
        ds_config = node.data
        
        symbol = ds_config.get('symbol')
        timeframe = ds_config.get('timeframe')
        start_date = ds_config.get('startDate')
        end_date = ds_config.get('endDate')
        
        df_output = load_data_for_ml(
            symbol=symbol,
            timeframe=timeframe,
            start_date_str=start_date,
            end_date_str=end_date
        )
        
        # Store symbol and timeframe in this node's metadata
        # This ensures it gets saved to the cache along with the data.
        context.node_metadata[node.id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return {"default": df_output}