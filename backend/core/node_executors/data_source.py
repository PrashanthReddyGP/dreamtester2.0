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
        
        # Replace with your actual data loading function
        df_output = load_data_for_ml(
            symbol=ds_config.get('symbol'),
            timeframe=ds_config.get('timeframe'),
            start_date_str=ds_config.get('startDate'),
            end_date_str=ds_config.get('endDate')
        )
        return {"default": df_output}