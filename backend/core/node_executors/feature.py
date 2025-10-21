from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any
from types import SimpleNamespace

from core.robust_idk_processor import calculate_indicators
from core.machinelearning import transform_features_for_backend

from core.types import IndicatorConfig

class FeatureExecutor(BaseNodeExecutor):
    """Executor for the 'feature' node. Calculates a technical indicator."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Feature Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            raise ValueError("Feature node requires an input.")
        
        # All single-input nodes will use this pattern.
        df_input = list(inputs.values())[0]['data'].copy()
        
        # Reformat feature data for the backend function
        indicator_config = [IndicatorConfig(id=node.id, name=node.data['name'], timeframe=node.data['timeframe'], params=node.data['params'])]
        
        indicator_tuples = transform_features_for_backend(indicator_config)
        
        mock_strategy = SimpleNamespace(indicators=indicator_tuples)
        
        df_output = calculate_indicators(mock_strategy, df_input)
        
        return {"default": df_output}