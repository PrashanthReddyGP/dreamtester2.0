from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any
from types import SimpleNamespace

from core.robust_idk_processor import calculate_indicators
from core.machinelearning import transform_features_for_backend
from .utils import resolve_parameters

from core.types import IndicatorConfig

class FeatureExecutor(BaseNodeExecutor):
    """Executor for the 'feature' node. Calculates a technical indicator."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        
        if not inputs:
            raise ValueError("Feature node requires an input.")
        
        # All single-input nodes will use this pattern.
        df_input = list(inputs.values())[0]['data'].copy()
        
        # Resolve templated parameters before using them
        # This is the key change. It checks node.data for any `{{...}}` placeholders
        # and replaces them with values from the context (injected by the loop node).
        resolved_node_data = resolve_parameters(node.data, context.pipeline_params)
        
        # Add a print statement for easy debugging during a loop run
        print(f"Executing Feature Node '{resolved_node_data.get('label')}' with resolved params: {resolved_node_data.get('params')}")
        
        # --- UPDATED: Use the resolved data from now on ---
        # Reformat feature data for the backend function
        indicator_config = [IndicatorConfig(
            id=node.id, 
            name=resolved_node_data['name'], 
            timeframe=resolved_node_data['timeframe'], 
            params=resolved_node_data['params']
        )]
        
        indicator_tuples = transform_features_for_backend(indicator_config)
        
        mock_strategy = SimpleNamespace(indicators=indicator_tuples)
        
        df_output = calculate_indicators(mock_strategy, df_input)
        
        return {"default": df_output}