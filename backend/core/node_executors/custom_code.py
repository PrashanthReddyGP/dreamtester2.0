from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
from typing import Dict, Any

class CustomCodeExecutor(BaseNodeExecutor):
    """Executor for 'customCode' node. Runs user-provided Python code."""
    
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Custom Code Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            df_input = pd.DataFrame() # Or handle as error if input is required
        else:
            df_input = list(inputs.values())[0]['data'].copy()
        
        custom_code = node.data.get('code', '')
        sub_type = node.data.get('subType', 'feature_engineering')
        
        if not custom_code.strip():
            return df_input # Pass through if no code
        
        local_namespace = {}
        exec_globals = globals().copy() 
        exec(custom_code, exec_globals, local_namespace)
        
        if sub_type == 'labeling':
            labeling_func = local_namespace.get('generate_labels')
            if not callable(labeling_func):
                raise NameError("Labeling script must have a 'generate_labels(data)' function.")
            
            labeling_func.__globals__.update(local_namespace)
            labels_series = labeling_func(df_input.copy())
            if not isinstance(labels_series, pd.Series):
                raise TypeError("'generate_labels' function must return a pandas Series.")
            
            df_output = df_input.copy()
            df_output['label'] = labels_series
            
            df_labeled = df_output.dropna(subset=['label'])
            label_counts = df_labeled['label'].value_counts()
            label_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}
            
            # Store metadata in the context
            context.node_metadata[node.id] = {"label_distribution": label_distribution}
            return {"default": df_output}
        
        else: # 'feature_engineering'
            process_func = local_namespace.get('process')
            if not callable(process_func):
                raise NameError("Feature Engineering script must have a 'process(data)' function.")
            
            process_func.__globals__.update(local_namespace)
            df_output = process_func(df_input.copy())
            if not isinstance(df_output, pd.DataFrame):
                raise TypeError("'process' function must return a pandas DataFrame.")
            return {"default": df_output}