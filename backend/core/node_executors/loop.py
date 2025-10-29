# src/executors/loop.py

from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput
import pandas as pd
import numpy as np
from typing import Any

class LoopNodeExecutor(BaseNodeExecutor):
    """
    Executor for the 'loop' node. This is a control-flow node.
    It manages the state of a loop and directs execution based on its state.
    """
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        node_id = node.id
        # Use a unique key in the context to store this specific loop's state
        loop_state_key = f"loop_state_{node_id}"

        # --- Step 1: Initialize or Retrieve Loop State ---
        if loop_state_key not in context.node_metadata:
            # First time hitting this node: Initialize the loop
            print(f"INITIALIZING LOOP for node {node_id}")
            loop_data = node.data
            values = []
            if loop_data['loopType'] == 'numeric_range':
                start = float(loop_data['numericStart'])
                end = float(loop_data['numericEnd'])
                step = float(loop_data['numericStep'])
                if step == 0: raise ValueError("Loop step cannot be zero.")
                # Use np.arange for float steps, add a small epsilon to include the end value
                values = list(np.arange(start, end + (step/100), step))
            elif loop_data['loopType'] == 'value_list':
                values = [s.strip() for s in loop_data['valueList'].split(',') if s.strip()]

            if not inputs.get('data-in'):
                raise ConnectionError(f"Loop node '{node_id}' must have a 'Data Input' connection.")

            context.node_metadata[loop_state_key] = {
                "variable_name": loop_data['variableName'],
                "values": values,
                "current_index": 0,
                "results": [],  # To accumulate results from each iteration
                # Store the initial data frame that enters the loop
                "initial_df": inputs['data-in']['data'].copy()
            }

        state = context.node_metadata[loop_state_key]

        # If this execution was triggered by the loop-back, it means an iteration just finished.
        if 'loop-back' in inputs:
            print(f"Loop {node_id}: Iteration {state['current_index']} completed.")
            # Here you could collect results from the finished iteration, e.g., from context.node_metadata
            # For now, we just advance the counter.
            state['current_index'] += 1

        current_index = state['current_index']

        # --- Step 2: Decide What to Do Next ---
        if current_index < len(state['values']):
            # --- CONTINUE LOOP: There are more values to iterate over ---
            current_value = state['values'][current_index]
            variable_name = state['variable_name']
            print(f"Loop {node_id}: Starting iteration {current_index} with {variable_name} = {current_value}")

            # INJECT the current loop variable into the pipeline context
            context.pipeline_params[variable_name] = current_value

            # Send the *original* input data into the loop body for this new iteration
            return {"loop-body": state['initial_df']}
        else:
            # --- END LOOP: All iterations are finished ---
            print(f"Loop {node_id}: All iterations finished. Exiting.")
            variable_name = state['variable_name']

            # CLEAN UP the injected loop variable from the context
            if variable_name in context.pipeline_params:
                del context.pipeline_params[variable_name]

            # Pass the data from the final iteration to the rest of the pipeline
            final_df = inputs.get('loop-back', {}).get('data', state['initial_df'])
            
            # TODO: Aggregate results from `state['results']` into a summary and add to metadata
            context.node_metadata[node.id] = {"message": f"Loop completed after {len(state['values'])} iterations."}
            
            return {"loop-end": final_df}