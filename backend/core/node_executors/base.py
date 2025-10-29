from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

# This defines the return type for all executors
# It can be a single DataFrame (for simple nodes) or a dict of them (for multi-output nodes)
ExecutorOutput = Dict[str, pd.DataFrame]

# Define the structure of the new rich input object
ExecutorInput = Dict[str, Any] 

class ExecutionContext:
    """
    A context object passed through the pipeline, holding shared state.
    This includes intermediate dataframes, generated metadata, and trained models.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        
        # Stores the primary DataFrame output of each node
        self.node_outputs: Dict[str, pd.DataFrame] = {}
        # Stores any special metadata a node wants to provide (e.g., metrics, charts)
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        # Stores trained model objects, keyed by their trainer node ID
        self.trained_models: Dict[str, Any] = {}
        
        # Stores pipeline-wide parameters, like symbol and timeframe from the source
        self.pipeline_params: Dict[str, Any] = {}

class BaseNodeExecutor(ABC):
    """Abstract Base Class for all node executors."""
    
    @abstractmethod
    def execute(
        self, 
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        """
        Executes the node's logic.
        
        Args:
            node: The node object from the frontend.
            inputs: A dictionary of parent DataFrames, keyed by the handle ID
                    on the current node they are connected to (e.g., 'a', 'b', 'train').
            context: The shared execution context for the pipeline run.
            
        Returns:
            A dictionary of output DataFrames, keyed by the source handle ID
            of this node (e.g., {'train': df_train, 'test': df_test}). For single-output
            nodes, this will be {'default': df_output}.
        """
        raise NotImplementedError
