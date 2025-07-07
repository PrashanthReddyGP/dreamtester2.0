import os
import sys
import inspect
import importlib.util
from core.basestrategy import BaseStrategy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

TEMP_STRATEGY_DIR = os.path.join(parent_dir, "temp_strategies")
os.makedirs(TEMP_STRATEGY_DIR, exist_ok=True)

def parse_file(job_id, file_name, strategy_code):

    """
    Dynamically imports and runs a backtest on a strategy.
    """
    temp_filepath = None
    
    try:
        # --- 1. Create a unique, temporary Python file ---
        # We use the job_id to ensure the filename is unique to this run.
        strategy_name = os.path.splitext(file_name)[0]
        unique_module_name = f"{strategy_name}_{job_id.replace('-', '_')}"
        temp_filepath = os.path.join(TEMP_STRATEGY_DIR, f"{unique_module_name}.py")
        
        with open(temp_filepath, "w") as f:
            f.write(strategy_code)

        # --- 2. Dynamically import the file as a module ---
        # This is the core magic of importlib
        spec = importlib.util.spec_from_file_location(unique_module_name, temp_filepath)
        strategy_module = importlib.util.module_from_spec(spec)
        
        # Add the parent directory of 'core' to the path so the import works
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        spec.loader.exec_module(strategy_module)
        
        if parent_dir in sys.path:
             sys.path.remove(parent_dir)
        
        sys.path.pop(0) # Clean up the path

        # --- 3. Find the class within the module ---
        # We assume the class name is the same as the original filename without .py
        StrategyClass = None
        
        for name, obj in inspect.getmembers(strategy_module):
            # We are looking for a member that is:
            # 1. A class (`inspect.isclass(obj)`)
            # 2. A subclass of BaseStrategy (`issubclass(obj, BaseStrategy)`)
            # 3. Not BaseStrategy itself (`obj is not BaseStrategy`)
            if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                StrategyClass = obj
                break # Found it, so we can exit the loop

        if not StrategyClass:
            # If the loop finishes and we haven't found a class, raise an error.
            raise ImportError(f"Could not find a valid subclass of 'BaseStrategy' in '{file_name}'")
            
        # --- 4. Instantiate the class and access attributes ---
        # Now you have a live Python object!
        strategy_instance = StrategyClass()
        
        return strategy_instance
        
    finally:
        # pass
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        pyc_path = os.path.join(TEMP_STRATEGY_DIR, "__pycache__", f"{unique_module_name}.cpython-311.pyc") # Example, version might differ
        
        if os.path.exists(pyc_path):
            try:
                os.remove(pyc_path)
            except OSError as e:
                print(f"Error removing .pyc file: {e}")