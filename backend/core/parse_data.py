import os
import sys
import inspect
import importlib.util
import threading

from core.basestrategy import BaseStrategy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

TEMP_STRATEGY_DIR = os.path.join(parent_dir, "temp_strategies")
os.makedirs(TEMP_STRATEGY_DIR, exist_ok=True)

sys_path_lock = threading.Lock()

def parse_file(job_id, file_name, strategy_code):
    """
    Dynamically imports a strategy from code in a thread-safe manner.
    """
    temp_filepath = None
    strategy_name = os.path.splitext(file_name)[0]
    unique_module_name = f"{strategy_name}_{job_id.replace('-', '_')}"
    
    # Use a try...finally block to GUARANTEE cleanup happens
    try:
        # --- 1. Create the temporary .py file ---
        temp_filepath = os.path.join(TEMP_STRATEGY_DIR, f"{unique_module_name}.py")
        with open(temp_filepath, "w", encoding='utf-8') as f:
            f.write(strategy_code)

        # --- 2. Prepare for import ---
        spec = importlib.util.spec_from_file_location(unique_module_name, temp_filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {file_name}")
            
        strategy_module = importlib.util.module_from_spec(spec)

        # --- 3. THE THREAD-SAFE PATH MODIFICATION ---
        with sys_path_lock:
            # This block ensures only one thread modifies sys.path at a time
            # We add the 'backend' directory so `from core.basestrategy` works
            sys.path.insert(0, parent_dir)
            try:
                # Execute the import while the path is temporarily modified
                spec.loader.exec_module(strategy_module)
            finally:
                # This inner finally guarantees the path is restored even if exec_module fails
                if sys.path[0] == parent_dir:
                    sys.path.pop(0)

        # --- 4. Find the class using introspection (this part is correct) ---
        StrategyClass = None
        for name, obj in inspect.getmembers(strategy_module):
            if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                StrategyClass = obj
                break

        if not StrategyClass:
            raise ImportError(f"Could not find a valid subclass of 'BaseStrategy' in '{file_name}'")
            
        # --- 5. Instantiate and return ---
        strategy_instance = StrategyClass()
        return strategy_instance, strategy_name
        
    finally:
        # --- 6. Cleanup the temporary files ---
        # This block runs regardless of success or failure in the `try` block.
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as e:
                print(f"Error removing temp .py file: {e}")
        
        # Cleanup the .pyc file if it exists in the __pycache__ directory
        pyc_path = os.path.join(TEMP_STRATEGY_DIR, "__pycache__", f"{unique_module_name}.cpython-311.pyc") # Adjust python version if needed
        if os.path.exists(pyc_path):
            try:
                os.remove(pyc_path)
            except OSError as e:
                print(f"Error removing .pyc file: {e}")