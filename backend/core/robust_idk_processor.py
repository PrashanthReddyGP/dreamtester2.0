# In the same file as your original calculate_indicators function
# (or refactor it to a more appropriate place)
from core.indicator_registry import INDICATOR_REGISTRY

def calculate_indicators(strategy_instance, df):
    """
    A refactored, data-driven function to calculate indicators.
    """
    indicators_to_run = strategy_instance.indicators
    print(f"--- Calculating {len(indicators_to_run)} indicators (Refactored) ---")
    
    for indicator_tuple in indicators_to_run:
        name, timeframe, params_list = indicator_tuple
        
        # Look up the indicator in our registry
        if name in INDICATOR_REGISTRY:
            indicator_info = INDICATOR_REGISTRY[name]
            func = indicator_info['function']
            expected_params_count = len(indicator_info['params'])
            
            # Validate parameter count
            if len(params_list) == expected_params_count:
                # Dynamically call the function with the parameters
                df = func(df, timeframe, *params_list)
            else:
                print(f"ERROR for {name}: Expected {expected_params_count} params, but got {len(params_list)}.")
        else:
            print(f"WARNING: Indicator '{name}' not found in registry. Skipping.")
            
    print("Indicators processed successfully")
    return df