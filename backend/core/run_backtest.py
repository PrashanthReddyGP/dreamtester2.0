import pandas as pd
import numpy as np
import traceback

def run_backtest_loop(strategy_name, strategy_instance, initialCapital, main_df, sub_dfs):
    print("  - Running event loop...")
    try:
        capital = initialCapital
        risk_percent = strategy_instance.risk_percent
        cash = (capital * risk_percent) / 100
        
        # Final Preparation and Assignment to Strategy Instance
        # Convert timestamp back to integer for compatibility with existing strategies
        main_df['timestamp'] = (main_df['timestamp'].astype(np.int64) // 10**6)
        
        # Assign main dataframe and its numpy arrays
        strategy_instance.df = main_df.copy()
        strategy_instance.open = main_df['open'].values
        strategy_instance.high = main_df['high'].values
        strategy_instance.low = main_df['low'].values
        strategy_instance.close = main_df['close'].values
        strategy_instance.volume = main_df['volume'].values
        strategy_instance.timestamp = main_df['timestamp'].values
        
        # Dynamically assign all merged sub-dataframe columns as numpy arrays
        for sub_timeframe in sub_dfs.keys():
            prefix = f"{sub_timeframe}_"
            for col_name in ['open', 'high', 'low', 'close', 'volume']:
                full_col_name = f"{prefix}{col_name}"
                if full_col_name in main_df.columns:
                    # Create a valid attribute name (e.g., '1h_open')
                    attr_name = full_col_name
                    setattr(strategy_instance, attr_name, main_df[full_col_name].values)
                    print(f"    - Assigned numpy array to strategy_instance.{attr_name}")
        
        n = len(strategy_instance.close)
        
        strategy_instance.trade_closed = np.zeros(n)
        strategy_instance.result = np.zeros(n)
        strategy_instance.exit_time = np.zeros(n)
        strategy_instance.open_trades = np.zeros(n)
        strategy_instance.signals = np.zeros(n)
        
        strategy_instance.capital, strategy_instance.risk_percent, strategy_instance.cash = capital, risk_percent, cash
        
        print(f"Running strategy: {strategy_instance}")
        
        # It does all the work and returns the complete, corrected 6-item tuple.
        return strategy_instance.run()
        
    except Exception as e:
        print(f"!!! ERROR during strategy.run() for '{strategy_name}': {e} !!!")
        print(traceback.format_exc()) # Print the full traceback for debugging
        
        # Return None to clearly signal a catastrophic failure in this run
        return None