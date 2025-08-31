import pandas as pd
import numpy as np
import traceback

def run_backtest_loop(strategy_name, strategy_instance, df, initialCapital):
    print("  - Running event loop...")
    try:
        capital = initialCapital
        risk_percent = strategy_instance.risk_percent
        cash = (capital * risk_percent) / 100
        
        df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6
            
        strategy_instance.df = df.copy()
        strategy_instance.open = df['open'].values
        strategy_instance.high = df['high'].values
        strategy_instance.low = df['low'].values
        strategy_instance.close = df['close'].values
        strategy_instance.volume = df['volume'].values

        strategy_instance.timestamp = df['timestamp'].values
        
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