import pandas as pd
import numpy as np

def run_backtest_loop(strategy_instance, df):
    print("  - Running event loop...")
    
    capital = 1000
    risk_percent = 1
    cash = (capital * risk_percent) / 100
    
    df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9  # Unix timestamp in seconds
    
    strategy_instance.df = df.copy()
    strategy_instance.open = df['open'].values
    strategy_instance.high = df['high'].values
    strategy_instance.low = df['low'].values
    strategy_instance.close = df['close'].values
    strategy_instance.timestamp = df['timestamp'].values
    
    n = len(strategy_instance.close)
    
    strategy_instance.trade_closed = np.zeros(n)
    strategy_instance.result = np.zeros(n)
    strategy_instance.exit_time = np.zeros(n)
    strategy_instance.open_trades = np.zeros(n)
    strategy_instance.signals = np.zeros(n)
    
    strategy_instance.capital, strategy_instance.risk_percent, strategy_instance.cash = capital, risk_percent, cash
        
    print(f"Running strategy: {strategy_instance}")
    
    try:
        equity, dfSignals, strategy_df = strategy_instance.run()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Store the equity and signals in a dictionary
        strategy_data = {
            'strategy_name': strategy_instance,
            'equity': equity,
            'ohlcv':df,
            'signals': dfSignals
        }
        
        # strategiesList.append(strategy_data)
        
        # self.finished.emit(strategiesList, initial_capital)
        
        # combined_df = pd.concat([combined_df, df], ignore_index=True)
        # combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        
        # combined_dfSignals = pd.concat([combined_dfSignals, dfSignals], ignore_index=True)
        
        return strategy_data
        
    except Exception as e:
        print(f"Error running strategy {strategy_instance}: {e}")

# if len(strategies) > 1:
    
#     combined_dfSignals = combined_dfSignals.sort_values(by='timestamp')
#     combined_df = combined_df.sort_values(by='timestamp')
    
#     combined_dfSignals.reset_index(drop=True, inplace=True)
#     combined_df.reset_index(drop=True, inplace=True)
    
#     combined_dfSignals = combined_dfSignals.copy()
#     combined_df = combined_df.copy()
    
#     complete_df = pd.merge(combined_df, combined_dfSignals, on='timestamp', how='outer')
    
#     complete_df = complete_df.sort_values(by='timestamp')
    
#     timestamps = (complete_df['timestamp'].astype(np.int64) // 10**9).values
#     results = complete_df['Result'].values
#     rrs = complete_df['RR'].values
#     reductions = complete_df['Reduction'].values
#     commissioned_returns_combined = complete_df['Commissioned Returns'].values
    
#     portfolio_equity = strategy_instance.portfolio(timestamps, results, rrs, reductions, commissioned_returns_combined)
    
#     complete_df = complete_df.drop_duplicates(subset=['timestamp'])
    
#     strategy_data = {
#         'strategy_name': 'Portfolio',
#         'equity': portfolio_equity,
#         'ohlcv':complete_df,
#         'signals': combined_dfSignals
#     }
    
#     strategiesList.append(strategy_data)
    
#     self.backtester_class.add_strategy_widget('Portfolio', complete_df)

# if type == 'backtest':
    
#     self.finished.emit(strategiesList, initial_capital)
    
#     # profiler = LineProfiler()
#     # profiler.add_function(self.backtester_class.plot_results)
#     # profiler.enable_by_count()
    
#     self.backtester_class.plot_results(strategiesList, initial_capital)
    
#     # profiler.disable_by_count()
#     # profiler.print_stats()  # Print the detailed report
    
#     self.backtester_class.metrics(strategiesList, initial_capital, type, threshold, row, above_thresholds)
#     self.backtester_class.on_backtest_finished(strategiesList, initial_capital)
    
#     result_queue.put(above_thresholds)
#     # profiler.disable()  # Stop profiling
#     # profiler.print_stats()  # Print the profiling results to the console
#     # profiler.dump_stats("Backtest_Opt.prof")
    
#     print("BACKTEST FINISHED")
    
# elif type == 'optimization':
    
#     self.finished.emit(strategiesList, initial_capital)
    
#     above_thresholds = self.backtester_class.metrics(strategiesList, initial_capital, type, threshold, row, above_thresholds)
    
#     result_queue.put(above_thresholds)