import pandas as pd
import numpy as np
import traceback

def run_backtest_loop(strategy_name, strategy_instance, initialCapital, main_df, sub_dfs):
    print("  - Running event loop...")
    try:
        capital = initialCapital
        risk_percent = strategy_instance.risk_percent
        cash = (capital * risk_percent) / 100
        
        # # --- 2. Pre-processing: Convert all timestamps to datetime objects ---
        # main_df['timestamp'] = pd.to_datetime(main_df['timestamp'], unit='ms')
        
        # for tf, df in sub_dfs.items():
        #     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # print(f"  -> Main DF ('{main_timeframe}') has {len(main_df)} rows before merging.")
        
        # # --- 3. Iteratively Merge Sub-DataFrames into the Main DataFrame ---
        # main_df_freq = main_df['timestamp'].diff().median()
        
        # for sub_timeframe, sub_df in sub_dfs.items():
        #     if sub_df.empty:
        #         print(f"  - Skipping empty sub-dataframe for timeframe '{sub_timeframe}'.")
        #         continue
            
        #     print(f"  - Merging sub-timeframe '{sub_timeframe}' into '{main_timeframe}'...")
        #     sub_df_freq = sub_df['timestamp'].diff().median()
            
        #     # SCENARIO 1: sub_df is lower frequency (e.g., 1D sub -> 15m main)
        #     if sub_df_freq > main_df_freq:
        #         print(f"    - Scenario: Lower freq sub_df ({sub_df_freq}) -> Higher freq main_df ({main_df_freq}).")
        #         print("    - Projecting data onto finer bars (with shift to prevent lookahead).")
                
        #         # Shift data by 1 to prevent lookahead bias (data from 10:00 1H bar is available at 11:00)
        #         data_cols_to_shift = [col for col in sub_df.columns if col != 'timestamp']
        #         sub_df[data_cols_to_shift] = sub_df[data_cols_to_shift].shift(1)
                
        #         # Rename columns dynamically (e.g., 'open' -> '1h_open')
        #         prefix = f"{sub_timeframe}_"
        #         sub_df_renamed = sub_df.add_prefix(prefix)
        #         sub_df_renamed = sub_df_renamed.rename(columns={f'{prefix}timestamp': 'timestamp'})
                
        #         # Drop first row which is now NaN after the shift
        #         sub_df_renamed = sub_df_renamed.dropna(subset=[f'{prefix}{col}' for col in data_cols_to_shift])
                
        #         # Use merge_asof for efficient point-in-time merging
        #         main_df = main_df.sort_values('timestamp')
        #         sub_df_renamed = sub_df_renamed.sort_values('timestamp')
                
        #         main_df = pd.merge_asof(main_df, sub_df_renamed, on='timestamp', direction='backward')
                
        #         # Back-fill any NaNs at the very beginning
        #         sub_columns = [col for col in main_df.columns if col.startswith(prefix)]
        #         main_df[sub_columns] = main_df[sub_columns].bfill()
                
        #     # SCENARIO 2: sub_df is higher frequency (e.g., 15m sub -> 1D main)
        #     else:
        #         print(f"    - Scenario: Higher freq sub_df ({sub_df_freq}) -> Lower freq main_df ({main_df_freq}).")
        #         print("    - Aggregating (resampling) finer data up to the main timeframe.")
                
        #         # Convert our timeframe string (e.g., '1h', '15m') to a pandas-compatible one ('1H', '15T')
        #         resample_freq_str = main_timeframe.upper().replace('M', 'T')
                
        #         sub_df_agg = sub_df.set_index('timestamp').resample(resample_freq_str).agg({
        #             'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        #         }).dropna()
                
        #         sub_df_agg = sub_df_agg.reset_index()
                
        #         # Rename columns dynamically before merging (e.g., 'open' -> '15m_open')
        #         prefix = f"{sub_timeframe}_"
        #         sub_df_agg_renamed = sub_df_agg.add_prefix(prefix)
        #         sub_df_agg_renamed = sub_df_agg_renamed.rename(columns={f'{prefix}timestamp': 'timestamp'})
                
        #         # Merge on the shared, resampled timestamp
        #         main_df = pd.merge(main_df, sub_df_agg_renamed, on='timestamp', how='left')
        
        # print(f"  -> After all merges, main_df has {len(main_df)} rows and columns: {list(main_df.columns)}")
        
        # --- 4. Final Preparation and Assignment to Strategy Instance ---
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