# backend/pipeline.py
import uuid
import traceback
import pandas as pd
import numpy as np

from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client


# --- NEW: The Batch Manager Function ---
def run_batch_manager(files_data: list[dict]):
    """
    This is the main background task. It orchestrates the entire batch.
    """
    print("--- BATCH MANAGER: Starting batch processing ---")
    
    client = get_client(exchange_name='binance') # Instantiate the client once

    strategies_results = []
    combined_df = pd.DataFrame()
    combined_dfSignals = pd.DataFrame()

    # Loop through the files one by one
    for file_info in files_data:
        job_id = str(uuid.uuid4()) # Generate a job_id here
        
        # --- Run the backtest SYNCHRONOUSLY within this background task ---
        # We are already in the background, so we can wait for each one to finish.
        single_result, strategy_instance = process_backtest_job(
            job_id=job_id,
            file_name=file_info['name'],
            file_content=file_info['content'],
            client=client
        )
        
        if single_result:
            strategies_results.append(single_result)
            
            # Combine the results as they complete
            # Assuming the result dict contains these keys
            df = single_result.get('ohlcv')
            if df is not None:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                combined_df.drop_duplicates(subset=['timestamp'], inplace=True)

            df_signals = single_result.get('signals')
            if df_signals is not None:
                combined_dfSignals = pd.concat([combined_dfSignals, df_signals], ignore_index=True)

    print("--- BATCH MANAGER: All backtests complete ---")

    if len(files_data) > 1:
        combined_dfSignals = combined_dfSignals.sort_values(by='timestamp')
        combined_df = combined_df.sort_values(by='timestamp')
                
        combined_dfSignals.reset_index(drop=True, inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        combined_dfSignals = combined_dfSignals.copy()
        combined_df = combined_df.copy()
        
        complete_df = pd.merge(combined_df, combined_dfSignals, on='timestamp', how='outer')
                
        complete_df = complete_df.sort_values(by='timestamp')
        
        timestamps = (complete_df['timestamp'].astype(np.int64) // 10**9).values
        results = complete_df['Result'].values
        rrs = complete_df['RR'].values
        reductions = complete_df['Reduction'].values
        commissioned_returns_combined = complete_df['Commissioned Returns'].values
        
        portfolio_equity = strategy_instance.portfolio(timestamps, results, rrs, reductions, commissioned_returns_combined)
        
        complete_df = complete_df.drop_duplicates(subset=['timestamp'])
        
        strategy_data = {
            'strategy_name': 'Portfolio',
            'equity': portfolio_equity,
            'ohlcv':complete_df,
            'signals': combined_dfSignals
        }
        
        strategies_results.append(strategy_data)
        
        print(f"Portfolio Equity: {round(portfolio_equity[-1] - portfolio_equity[0])}")
    
#     if type == 'backtest':
        
#         self.backtester_class.plot_results(strategiesList, initial_capital)
        
#         # profiler.disable_by_count()
#         # profiler.print_stats()  # Print the detailed report
        
#         self.backtester_class.metrics(strategiesList, initial_capital, type, threshold, row, above_thresholds)
#         self.backtester_class.on_backtest_finished(strategiesList, initial_capital)
        
#         result_queue.put(above_thresholds)
#         # profiler.disable()  # Stop profiling
#         # profiler.print_stats()  # Print the profiling results to the console
#         # profiler.dump_stats("Backtest_Opt.prof")
        
#         print("BACKTEST FINISHED")
        
#     elif type == 'optimization':
        
#         self.finished.emit(strategiesList, initial_capital)
        
#         above_thresholds = self.backtester_class.metrics(strategiesList, initial_capital, type, threshold, row, above_thresholds)
        
#         result_queue.put(above_thresholds)
        

# process_backtest_job should now RETURN the results instead of just printing
def process_backtest_job(job_id: str, file_name: str, file_content: str, client):
    """
    Processes a SINGLE backtest job and returns its results.
    """
    try:
        print(f"WORKER: Picked up job {job_id} for strategy '{file_name}'.")
        
        # This function should contain all the logic for ONE backtest:
        # parse_file, fetch_candlestick_data, calculate_indicators, run_backtest_loop
        strategy_data, strategy_instance = run_single_backtest(job_id, file_name, file_content, client)
        
        print(f"WORKER: Job {job_id} completed successfully.")
        return strategy_data, strategy_instance
        
    except Exception as e:
        print(f"!!! CRITICAL ERROR IN BACKGROUND TASK job_id={job_id} !!!")
        traceback.print_exc()
        return None # Return None on failure


# This is a placeholder for your actual backtest engine
def run_single_backtest(job_id: str, file_name:str, strategy_code: str, client):
    """
    This function contains the core logic for one backtest.
    1. Parses the code for config.
    2. Fetches data.
    3. Calculates indicators.
    4. Runs the simulation loop.
    5. Calculates final metrics.
    """
    print(f"--- Starting backtest ---")
    
    # 1. Parse the code of config
    strategy_instance = parse_file(job_id, file_name, strategy_code)
    
    # 2. Fetch data
    asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    
    # 3. Calculate indicators
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    
    # 4. Run simulation
    strategy_data = run_backtest_loop(strategy_instance, ohlcv_idk)
    
    # 5. Calculate metrics (simulated)
    calculate_metrics(strategy_data)
        
    print(f"--- Backtest for {asset} finished ---")
    
    # In a real app, this would return the full results object
    return strategy_data, strategy_instance