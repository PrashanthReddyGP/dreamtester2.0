# backend/pipeline.py
import uuid
import asyncio
import traceback
import pandas as pd
import numpy as np
from datetime import time, timedelta 

from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from database import LatestResult, SessionLocal, save_backtest_results, update_job_status, fail_job
from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client
from core.data_manager import db_write_lock 



def send_websocket_update(manager: any, batch_id: str, message: dict):
    """
    Safely runs the async send_json_to_batch function from a synchronous context.
    """
    # Get the current event loop or create a new one if none exists
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Schedule the async task to be run
    loop.run_until_complete(manager.send_json_to_batch(batch_id, message))




# --- NEW: The Batch Manager Function ---
def run_batch_manager(batch_id: str, files_data: list[dict], manager: any):
    """
    This is the main background task. It orchestrates the entire batch IN PARALLEL
    and sends live updates via WebSocket.
    """
    print(f"--- BATCH MANAGER ({batch_id}): Starting PARALLEL batch processing ---")
    
    # --- 2. SEND INITIAL STATUS UPDATE ---
    send_websocket_update(manager, batch_id, {
        "type": "log", "payload": {"level": "INFO", "message": f"Starting batch of {len(files_data)} strategies..."}
    })
    update_job_status(batch_id, "running", f"Starting batch of {len(files_data)} strategies...")

    client = get_client(exchange_name='binance') # Instantiate the client once

    initialCapital = 1000 # Make this dynamic later
    
    strategies_results = []
    strategies_metrics = []
    strategies_returns = []

    raw_results = []

    # --- 2. Use a ThreadPoolExecutor to run jobs in parallel ---
    # `with ThreadPoolExecutor() as executor:` manages the pool lifecycle automatically.
    # You can specify `max_workers=N` to control the number of threads. Default is usually good.
    with ThreadPoolExecutor() as executor:
        
        # Create a dictionary to map a "future" object back to its file_name
        future_to_file = {
            executor.submit(process_backtest_job, batch_id, manager, str(uuid.uuid4()), file['name'], file['content'], initialCapital, client): file
            for file in files_data
        }
        
        # # --- 3. Submit all jobs to the pool ---
        # for file_info in files_data:
        #     job_id = str(uuid.uuid4())
            
        #     # `executor.submit()` immediately returns a "future" object
        #     # and starts running the task in a background thread.
        #     future = executor.submit(
        #         process_backtest_job, 
        #         job_id=job_id,
        #         file_name=file_info['name'],
        #         file_content=file_info['content'],
        #         initialCapital=initialCapital,
        #         client=client
        #     )
        #     future_to_file[future] = file_info['name']

        # --- 4. Collect results as they are completed ---
        # `as_completed(futures)` yields results as soon as they are ready,
        # not necessarily in the order they were submitted.
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                # `future.result()` gets the return value from process_backtest_job
                # --- THE FIX IS HERE ---
                
                result_tuple = future.result()
                
                # Check if the job failed and returned None
                if result_tuple is None:
                    print(f"--> Manager skipping failed job: {file_info['name']}")
                    send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": f"{file_info['name']}: Skipping failed job!"}})
                    continue # Move to the next completed future
                
                # The worker has already sent the update, so we just collect the result
                raw_results.append(result_tuple)
                print(f"  -> Manager collected result for: {file_info['name']}")
                                
                # If we get here, the job was successful, so we can safely unpack
                single_result, strategy_instance, single_metric, single_returns = result_tuple
                
                send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"Finished processing: {file_info['name']}"}})
                
                if single_result:
                    print(f"  -> Finished processing: {file_info['name']}")
                    single_result['strategy_instance'] = strategy_instance 
                    strategies_results.append(single_result)
                    strategies_metrics.append(single_metric)
                    strategies_returns.append(single_returns)

            except Exception as exc:
                error_msg = f"Fatal error in manager for {file_info['name']}: {exc}"
                print(f"!!! {error_msg} !!!")
                traceback.print_exc()
                fail_job(batch_id, error_msg)
                send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": error_msg}})

    # --- 5. Now, all backtests are done. Proceed with combining the results. ---
    print("--- BATCH MANAGER: All backtests complete. Preparing final results... ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"All backtests complete. Preparing final results..."}})

    if not raw_results:
        fail_job(batch_id, "No strategies completed successfully.")
        send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": "No strategies completed successfully."}})
        return
        
    # Initialize the combined dataframes
    combined_df = pd.DataFrame()
    combined_dfSignals = pd.DataFrame()

    for result in strategies_results:
        df = result.get('ohlcv')
        if df is not None:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        df_signals = result.get('signals')
        if df_signals is not None:
            combined_dfSignals = pd.concat([combined_dfSignals, df_signals], ignore_index=True)
            
    combined_df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Let's clean up the `strategies_results` list before we use it.
    # We'll replace the full instance object with just its name.
    processed_results = []
    
    for res, metric, returns in zip(strategies_results, strategies_metrics, strategies_returns):
                
        # Assuming your run_single_backtest returns a dict with 'strategy_name', 'equity', etc.
        equity_values = res['equity'].tolist()
        
        timestamps_ms = (res['ohlcv']['timestamp'].astype(np.int64) // 10**9).tolist()
        
        # Ensure timestamps and equity have the same length
        min_len = min(len(timestamps_ms), len(equity_values))
        
        signals_df = res.get('signals')
        
        if signals_df is not None:
            # Create a copy to avoid SettingWithCopyWarning
            temp_signals = signals_df.copy()

            temp_signals['timestamp'] = temp_signals['timestamp'].astype(np.int64) // 10**9
            temp_signals['Exit_Time'] = temp_signals['Exit_Time'].astype(np.int64) // 10**9

            signals_json = temp_signals.to_dict(orient='records')
        else:
            signals_json = []

        processed_results.append({
            "strategy_name": res['strategy_name'],
            "equity_curve": list(zip(timestamps_ms[:min_len], equity_values[:min_len])),
            "metrics": metric,
            "trades": signals_json,
            "monthly_returns": returns
        })

    if len(strategies_results) > 1:
        combined_dfSignals = combined_dfSignals.sort_values(by='timestamp').reset_index(drop=True)
        combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
        
        complete_df = pd.merge(combined_df, combined_dfSignals, on='timestamp', how='outer')
        complete_df = complete_df.sort_values(by='timestamp')
        
        # Get the strategy instance from one of the results to call the portfolio method
        # (This confirms the refactoring to a static method is a good idea)
        sample_instance = strategies_results[0]['strategy_instance']
        
        timestamps = (complete_df['timestamp'].astype(np.int64) // 10**6).values
        # Fill NaN values that may result from the 'outer' merge before processing
        results = complete_df['Result'].values 
        rrs = complete_df['RR'].values
        reductions = complete_df['Reduction'].values
        commissioned_returns_combined = complete_df['Commissioned Returns'].values
        
        portfolio_equity = sample_instance.portfolio(timestamps, results, rrs, reductions, commissioned_returns_combined)
        
        print("Portfolio:", portfolio_equity[0], portfolio_equity[-1])

        # Calculate how many rows are missing from the start
        len_diff = len(complete_df) - len(portfolio_equity)
        
        # Create a new pandas Series for the equity.
        # We pad the beginning with the initial_capital value for the rows
        # that were part of the indicator lookback period.
        aligned_equity = pd.Series(
            [initialCapital] * len_diff + portfolio_equity, 
            index=complete_df.index
        )
        
        # Create the portfolio result object
        equity_values = aligned_equity
        
        # The timestamps for the portfolio are from the combined `complete_df`
        timestamps_ms = (complete_df['timestamp'].astype(np.int64) // 10**9).tolist()
                
        min_len = min(len(timestamps_ms), len(equity_values))

        signals = combined_dfSignals[combined_dfSignals['Signal'] != 0]
        
        strategy_data = {
            'strategy_name': 'Portfolio',
            'equity': aligned_equity,
            'ohlcv':complete_df,
            'signals': signals
        }
        
        strategy_metrics, returns = calculate_metrics(strategy_data, initialCapital)

        signals_df = signals.copy()
        
        signals_df['timestamp'] = signals_df['timestamp'].astype(np.int64) // 10**9
        signals_df['Exit_Time'] = signals_df['Exit_Time'].astype(np.int64) // 10**9

        signals_json = signals_df.to_dict(orient='records')

        portfolio_result = {
            'strategy_name': 'Portfolio', # Give it a special name
            'equity_curve': list(zip(timestamps_ms[:min_len], equity_values[:min_len])),
            'metrics': strategy_metrics,
            'trades': signals_json,
            "monthly_returns": returns
        }
        
        # Add the portfolio to the list of results
        processed_results.insert(0, portfolio_result) # Insert at the beginning
        
    # --- PREPARE THE FINAL JSON PAYLOAD ---
    final_payload = {
        "name": "Batch Result",
        "strategies_results": processed_results,
        "initial_capital": 1000
    }

    # --- THE FIX: Sanitize the dictionary before saving ---
    serializable_payload = convert_to_json_serializable(final_payload)
    
    try:
        save_backtest_results(batch_id= batch_id, results= serializable_payload)
        save_latest_result(serializable_payload)
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"ERROR in Portfolio: {e}"
        print(f"!!! CRITICAL ERROR IN BACKGROUND TASK for Portfolio !!!\n{tb_str}")
        
        fail_job(batch_id, error_msg)
        send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": error_msg}})
        return None
    
    update_job_status(batch_id, "completed", "Portfolio calculated. Batch complete.")
    send_websocket_update(manager, batch_id, {"type": "batch_complete", "payload": {"message": "All tasks finished."}})
    
    print(f"--- BATCH MANAGER: Results for batch {batch_id} saved to database. ---")

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
def process_backtest_job(batch_id: str, manager: any, job_id: str, file_name: str, file_content: str, initialCapital: int, client):
    """
    Processes a SINGLE backtest job, sends live updates, and returns its results.
    """
    try:
        log_msg = f"Processing: {file_name}"
        send_websocket_update(manager, batch_id, {"type": "log", "payload": {"level": "INFO", "message": log_msg}})
        update_job_status(batch_id, "running", log_msg)
        
        # This function should contain all the logic for ONE backtest:
        # parse_file, fetch_candlestick_data, calculate_indicators, run_backtest_loop
        strategy_data, strategy_instance, strategy_metrics, monthly_returns = run_single_backtest(batch_id, manager, job_id, file_name, file_content, initialCapital, client)
        
        single_result_payload = prepare_strategy_payload(strategy_data, strategy_metrics, monthly_returns)
        send_websocket_update(manager, batch_id, {
            "type": "strategy_completed",
            "payload": convert_to_json_serializable(single_result_payload)
        })

        print(f"WORKER: Job {job_id} completed successfully.")
        
        # send_websocket_update(manager, batch_id, {
        #     "type": "strategy_completed",
        #     "payload": convert_to_json_serializable(single_result_payload)
        # })
        
        return strategy_data, strategy_instance, strategy_metrics, monthly_returns
        
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"ERROR in {file_name}: {e}"
        print(f"!!! CRITICAL ERROR IN BACKGROUND TASK job_id={job_id} !!!\n{tb_str}")
        
        fail_job(batch_id, error_msg)
        send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": error_msg}})
        return None


# This is a placeholder for your actual backtest engine
def run_single_backtest(batch_id: str, manager: any, job_id: str, file_name:str, strategy_code: str, initialCapital: int, client):
    """
    This function contains the core logic for one backtest.
    1. Parses the code for config.
    2. Fetches data.
    3. Calculates indicators.
    4. Runs the simulation loop.
    5. Calculates final metrics.
    """
    print(f"--- Starting backtest ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Starting Backtest"}})

    # 1. Parse the code of config
    strategy_instance, strategy_name = parse_file(job_id, file_name, strategy_code)
    
    # 2. Fetch data
    asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Ohlcv Fetch Successful"}})

    # 3. Calculate indicators
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Indicators Calculation Successful"}})

    # 4. Run simulation
    strategy_data = run_backtest_loop(strategy_name, strategy_instance, ohlcv_idk, initialCapital)
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Backtestloop Successful"}})

    # 5. Calculate metrics (simulated)
    strategy_metrics, monthly_returns = calculate_metrics(strategy_data, initialCapital)
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Metrics Calculation Successful"}})

    print(f"--- Backtest for {asset} finished ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Backtest for {file_name} finished"}})

    # In a real app, this would return the full results object
    return strategy_data, strategy_instance, strategy_metrics, monthly_returns

# def save_latest_result(results: dict):
#     with db_write_lock:

#         print("--- Acquiring DB lock to save final results ---")        
        
#         db = SessionLocal()
#         try:
#             # The 'UPSERT' statement for SQLite
#             stmt = sqlite_insert(LatestResult).values(id=1, results_data=results)
            
#             # If the row with id=1 already exists, update the 'results_data' column
#             stmt = stmt.on_conflict_do_update(
#                 index_elements=['id'],
#                 set_={'results_data': stmt.excluded.results_data}
#             )
            
#             db.execute(stmt)
#             db.commit()
#             print("--- Latest backtest result has been updated. ---")
        
#         finally:
#             db.close()
            
#     print("--- Released DB lock ---")
        
# --- Helper function to make a dictionary JSON-serializable ---
def convert_to_json_serializable(obj):
    """
    Recursively traverses a dictionary or list to convert special data types
    to standard Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(elem) for elem in obj]
    
    # --- Date and Time type conversions ---
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (timedelta, time)): # <-- THE FIX IS HERE
        # Convert timedelta or time objects to a string representation
        return str(obj)
    elif pd.isna(obj):
        return None
        
    # --- Numpy type conversions ---
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
        
    else:
        return obj
    

def prepare_strategy_payload(strategy_data, metrics, monthly_returns, is_portfolio=False):
    """Prepares the JSON-friendly payload for a single strategy or portfolio."""
    
    equity_values = np.array(strategy_data['equity']).tolist()
    # Timestamps should be from OHLCV and converted to milliseconds
    timestamps_ms = (strategy_data['ohlcv']['timestamp'].astype(np.int64) // 10**6).tolist()
    min_len = min(len(timestamps_ms), len(equity_values))
    equity_curve = list(zip(timestamps_ms[:min_len], equity_values[:min_len]))
    
    signals_df = strategy_data.get('signals')
    trades_json = []
    if signals_df is not None and not signals_df.empty:
        formatted_trades_df = format_datetime_columns_for_json(signals_df)
        trades_json = formatted_trades_df.to_dict(orient='records')
        
    return {
        "strategy_name": strategy_data['strategy_name'],
        "equity_curve": equity_curve,
        "metrics": metrics,
        "trades": trades_json,
        "monthly_returns": monthly_returns
    }
    
def format_datetime_columns_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically finds and formats timestamp-like columns in a DataFrame
    for safe JSON conversion. It intelligently detects whether the timestamp
    is in seconds or milliseconds and handles NaT/NaN values.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A new DataFrame with time-related columns formatted as strings.
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying the original DataFrame in-place
    formatted_df = df.copy()

    for col_name in formatted_df.columns:
        # Heuristic: check if column name suggests it's a timestamp.
        is_time_column_name = 'time' in col_name.lower() or 'date' in col_name.lower()
        
        if not is_time_column_name:
            continue

        # Check if the data is numeric, which indicates a Unix timestamp
        if pd.api.types.is_numeric_dtype(formatted_df[col_name]):
            print(f"  --> Formatting numeric time column: '{col_name}'")
            
            # Use the median for a robust check against outliers/zeros
            sample_values = formatted_df[col_name][formatted_df[col_name] > 0]
            if sample_values.empty:
                formatted_df[col_name] = None # Or '' if you prefer
                continue
            
            median_value = sample_values.median()
            
            # If the number is smaller than a 12-digit number, it's likely seconds. Otherwise, milliseconds.
            unit_to_use = 's' if median_value < 10**12 else 'ms'
            
            # Convert to datetime objects
            dt_series = pd.to_datetime(
                formatted_df[col_name], 
                unit=unit_to_use,
                errors='coerce'
            )
            
            # Format to a string and replace NaT with None
            str_series = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S')
            formatted_df[col_name] = str_series.replace({pd.NaT: None})
        
        # Also handle columns that might already be datetime objects
        elif pd.api.types.is_datetime64_any_dtype(formatted_df[col_name]):
             print(f"  --> Formatting datetime object column: '{col_name}'")
             str_series = formatted_df[col_name].dt.strftime('%Y-%m-%d %H:%M:%S')
             formatted_df[col_name] = str_series.replace({pd.NaT: None})

    return formatted_df


def save_latest_result(results: dict):
    with db_write_lock:

        print("--- Acquiring DB lock to save final results ---")        
        
        db = SessionLocal()
        try:
            # The 'UPSERT' statement for SQLite
            stmt = sqlite_insert(LatestResult).values(id=1, results_data=results)
            
            # If the row with id=1 already exists, update the 'results_data' column
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={'results_data': stmt.excluded.results_data}
            )
            
            db.execute(stmt)
            db.commit()
            print("--- Latest backtest result has been updated. ---")
        
        finally:
            db.close()
            
    print("--- Released DB lock ---")