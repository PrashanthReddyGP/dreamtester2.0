# backend/pipeline.py
import uuid
import asyncio
import traceback
import pandas as pd
import numpy as np
from datetime import time, timedelta 

from concurrent.futures import ThreadPoolExecutor, as_completed

from database import SessionLocal, save_backtest_results, update_job_status, fail_job
from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client



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
    Orchestrates the batch in parallel. It now primarily collects raw data
    for the final portfolio calculation, as workers send their own UI updates.
    """
    print(f"--- BATCH MANAGER ({batch_id}): Starting PARALLEL batch processing ---")
    
    # --- 2. SEND INITIAL STATUS UPDATE ---
    send_websocket_update(manager, batch_id, {
        "type": "log", "payload": {"level": "INFO", "message": f"Starting batch of {len(files_data)} strategies..."}
    })
    
    update_job_status(batch_id, "running", f"Starting batch of {len(files_data)} strategies...")

    client = get_client(exchange_name='binance') # Instantiate the client once

    initialCapital = 1000 # Make this dynamic later

    raw_results_for_portfolio = []

    # --- 2. Use a ThreadPoolExecutor to run jobs in parallel ---
    # `with ThreadPoolExecutor() as executor:` manages the pool lifecycle automatically.
    # You can specify `max_workers=N` to control the number of threads. Default is usually good.
    with ThreadPoolExecutor() as executor:
        
        # Create a dictionary to map a "future" object back to its file_name
        future_to_file = {
            executor.submit(process_backtest_job, batch_id, manager, str(uuid.uuid4()), file['name'], file['content'], initialCapital, client): file
            for file in files_data
        }
        
        # --- Collect results as they are completed ---
        # `as_completed(futures)` yields results as soon as they are ready,
        # not necessarily in the order they were submitted.
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            
            try:
                result_tuple = future.result()
                
                # Check if the job failed and returned None
                if result_tuple is None:
                    print(f"--> Manager skipping failed job: {file_info['name']}")
                    send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": f"{file_info['name']}: Skipping failed job!"}})
                    continue
                
                # The worker has already sent the update, so we just collect the result
                raw_results_for_portfolio.append(result_tuple)
                print(f"  -> Manager collected result for: {file_info['name']}")
                                
            except Exception as exc:
                error_msg = f"Fatal error in manager for {file_info['name']}: {exc}"
                print(f"!!! {error_msg} !!!")
                traceback.print_exc()
                fail_job(batch_id, error_msg)
                send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": error_msg}})

    # --- 5. Now, all backtests are done. Proceed with combining the results. ---
    print("--- BATCH MANAGER: All backtests complete. Preparing final results... ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"All backtests complete. Preparing final results..."}})

    if not raw_results_for_portfolio:
        fail_job(batch_id, "No strategies completed successfully.")
        send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": "No strategies completed successfully."}})
        return
    
    # Let's clean up the `strategies_results` list before we use it.
    # We'll replace the full instance object with just its name.
    all_processed_results_for_db  = []
    
    for res_tuple  in raw_results_for_portfolio:
        
        strategy_data, _, metrics, monthly_returns = res_tuple
        
        # Use helper to prepare the serializable dict for this strategy
        all_processed_results_for_db.append(
            prepare_strategy_payload(strategy_data, metrics, monthly_returns)
        )

        # # Assuming your run_single_backtest returns a dict with 'strategy_name', 'equity', etc.
        # equity_values = res['equity'].tolist()
        
        # timestamps_ms = (res['ohlcv']['timestamp'].astype(np.int64) // 10**9).tolist()
        
        # # Ensure timestamps and equity have the same length
        # min_len = min(len(timestamps_ms), len(equity_values))
        
        # signals_df = res.get('signals')
        
        # if signals_df is not None:
        #     # Create a copy to avoid SettingWithCopyWarning
        #     temp_signals = signals_df.copy()

        #     temp_signals['timestamp'] = temp_signals['timestamp'].astype(np.int64) // 10**9
        #     temp_signals['Exit_Time'] = temp_signals['Exit_Time'].astype(np.int64) // 10**9

        #     signals_json = temp_signals.to_dict(orient='records')
        # else:
        #     signals_json = []

        # processed_results.append({
        #     "strategy_name": res['strategy_name'],
        #     "equity_curve": list(zip(timestamps_ms[:min_len], equity_values[:min_len])),
        #     "metrics": metric,
        #     "trades": signals_json,
        #     "monthly_returns": returns
        # })

    if len(raw_results_for_portfolio) > 1:
        
        print("--- BATCH MANAGER: Calculating portfolio... ---")
        send_websocket_update(manager, batch_id, {"type": "log", "payload": {"level": "INFO", "message": "Calculating portfolio performance..."}})

        try:
            combined_df, combined_dfSignals = pd.DataFrame(), pd.DataFrame()
            
            for res_tuple in raw_results_for_portfolio:
                result, _, _, _ = res_tuple
                df, df_signals = result.get('ohlcv'), result.get('signals')
                
                if df is not None: 
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    
                if df_signals is not None: 
                    combined_dfSignals = pd.concat([combined_dfSignals, df_signals], ignore_index=True)
            
            combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
            
            combined_dfSignals = combined_dfSignals.sort_values(by='timestamp').reset_index(drop=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            
            combined_dfSignals['timestamp'] = pd.to_datetime(combined_dfSignals['timestamp'])
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            
            complete_df = pd.merge(combined_df, combined_dfSignals, on='timestamp', how='outer').sort_values(by='timestamp')
            
            sample_instance = raw_results_for_portfolio[0][1] # Get a sample instance
            
            portfolio_equity = sample_instance.portfolio(
                (complete_df['timestamp'].astype(np.int64) // 10**9).values,
                complete_df['Result'].values, complete_df['RR'].values,
                complete_df['Reduction'].values, complete_df['Commissioned Returns'].values
            )
            
            len_diff = len(complete_df) - len(portfolio_equity)
            aligned_equity = pd.Series([initialCapital] * len_diff + portfolio_equity, index=complete_df.index)
            
            portfolio_strategy_data = {'strategy_name': 'Portfolio', 'equity': aligned_equity, 'ohlcv': complete_df, 'signals': combined_dfSignals[combined_dfSignals['Signal'] != 0]}
            portfolio_metrics, portfolio_returns = calculate_metrics(portfolio_strategy_data, initialCapital)

            # Prepare payload for both WebSocket and final DB save
            portfolio_payload = prepare_strategy_payload(portfolio_strategy_data, portfolio_metrics, portfolio_returns)
            
            # Send the portfolio result to the UI
            send_websocket_update(manager, batch_id, {
                "type": "strategy_result", 
                "payload": convert_to_json_serializable(portfolio_payload)
            })
            
            # Add portfolio to the beginning of the list for the final DB save
            all_processed_results_for_db.insert(0, portfolio_payload)

        except Exception as e:
            error_msg = f"ERROR calculating portfolio: {traceback.format_exc()}"
            print(f"!!! CRITICAL PORTFOLIO ERROR !!!\n{error_msg}")
            fail_job(batch_id, error_msg)
            send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": "Failed to calculate portfolio."}})

        
    # --- PREPARE THE FINAL JSON PAYLOAD ---
    final_db_payload = {
        "name": "Batch Result",
        "strategies_results": all_processed_results_for_db,
        "initial_capital": initialCapital
    }

    save_backtest_results(batch_id=batch_id, results=convert_to_json_serializable(final_db_payload))
    update_job_status(batch_id, "completed", "Batch complete.")
    send_websocket_update(manager, batch_id, {"type": "batch_complete", "payload": {"message": "All tasks finished."}})
    
    print(f"--- BATCH MANAGER: Job {batch_id} finished and saved. ---")
        

# process_backtest_job should now RETURN the results instead of just printing
def process_backtest_job(batch_id: str, manager: any, job_id: str, file_name: str, file_content: str, initialCapital: int, client):
    """
    Processes a SINGLE backtest job, sends live updates, and returns its results.
    """
    try:
        log_msg = f"Starting backtest for: {file_name}"
        send_websocket_update(manager, batch_id, {"type": "log", "payload": {"level": "INFO", "message": log_msg}})
        update_job_status(batch_id, "running", log_msg)
        
        # This pure function does all the heavy lifting
        strategy_data, strategy_instance, strategy_metrics, monthly_returns = run_single_backtest(batch_id, manager, job_id, file_name, file_content, initialCapital, client)
        
        # Prepare the payload for the UI
        single_result_payload = prepare_strategy_payload(strategy_data, strategy_metrics, monthly_returns)
        
        # Send the result for this single strategy IMMEDIATELY to the UI
        send_websocket_update(manager, batch_id, {
            "type": "strategy_result",
            "payload": convert_to_json_serializable(single_result_payload)
        })
        
        log_msg_done = f"Finished processing: {file_name}"
        send_websocket_update(manager, batch_id, {"type": "log", "payload": {"level": "SUCCESS", "message": log_msg_done}})
        print(f"WORKER: Job {job_id} for {file_name} completed and sent update.")
        
        # Return the raw data for the manager to use later for the portfolio
        return strategy_data, strategy_instance, strategy_metrics, monthly_returns
        
    except Exception as e:
        error_msg = f"ERROR in {file_name}: {traceback.format_exc()}"
        print(f"!!! CRITICAL ERROR IN WORKER job_id={job_id} !!!\n{error_msg}")
        fail_job(batch_id, error_msg)
        send_websocket_update(manager, batch_id, {"type": "error", "payload": {"message": f"Failed in {file_name}. See server logs."}})
        return None

# This is a placeholder for your actual backtest engine
def run_single_backtest(batch_id: str, manager: any, job_id: str, file_name:str, strategy_code: str, initialCapital: int, client):
    
    print(f"--- Running core backtest for {file_name} ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"Running core backtest for {file_name}"}})

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

    print(f"--- Core backtest for {asset} finished ---")
    send_websocket_update(manager, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Core backtest for {file_name} finished"}})

    # In a real app, this would return the full results object
    return strategy_data, strategy_instance, strategy_metrics, monthly_returns

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
    elif isinstance(obj, (timedelta, time)):
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
    timestamps_ms = (strategy_data['ohlcv']['timestamp'].astype(np.int64) // 10**9).tolist()
    
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

    if df.empty:
        return df

    # Create a copy to avoid modifying the original DataFrame in-place
    formatted_df = df.copy()

    formatted_df['timestamp'] = formatted_df['timestamp'].astype(np.int64) // 10**9
    formatted_df['Exit_Time'] = formatted_df['Exit_Time'].astype(np.int64) // 10**9

    # for col_name in formatted_df.columns:
    #     # Heuristic: check if column name suggests it's a timestamp.
    #     is_time_column_name = 'time' in col_name.lower() or 'date' in col_name.lower()
        
    #     if not is_time_column_name:
    #         continue

    #     # Check if the data is numeric, which indicates a Unix timestamp
    #     if pd.api.types.is_numeric_dtype(formatted_df[col_name]):
    #         print(f"  --> Formatting numeric time column: '{col_name}'")
            
    #         # Use the median for a robust check against outliers/zeros
    #         sample_values = formatted_df[col_name][formatted_df[col_name] > 0]
    #         if sample_values.empty:
    #             formatted_df[col_name] = None # Or '' if you prefer
    #             continue
            
    #         median_value = sample_values.median()
            
    #         # If the number is smaller than a 12-digit number, it's likely seconds. Otherwise, milliseconds.
    #         unit_to_use = 's' if median_value < 10**12 else 'ms'
            
    #         # Convert to datetime objects
    #         dt_series = pd.to_datetime(
    #             formatted_df[col_name], 
    #             unit=unit_to_use,
    #             errors='coerce'
    #         )
            
    #         # Format to a string and replace NaT with None
    #         str_series = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S')
    #         formatted_df[col_name] = str_series.replace({pd.NaT: None})
        
    #     # Also handle columns that might already be datetime objects
    #     elif pd.api.types.is_datetime64_any_dtype(formatted_df[col_name]):
    #          print(f"  --> Formatting datetime object column: '{col_name}'")
    #          str_series = formatted_df[col_name].dt.strftime('%Y-%m-%d %H:%M:%S')
    #          formatted_df[col_name] = str_series.replace({pd.NaT: None})

    return formatted_df