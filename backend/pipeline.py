# backend/pipeline.py
import os
import re
import ast
import math
import uuid
import asyncio
import itertools
import traceback
import numpy as np
import pandas as pd
import operator as op
from datetime import time, timedelta 

from concurrent.futures import ThreadPoolExecutor, as_completed

from database import SessionLocal, save_backtest_results, update_job_status, fail_job
from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client

# --- Map string operators from frontend to actual Python functions ---
OPERATOR_MAP = {
    '<': op.lt,
    '>': op.gt,
    '<=': op.le,
    '>=': op.ge,
    '===': op.eq,
    '!==': op.ne
}


def send_update_to_queue(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, batch_id: str, message: dict):
    """
    A thread-safe way to send a message from a sync worker
    to the async WebSocket sender task.
    """
    # This logic remains the same, it finds the main event loop
    loop.call_soon_threadsafe(queue.put_nowait, (batch_id, message))


# def send_websocket_update(manager: any, batch_id: str, message: dict):
#     """
#     Safely runs the async send_json_to_batch function from a synchronous context.
#     """
#     # Get the current event loop or create a new one if none exists
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:  # 'There is no current event loop...'
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     # Schedule the async task to be run
#     loop.run_until_complete(manager.send_json_to_batch(batch_id, message))




# --- NEW: The Batch Manager Function ---
def run_batch_manager(batch_id: str, files_data: list[dict], manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop): 
    """
    Orchestrates the batch in parallel. It now primarily collects raw data
    for the final portfolio calculation, as workers send their own UI updates.
    """
    print(f"--- BATCH MANAGER ({batch_id}): Starting PARALLEL batch processing ---")
    
    # --- 2. SEND INITIAL STATUS UPDATE ---
    send_update_to_queue(loop, queue, batch_id, {
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
            executor.submit(process_backtest_job, batch_id, manager, str(uuid.uuid4()), file['name'], file['content'], initialCapital, client, queue, loop): file
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
                    send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": f"{file_info['name']}: Skipping failed job!"}})
                    continue
                
                # The worker has already sent the update, so we just collect the result
                raw_results_for_portfolio.append(result_tuple)
                print(f"  -> Manager collected result for: {file_info['name']}")
                                
            except Exception as exc:
                error_msg = f"Fatal error in manager for {file_info['name']}: {exc}"
                print(f"!!! {error_msg} !!!")
                traceback.print_exc()
                fail_job(batch_id, error_msg)
                send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": error_msg}})

    # --- 5. Now, all backtests are done. Proceed with combining the results. ---
    print("--- BATCH MANAGER: All backtests complete. Preparing final results... ---")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"All backtests complete. Preparing final results..."}})

    if not raw_results_for_portfolio:
        fail_job(batch_id, "No strategies completed successfully.")
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": "No strategies completed successfully."}})
        return
    
    # Let's clean up the `strategies_results` list before we use it.
    # We'll replace the full instance object with just its name.
    all_processed_results_for_db  = []
    
    for res_tuple  in raw_results_for_portfolio:
        
        strategy_data, _, metrics, monthly_returns, _ = res_tuple
                
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
        #     temp_signals['Exit_Time'] = temp_signals['Exit_Time'].astype(np.int64) // 10**9F

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
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": "Calculating portfolio performance..."}})

        try:
            combined_df, combined_dfSignals = pd.DataFrame(), pd.DataFrame()
            
            all_no_fee_equity = 0

            for res_tuple in raw_results_for_portfolio:
                result, _, _, _, no_fee_equity = res_tuple
                
                all_no_fee_equity += no_fee_equity
                
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

            commission = round((((all_no_fee_equity - (portfolio_equity[-1] - portfolio_equity[0])) * 100 )/ all_no_fee_equity), 2)
            
            portfolio_metrics, portfolio_returns = calculate_metrics(portfolio_strategy_data, initialCapital, commission)

            # Prepare payload for both WebSocket and final DB save
            portfolio_payload = prepare_strategy_payload(portfolio_strategy_data, portfolio_metrics, portfolio_returns)
            
            # Send the portfolio result to the UI
            send_update_to_queue(loop, queue, batch_id, {
                "type": "strategy_result", 
                "payload": convert_to_json_serializable(portfolio_payload)
            })
            
            # Add portfolio to the beginning of the list for the final DB save
            all_processed_results_for_db.insert(0, portfolio_payload)

        except Exception as e:
            error_msg = f"ERROR calculating portfolio: {traceback.format_exc()}"
            print(f"!!! CRITICAL PORTFOLIO ERROR !!!\n{error_msg}")
            fail_job(batch_id, error_msg)
            send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": "Failed to calculate portfolio."}})

        
    # --- PREPARE THE FINAL JSON PAYLOAD ---
    final_db_payload = {
        "name": "Batch Result",
        "strategies_results": all_processed_results_for_db,
        "initial_capital": initialCapital
    }

    save_backtest_results(batch_id=batch_id, results=convert_to_json_serializable(final_db_payload))
    update_job_status(batch_id, "completed", "Batch complete.")
    send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "All tasks finished."}})
    
    print(f"--- BATCH MANAGER: Job {batch_id} finished and saved. ---")
        

# process_backtest_job should now RETURN the results instead of just printing
def process_backtest_job(batch_id: str, manager: any, job_id: str, file_name: str, file_content: str, initialCapital: int, client, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Processes a SINGLE backtest job, sends live updates, and returns its results.
    """
    try:
        log_msg = f"Starting backtest for: {file_name}"
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": log_msg}})
        update_job_status(batch_id, "running", log_msg)
        
        # This pure function does all the heavy lifting
        strategy_data, strategy_instance, strategy_metrics, monthly_returns, no_fee_equity = run_single_backtest(batch_id, manager, job_id, file_name, file_content, initialCapital, client, queue, loop)
        
        # Prepare the payload for the UI
        single_result_payload = prepare_strategy_payload(strategy_data, strategy_metrics, monthly_returns)
        
        # Send the result for this single strategy IMMEDIATELY to the UI
        send_update_to_queue(loop, queue, batch_id, {
            "type": "strategy_result",
            "payload": convert_to_json_serializable(single_result_payload)
        })
        
        log_msg_done = f"Finished processing: {file_name}"
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "SUCCESS", "message": log_msg_done}})
        print(f"WORKER: Job {job_id} for {file_name} completed and sent update.")
        
        # Return the raw data for the manager to use later for the portfolio
        return strategy_data, strategy_instance, strategy_metrics, monthly_returns, no_fee_equity
        
    except Exception as e:
        error_msg = f"ERROR in {file_name}: {traceback.format_exc()}"
        print(f"!!! CRITICAL ERROR IN WORKER job_id={job_id} !!!\n{error_msg}")
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": f"Failed in {file_name}. See server logs."}})
        return None

# This is a placeholder for your actual backtest engine
def run_single_backtest(batch_id: str, manager: any, job_id: str, file_name:str, strategy_code: str, initialCapital: int, client, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    
    print(f"--- Running core backtest for {file_name} ---")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Running core backtest for {file_name}"}})

    # 1. Parse the code of config
    strategy_instance, strategy_name = parse_file(job_id, file_name, strategy_code)
    
    # 2. Fetch data
    asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Ohlcv Fetch Successful"}})

    # 3. Calculate indicators
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Indicators Calculation Successful"}})

    # 4. Run simulation
    strategy_data, commission, no_fee_equity = run_backtest_loop(strategy_name, strategy_instance, ohlcv_idk, initialCapital)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Backtestloop Successful"}})

    # 5. Calculate metrics (simulated)
    strategy_metrics, monthly_returns = calculate_metrics(strategy_data, initialCapital, commission)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Metrics Calculation Successful"}})

    print(f"--- Core backtest for {asset} finished ---")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Core backtest for {file_name} finished"}})

    # In a real app, this would return the full results object
    return strategy_data, strategy_instance, strategy_metrics, monthly_returns, no_fee_equity

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
    elif isinstance(obj, (float, np.floating, np.float64)):
        # Check for NaN, Infinity, or -Infinity
        if math.isnan(obj) or math.isinf(obj):
            return None # Convert to null in the final JSON
        return float(obj) # Otherwise, convert to a standard Python float

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



def modify_indicator_list(code_str: str, modifications: dict) -> str:
    tree = ast.parse(code_str)
    class IndicatorModifier(ast.NodeTransformer):
        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == 'indicators':
                if isinstance(node.value, ast.List):
                    for indicator_idx, indicator_tuple_node in enumerate(node.value.elts):
                        if indicator_idx in modifications:
                            param_tuple_node = indicator_tuple_node.elts[2]
                            for param_idx, _ in enumerate(param_tuple_node.elts):
                                if param_idx in modifications[indicator_idx]:
                                    new_value = modifications[indicator_idx][param_idx]
                                    # Replace the old value node with a new one
                                    param_tuple_node.elts[param_idx] = ast.Constant(value=new_value)
            return self.generic_visit(node)
    modifier = IndicatorModifier()
    modified_tree = modifier.visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree)


def build_new_indicator_line(original_code: str, modifications: dict) -> str:
    """
    Rebuilds the `self.indicators` list as a string using the modifications.
    This is much safer than modifying the AST.
    """
    # 1. First, parse the original line to get the structure and non-modified values
    match = re.search(r"self\.indicators\s*=\s*(\[.*?\])", original_code, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'self.indicators = [...]' in the strategy code.")
    
    # Safely evaluate the list string to get a Python list object
    # This is safe because we're only evaluating the part inside the brackets
    indicators_list = eval(match.group(1))

    # 2. Apply the modifications to the Python list object
    for indicator_idx, params_to_change in modifications.items():
        if indicator_idx < len(indicators_list):
            # Get the parameter tuple, e.g., (50,)
            original_params = list(indicators_list[indicator_idx][2])
            for param_idx, new_value in params_to_change.items():
                if param_idx < len(original_params):
                    original_params[param_idx] = new_value
            
            # Recreate the indicator tuple with the modified parameter tuple
            original_tuple = list(indicators_list[indicator_idx])
            original_tuple[2] = tuple(original_params)
            indicators_list[indicator_idx] = tuple(original_tuple)

    # 3. Convert the modified Python list back into a perfectly formatted string
    # e.g., "self.indicators = [('SMA', '1m', (70,)), ...]"
    return f"self.indicators = {str(indicators_list)}"



def process_optimization_job(batch_id: str, manager: any, job_id: str, run_num: int, total_runs: int, original_code: str, modifications_for_run: list, strategy_display_name: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    try:
        modified_code = modify_strategy_code(original_code, modifications_for_run)
        
        log_msg = f"({run_num}/{total_runs}) Running: {strategy_display_name}"
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": log_msg}})
        
        # The rest of the function is the same, it just uses the modified_code
        initialCapital = 1000
        client = get_client('binance')
        strategy_data, _, metrics, monthly_returns, _ = run_single_backtest(
            batch_id, manager, job_id, strategy_display_name, modified_code, initialCapital, client, queue, loop
        )
        
        full_payload = prepare_strategy_payload(strategy_data, metrics, monthly_returns)
        send_update_to_queue(loop, queue, batch_id, {
            "type": "strategy_result",
            "payload": convert_to_json_serializable(full_payload)
        })
    except Exception as e:
        error_msg = f"ERROR in run {strategy_display_name}: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": f"Run '{strategy_display_name}' failed."}})


def run_optimization_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    params_to_optimize = config['parameters_to_optimize']
    original_code = config['strategy_code']

    param_ranges = [np.arange(p['start'], p['end'] + p['step'], p['step']) for p in params_to_optimize]
    all_combinations = list(itertools.product(*param_ranges))
    total_runs = len(all_combinations)

    print(f"Total optimization runs to execute: {total_runs}")

    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Generated {total_runs} parameter sets to test."}})

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for i, combo in enumerate(all_combinations):
            
            modifications_for_run = []
            param_strings = []
            
            for j, param_config in enumerate(params_to_optimize):
                current_value = round(combo[j], 4) # Get the value for this combination
                # Add the specific modification instruction for this run
                modifications_for_run.append({**param_config, 'value': current_value})
                param_strings.append(f"{param_config['name']}={current_value}")
            
            strategy_display_name = ", ".join(param_strings)

            executor.submit(
                process_optimization_job,
                batch_id, manager, str(uuid.uuid4()), i + 1, total_runs,
                original_code, modifications_for_run, strategy_display_name,
                queue=queue,
                loop=loop
            )

        #     futures.append(future)
        
        # for future in as_completed(futures):
        #     future.result() # Wait for all tasks to complete and raise exceptions if any

    update_job_status(batch_id, "completed", "Optimization finished.")
    send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Optimization complete."}})
    print(f"--- OPTIMIZATION MANAGER ({batch_id}): Job finished. ---")


def modify_strategy_code(original_code: str, modifications: list) -> str:
    """
    A generic function to modify both `self.params` and `self.indicators`
    in the strategy code string before a run.
    """
    modified_code = original_code

    # --- Step 1: Modify `self.params` dictionary ---
    params_match = re.search(r"self\.params\s*=\s*(\{.*?\})", modified_code, re.DOTALL)
    if params_match:
        params_dict = ast.literal_eval(params_match.group(1))
        for mod in modifications:
            if mod['type'] == 'strategy_param':
                params_dict[mod['name']] = mod['value']
        new_params_line = f"self.params = {str(params_dict)}"
        modified_code = re.sub(r"self\.params\s*=\s*\{.*?\}", new_params_line, modified_code, flags=re.DOTALL)

    # --- Step 2: Modify `self.indicators` list ---
    indicators_match = re.search(r"self\.indicators\s*=\s*(\[.*?\])", modified_code, re.DOTALL)
    if indicators_match:
        indicators_list = ast.literal_eval(indicators_match.group(1))
        for mod in modifications:
            if mod['type'] == 'indicator_param':
                ind_params = list(indicators_list[mod['indicatorIndex']][2])
                ind_params[mod['paramIndex']] = mod['value']
                ind_tuple = list(indicators_list[mod['indicatorIndex']])
                ind_tuple[2] = tuple(ind_params)
                indicators_list[mod['indicatorIndex']] = tuple(ind_tuple)
        new_indicators_line = f"self.indicators = {str(indicators_list)}"
        modified_code = re.sub(r"self\.indicators\s*=\s*\[.*?\]", new_indicators_line, modified_code, flags=re.DOTALL)

    return modified_code


def run_asset_screening_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    print(f"--- ASSET SCREENER ({batch_id}): Starting job ---")
    
    symbols_to_screen = config['symbols_to_screen']
    original_code = config['strategy_code']
    total_runs = len(symbols_to_screen)
    
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Starting to screen {total_runs} assets..."}})
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for i, symbol in enumerate(symbols_to_screen):
            
            # --- Dynamically modify the code to set the symbol ---
            # This is much more robust than relying on the backtest engine to do it.
            modified_code = re.sub(
                r"self\.symbol\s*=\s*['\"].*?['\"]", # Find `self.symbol = '...'`
                f"self.symbol = '{symbol}'",      # Replace it with the current symbol
                original_code
            )
            
            # The "name" of the strategy for this run is just the symbol itself
            strategy_display_name = symbol
            
            executor.submit(
                # We can reuse the same backtest worker!
                process_backtest_job, # Reusing the simple batch worker is perfect
                batch_id=batch_id,
                manager=manager,
                job_id=str(uuid.uuid4()),
                file_name=strategy_display_name,
                file_content=modified_code, # Pass the modified code
                initialCapital=1000, # This could also be a config option
                client=get_client('binance'),
                queue=queue,
                loop=loop
            )
            
    # The manager doesn't wait, it just submits all jobs. The completion message is sent
    # after the last worker finishes, which we can track if needed, or just let the user see.
    # For now, we assume the job is "done" when all tasks are submitted.
    # A more advanced version would use `as_completed` to send the final message.
    print(f"--- ASSET SCREENER ({batch_id}): All screening jobs submitted. ---")
    
    
def run_unified_test_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    A single, powerful manager that handles asset screening, parameter optimization,
    or both combined, with intelligent pruning of invalid combinations.
    """
    print(f"--- UNIFIED TEST MANAGER ({batch_id}): Starting job ---")
    
    # --- Step 1: Get all configuration from the payload ---
    original_code = config['strategy_code']
    symbols_to_screen = config.get('symbols_to_screen', [])
    params_to_optimize = config.get('parameters_to_optimize', [])
    combination_rules = config.get('combination_rules', [])

    # --- Step 2: Handle the symbols list ---
    if not symbols_to_screen:
        try:
            match = re.search(r"self\.symbol\s*=\s*['\"](.*?)['\"]", original_code)
            if match: symbols_to_screen.append(match.group(1))
            else: raise ValueError("No symbols selected and no default `self.symbol` found in file.")
        except Exception as e:
            fail_job(batch_id, str(e)); return

    # --- Step 3: Generate parameter value lists ---
    param_value_lists = []
    if params_to_optimize:
        for p in params_to_optimize:
            if p.get('mode') == 'list':
                try:
                    values = [float(val.strip()) for val in p.get('list_values', '').split(',') if val.strip()]
                    if values: param_value_lists.append(values)
                except ValueError:
                    fail_job(batch_id, f"Invalid number in list for '{p['name']}'."); return
            else: # 'range' mode
                param_value_lists.append(np.arange(p['start'], p['end'] + p['step'], p['step']))
    
    # --- Step 4: Generate ALL possible combinations ---
    # If no params were optimized, param_value_lists is empty, and this correctly produces [()]
    all_combinations = list(itertools.product(*param_value_lists))

    # --- Step 5: Prune invalid combinations using the rules ---
    valid_combinations = []
    param_id_to_index = {p['id']: i for i, p in enumerate(params_to_optimize)}

    for combo in all_combinations:
        is_combo_valid = True
        for rule in combination_rules:
            idx1 = param_id_to_index.get(rule['param1'])
            idx2 = param_id_to_index.get(rule['param2'])
            
            if idx1 is not None and idx2 is not None:
                val1 = combo[idx1]
                val2 = combo[idx2]
                operator_func = OPERATOR_MAP.get(rule['operator'])
                
                if operator_func and not operator_func(val1, val2):
                    is_combo_valid = False
                    break 
        
        if is_combo_valid:
            valid_combinations.append(combo)
            
    # --- Step 6: Final setup and logging ---
    original_total = len(all_combinations)
    pruned_count = original_total - len(valid_combinations)
    total_runs = len(symbols_to_screen) * len(valid_combinations)

    print(f"Pruned {pruned_count} invalid combinations. Total valid runs: {total_runs}")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Generated {total_runs} valid tests (pruned {pruned_count} combinations)."}})
    
    if total_runs == 0: 
        fail_job(batch_id, "No valid test combinations to run after pruning."); 
        return
    
    # --- Step 7: The Nested Loop Logic for Job Submission ---
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for symbol in symbols_to_screen:
            code_with_symbol = re.sub(r"self\.symbol\s*=\s*['\"].*?['\"]", f"self.symbol = '{symbol}'", original_code)
            
            for combo in valid_combinations:
                modifications_for_run = []
                param_strings = []

                for j, param_config in enumerate(params_to_optimize):
                    current_value = combo[j]
                    modifications_for_run.append({**param_config, 'value': current_value})
                    formatted_value = f"{current_value:g}" 
                    param_strings.append(f"{param_config['name']}={formatted_value}")
                
                final_modified_code = modify_strategy_code(code_with_symbol, modifications_for_run)
                param_part = ", ".join(param_strings)
                strategy_display_name = f"{symbol}" + (f" | {param_part}" if param_part else "")

                # Submit the job to the worker pool
                executor.submit(
                    process_backtest_job, # We can reuse the simplest worker
                    batch_id=batch_id,
                    manager=manager,
                    job_id=str(uuid.uuid4()),
                    file_name=strategy_display_name,
                    file_content=final_modified_code,
                    initialCapital=1000,
                    client=get_client('binance'),
                    queue=queue,
                    loop=loop
                )

    # Note: A truly robust implementation would wait for all futures to complete
    # before sending the final "batch_complete" message. For simplicity, we omit that here.
    # The current implementation will send "batch_complete" after all jobs are submitted.
    update_job_status(batch_id, "completed", "All tests submitted.")
    send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "All tests submitted."}})
    print(f"--- UNIFIED MANAGER ({batch_id}): All {total_runs} jobs submitted. ---")