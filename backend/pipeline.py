# backend/pipeline.py
import io
import os
import re
import ast
import math
import uuid
import asyncio
import itertools
import traceback
from enum import Enum
import numpy as np
import pandas as pd
import operator as op
from typing import Optional
from datetime import time, timedelta 

from concurrent.futures import ThreadPoolExecutor, as_completed

from database import SessionLocal, save_backtest_results, update_job_status, fail_job
from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client
from core.basestrategy import BaseStrategy

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




# --- The Batch Manager Function ---
def run_batch_manager(batch_id: str, files_data: list[dict], manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event): 
    """
    Orchestrates the batch in parallel. It now primarily collects raw data
    for the final portfolio calculation, as workers send their own UI updates.
    """
    async def wait_for_connection():
        print(f"[{batch_id}] Batch manager started, waiting for client to connect...")
        await connection_event.wait()
        print(f"[{batch_id}] Client connected! Proceeding with batch...")

    future = asyncio.run_coroutine_threadsafe(wait_for_connection(), loop)
    future.result() # This blocks until the client connects
    
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
def run_single_backtest(batch_id: str, manager: any, job_id: str, file_name:str, strategy_code: str, initialCapital: int, client, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, ohlcv_df: Optional[pd.DataFrame] = None):
    
    print(f"--- Running core backtest for {file_name} ---")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Running core backtest for {file_name}"}})

    # 1. Parse the code of config
    strategy_instance, strategy_name = parse_file(job_id, file_name, strategy_code)
    
    # 2. Get OHLCV data (either from CSV or from exchange)
    if ohlcv_df is not None:
        print(f"--- Using pre-loaded OHLCV data for {file_name} ---")
        ohlcv = ohlcv_df
        asset = strategy_instance.symbol
    else:
        print(f"--- Fetching OHLCV data from exchange for {file_name} ---")
        client = get_client('binance')
        asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    
    # 2. Fetch data
    # asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Ohlcv Fetch Successful"}})

    # 3. Calculate indicators
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Indicators Calculation Successful"}})
    pd.set_option('display.max_columns', None)
    
    # # 4. Clean the data before backtesting
    # # Numba cannot handle NaN values. We must remove any rows that have
    # # incomplete data, which typically occurs at the beginning of the
    # # DataFrame where long-period indicators haven't "warmed up" yet.
    # original_rows = len(ohlcv_idk)
    # ohlcv_cleaned = ohlcv_idk.dropna()
    # cleaned_rows = len(ohlcv_cleaned)
    
    # if original_rows > cleaned_rows:
    #     print(f"--- Data Cleaning: Dropped {original_rows - cleaned_rows} rows with NaN values. ---")
    
    # # Check if any data remains after cleaning
    # if ohlcv_cleaned.empty:
    #     raise ValueError("No data remained after cleaning NaN values. Your date range might be too short for the indicators used (e.g., SMA 200).")


    # 4. Run simulation
    strategy_data, commission, no_fee_equity = run_backtest_loop(strategy_name, strategy_instance, ohlcv_idk, initialCapital)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Backtestloop Successful"}})
    
    # 5. Calculate metrics (simulated)
    strategy_metrics, monthly_returns = calculate_metrics(strategy_data, initialCapital, commission)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Metrics Calculation Successful"}})
    
    print(f"--- Core backtest for {asset} finished ---")
    # print(f"  -> Strategy Metrics: {strategy_metrics}")
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
    elif isinstance(obj, BaseStrategy):
        # Convert the strategy object to its class name (e.g., "EURUSD_Short")
        return obj.__class__.__name__
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
    elif isinstance(obj, Enum):
        return obj.name
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
    
    rename_map = {
        'Entry': 'entry',
        'Take Profit': 'take_profit',
        'Stop Loss': 'stop_loss',
    }
    
    formatted_df.rename(columns=rename_map, inplace=True, errors='ignore')

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


def run_optimization_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event):
    
    async def wait_for_connection():
        print(f"[{batch_id}] Batch manager started, waiting for client to connect...")
        await connection_event.wait()
        print(f"[{batch_id}] Client connected! Proceeding with batch...")

    future = asyncio.run_coroutine_threadsafe(wait_for_connection(), loop)
    future.result() # This blocks until the client connects
    
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
    
    
def run_unified_test_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event):
    """
    A single, powerful manager that handles asset screening, parameter optimization,
    or both combined, with intelligent pruning of invalid combinations.
    """
    async def wait_for_connection():
        print(f"[{batch_id}] Batch manager started, waiting for client to connect...")
        await connection_event.wait()
        print(f"[{batch_id}] Client connected! Proceeding with batch...")

    future = asyncio.run_coroutine_threadsafe(wait_for_connection(), loop)
    future.result() # This blocks until the client connects
    
    test_type = config.get("test_type")
    
    send_update_to_queue(loop, queue, batch_id, {
        "type": "batch_info",
        # Add a default test_type if it doesn't exist for standard optimizations
        "payload": { "config": {**config, "test_type": test_type or "standard_optimization"} }
    })
    
    if test_type == "data_segmentation":
        # Call the new, dedicated manager for this test type
        run_data_segmentation_manager(batch_id, config, manager, queue, loop)
        return # Important: exit after calling the specific manager

    print(f"[{batch_id}] Starting standard optimization/screening test...")
    
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
    
    
def run_local_backtest_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    try:
        strategy_name = config['strategy_name']
        strategy_code = config['strategy_code']
        csv_data = config['csv_data']

        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Starting backtest for {strategy_name} with local CSV data..."}})
        
        # --- Parse and Validate CSV Data ---
        # Use io.StringIO to treat the string data as a file
        csv_file = io.StringIO(csv_data)
        ohlcv_df = pd.read_csv(csv_file)

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
        if not all(col in ohlcv_df.columns for col in required_columns):
            raise ValueError(f"CSV is missing one of the required columns: {required_columns}")

        # Convert timestamp to a consistent format (datetime objects)
        # This is very robust and handles Unix seconds, milliseconds, or ISO strings.
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='s', errors='coerce').fillna(
                                pd.to_datetime(ohlcv_df['timestamp'], unit='ms', errors='coerce')).fillna(
                                pd.to_datetime(ohlcv_df['timestamp'], errors='coerce'))
        
        # Ensure timezone is consistent (UTC is best practice)
        if ohlcv_df['timestamp'].dt.tz is None:
            ohlcv_df['timestamp'] = ohlcv_df['timestamp'].dt.tz_localize('UTC')
        else:
            ohlcv_df['timestamp'] = ohlcv_df['timestamp'].dt.tz_convert('UTC')

        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce')
        
        ohlcv_df.dropna(inplace=True)
        ohlcv_df.sort_values(by='timestamp', inplace=True)

        # Now, call the universal worker, passing the pre-loaded DataFrame
        universal_backtest_worker(
            batch_id, manager, config['strategy_name'], config['strategy_code'], 1000, client=get_client('binance'),
            queue=queue, loop=loop, ohlcv_df=ohlcv_df
        )
                
        update_job_status(batch_id, "completed", "Local backtest complete.")
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Local backtest complete."}})

    except Exception as e:
        error_msg = f"Failed to process local backtest: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})

def universal_backtest_worker(
    batch_id: str, 
    manager: any,
    display_name: str, 
    code_str: str, 
    capital: int,
    client, 
    queue: asyncio.Queue, 
    loop: asyncio.AbstractEventLoop, 
    ohlcv_df: Optional[pd.DataFrame] = None
):
    """
    This worker runs one backtest and sends the result. It can be called by any manager.
    It returns the raw data for potential portfolio calculation.
    """
    job_id = str(uuid.uuid4())
    try:
        # We reuse the core backtest engine, passing the optional DataFrame
        strategy_data, strategy_instance, metrics, monthly_returns, no_fee_equity = run_single_backtest(
            batch_id, manager, job_id, display_name, code_str, capital, client, queue, loop, ohlcv_df=ohlcv_df
        )
        
        full_payload = prepare_strategy_payload(strategy_data, metrics, monthly_returns)
        full_payload['strategy_name'] = display_name 

        send_update_to_queue(loop, queue, batch_id, {
            "type": "strategy_result",
            "payload": convert_to_json_serializable(full_payload)
        })
                
        # Return raw data for portfolio manager if needed
        return strategy_data, strategy_instance, metrics, monthly_returns, no_fee_equity

    except Exception as e:
        error_msg = f"ERROR in run '{display_name}': {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": f"Run '{display_name}' failed."}})
        return None


def run_data_segmentation_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Manages a Data Segmentation test.
    1. Fetches full data for the primary symbol.
    2. Splits data into Training, Validation, and Testing sets.
    3. Runs a parameter optimization on the Training set.
    4. Takes the top N results and runs them on Validation and Testing sets.
    """
    try:
        send_update_to_queue(loop, queue, batch_id, {
            "type": "batch_info", 
            "payload": { "config": config }
        })
        
        print(f"--- DATA SEGMENTATION ({batch_id}): Starting job ---")
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Starting Data Segmentation Test..."}})
        
        original_code = config['strategy_code']
        primary_symbol = config.get('symbols_to_screen', [])[0]
        optimization_metric = config['optimization_metric']
        top_n_sets = config['top_n_sets']
        
        temp_code = re.sub(r"self\.symbol\s*=\s*['\"].*?['\"]", f"self.symbol = '{primary_symbol}'", original_code)
        strategy_instance, _ = parse_file(str(uuid.uuid4()), "temp", temp_code)
        client = get_client('binance')
        _, full_ohlcv_df = fetch_candlestick_data(client, strategy_instance)

        total_rows = len(full_ohlcv_df)
        train_end_idx = int(total_rows * (config['training_pct'] / 100))
        validation_end_idx = train_end_idx + int(total_rows * (config['validation_pct'] / 100))

        train_df = full_ohlcv_df.iloc[:train_end_idx].reset_index(drop=True)
        validation_df = full_ohlcv_df.iloc[train_end_idx:validation_end_idx].reset_index(drop=True)
        test_df = full_ohlcv_df.iloc[validation_end_idx:].reset_index(drop=True)

        log_msg = (f"Data split: Training ({len(train_df)}), Validation ({len(validation_df)}), Testing ({len(test_df)})")
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": log_msg}})
        
        # --- 4. Run Optimization on Training Data (Unchanged) ---
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Phase 1: Optimizing on Training data..."}})
        training_results = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            param_combos = generate_and_prune_combinations(config)
            if not param_combos: raise ValueError("No valid parameter combinations.")
            
            futures = {executor.submit(universal_backtest_worker, batch_id, manager, build_modifications_and_name(c, config['parameters_to_optimize'], primary_symbol)[1], modify_strategy_code(temp_code, build_modifications_and_name(c, config['parameters_to_optimize'],"")[0]), 1000, client, queue, loop, train_df.copy()): c for c in param_combos}
            for future in as_completed(futures):
                result = future.result()
                if result: training_results.append((result, futures[future]))
        
        if not training_results: raise ValueError("Optimization on training data yielded no runs.")

        # --- 5. Rank Results and Select Top N (Unchanged) ---
        is_reverse_sort = optimization_metric != 'Max Drawdown [%]'
        training_results.sort(key=lambda x: x[0][2].get(optimization_metric, -9999), reverse=is_reverse_sort)
        top_n_from_training = training_results[:top_n_sets]
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 1 Complete. Selected top {len(top_n_from_training)} parameter sets for validation."}})

        # --- MODIFIED: Split Step 6 into three parts ---

        # --- 6. Run Top N on Validation Data ONLY ---
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Phase 2: Verifying top sets on Validation data..."}})
        
        validation_results = []
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {}
            for _, combo in top_n_from_training:
                modifications, display_name = build_modifications_and_name(combo, config['parameters_to_optimize'], primary_symbol)
                modified_code = modify_strategy_code(temp_code, modifications)
                future = executor.submit(universal_backtest_worker, batch_id, manager, f"{display_name} [Validation]", modified_code, 1000, client, queue, loop, validation_df.copy())
                futures[future] = combo # Store the combo to link it to the result
            
            for future in as_completed(futures):
                result = future.result()
                if result: validation_results.append((result, futures[future]))

        # --- 7. Filter validation results to find qualified sets ---
        qualified_for_testing = []
        for result_tuple, combo in validation_results:
            # The qualification criteria: must be profitable in the validation period.
            # You can make this criteria stricter (e.g., Sharpe > 0.5)
            
            # Get the metrics dictionary for the validation run
            validation_metrics = result_tuple[2]
            
            # Define our qualification criteria
            is_profitable = validation_metrics.get('Net_Profit', 0) > 0
            has_acceptable_drawdown = validation_metrics.get('Max_Drawdown', 100) < 30 # Max DD must be less than 30%
            is_consistent = validation_metrics.get('Profit_Factor', 0) > 1.1 # Must be better than a coin flip
            traded_enough = validation_metrics.get('Closed_Trades', 0) >= 5 # Must have at least 5 trades
            
            # The strategy only qualifies if ALL criteria are met
            if is_profitable and has_acceptable_drawdown and is_consistent and traded_enough:
                qualified_for_testing.append(combo)
        
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 2 Complete. {len(qualified_for_testing)} sets qualified for final testing."}})

        if not qualified_for_testing:
            print(f"[{batch_id}] No parameter sets passed the validation criteria.")
        else:
            # --- 8. Run qualified sets on Testing Data ---
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Phase 3: Running qualified sets on Testing (Out-of-Sample) data..."}})
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                for combo in qualified_for_testing:
                    modifications, display_name = build_modifications_and_name(combo, config['parameters_to_optimize'], primary_symbol)
                    modified_code = modify_strategy_code(temp_code, modifications)
                    executor.submit(universal_backtest_worker, batch_id, manager, f"{display_name} [Testing]", modified_code, 1000, client, queue, loop, test_df.copy())

    except Exception as e:
        error_msg = f"Data Segmentation test failed: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})
    finally:
        update_job_status(batch_id, "completed", "Data Segmentation test finished.")
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Data Segmentation test complete."}})
        print(f"--- DATA SEGMENTATION ({batch_id}): Job finished. ---")

# --- 2. Helper Functions for the New Manager ---
# We need to extract the combination generation logic from `run_unified_test_manager`
# so we can reuse it.

def generate_and_prune_combinations(config: dict) -> list:
    """
    Generates and prunes parameter combinations based on the config.
    Extracted from run_unified_test_manager for reusability.
    """
    params_to_optimize = config.get('parameters_to_optimize', [])
    combination_rules = config.get('combination_rules', [])
    
    param_value_lists = []
    if params_to_optimize:
        for p in params_to_optimize:
            if p.get('mode') == 'list':
                try:
                    values = [float(val.strip()) for val in p.get('list_values', '').split(',') if val.strip()]
                    if values: param_value_lists.append(values)
                except ValueError:
                    raise ValueError(f"Invalid number in list for '{p['name']}'.")
            else: # 'range' mode
                param_value_lists.append(np.arange(p['start'], p['end'] + p['step'], p['step']))
    
    all_combinations = list(itertools.product(*param_value_lists))
    if not params_to_optimize: # Handle case with no params (itertools gives [()])
        return [()] 

    # Prune invalid combinations
    valid_combinations = []
    param_id_to_index = {p['id']: i for i, p in enumerate(params_to_optimize)}

    for combo in all_combinations:
        is_combo_valid = True
        for rule in combination_rules:
            idx1 = param_id_to_index.get(rule['param1'])
            idx2 = param_id_to_index.get(rule['param2'])
            if idx1 is not None and idx2 is not None:
                val1, val2 = combo[idx1], combo[idx2]
                operator_func = OPERATOR_MAP.get(rule['operator'])
                if operator_func and not operator_func(val1, val2):
                    is_combo_valid = False
                    break
        if is_combo_valid:
            valid_combinations.append(combo)
            
    return valid_combinations

def build_modifications_and_name(combo: tuple, params_config: list, symbol: str) -> tuple[list, str]:
    """Builds modification list and display name from a parameter combo."""
    modifications = []
    param_strings = []
    for i, param_conf in enumerate(params_config):
        current_value = combo[i]
        modifications.append({**param_conf, 'value': current_value})
        formatted_value = f"{current_value:g}"
        param_strings.append(f"{param_conf['name']}={formatted_value}")
    
    param_part = ", ".join(param_strings)
    display_name = f"{symbol}" + (f" | {param_part}" if param_part else "")
    return modifications, display_name

def build_modifications_and_name_from_str(combo_str: str, params_config: list, symbol: str) -> tuple[list, str]:
    """Reverse engineers the modifications list from the display name string."""
    modifications = []
    param_values = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in combo_str.split(', ')}
    
    for conf in params_config:
        if conf['name'] in param_values:
            modifications.append({**conf, 'value': param_values[conf['name']]})
            
    display_name = f"{symbol}" + (f" | {combo_str}" if combo_str else "")
    return modifications, display_name