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
def run_batch_manager(batch_id: str, files_data: list[dict], use_training_set: bool, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event): 
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
    
    print(f"Starting batch manager for {batch_id}. Training set only: {use_training_set}")
    
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
            executor.submit(process_backtest_job, batch_id, manager, str(uuid.uuid4()), file['name'], file['content'], initialCapital, client, queue, loop, send_ui_update = True, use_training_set = use_training_set): file
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
        
        strategy_data, _, metrics, monthly_returns, _, trade_events_df = res_tuple
                
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
            
            # Collect all timestamp series
            all_timestamps = [
                res_tuple[0]['ohlcv']['timestamp'] 
                for res_tuple in raw_results_for_portfolio 
                if not res_tuple[0]['ohlcv'].empty
            ]
            if not all_timestamps: raise ValueError("No valid OHLCV data.")
            
            master_timestamps_array = np.sort(pd.concat(all_timestamps).unique())
            base_ohlcv_df = pd.DataFrame({'timestamp': master_timestamps_array})
            
            all_no_fee_equity_gain = 0.0
            
            # --- Step 2: Prepare a Master List of Entry and Exit Events ---
            all_entry_events = []
            all_exit_events = []
            all_trade_events_for_log = [] # For building the final, combined trade log
            trade_id_counter = 0
            
            for res_tuple in raw_results_for_portfolio:
                
                # Unpack the new 6-item tuple
                strategy_data, _, _, _, no_fee_equity_array, trade_events = res_tuple
                
                # Sum up the no-fee gain for the final portfolio commission calculation
                if len(no_fee_equity_array) > 1:
                    all_no_fee_equity_gain += no_fee_equity_array[-1] - no_fee_equity_array[0]
                
                if trade_events.empty:
                    continue

                # Give each trade a globally unique ID
                trade_events['trade_id'] = range(trade_id_counter, trade_id_counter + len(trade_events))
                trade_id_counter += len(trade_events)
                all_trade_events_for_log.append(trade_events)
                
                entry_df = trade_events[[
                    'timestamp', 'trade_id', 'Signal', 'Entry', 'Stop Loss', 'Take Profit', 'Risk'
                ]].copy()
                entry_df.rename(columns={
                    'timestamp': 'event_timestamp', 'Signal': 'entry_direction', 
                    'Entry': 'entry_price', 'Stop Loss': 'entry_sl', 'Take Profit': 'entry_tp', 'Risk': 'risk_percent'
                }, inplace=True)
                all_entry_events.append(entry_df)

                # Create and collect Exit Events
                exit_df = trade_events[['Exit_Time', 'trade_id', 'Result']].copy()
                exit_df.rename(columns={'Exit_Time': 'event_timestamp', 'Result': 'exit_result'}, inplace=True)
                all_exit_events.append(exit_df)
            
            master_entries = pd.concat(all_entry_events, ignore_index=True) if all_entry_events else pd.DataFrame()
            master_exits = pd.concat(all_exit_events, ignore_index=True) if all_exit_events else pd.DataFrame()

            entries_agg = master_entries.groupby('event_timestamp').agg(list).reset_index() if not master_entries.empty else pd.DataFrame(columns=['event_timestamp'])
            exits_agg = master_exits.groupby('event_timestamp').agg(list).reset_index() if not master_exits.empty else pd.DataFrame(columns=['event_timestamp'])
            
            complete_df = pd.merge(base_ohlcv_df, entries_agg, left_on='timestamp', right_on='event_timestamp', how='left')
            complete_df = pd.merge(complete_df, exits_agg, left_on='timestamp', right_on='event_timestamp', how='left', suffixes=('_entry', '_exit'))

            if 'event_timestamp_entry' in complete_df.columns: complete_df.drop(columns=['event_timestamp_entry'], inplace=True)
            if 'event_timestamp_exit' in complete_df.columns: complete_df.drop(columns=['event_timestamp_exit'], inplace=True)
            
            # --- Step 4: Call the NEW Portfolio Simulator (MODIFIED) ---
            sample_instance = raw_results_for_portfolio[0][1]
            
            portfolio_equity, open_trade_count, closed_trades_log = sample_instance.portfolio(
                complete_df['timestamp'].values,
                complete_df['trade_id_entry'].values,
                complete_df['entry_direction'].values,
                complete_df['entry_price'].values,
                complete_df['entry_sl'].values,
                complete_df['entry_tp'].values,
                complete_df['risk_percent'].values,
                complete_df['trade_id_exit'].values,
                complete_df['exit_result'].values
            )

            # --- Step 5: Process Results with the NEW, SIMPLIFIED LOGIC ---
            aligned_equity = pd.Series(portfolio_equity, index=complete_df.index)
            complete_df['Portfolio_Open_Trades'] = open_trade_count

            # Create the combined trade log DataFrame
            all_trade_events_df = pd.concat(all_trade_events_for_log, ignore_index=True) if all_trade_events_for_log else pd.DataFrame()
            
            if closed_trades_log:
                # Convert the log from the portfolio into a DataFrame
                returns_df = pd.DataFrame(closed_trades_log)
                
                # Merge the true returns back into the main trade log
                all_trade_events_df = pd.merge(all_trade_events_df, returns_df, on='trade_id', how='left')
                
                # Overwrite the old, inaccurate returns columns
                # Note: 'Returns' (gross) and 'Commissioned Returns' (net) are now accurate
                all_trade_events_df['Returns'] = all_trade_events_df['final_gross_return'].round(2)
                all_trade_events_df['Commissioned Returns'] = all_trade_events_df['final_net_return'].round(2)
                
                # Clean up helper columns
                all_trade_events_df.drop(columns=['final_gross_return', 'final_net_return'], inplace=True)

            # Update the 'Open_Trades' column (same as before)
            if not all_trade_events_df.empty:
                open_trades_lookup = complete_df[['timestamp', 'Portfolio_Open_Trades']]
                all_trade_events_df = pd.merge(all_trade_events_df, open_trades_lookup, on='timestamp', how='left')
                all_trade_events_df.drop(columns=['Open_Trades'], inplace=True, errors='ignore')
                all_trade_events_df.rename(columns={'Portfolio_Open_Trades': 'Open_Trades'}, inplace=True)
                
            portfolio_strategy_data = {
                'strategy_name': 'Portfolio', 
                'equity': aligned_equity, 
                'ohlcv': complete_df, 
                'signals': all_trade_events_df # Use the combined events for the trade list
            }
            
            portfolio_net_profit = aligned_equity.iloc[-1] - aligned_equity.iloc[0]
            
            if all_no_fee_equity_gain > 0:
                commission = round(((all_no_fee_equity_gain - portfolio_net_profit) * 100) / all_no_fee_equity_gain, 2)
            else:
                commission = 0.0
            
            portfolio_metrics, portfolio_returns = calculate_metrics(portfolio_strategy_data, initialCapital, commission)
            
            portfolio_payload = prepare_strategy_payload(portfolio_strategy_data, portfolio_metrics, portfolio_returns)
            
            send_update_to_queue(loop, queue, batch_id, {
                "type": "strategy_result", 
                "payload": convert_to_json_serializable(portfolio_payload)
            })
            
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
def process_backtest_job(batch_id: str, manager: any, job_id: str, file_name: str, file_content: str, initialCapital: int, client, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, send_ui_update: bool = True, use_training_set: bool = False):
    """
    Processes a SINGLE backtest job, sends live updates, and returns its results.
    """
    try:
        log_msg = f"Starting backtest for: {file_name}"
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": log_msg}})
        update_job_status(batch_id, "running", log_msg)
        
        # This pure function does all the heavy lifting
        strategy_data, strategy_instance, strategy_metrics, monthly_returns, no_fee_equity, trade_events_df = run_single_backtest(
            batch_id, manager, job_id, file_name, file_content, initialCapital, client, queue, loop, use_training_set=use_training_set
            )
        
        if send_ui_update:
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
        return strategy_data, strategy_instance, strategy_metrics, monthly_returns, no_fee_equity, trade_events_df
        
    except Exception as e:
        error_msg = f"ERROR in {file_name}: {traceback.format_exc()}"
        print(f"!!! CRITICAL ERROR IN WORKER job_id={job_id} !!!\n{error_msg}")
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": f"Failed in {file_name}. See server logs."}})
        return None

# This is a placeholder for your actual backtest engine
def run_single_backtest(batch_id: str, manager: any, job_id: str, file_name:str, strategy_code: str, initialCapital: int, client, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, ohlcv_df: Optional[pd.DataFrame] = None, use_training_set: bool = False):
    
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
    
    # If use_training_set is True, filter the DataFrame to include only 70% of the data
    if use_training_set:
        print(f"Initial OHLCV rows: {len(ohlcv)}")
        split_index = int(len(ohlcv) * 0.7)
        ohlcv = ohlcv.iloc[:split_index]
        print(f"Using training set only: {len(ohlcv)} rows")
    
    # 2. Fetch data
    # asset, ohlcv = fetch_candlestick_data(client, strategy_instance)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Ohlcv Fetch Successful"}})

    # 3. Calculate indicators
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Indicators Calculation Successful"}})
    pd.set_option('display.max_columns', None)
    
    # --- Step 4: Run the backtest simulation loop ---
    run_result = run_backtest_loop(strategy_name, strategy_instance, ohlcv_idk, initialCapital)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Backtest loop finished."}})
    
    # --- Step 5: Check for a complete failure from the backtest loop ---
    if run_result is None:
        print(f"--- Core backtest for {file_name} failed. Creating dummy result. ---")
        # Create a dictionary for the UI payload
        failed_strategy_data = {
            'strategy_name': f"{file_name} [FAILED]",
            'equity': np.array([initialCapital] * len(ohlcv_idk)),
            'ohlcv': ohlcv_idk,
            'signals': pd.DataFrame()
        }
        # We must still return a valid 6-item tuple to the calling manager
        return (
            failed_strategy_data, 
            strategy_instance, 
            {}, # Empty metrics
            [], # Empty monthly returns
            np.array([]), # Empty no-fee equity
            pd.DataFrame() # Empty trade events
        )

    # --- Step 6: Unpack the successful result ---
    (
        corrected_equity, 
        dfSignals, 
        df_full, 
        commission, 
        corrected_no_fee_equity, 
        trade_events_df
    ) = run_result
    
    # --- Step 7: Assemble the `strategy_data` dictionary for metrics ---
    # This dictionary is the primary data structure used by downstream functions.
    strategy_data = {
        'strategy_name': strategy_name,
        'equity': corrected_equity,
        'ohlcv': df_full,
        'signals': dfSignals
    }
    
    # 5. Calculate metrics (simulated)
    strategy_metrics, monthly_returns = calculate_metrics(strategy_data, initialCapital, commission)
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Metrics Calculation Successful"}})
    
    print(f"--- Core backtest for {asset} finished ---")
    # print(f"  -> Strategy Metrics: {strategy_metrics}")
    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"{file_name}: Core backtest for {file_name} finished"}})

    # In a real app, this would return the full results object
    return strategy_data, strategy_instance, strategy_metrics, monthly_returns, corrected_no_fee_equity, trade_events_df

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
        strategy_data, _, metrics, monthly_returns, _, trade_events_df = run_single_backtest(
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
                loop=loop,
                send_ui_update=True # We want immediate updates for each asset
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
                    loop=loop,
                    send_ui_update=True # Immediate updates for each test
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
        strategy_data, strategy_instance, metrics, monthly_returns, no_fee_equity, trade_events_df = run_single_backtest(
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
            
            futures = {}
            for c in param_combos:
                # Call the helper function ONCE to get both pieces of data
                modifications, display_name = build_modifications_and_name(c, config['parameters_to_optimize'], primary_symbol)
                modified_code = modify_strategy_code(temp_code, modifications)
                
                # Submit the job with the clean variables
                future = executor.submit(
                    universal_backtest_worker,
                    batch_id, manager, display_name, modified_code, 1000,
                    client, queue, loop, train_df.copy()
                )
                futures[future] = c
            
            for future in as_completed(futures):
                result = future.result()
                if result: training_results.append((result, futures[future]))
        
        if not training_results: raise ValueError("Optimization on training data yielded no runs.")
        
        # --- Create a dictionary for easy lookup of training metrics by combo ---
        training_metrics_map = { combo: result[2] for result, combo in training_results }
        
        # --- 5. Rank Results and Select Top N ---
        is_reverse_sort = optimization_metric != 'Max_Drawdown'
        training_results.sort(key=lambda x: x[0][2].get(optimization_metric, -9999), reverse=is_reverse_sort)
        top_n_from_training = training_results[:top_n_sets]
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 1 Complete. Selected top {len(top_n_from_training)} parameter sets for validation."}})

        # --- Split Step 6 into three parts ---

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
                
        # --- Create a dictionary for easy lookup of validation metrics by combo ---
        validation_metrics_map = { combo: result[2] for result, combo in validation_results }

        # --- 7. Filter validation results to find qualified sets ---
        qualified_for_testing = []
        
        for result_tuple, combo in validation_results:
            # The qualification criteria: must be profitable in the validation period.
            # You can make this criteria stricter (e.g., Sharpe > 0.5)
            
            # Get metrics from both periods
            validation_metrics = result_tuple[2]
            training_metrics = training_metrics_map.get(combo)

            # --- Define acceptable degradation percentages ---
            MAX_SHARPE_DEGRADATION_PCT = 40.0
            MAX_PROFIT_FACTOR_DEGRADATION_PCT = 30.0
            MAX_DRAWDOWN_INCREASE_PCT = 50.0 # Allow DD to increase by up to 50%

            # --- Perform Checks ---
            training_sharpe = training_metrics.get('Sharpe_Ratio', -100)
            validation_sharpe = validation_metrics.get('Sharpe_Ratio', -100)

            training_pf = training_metrics.get('Profit_Factor', 0)
            validation_pf = validation_metrics.get('Profit_Factor', 0)

            training_dd = training_metrics.get('Max_Drawdown', 0)
            validation_dd = validation_metrics.get('Max_Drawdown', 0)


            # Check 1: Sharpe Ratio Degradation
            sharpe_degradation = ((training_sharpe - validation_sharpe) / (abs(training_sharpe) + 1e-9)) * 100
            is_sharpe_ok = not (training_sharpe > 0.1 and sharpe_degradation > MAX_SHARPE_DEGRADATION_PCT)

            # Check 2: Profit Factor Degradation
            pf_degradation = ((training_pf - validation_pf) / (abs(training_pf) + 1e-9)) * 100
            is_pf_ok = not (training_pf > 1 and pf_degradation > MAX_PROFIT_FACTOR_DEGRADATION_PCT)

            # Check 3: Max Drawdown Increase
            dd_increase = ((validation_dd - training_dd) / (abs(training_dd) + 1e-9)) * 100
            is_dd_ok = dd_increase < MAX_DRAWDOWN_INCREASE_PCT

            # Qualify only if all checks pass
            if is_sharpe_ok and is_pf_ok and is_dd_ok:
                qualified_for_testing.append(combo)
            else:
                # Use the correct helper function to build the name from the combo tuple
                _, display_name = build_modifications_and_name(combo, config['parameters_to_optimize'], '')
                param_set_str = display_name.strip(" | ")
                
                # Build a list of specific failure reasons
                failure_reasons = []
                if not is_sharpe_ok:
                    failure_reasons.append(f"Sharpe degradation: {sharpe_degradation:.1f}%")
                if not is_pf_ok:
                    failure_reasons.append(f"PF degradation: {pf_degradation:.1f}%")
                if not is_dd_ok:
                    failure_reasons.append(f"DD increase: {dd_increase:.1f}%")
                reasons_str = ", ".join(failure_reasons)

                log_message = f"'{param_set_str}' DISQUALIFIED. Reason(s): {reasons_str}"
                
                print(f"[{batch_id}] {log_message}")
                send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": log_message}})

        # --- 8. Run Qualified Sets on Testing Data ---
        testing_results = [] # <- List to hold the final results
        if not qualified_for_testing:
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": "No parameter sets qualified for the final Testing phase."}})
        else:
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 3: Running {len(qualified_for_testing)} qualified set(s) on the final Testing data..."}})
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                futures = {}
                for combo in qualified_for_testing:
                    modifications, display_name = build_modifications_and_name(combo, config['parameters_to_optimize'], primary_symbol)
                    modified_code = modify_strategy_code(temp_code, modifications)
                    future = executor.submit(universal_backtest_worker, batch_id, manager, f"{display_name} [Testing]", modified_code, 1000, client, queue, loop, test_df.copy())
                    futures[future] = combo # <-- Map the future back to its combo
                
                # --- Collect the testing results ---
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        testing_results.append((result, futures[future]))
        
        # --- Step 9 - Final Analysis and Reporting ---
        if testing_results:
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "SUCCESS", "message": "--- Final Performance Analysis ---"}})

            for testing_result_tuple, combo in testing_results:
                testing_metrics = testing_result_tuple[2]
                training_metrics = training_metrics_map.get(combo)
                validation_metrics = validation_metrics_map.get(combo)

                if not training_metrics or not validation_metrics: continue

                # Get key metrics from all three stages
                t_sharpe = training_metrics.get('Sharpe_Ratio', 0)
                v_sharpe = validation_metrics.get('Sharpe_Ratio', 0)
                f_sharpe = testing_metrics.get('Sharpe_Ratio', 0) # f for final/testing

                t_dd = training_metrics.get('Max_Drawdown', 0)
                v_dd = validation_metrics.get('Max_Drawdown', 0)
                f_dd = testing_metrics.get('Max_Drawdown', 0)

                # Build the display name
                _, display_name = build_modifications_and_name(combo, config['parameters_to_optimize'], '')
                param_set_str = display_name.strip(" | ")

                # Send a series of clear, formatted log messages for each finalist
                send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "SUCCESS", "message": f"Report for: '{param_set_str}'"}})
                send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": f"  Sharpe Path (Train -> Valid -> Test): {t_sharpe:.2f} -> {v_sharpe:.2f} -> {f_sharpe:.2f}"}})
                send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "INFO", "message": f"  Max DD Path (Train -> Valid -> Test): {t_dd:.2f}% -> {v_dd:.2f}% -> {f_dd:.2f}%"}})
    
    except Exception as e:
        error_msg = f"Data Segmentation test failed: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})
    finally:
        # NOTE: The job will now complete only AFTER the testing runs are submitted.
        # For a truly complete picture, you would collect these futures as well before
        # sending the batch_complete message, but this "fire-and-forget" is acceptable.
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


def run_hedge_optimization_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event):
    try:
        # Wait for client connection
        async def wait_for_connection():
            await connection_event.wait()
        future = asyncio.run_coroutine_threadsafe(wait_for_connection(), loop)
        future.result()
        
        send_update_to_queue(loop, queue, batch_id, { "type": "batch_info", "payload": { "config": config } })
        
        top_n = config['top_n_candidates']
        portfolio_metric = config['portfolio_metric']
        primary_symbol = config['symbols_to_screen'][0]
        base_metric_name = portfolio_metric.replace('Portfolio ', '')

        # --- PHASE 1: INDIVIDUAL OPTIMIZATION & PRE-FILTERING ---
        # --- Pass the base metric name to the helper ---
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 1: Optimizing Strategy A (Top {top_n})..."}})
        top_n_a, ohlcv_a = run_individual_optimization(batch_id, config['strategy_a'], primary_symbol, top_n, base_metric_name, manager, queue, loop)
        
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 1: Optimizing Strategy B (Top {top_n})..."}})
        top_n_b, ohlcv_b = run_individual_optimization(batch_id, config['strategy_b'], primary_symbol, top_n, base_metric_name, manager, queue, loop)
        
        if not top_n_a or not top_n_b:
            raise ValueError("Individual optimization for one or both strategies failed to produce results.")
        
        # --- PHASE 2: PAIRWISE COMBINATION & PORTFOLIO ANALYSIS ---
        total_pairs = len(top_n_a) * len(top_n_b)
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 2: Analyzing {total_pairs} hedge combinations..."}})
        
        master_timestamps_array = np.sort(pd.concat([ohlcv_a['timestamp'], ohlcv_b['timestamp']]).unique())
        master_ohlcv_df = pd.DataFrame({'timestamp': master_timestamps_array})

        # This list will now store tuples of (payload, ohlcv_df)
        portfolio_results_with_data = [] 
        with ThreadPoolExecutor() as executor:
            futures = {}
            for res_a in top_n_a:
                for res_b in top_n_b:
                    # Pass the master ohlcv_df to the combination function
                    future = executor.submit(combine_and_evaluate_pair, res_a, res_b, master_ohlcv_df)
                    futures[future] = (res_a, res_b)
            
            for future in as_completed(futures):
                # The result is now a tuple
                portfolio_result_tuple = future.result()
                if portfolio_result_tuple:
                    portfolio_results_with_data.append(portfolio_result_tuple)

        # --- PHASE 3: FINAL RANKING & DURABILITY TESTING ---
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Phase 3: Ranking portfolio results..."}})
        
        portfolio_metric = config['portfolio_metric']
        is_reverse_sort = 'Max_Drawdown' not in portfolio_metric
        
        # Sort using the frontend payload (the first element of the tuple)
        portfolio_results_with_data.sort(
            key=lambda x: x[0]['metrics'].get(portfolio_metric.replace('Portfolio ', ''), -9999),
            reverse=is_reverse_sort
        )

        # The single best hedge combination's data
        best_hedge_payload, best_hedge_ohlcv = portfolio_results_with_data[0]
        
        # Send the best portfolio result's PAYLOAD to the UI
        send_update_to_queue(loop, queue, batch_id, { "type": "strategy_result", "payload": convert_to_json_serializable(best_hedge_payload) })

        final_analysis_config = config.get('final_analysis')
        
        if final_analysis_config and final_analysis_config['type'] != 'none':
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 4: Running {final_analysis_config['type']} on the best hedge portfolio..."}})
            
            # Create the equity_df using the payload and the SEPARATE ohlcv data
            equity_df = pd.DataFrame({
                'timestamp': best_hedge_ohlcv['timestamp'],
                'equity': best_hedge_payload['equity_curve'] # This might need adjustment based on how equity is stored
            })
            # A more robust way if equity_curve is a list of [timestamp, value] pairs:
            equity_curve_data = best_hedge_payload['equity_curve']
            equity_df = pd.DataFrame(equity_curve_data, columns=['timestamp_ms', 'equity'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp_ms'], unit='ms')
            equity_df = equity_df.set_index('timestamp').drop(columns=['timestamp_ms'])


            if final_analysis_config['type'] == 'data_segmentation':
                run_equity_curve_data_segmentation(batch_id, equity_df, final_analysis_config, queue, loop)
            elif final_analysis_config['type'] == 'walk_forward':
                run_equity_curve_walk_forward(batch_id, equity_df, final_analysis_config, queue, loop)
        else:
            # Get the number of results to return from the config
            num_to_return = config.get('num_results_to_return', 50) # Default to 50 if not provided
            
            # If no final analysis, send the other top results for comparison
            for payload, _ in portfolio_results_with_data[1:num_to_return]:
                send_update_to_queue(loop, queue, batch_id, { "type": "strategy_result", "payload": convert_to_json_serializable(payload) })

    except Exception as e:
        print(f"ERROR in Hedge Optimization Manager: {traceback.format_exc()}")
    finally:
        update_job_status(batch_id, "completed", "Hedge optimization finished.")
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Hedge optimization complete."}})


def run_individual_optimization(batch_id, strategy_config, symbol, top_n, metric, manager, queue, loop):
    """
    Runs a full optimization for a single strategy by submitting jobs to the
    standard backtest worker, but with UI updates disabled.
    Returns the raw results of the top N best-performing parameter sets.
    """
    # 1. Generate parameter combinations (unchanged)
    original_code = strategy_config['strategy_code']
    params_to_optimize = strategy_config.get('parameters_to_optimize', [])
    code_with_symbol = re.sub(r"self\.symbol\s*=\s*['\"].*?['\"]", f"self.symbol = '{symbol}'", original_code)
    param_combos = generate_and_prune_combinations(strategy_config)
    
    if not param_combos:
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "WARNING", "message": "No valid parameter combinations found."}})
        return []

    send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Generated {len(param_combos)} parameter sets to test..."}})
    
    # 2. Submit jobs to the standard worker and collect results (much cleaner)
    lightweight_results = [] 
    representative_ohlcv_df = None 
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {}
        for combo in param_combos:
            modifications, display_name = build_modifications_and_name(combo, params_to_optimize, symbol)
            modified_code = modify_strategy_code(code_with_symbol, modifications)
            
            future = executor.submit(
                process_backtest_job,
                batch_id, 
                manager, 
                str(uuid.uuid4()), 
                display_name, 
                modified_code,
                1000, 
                get_client('binance'), 
                queue, 
                loop,
                send_ui_update=False
            )
            
            futures[future] = display_name
        
        for future in as_completed(futures):
            try:
                result_tuple = future.result()
                if result_tuple:
                    strategy_data, instance, metrics, monthly, no_fee, trade_events_df = result_tuple
                    
                    # If we haven't found a timeline yet and this one is valid, grab it.
                    if representative_ohlcv_df is None and not strategy_data['ohlcv'].empty:
                        representative_ohlcv_df = strategy_data['ohlcv'][['timestamp']].copy()
                    
                    # The "lightweight" result now includes the full trade_events_df,
                    # which is small but contains all necessary columns for portfolio calcs.
                    lightweight_tuple = (
                        strategy_data['strategy_name'], # Pass name as a simple string
                        instance,
                        metrics,
                        monthly,
                        no_fee,
                        trade_events_df
                    )
                    lightweight_results.append(lightweight_tuple)
            except Exception as e:
                print(f"!!! Error in individual optimization sub-task for {futures[future]}: {e} !!!")

    if not lightweight_results:
        return [], None

    # 3. Rank the results and return the top N (unchanged)
    LOWER_IS_BETTER_METRICS = {'Max_Drawdown', 'Max_Drawdown_Duration_days'}
    is_reverse_sort = metric not in LOWER_IS_BETTER_METRICS
    
    lightweight_results.sort(
        key=lambda x: x[2].get(metric.replace('Portfolio ', ''), -9999 if is_reverse_sort else 9999),
        reverse=is_reverse_sort
    )
    
    # Return the top N lightweight results AND the single master OHLCV DataFrame
    return lightweight_results[:top_n], representative_ohlcv_df

def combine_and_evaluate_pair(result_a, result_b, base_ohlcv_df):
    """
    Combines two LIGHTWEIGHT backtest results using a master OHLCV timeline.
    This function now uses the same detailed portfolio simulation logic as run_batch_manager.
    """
    # Define names here so they are available in the except block
    name_a = "Strategy A"
    name_b = "Strategy B"
    try:
        # 1. Unpack the lightweight data for both strategies
        name_a, instance_a, _, _, no_fee_equity_array_a, events_a = result_a
        name_b, instance_b, _, _, no_fee_equity_array_b, events_b = result_b

        if events_a.empty and events_b.empty:
            print(f"--- NOTE: No trades found for pair ({name_a}) + ({name_b}). Skipping. ---")
            return None, None

        # --- Step 2: Prepare a Master List of Entry and Exit Events (same as run_batch_manager) ---
        all_entry_events = []
        all_exit_events = []
        all_trade_events_for_log = []
        trade_id_counter = 0
        
        # Process Strategy A's trades
        if not events_a.empty:
            events_a_copy = events_a.copy()
            events_a_copy['trade_id'] = range(trade_id_counter, trade_id_counter + len(events_a_copy))
            trade_id_counter += len(events_a_copy)
            all_trade_events_for_log.append(events_a_copy)
        
        # Process Strategy B's trades, ensuring unique IDs
        if not events_b.empty:
            events_b_copy = events_b.copy()
            events_b_copy['trade_id'] = range(trade_id_counter, trade_id_counter + len(events_b_copy))
            trade_id_counter += len(events_b_copy)
            all_trade_events_for_log.append(events_b_copy)

        for trade_events in all_trade_events_for_log:
            entry_df = trade_events[['timestamp', 'trade_id', 'Signal', 'Entry', 'Stop Loss', 'Take Profit']].copy()
            entry_df.rename(columns={
                'timestamp': 'event_timestamp', 'Signal': 'entry_direction',
                'Entry': 'entry_price', 'Stop Loss': 'entry_sl', 'Take Profit': 'entry_tp'
            }, inplace=True)
            all_entry_events.append(entry_df)

            exit_df = trade_events[['Exit_Time', 'trade_id', 'Result']].copy()
            exit_df.rename(columns={'Exit_Time': 'event_timestamp', 'Result': 'exit_result'}, inplace=True)
            all_exit_events.append(exit_df)

        # --- Step 3: Build the complete timeline DataFrame (same as run_batch_manager) ---
        master_entries = pd.concat(all_entry_events, ignore_index=True) if all_entry_events else pd.DataFrame()
        master_exits = pd.concat(all_exit_events, ignore_index=True) if all_exit_events else pd.DataFrame()

        entries_agg = master_entries.groupby('event_timestamp').agg(list).reset_index() if not master_entries.empty else pd.DataFrame(columns=['event_timestamp'])
        exits_agg = master_exits.groupby('event_timestamp').agg(list).reset_index() if not master_exits.empty else pd.DataFrame(columns=['event_timestamp'])

        complete_df = pd.merge(base_ohlcv_df.copy(), entries_agg, left_on='timestamp', right_on='event_timestamp', how='left')
        complete_df = pd.merge(complete_df, exits_agg, left_on='timestamp', right_on='event_timestamp', how='left', suffixes=('_entry', '_exit'))

        if 'event_timestamp_entry' in complete_df.columns: complete_df.drop(columns=['event_timestamp_entry'], inplace=True)
        if 'event_timestamp_exit' in complete_df.columns: complete_df.drop(columns=['event_timestamp_exit'], inplace=True)

        # --- Step 4: Call the Portfolio Simulator with the new, detailed signature ---
        sample_instance = instance_a  # We just need one instance to call the method
        portfolio_equity, open_trade_count, closed_trades_log = sample_instance.portfolio(
            complete_df['timestamp'].values,
            complete_df['trade_id_entry'].values,
            complete_df['entry_direction'].values,
            complete_df['entry_price'].values,
            complete_df['entry_sl'].values,
            complete_df['entry_tp'].values,
            complete_df['trade_id_exit'].values,
            complete_df['exit_result'].values
        )

        # --- Step 5: Process Results with the new logic (same as run_batch_manager) ---
        aligned_equity = pd.Series(portfolio_equity, index=complete_df.index)
        complete_df['Portfolio_Open_Trades'] = open_trade_count
        
        all_trade_events_df = pd.concat(all_trade_events_for_log, ignore_index=True) if all_trade_events_for_log else pd.DataFrame()
        
        if closed_trades_log:
            returns_df = pd.DataFrame(closed_trades_log)
            all_trade_events_df = pd.merge(all_trade_events_df, returns_df, on='trade_id', how='left')
            all_trade_events_df['Returns'] = all_trade_events_df['final_gross_return'].round(2)
            all_trade_events_df['Commissioned Returns'] = all_trade_events_df['final_net_return'].round(2)
            all_trade_events_df.drop(columns=['final_gross_return', 'final_net_return'], inplace=True)

        # Update the 'Open_Trades' column for the final trade log
        if not all_trade_events_df.empty:
            open_trades_lookup = complete_df[['timestamp', 'Portfolio_Open_Trades']]
            all_trade_events_df = pd.merge(all_trade_events_df, open_trades_lookup, on='timestamp', how='left')
            all_trade_events_df.drop(columns=['Open_Trades'], inplace=True, errors='ignore')
            all_trade_events_df.rename(columns={'Portfolio_Open_Trades': 'Open_Trades'}, inplace=True)
            
        portfolio_name = f"Hedge: ({name_a}) + ({name_b})"
        portfolio_strategy_data = {
            'strategy_name': portfolio_name,
            'equity': aligned_equity,
            'ohlcv': complete_df,
            'signals': all_trade_events_df
        }

        # --- Step 6: Recalculate portfolio-level commission ---
        no_fee_equity_gain_a = no_fee_equity_array_a[-1] - no_fee_equity_array_a[0] if len(no_fee_equity_array_a) > 1 else 0
        no_fee_equity_gain_b = no_fee_equity_array_b[-1] - no_fee_equity_array_b[0] if len(no_fee_equity_array_b) > 1 else 0
        total_no_fee_equity_gain = no_fee_equity_gain_a + no_fee_equity_gain_b
        
        portfolio_net_profit = aligned_equity.iloc[-1] - aligned_equity.iloc[0]

        if total_no_fee_equity_gain > 0:
            commission = round(((total_no_fee_equity_gain - portfolio_net_profit) * 100) / total_no_fee_equity_gain, 2)
        else:
            commission = 0.0

        # --- Step 7: Recalculate metrics and prepare final payload ---
        portfolio_metrics, portfolio_returns = calculate_metrics(portfolio_strategy_data, 1000, commission)
        frontend_payload = prepare_strategy_payload(portfolio_strategy_data, portfolio_metrics, portfolio_returns)

        # 8. Return both the payload and the base OHLCV for potential backend use
        return frontend_payload, complete_df

    except Exception as e:
        # It's good practice to handle errors within the worker
        print(f"!!! ERROR combining pair: {name_a} + {name_b} !!!")
        print(traceback.format_exc())
        return None, None # Return None to indicate failure

# --- NEW HELPER FUNCTIONS FOR DURABILITY TESTING ON AN EQUITY CURVE ---

def run_equity_curve_data_segmentation(batch_id, equity_df, config, queue, loop):
    """Takes a combined equity curve and runs a simple Train/Test split on it."""
    total_rows = len(equity_df)
    train_end_idx = int(total_rows * (config['training_pct'] / 100))
    
    train_equity = equity_df.iloc[:train_end_idx]
    test_equity = equity_df.iloc[train_end_idx:]
    
    initial_capital = equity_df['equity'].iloc[0]
    
    # Calculate and send metrics for each segment
    train_metrics = calculate_metrics_from_equity(train_equity, 'Hedge Portfolio [Training]', initial_capital=initial_capital)
    test_metrics = calculate_metrics_from_equity(test_equity, 'Hedge Portfolio [Testing]', initial_capital=initial_capital)
    
    send_update_to_queue(loop, queue, batch_id, { "type": "strategy_result", "payload": convert_to_json_serializable(train_metrics) })
    send_update_to_queue(loop, queue, batch_id, { "type": "strategy_result", "payload": convert_to_json_serializable(test_metrics) })

def run_equity_curve_walk_forward(batch_id, equity_df, config, queue, loop):
    """
    Takes a combined equity curve and runs a Walk-Forward analysis on it by
    calculating metrics on rolling out-of-sample windows.
    """
    try:
        # --- Config Extraction ---
        train_delta = pd.Timedelta(**{config['training_period_unit']: config['training_period_length']})
        test_delta = pd.Timedelta(**{config['testing_period_unit']: config['testing_period_length']})
        step_delta = test_delta * (config['step_forward_size_pct'] / 100)
        
        initial_capital = equity_df['equity'].iloc[0]

        # --- The Walk-Forward Loop ---
        window_start_date = equity_df.index.min()
        
        iteration = 1
        while True:
            # Define current window boundaries
            train_end = window_start_date + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > equity_df.index.max():
                break # Stop if the next test period goes beyond our data

            # Slice the equity curve for the out-of-sample (testing) period
            out_of_sample_equity_slice = equity_df[(equity_df.index >= test_start) & (equity_df.index < test_end)]

            if out_of_sample_equity_slice.empty:
                window_start_date += step_delta
                iteration += 1
                continue

            # Calculate metrics for JUST this OOS slice
            oos_period_name = f"WFA Period {iteration} (OOS)"
            oos_metrics_payload = calculate_metrics_from_equity(out_of_sample_equity_slice, oos_period_name, initial_capital)

            if oos_metrics_payload:
                send_update_to_queue(loop, queue, batch_id, {
                    "type": "strategy_result",
                    "payload": convert_to_json_serializable(oos_metrics_payload)
                })

            # Advance the window for the next iteration
            window_start_date += step_delta
            iteration += 1

        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Walk-Forward analysis on the best pair is complete."}})

    except Exception as e:
        error_msg = f"Equity Curve Walk-Forward failed: {traceback.format_exc()}"
        print(error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})

def calculate_metrics_from_equity(equity_df: pd.DataFrame, name: str, initial_capital: int):
    """
    Calculates performance metrics directly from a pandas Series of equity values
    with a DatetimeIndex. Returns a dictionary in the same format as prepare_strategy_payload.
    """
    if equity_df.empty or len(equity_df) < 2:
        return None

    equity_array = equity_df['equity'].to_numpy()
    
    # --- Basic Metrics ---
    net_profit = equity_array[-1] - equity_array[0]
    profit_percentage = (net_profit / equity_array[0]) * 100 if equity_array[0] != 0 else 0

    # --- Drawdown Calculation ---
    peaks = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - peaks) / peaks
    max_drawdown = np.min(drawdowns) * 100 if np.all(peaks > 0) else 0

    # --- Sharpe Ratio Calculation (Annualized from daily data) ---
    daily_returns = equity_df['equity'].pct_change().dropna()
    if not daily_returns.empty:
        # Assuming 252 trading days in a year
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # --- Monthly Returns ---
    # Resample equity to monthly, get the last day's value, then calculate percentage change
    monthly_equity = equity_df['equity'].resample('M').last()
    monthly_pct_returns = monthly_equity.pct_change().dropna()
    
    monthly_returns_data = []
    for timestamp, pct_return in monthly_pct_returns.items():
        month_name = timestamp.strftime('%b %Y')
        monthly_returns_data.append({
            "Month": month_name,
            "Returns (%)": pct_return * 100,
        })
    
    # --- Assemble the metrics dictionary (simplified version) ---
    metrics = {
        "Net_Profit": net_profit,
        "Profit_Percentage": round(profit_percentage, 2),
        "Max_Drawdown": round(abs(max_drawdown), 2),
        "Sharpe_Ratio": round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else 0.0,
        # Add other relevant metrics here if needed (e.g., Calmar)
    }

    # --- Prepare the payload in the standard format for the frontend ---
    timestamps_ms = (equity_df.index.astype(np.int64) // 10**6).tolist()
    equity_values = equity_array.tolist()
    
    payload = {
        "strategy_name": name,
        "equity_curve": list(zip(timestamps_ms, equity_values)),
        "metrics": metrics,
        "trades": [], # No individual trades when analyzing a combined curve
        "monthly_returns": monthly_returns_data
    }
    return payload