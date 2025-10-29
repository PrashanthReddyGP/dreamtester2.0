# backend/worker.py
import time
import os
import re
import ast
import traceback
import numpy as np
import itertools

# IMPORTANT: This worker runs in a separate process. It cannot access
# the FastAPI app state directly. It communicates ONLY through queues.

# Assume your core logic is importable
from core.connect_to_brokerage import get_client
from core.fetch_ohlcv_data import fetch_candlestick_data
from core.parse_data import parse_file
from core.process_indicators import calculate_indicators
from core.run_backtest import run_backtest_loop
from core.metrics import calculate_metrics

# --- Your helper functions can be moved here or imported ---
def modify_strategy_code(original_code, modifications):
    # ... logic from previous answers
    pass

def prepare_strategy_payload(strategy_data, metrics, monthly_returns):
    # ... logic from previous answers
    pass

def convert_to_json_serializable(obj):
    # ... logic from previous answers
    pass

def run_single_backtest_logic(display_name, code_str, capital):
    """
    This function contains the pure logic of a single backtest.
    It does NOT interact with queues. It just returns the result or raises an error.
    """
    sanitized_name = re.sub(r'[ |/\\:*?"<>|=]', '_', display_name)
    job_id = "local_job" # Job ID is less critical here

    # This is your core backtest pipeline
    client = get_client('binance', market_type='future')
    strategy_instance, _ = parse_file(job_id, sanitized_name, code_str)
    _, ohlcv = fetch_candlestick_data(client, strategy_instance)
    ohlcv_idk = calculate_indicators(strategy_instance, ohlcv)
    strategy_data, commission, _ = run_backtest_loop(display_name, strategy_instance, ohlcv_idk, capital)
    metrics, monthly_returns = calculate_metrics(strategy_data, capital, commission)
    
    # Prepare the final payload
    full_payload = prepare_strategy_payload(strategy_data, metrics, monthly_returns)
    full_payload['strategy_name'] = display_name
    return convert_to_json_serializable(full_payload)


def worker_process_loop(job_queue, result_queue):
    """

    The main loop for each worker process.
    It continuously pulls jobs from the job_queue, executes them,
    and puts the results into the result_queue.
    """
    print(f"--- Worker process {os.getpid()} started ---")
    while True:
        try:
            # This will block until a job is available
            batch_id, job_config = job_queue.get()
            
            # Use a "poison pill" to gracefully shut down the worker
            if job_config is None:
                print(f"--- Worker process {os.getpid()} shutting down ---")
                break

            # Send a log message back to the main process via the result queue
            log_message = {"type": "log", "payload": {"message": f"Running: {job_config['display_name']}"}}
            result_queue.put((batch_id, log_message))
            
            # Execute the backtest
            result_payload = run_single_backtest_logic(
                job_config['display_name'], 
                job_config['code_str'], 
                job_config['capital']
            )

            # Send the result message back
            result_message = {"type": "strategy_result", "payload": result_payload}
            result_queue.put((batch_id, result_message))

        except Exception as e:
            # If an error occurs, send an error message back
            print(f"!!! WORKER ERROR in {os.getpid()}: {traceback.format_exc()}")
            error_message = {"type": "error", "payload": {"message": f"Error in worker: {e}"}}
            result_queue.put((batch_id, error_message))