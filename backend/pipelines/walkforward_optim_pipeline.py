import pandas as pd
from datetime import timedelta
import re
import uuid
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import traceback

from core.metrics import calculate_metrics
from pipeline import send_update_to_queue, parse_file, get_client, fetch_candlestick_data, generate_and_prune_combinations, universal_backtest_worker, modify_strategy_code, build_modifications_and_name, update_job_status, fail_job


def run_walk_forward_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Manages a Walk-Forward Analysis test.
    1. Fetches full data for the primary symbol.
    2. Iteratively slices data into expanding/rolling Training and Testing windows.
    3. In each "walk", it optimizes on Training data to find the best parameter set.
    4. Runs that single best set on the subsequent unseen Testing data.
    5. Aggregates all out-of-sample (testing) results into a final performance report.
    """
    try:
        send_update_to_queue(loop, queue, batch_id, {
            "type": "batch_info", 
            "payload": { "config": config }
        })
        
        print(f"--- WALK-FORWARD ({batch_id}): Starting job ---")
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "Starting Walk-Forward Analysis..."}})
        
        # --- 1. Initial Setup & Data Fetching ---
        original_code = config['strategy_code']
        primary_symbol = config.get('symbols_to_screen', [])[0]
        optimization_metric = config['optimization_metric']
        
        temp_code = re.sub(r"self\.symbol\s*=\s*['\"].*?['\"]", f"self.symbol = '{primary_symbol}'", original_code)
        strategy_instance, _ = parse_file(str(uuid.uuid4()), "temp", temp_code)
        client = get_client('binance')
        _, full_ohlcv_df = fetch_candlestick_data(client, strategy_instance)
        
        if 'timestamp' not in full_ohlcv_df.columns:
            raise ValueError("OHLCV DataFrame must have a 'timestamp' column.")
            
        full_ohlcv_df['timestamp'] = pd.to_datetime(full_ohlcv_df['timestamp'], unit='ms')
        
        # --- 2. Define Walk-Forward Windows and Parameters ---
        def get_timedelta(length, unit):
            if unit == 'days': return timedelta(days=length)
            if unit == 'weeks': return timedelta(weeks=length)
            if unit == 'months': return timedelta(days=length * 30) # Approx
            raise ValueError(f"Invalid time unit: {unit}")
        
        train_delta = get_timedelta(config['training_period_length'], config['training_period_unit'])
        test_delta = get_timedelta(config['testing_period_length'], config['testing_period_unit'])
        step_delta = test_delta * (config['step_forward_pct'] / 100)
        is_anchored = config['is_anchored']
        
        start_date = full_ohlcv_df['timestamp'].min()
        end_date = full_ohlcv_df['timestamp'].max()
        
        all_out_of_sample_results = []
        
        # --- 3. Main Walk-Forward Loop ---
        current_train_start = start_date
        current_train_end = start_date + train_delta
        fold_num = 1
        
        while current_train_end + test_delta <= end_date:
            current_test_start = current_train_end
            current_test_end = current_train_end + test_delta
            
            # Slice DataFrames for the current fold
            train_df = full_ohlcv_df[(full_ohlcv_df['timestamp'] >= current_train_start) & (full_ohlcv_df['timestamp'] < current_train_end)].reset_index(drop=True)
            test_df = full_ohlcv_df[(full_ohlcv_df['timestamp'] >= current_test_start) & (full_ohlcv_df['timestamp'] < current_test_end)].reset_index(drop=True)
            
            if train_df.empty or test_df.empty:
                print(f"Fold {fold_num}: Skipping due to empty data slice.")
                # Advance windows to the next step
                current_train_end += step_delta
                if not is_anchored:
                    current_train_start += step_delta
                fold_num += 1
                continue
            
            log_msg = (f"--- FOLD {fold_num} --- \n"
                        f"  Training: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()} ({len(train_df)} rows)\n"
                        f"  Testing:  {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()} ({len(test_df)} rows)")
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": log_msg}})
            
            # --- 3a. Optimize on Training Data ---
            training_results = []
            param_combos = generate_and_prune_combinations(config)
            if not param_combos: raise ValueError("No valid parameter combinations.")
            
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                futures = {
                    executor.submit(
                        universal_backtest_worker,
                        batch_id, manager, f"Fold {fold_num} Train", 
                        modify_strategy_code(temp_code, build_modifications_and_name(c, config['parameters_to_optimize'], primary_symbol)[0]),
                        1000, client, queue, loop, train_df.copy(), is_verbose=False # Quieter logs for inner runs
                    ): c for c in param_combos
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        training_results.append(result)
                        
            if not training_results:
                send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"level": "WARNING", "message": f"Fold {fold_num}: No profitable results in training period. Skipping."}})
                # Advance windows and continue to next fold
                current_train_end += step_delta
                if not is_anchored:
                    current_train_start += step_delta
                fold_num += 1
                continue
                
            # --- 3b. Find Best Parameter Set ---
            is_reverse_sort = optimization_metric not in ['Max_Drawdown', 'Max_Drawdown_Duration']
            training_results.sort(key=lambda x: x[2].get(optimization_metric, -9999), reverse=is_reverse_sort)
            best_training_run = training_results[0]
            best_params_instance = best_training_run[1] # The strategy instance holds the best params
            
            _, display_name = build_modifications_and_name(best_params_instance.get_params_tuple(), config['parameters_to_optimize'], '')
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Fold {fold_num}: Best params found: {display_name.strip(' | ')}"}})
            
            # --- 3c. Run Best Set on Testing (Out-of-Sample) Data ---
            # Re-create the code with the single best set of parameters
            modifications, _ = build_modifications_and_name(best_params_instance.get_params_tuple(), config['parameters_to_optimize'], primary_symbol)
            best_code = modify_strategy_code(temp_code, modifications)
            
            # Run the single backtest
            oos_result = universal_backtest_worker(
                batch_id, manager, f"Fold {fold_num} Test [OOS]", best_code,
                1000, client, queue, loop, test_df.copy(), is_verbose=True # Verbose for the important OOS run
            )
            
            if oos_result:
                all_out_of_sample_results.append(oos_result)
                
            # --- 3d. Advance the time windows for the next loop iteration ---
            current_train_end += step_delta
            if not is_anchored:
                current_train_start += step_delta
            fold_num += 1
            
        # --- 4. Final Aggregation and Reporting ---
        if not all_out_of_sample_results:
            raise ValueError("Walk-Forward Analysis produced no out-of-sample results.")
        
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": "All folds complete. Aggregating out-of-sample results..."}})
        
        # This is a for reporting function which will:
        # 1. Stitch together the trade logs from each `oos_result`.
        # 2. Recalculate portfolio metrics on the combined equity curve.
        # 3. Send the final combined report to the frontend.
        combine_walk_forward_results(batch_id, all_out_of_sample_results, queue, loop)
        
    except Exception as e:
        error_msg = f"Walk-Forward Analysis failed: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})
    finally:
        update_job_status(batch_id, "completed", "Walk-Forward Analysis finished.")
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Walk-Forward Analysis complete."}})
        print(f"--- WALK-FORWARD ({batch_id}): Job finished. ---")


def combine_walk_forward_results(batch_id, oos_results, queue, loop):
    """
    Combines the results from individual out-of-sample (OOS) folds
    into a single, continuous performance report.
    
    Args:
        batch_id (str): The unique ID for this job.
        oos_results (list): A list of result tuples from universal_backtest_worker for each OOS period.
        manager: The WebSocket connection manager.
        queue: The asyncio queue for sending WebSocket messages.
        loop: The asyncio event loop.
    """
    if not oos_results:
        print(f"[{batch_id}] No out-of-sample results to combine.")
        return
    
    # --- Step 1: Combine all individual trade logs into one DataFrame ---
    all_trades = pd.concat([result[0]['trades'] for result in oos_results if not result[0]['trades'].empty])
    all_trades.sort_values(by='entry_time', inplace=True)
    all_trades.reset_index(drop=True, inplace=True)
    
    # --- Step 2: Stitch together the equity curves ---
    # This is the most important part. We create a continuous equity curve by
    # adjusting the start of each new curve to match the end of the previous one.
    combined_equity_curve = pd.DataFrame()
    
    # Start with the initial capital from the very first backtest run.
    initial_capital = oos_results[0][1].capital 
    last_equity = initial_capital
    
    for result in oos_results:
        # result[0] is the 'strategy_data' dictionary
        equity_df = result[0]['equity_curve'].copy()
        if not equity_df.empty:
            # Adjust the equity of the current fold based on the end of the last one
            equity_df['equity'] = equity_df['equity'] - equity_df['equity'].iloc[0] + last_equity
            last_equity = equity_df['equity'].iloc[-1]
            combined_equity_curve = pd.concat([combined_equity_curve, equity_df])
    
    combined_equity_curve.reset_index(drop=True, inplace=True)
    
    # --- Step 3: Recalculate performance metrics on the COMBINED results ---
    # This gives you the true performance of the walk-forward process.
    final_metrics = calculate_metrics(
        trades=all_trades,
        equity_curve=combined_equity_curve,
        initial_capital=initial_capital
    )
    
    # --- Step 4: Prepare the final payload and send it to the frontend ---
    final_instance = oos_results[-1][1] # Use the last instance as a representative sample
    
    # Sanitize the DataFrames for JSON serialization
    final_trades_json = all_trades.astype(object).where(pd.notnull(all_trades), None).to_dict(orient='records')
    final_equity_json = combined_equity_curve.astype(object).where(pd.notnull(combined_equity_curve), None).to_dict(orient='records')
    
    final_strategy_data = {
        'trades': final_trades_json,
        'equity_curve': final_equity_json
    }
    
    # Send the final, combined result as a single "backtest_result" message
    send_update_to_queue(loop, queue, batch_id, {
        "type": "backtest_result",
        "payload": {
            "strategy_data": final_strategy_data,
            "strategy_name": f"{final_instance.name} [Walk-Forward OOS]",
            "metrics": final_metrics,
            "logs": ["This is the combined out-of-sample performance from all walk-forward folds."],
            "is_final_report": True # A special flag for the UI to identify this as the main result
        }
    })
    
    print(f"[{batch_id}] Successfully combined walk-forward results and sent final report.")