# backend/pipeline.py
import os
import re
import uuid
import asyncio
import traceback
from enum import Enum
from itertools import product
import numpy as np
import pandas as pd
from datetime import time, timedelta 

from concurrent.futures import ThreadPoolExecutor, as_completed

from database import update_job_status, fail_job
from core.metrics import calculate_metrics
from core.connect_to_brokerage import get_client
from core.basestrategy import BaseStrategy

from pipeline import generate_and_prune_combinations, build_modifications_and_name, modify_strategy_code, process_backtest_job, send_update_to_queue, parse_file, fetch_candlestick_data, convert_to_json_serializable, prepare_strategy_payload
from core.params_heatmap import generate_parameter_heatmap

# This helper is fine as is, no changes needed.
def _run_single_backtest(batch_id, manager, queue, loop, client, strategy_config, symbol, combo, data_df, send_ui_update=False):
    params_to_optimize = strategy_config.get('parameters_to_optimize', [])
    original_code = strategy_config['strategy_code']
    modifications, display_name = build_modifications_and_name(combo, params_to_optimize, symbol)
    modified_code = modify_strategy_code(original_code, modifications)
    
    return process_backtest_job(
        batch_id, manager, str(uuid.uuid4()), display_name, modified_code,
        1000, client, queue, loop, 
        send_ui_update=send_ui_update, ohlcv_df=data_df
    )

# The main manager now uses the corrected data flow
def run_hedge_optimization_manager(batch_id: str, config: dict, manager: any, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, connection_event: asyncio.Event):
    """
    Manages a Hedge Optimization test with proper data segmentation.
    This version is surgically corrected to match the data flow of the old, working manager.
    """
    try:
        async def wait_for_connection():
            await connection_event.wait()
        future = asyncio.run_coroutine_threadsafe(wait_for_connection(), loop)
        future.result()
        
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_info", "payload": {"config": config}})
        client = get_client('binance')
        primary_symbol = config['symbols_to_screen'][0]
        final_analysis_config = config.get('final_analysis', {})
        use_segmentation = final_analysis_config.get('type') == 'data_segmentation'
        num_to_return = config.get('num_results_to_return', 10)

        # Phase 0: Data Prep
        train_dfs, test_dfs = {}, {}
        for s_key in ['a', 'b']:
            strategy_config = config[f'strategy_{s_key}']
            temp_code = re.sub(r"self\.symbol\s*=\s*['\"].*?['\"]", f"self.symbol = '{primary_symbol}'", strategy_config['strategy_code'])
            instance, _ = parse_file(str(uuid.uuid4()), "temp", temp_code)
            _, ohlcv_df = fetch_candlestick_data(client, instance, ticks=False)
            
            if instance.ticks and ticks_df is None:
                _, ticks_df = fetch_candlestick_data(client, instance, ticks=True)
            else:
                ticks_df = ticks_df
            
            if use_segmentation:
                total_rows = len(ohlcv_df)
                train_end_idx = int(total_rows * (final_analysis_config['training_pct'] / 100))
                train_dfs[s_key] = ohlcv_df.iloc[:train_end_idx].reset_index(drop=True)
                test_dfs[s_key] = ohlcv_df.iloc[train_end_idx:].reset_index(drop=True)
            else:
                train_dfs[s_key] = ohlcv_df

        # Phase 1: Individual Optimization
        top_n_results_with_combos = {}
        ohlcv_dfs = {}
        for s_key in ['a', 'b']:
            log_phase = "training data" if use_segmentation else "entire data"
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 1: Optimizing Strategy {s_key.upper()} on {log_phase}..."}})
            
            results_with_combos, ohlcv_df = _run_single_strategy_optimization_with_combos(
                batch_id, manager, queue, loop, client,
                strategy_config=config[f'strategy_{s_key}'], symbol=primary_symbol, 
                data_df=train_dfs[s_key].copy(), top_n=config['top_n_candidates'], 
                metric=config['portfolio_metric'],
                strategy_key=s_key
            )
            
            if not results_with_combos: raise ValueError(f"Optimization for Strategy {s_key.upper()} yielded no results.")
            top_n_results_with_combos[s_key] = results_with_combos
            ohlcv_dfs[s_key] = ohlcv_df

        # Phase 2: Pairwise Combination
        log_phase = "training data" if use_segmentation else "entire data"
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 2: Combining and evaluating pairs on {log_phase}..."}})
        
        if ohlcv_dfs['a'] is None or ohlcv_dfs['b'] is None: raise ValueError("Could not determine a representative OHLCV timeline.")
        
        master_train_timestamps = np.sort(pd.concat([ohlcv_dfs['a']['timestamp'], ohlcv_dfs['b']['timestamp']]).unique())
        master_train_ohlcv_df = pd.DataFrame({'timestamp': master_train_timestamps})
        
        training_portfolios_with_data = []
        with ThreadPoolExecutor() as executor:
            futures = {}
            for res_a, combo_a in top_n_results_with_combos['a']:
                for res_b, combo_b in top_n_results_with_combos['b']:
                    future = executor.submit(combine_and_evaluate_pair, res_a, res_b, master_train_ohlcv_df.copy())
                    futures[future] = (combo_a, combo_b) # Store the actual combo tuples
            
            for future in as_completed(futures):
                portfolio_result_tuple = future.result()
                if portfolio_result_tuple:
                    combos = futures[future]
                    # Store the clean payload along with the actual combo tuples that created it
                    training_portfolios_with_data.append( (portfolio_result_tuple[0], combos[0], combos[1]) )

        # Phase 3: Ranking and Reporting Training Results
        send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 3: Ranking portfolios from {log_phase}..."}})
        if not training_portfolios_with_data: raise ValueError("Combining pairs yielded no portfolios.")
        
        portfolio_metric = config['portfolio_metric']
        is_reverse_sort = 'Max_Drawdown' not in portfolio_metric
        metric_key = portfolio_metric.replace('Portfolio ', '')
        training_portfolios_with_data.sort(
            key=lambda x: x[0]['metrics'].get(metric_key, -9999 if is_reverse_sort else 9999),
            reverse=is_reverse_sort
        )
        
        for payload, _, _ in training_portfolios_with_data[:num_to_return]:
            display_payload = payload.copy()
            if use_segmentation: display_payload['strategy_name'] += " [Training]"
            send_update_to_queue(loop, queue, batch_id, {"type": "strategy_result", "payload": convert_to_json_serializable(display_payload)})
        
        # --- PHASE 4: FINAL EVALUATION ON UNSEEN TEST DATA (TOP N) ---
        if use_segmentation:
            send_update_to_queue(loop, queue, batch_id, {"type": "log", "payload": {"message": f"Phase 4: Running top {num_to_return} portfolios on unseen Test data..."}})
            
            top_n_portfolios_to_test = training_portfolios_with_data[:num_to_return]

            # The loop structure was correct, the data passed to it was the issue.
            for _, combo_a, combo_b in top_n_portfolios_to_test:
                # Debugging print to confirm you have the right combos
                print(f"--- TESTING COMBO: A={combo_a}, B={combo_b} ---")

                raw_test_result_a = _run_single_backtest(batch_id, manager, queue, loop, client, config['strategy_a'], primary_symbol, combo_a, test_dfs['a'].copy())
                raw_test_result_b = _run_single_backtest(batch_id, manager, queue, loop, client, config['strategy_b'], primary_symbol, combo_b, test_dfs['b'].copy())
                
                if raw_test_result_a and raw_test_result_b:
                    lightweight_tuple_a = (raw_test_result_a[0]['strategy_name'], raw_test_result_a[1], raw_test_result_a[2], raw_test_result_a[3], raw_test_result_a[4], raw_test_result_a[5])
                    lightweight_tuple_b = (raw_test_result_b[0]['strategy_name'], raw_test_result_b[1], raw_test_result_b[2], raw_test_result_b[3], raw_test_result_b[4], raw_test_result_b[5])
                    
                    master_test_timestamps = np.sort(pd.concat([test_dfs['a']['timestamp'], test_dfs['b']['timestamp']]).unique())
                    master_test_ohlcv_df = pd.DataFrame({'timestamp': master_test_timestamps})
                    
                    final_payload, _ = combine_and_evaluate_pair(lightweight_tuple_a, lightweight_tuple_b, master_test_ohlcv_df.copy())
                    
                    if final_payload:
                        final_payload['strategy_name'] += " [Testing]"
                        send_update_to_queue(loop, queue, batch_id, {"type": "strategy_result", "payload": convert_to_json_serializable(final_payload)})
        
    except Exception as e:
        error_msg = f"ERROR in Hedge Optimization Manager: {traceback.format_exc()}"
        print(error_msg)
        fail_job(batch_id, error_msg)
        send_update_to_queue(loop, queue, batch_id, {"type": "error", "payload": {"message": str(e)}})
    finally:
        update_job_status(batch_id, "completed", "Hedge optimization finished.")
        send_update_to_queue(loop, queue, batch_id, {"type": "batch_complete", "payload": {"message": "Hedge optimization complete."}})

# --- We need a new helper that returns combos as well ---
def _run_single_strategy_optimization_with_combos(batch_id, manager, queue, loop, client, strategy_config, symbol, data_df, top_n, metric, strategy_key):
    """ This is a new helper that returns combos along with the results. """
    
    param_combos = generate_and_prune_combinations(strategy_config)
    if not param_combos: return [], None
    
    results_with_combos = []
    representative_ohlcv_df = None
    
    all_results_for_heatmap = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        
        futures = {
            executor.submit(
                _run_single_backtest, batch_id, manager, queue, loop, client, 
                strategy_config, symbol, combo, data_df.copy(), send_ui_update=False
            ): combo for combo in param_combos
        }
        
        for future in as_completed(futures):
            
            result_tuple = future.result()
            
            if result_tuple:
                strategy_data = result_tuple[0]
                
                metrics = result_tuple[2]
                combo = futures[future]
                
                # Collect data for the heatmap
                metric_value = metrics.get(metric)
                
                if metric_value is not None:
                    all_results_for_heatmap.append((combo, metric_value))
                
                if representative_ohlcv_df is None and not strategy_data['ohlcv'].empty:
                    representative_ohlcv_df = strategy_data['ohlcv'][['timestamp']].copy()
                
                lightweight_tuple = (
                    strategy_data['strategy_name'], # Pass name as a simple string
                    result_tuple[1],
                    metrics,
                    result_tuple[3],
                    result_tuple[4],
                    result_tuple[5]
                )
                
                combo = futures[future]
                
                results_with_combos.append((lightweight_tuple, combo))
    
    if all_results_for_heatmap:
        generate_parameter_heatmap(
            results_data=all_results_for_heatmap,
            params_to_optimize=strategy_config.get('parameters_to_optimize', []),
            metric=metric,
            batch_id=batch_id,
            strategy_key=strategy_key
        )
    
    if not results_with_combos: return [], None
    
    print(f"--- Calculating robustness scores for Strategy {strategy_key.upper()}... ---")
    
    # 1. Calculate robustness scores for all completed backtests
    # The format of `results_with_combos` is [(lightweight_tuple, combo), ...]
    # The lightweight_tuple is (name, instance, metrics_dict, ...)
    robustness_data = calculate_robustness_for_all_candidates(
        results_with_combos,
        strategy_config.get('parameters_to_optimize', [])
    )
    
    if not robustness_data:
        print("--- WARNING: Could not calculate robustness scores. Falling back to simple sort. ---")
        # Fallback to the old sorting method if scoring fails
        is_reverse_sort = 'Max_Drawdown' not in metric
        results_with_combos.sort(key=lambda x: x[0][2].get(metric, -9999 if is_reverse_sort else 9999), reverse=is_reverse_sort)
        return results_with_combos[:top_n], representative_ohlcv_df

    # 2. Sort by the new robustness score (descending)
    robustness_data.sort(key=lambda x: x[2], reverse=True)

    # 3. Get the combos of the top N most robust candidates
    top_robust_combos = [item[0] for item in robustness_data[:top_n]]

    # 4. Filter the original `results_with_combos` to only include these top robust candidates
    # We maintain the original data structure that the rest of the pipeline expects.
    final_candidates = [res for res in results_with_combos if res[1] in top_robust_combos]
    
    is_reverse_sort = 'Max_Drawdown' not in metric
    
    final_candidates.sort(key=lambda x: x[0][2].get(metric, -9999 if is_reverse_sort else 9999), reverse=is_reverse_sort)
    
    print(f"--- Top robust candidates for Strategy {strategy_key.upper()}: {[c[1] for c in final_candidates]} ---")
    
    return final_candidates, representative_ohlcv_df


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
            entry_df = trade_events[['timestamp', 'trade_id', 'Signal', 'Entry', 'Stop Loss', 'Take Profit', 'Risk']].copy()
            entry_df.rename(columns={
                'timestamp': 'event_timestamp', 'Signal': 'entry_direction',
                'Entry': 'entry_price', 'Stop Loss': 'entry_sl', 'Take Profit': 'entry_tp', 'Risk': 'risk_percent'
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
            complete_df['risk_percent'].values,
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
    

# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import TwoSlopeNorm

# def generate_parameter_heatmap(results_data, params_to_optimize, metric, batch_id, strategy_key):
#     """
#     Generates and saves a heatmap grid.
#     - 2 params: single 2D heatmap
#     - 3 params: single row of heatmaps (1D facet)
#     - 4 params: a 2D grid of heatmaps (2D facet grid)
#     """
#     if not results_data:
#         print("--- NOTE: No results data to generate a heatmap. ---")
#         return

#     param_names = [p['name'] for p in params_to_optimize]
#     num_params = len(param_names)

#     if num_params < 2:
#         print("--- NOTE: Not enough parameters (<2) to generate a heatmap. ---")
#         return

#     # --- Data Preparation (same as before) ---
#     records = []
#     for combo_tuple, metric_value in results_data:
#         record = dict(zip(param_names, combo_tuple))
#         record[metric] = metric_value
#         records.append(record)
    
#     if not records:
#         print("--- NOTE: No valid records to create a heatmap. ---")
#         return
        
#     df = pd.DataFrame(records)
#     # 1. Find the true min and max values across all results for the metric
#     if df.empty or metric not in df.columns:
#         print("--- WARNING: Metric column not found or DataFrame is empty. Cannot generate heatmap. ---")
#         return
        
#     global_min = df[metric].min()
#     global_max = df[metric].max()
    
#     # 2. Handle edge cases where all data is on one side of zero
#     if global_min >= 0: global_min = 0 # If no losses, anchor min at 0
#     if global_max <= 0: global_max = 0 # If no profits, anchor max at 0
    
#     # If there's no range, create a small default one to avoid errors
#     if global_min == global_max:
#         global_min -= 1
#         global_max += 1

#     # 3. Create the TwoSlopeNorm normalizer object
#     # This is the core of the fix. It maps the colors based on the actual data range
#     # on either side of the center point (0).
#     norm = TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)

#     print(f"--- Heatmap ASYMMETRIC color scale set for '{metric}': Min={global_min:.2f}, Center=0, Max={global_max:.2f} ---")
    
#     # Assign roles based on parameter order
#     x_param = param_names[0]
#     y_param = param_names[1]
#     col_facet_param = param_names[2] if num_params >= 3 else None
#     row_facet_param = param_names[3] if num_params >= 4 else None
    
#     # Get unique values for faceting and ensure they are sorted correctly
#     col_values = [str(v) for v in sorted(df[col_facet_param].unique())] if col_facet_param else [None]
#     row_values = [str(v) for v in sorted(df[row_facet_param].unique())] if row_facet_param else [None]
    
#     # Determine grid size
#     n_cols = len(col_values)
#     n_rows = len(row_values)
    
#     # <<< DYNAMIC FIGSIZE ADJUSTMENT >>>
#     # Adjust base size based on density to keep plots readable
#     base_width_per_plot = 7
#     base_height_per_plot = 6
    
#     # Reduce size slightly for very large grids to prevent huge images
#     if n_cols > 3:
#         base_width_per_plot = 6
#     if n_rows > 3:
#         base_height_per_plot = 5
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * base_width_per_plot, n_rows * base_height_per_plot), squeeze=False)
    
#     # Iterate through the 2D grid of facets
#     for i, row_val in enumerate(row_values):
#         for j, col_val in enumerate(col_values):
#             ax = axes[i, j]
            
#             # Filter data for the current subplot
#             subset_df = df.copy()
#             title_parts = []
            
#             if row_facet_param:
#                 subset_df = subset_df[subset_df[row_facet_param].astype(str) == row_val]
#                 title_parts.append(f"{row_facet_param} = {row_val}")
#             if col_facet_param:
#                 subset_df = subset_df[subset_df[col_facet_param].astype(str) == col_val]
#                 title_parts.append(f"{col_facet_param} = {col_val}")
            
#             ax.set_title(" | ".join(title_parts))
            
#             # Check if there's enough data to form a 2D grid for the heatmap
#             if subset_df.empty or subset_df[x_param].nunique() < 2 or subset_df[y_param].nunique() < 1:
#                 ax.text(0.5, 0.5, 'Not enough data to form a 2D plot', ha='center', va='center')
#                 continue
            
#             try:
#                 pivot_df = subset_df.pivot_table(index=y_param, columns=x_param, values=metric)
#                 if not pivot_df.empty:
#                     sns.heatmap(
#                         pivot_df, 
#                         ax=ax, 
#                         cmap="RdYlGn", 
#                         annot=True, 
#                         fmt=".2f",
#                         norm=norm  # Use our new asymmetric normalizer
#                     )
#                     ax.invert_yaxis()
#             except Exception as e:
#                 ax.text(0.5, 0.5, f'Error plotting:\n{e}', ha='center', va='center')

#     # --- Final Figure Formatting ---
#     main_title = f'Optimization Heatmap for Strategy {strategy_key.upper()} ({metric})'
#     if row_facet_param:
#         main_title += f'\nRows: {row_facet_param} | Columns: {col_facet_param}'
#     elif col_facet_param:
#         main_title += f'\nFaceted by {col_facet_param}'

#     fig.suptitle(main_title, fontsize=16)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save the figure
#     plot_dir = os.path.join('params_distribution')
#     os.makedirs(plot_dir, exist_ok=True)
#     filename = f"heatmap_strat_{strategy_key}.png"
#     filepath = os.path.join(plot_dir, filename)
#     plt.savefig(filepath)
#     plt.close(fig)
#     print(f"--- Heatmap saved to {filepath} ---")
    

def calculate_robustness_for_all_candidates(all_results, params_to_optimize):
    """
    Calculates a robustness score for every parameter combination based on its own
    performance and the performance of its immediate neighbors.

    Returns a list of tuples: (combo, original_metric, robustness_score)
    """
    if not all_results:
        return []

    param_names = [p['name'] for p in params_to_optimize]
    
    # --- A. Create a grid-like dictionary for fast neighbor lookups ---
    # `results_grid` will look like: {(val1, val2, ...): metric_value}
    results_grid = {res[1]: res[0][2] for res in all_results} # Assuming res format is (lightweight_tuple, combo)
    metric_key = list(all_results[0][0][2].keys())[0] # Dynamically get the metric name (e.g., 'Equity_Efficiency_Rate')
    for combo, data in results_grid.items():
        results_grid[combo] = data.get(metric_key, -np.inf)

    # --- B. Create a map of each parameter to its unique, sorted values ---
    # `param_value_map` will look like: {'rr': [1.0, 2.0, 3.0], 'SMA 1 #1': [20.0, 50.0]}
    param_value_map = {name: sorted(list(set(c[i] for c in results_grid.keys()))) for i, name in enumerate(param_names)}

    scored_results = []
    for combo, metric_value in results_grid.items():
        
        # --- C. For each candidate, find all its neighbors ---
        neighbor_combos = []
        # Get the index of the current value for each parameter
        current_indices = [param_value_map[name].index(val) for name, val in zip(param_names, combo)]

        # Generate all possible index offsets (-1, 0, 1) for each parameter
        # This creates the N-dimensional "neighborhood"
        index_offsets = product([-1, 0, 1], repeat=len(param_names))

        for offsets in index_offsets:
            if all(o == 0 for o in offsets): # Skip the candidate itself
                continue

            new_indices = [curr + off for curr, off in zip(current_indices, offsets)]
            
            # Check if the neighbor's indices are valid (not out of bounds)
            if all(0 <= idx < len(param_value_map[name]) for idx, name in zip(new_indices, param_names)):
                neighbor_combo = tuple(param_value_map[name][idx] for idx, name in zip(new_indices, param_names))
                neighbor_combos.append(neighbor_combo)

        # --- D. Calculate neighborhood statistics ---
        neighbor_metrics = [results_grid.get(nc, np.nan) for nc in neighbor_combos]
        neighbor_metrics = [m for m in neighbor_metrics if not np.isnan(m)] # Filter out non-existent neighbors

        if not neighbor_metrics: # Handle edge cases with no neighbors
            avg_neighbor_metric = metric_value # Assume neighbors are as good as the point itself
            std_neighbor_metric = 0
        else:
            avg_neighbor_metric = np.mean(neighbor_metrics)
            std_neighbor_metric = np.std(neighbor_metrics)

        # --- E. Calculate the final Robustness Score ---
        # Weights can be tuned. Higher `w_stability` penalizes spiky peaks more.
        w_self = 0.5
        w_neighborhood = 0.5
        w_stability = 0.2 
        
        # Normalize the standard deviation by the average to make the penalty scale-invariant
        instability_penalty = (std_neighbor_metric / (abs(avg_neighbor_metric) + 1e-6))
        
        robustness_score = (w_self * metric_value) + \
                           (w_neighborhood * avg_neighbor_metric) - \
                           (w_stability * instability_penalty)

        scored_results.append((combo, metric_value, robustness_score))

    return scored_results