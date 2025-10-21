import asyncio
import traceback
import pandas as pd
from datetime import datetime, timedelta
from types import SimpleNamespace # A handy tool for creating simple objects
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import io
from contextlib import redirect_stdout
from sklearn.metrics import classification_report, confusion_matrix
import vectorbt as vbt

# --- Core Logic Imports ---
from core.data_manager import get_ohlcv
from core.connect_to_brokerage import get_client

from core.indicator_registry import INDICATOR_REGISTRY 
from core.robust_idk_processor import calculate_indicators
from core.ml_models import get_model, get_scaler

from .state import WORKFLOW_CACHE

def transform_features_for_backend(features_config: list) -> list:
    """
    Converts the feature config from the frontend into the tuple format
    required by the `calculate_indicators` function.
    """
    backend_format_indicators = []
    
    # 'features_config' is a list of 'IndicatorConfig' Pydantic model instances
    for feature in features_config:
        # Use dot notation to access attributes of the Pydantic model
        name = feature.name
        timeframe = feature.timeframe
        if name in INDICATOR_REGISTRY:
            param_order = INDICATOR_REGISTRY[name]['params']
            
            # Use dot notation for 'params', which is a dictionary.
            # Then, you can use standard dictionary access on it.
            params_list = [feature.params[p_name] for p_name in param_order]
            
            backend_format_indicators.append((name, timeframe, params_list))
        else:
            print(f"Warning: Feature '{name}' from config not found in registry. It will be skipped.")
            
    return backend_format_indicators

def load_data_for_ml(symbol: str, timeframe: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    This function's ONLY responsibility is to fetch and return raw OHLCV data.
    It no longer calculates indicators.
    """
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error parsing dates: {e}. Using default date range.")
        # Handle cases where date might be in a different format or invalid
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3) # Default to 3 years of data

    print(f"--- Fetching OHLCV data for Asset: {symbol}, Timeframe: {timeframe}, StartDate: {start_date.date()}, EndDate: {end_date.date()} ---")
    
    # We can hardcode 'binance' for now or make it part of the config later
    client = get_client('binance') 
    df = get_ohlcv(client, symbol, timeframe, start_date, end_date)
    
    # Ensure correct data types
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    
    # if 'timestamp' in df.columns:
    #     df['timestamp'] = pd.to_datetime(df['timestamp'])
        # df.set_index('timestamp', inplace=True)
    
    return df

def prepare_data_for_ml(df: pd.DataFrame):
    """
    Prepares the feature-rich DataFrame for machine learning by:
    1. Handling infinite values.
    2. Dropping rows with NaNs (created by indicators).
    3. Separating features (X) from the label (y).
    """
    # Replace infinite values with NaN so they can be dropped
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with any NaN values. This is crucial as indicators
    # create NaNs at the start of the series.
    df_cleaned = df.dropna()
    
    # Separate the label from the features
    y = df_cleaned['label']
    
    # Drop non-feature columns. Keep OHLCV for now if they are to be used as features.
    # If not, add them to the list to be dropped.
    X = df_cleaned.drop(columns=['label']) 
    
    return X, y, df_cleaned.index

def run_ml_pipeline_manager(batch_id: str, config: dict, manager, queue, loop):
    """
    Manages the entire ML pipeline from data prep to backtesting.
    This is the main orchestrator.
    """
    async def send_log(message: str, level: str = 'INFO'):
        """
        Sends a log message to the frontend, correctly formatted with a payload.
        """
        for line in message.strip().split('\n'):
            if line.strip():
                # This now matches the WebSocketManager's expected format
                payload = {"level": level.upper(), "message": line.strip()}
                await queue.put((batch_id, {"type": "log", "payload": payload}))

    async def main_logic():
        try:
            await send_log("✅ ML Pipeline job started.")
            await send_log(f"Configuration received for model: {config['model']['name']}")
            
            # --- STEP 1: RETRIEVE DATA FROM CACHE (INSTEAD OF LOADING/CALCULATING) ---
            workflow_id = config.get('workflow_id')
            if not workflow_id or workflow_id not in WORKFLOW_CACHE or 'engineered_df' not in WORKFLOW_CACHE[workflow_id]:
                await send_log("❌ FATAL ERROR: Could not find pre-processed data in the workflow session. Stopping.", level='ERROR')
                return
            
            await send_log("Found pre-processed data in workflow session. Proceeding...")
            # We start directly with the engineered DataFrame
            df_engineered = WORKFLOW_CACHE[workflow_id]['engineered_df'].copy()
            
            # --- STEP 2: LABEL GENERATION (Previously Step 3) ---
            await send_log("Generating labels from user-defined logic...")
            user_code = config['problem_definition']['custom_code']
            execution_scope = {}
            try:
                exec(user_code, execution_scope)
                labeling_func = execution_scope.get('generate_labels')
                if not callable(labeling_func):
                    raise ValueError("'generate_labels' function not found.")
                
                labels = labeling_func(df_engineered.copy())
                if not isinstance(labels, pd.Series):
                    raise TypeError("'generate_labels' must return a Pandas Series.")
                
                df_engineered['label'] = labels
                await send_log(f"Labels generated. Distribution:\n{labels.value_counts().to_string(na_rep='NaNs')}")
                
            except Exception as e:
                error_msg = f"Error executing user's labeling logic: {e}"
                await send_log(error_msg, level='ERROR') 
                print(traceback.format_exc()) 
                raise
            
            # --- STEP 3: FINAL DATA PREPARATION FOR ML (Previously Step 4) ---
            await send_log("Preparing data for model training (handling NaNs, separating X/y)...")
            
            # We now use our labeled, engineered DataFrame
            X, y, valid_indices = prepare_data_for_ml(df_engineered) 
            
            if X.empty:
                await send_log("❌ ERROR: No data remaining after cleaning NaNs. Stopping.")
                return
            
            await send_log(f"Data prepared. Training on {len(X)} samples.")
            
            # --- Step 5: Model Training Loop ---
            validation_cfg = config['validation']
            model_cfg = config['model']
            preprocessing_cfg = config['preprocessing']
            
            await send_log(f"Initializing model training with '{validation_cfg['method']}' validation...")
            
            all_predictions = pd.Series(index=y.index, dtype=float)
            n_splits = 1
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                await send_log(f"--- Processing Fold {fold + 1}/{n_splits} ---")
                
                # Keep X_train as a DataFrame to preserve feature names for fitting
                X_train_df, X_test_df = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # --- 5a. Preprocessing (Scaling) ---
                scaler = get_scaler(preprocessing_cfg['scaler'])
                if scaler:
                    await send_log(f"Applying {preprocessing_cfg['scaler']}...")
                    # Fit on the DataFrame, transform returns a NumPy array
                    X_train_scaled = scaler.fit_transform(X_train_df)
                    X_test_scaled = scaler.transform(X_test_df)
                    # Re-create DataFrames with original columns and index to prevent the warning
                    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=X_train_df.columns)
                    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=X_test_df.columns)
                else:
                    X_train_scaled, X_test_scaled = X_train_df, X_test_df
                
                model = get_model(model_cfg['name'])
                
                await send_log(f"Training {model_cfg['name']} on {len(X_train_scaled)} samples...")
                
                # Create an in-memory text buffer
                log_capture_buffer = io.StringIO()
                
                try:
                    # Use redirect_stdout to capture all print statements from the enclosed block
                    with redirect_stdout(log_capture_buffer):
                        # Run the blocking function in the executor
                        await loop.run_in_executor(
                            None, 
                            lambda: model.fit(X_train_scaled, y_train)
                        )
                    
                    # After the block, get the captured content and send it as a proper log
                    captured_logs = log_capture_buffer.getvalue()
                    await send_log(f"--- Model Training Output (Fold {fold + 1}) ---\n{captured_logs}")
                
                finally:
                    log_capture_buffer.close() # Always close the buffer
                
                await send_log("Training for this fold is complete.")
                
                # --- Prediction (can also be verbose, so we capture its logs too) ---
                await send_log("Generating predictions for the test set...")
                
                pred_probas = await loop.run_in_executor(
                    None,
                    lambda: model.predict_proba(X_test_scaled)
                )
                
                # First, get the index of the highest probability
                predicted_indices = np.argmax(pred_probas, axis=1)
                
                # Then, use the model's `classes_` attribute to map the index back to the original label
                predictions_for_fold = model.classes_[predicted_indices]
                
                fold_predictions = pd.Series(predictions_for_fold, index=y_test.index)
                all_predictions.update(fold_predictions)
            
            await send_log("All training folds completed.")
            await send_log(f"Generated {all_predictions.count()} predictions out of {len(y)} possible.")
            
            # --- Step 6: Backtesting and Performance Analysis (REVISED) ---
            await send_log("Running backtest and generating performance metrics...")
            
            # First, create the series of valid (non-NaN) predictions
            valid_predictions = all_predictions.dropna()
            
            # --- SAFETY CHECK: Abort if no predictions were generated ---
            if valid_predictions.empty:
                await send_log("⚠️ No valid predictions were generated after training. Cannot run backtest. Stopping.", level='WARNING')
                # We still need to send a "complete" message so the frontend doesn't hang
                final_payload = { "strategy_name": "ML Model", "metrics": {"error": "No trades to analyze"}, "equity_curve": [], "trades": [], "model_analysis": {}, "run_config": {} }
                await queue.put((batch_id, {"type": "strategy_result", "payload": final_payload}))
                await send_log("✅ ML Pipeline finished (aborted due to no predictions).", level='SUCCESS')
                return # Stop execution here
            
            # Second, use the index of valid_predictions to select the corresponding true labels
            true_labels_for_preds = y.loc[valid_predictions.index]
            
            # Get all unique labels that exist in the ground truth
            all_possible_labels = np.union1d(true_labels_for_preds.unique(), valid_predictions.unique())
            
            # Sort them for consistent ordering
            all_possible_labels.sort() 
            
            # 1. Establish the price series as the source of truth for the index.
            #    This index might be sparse because of dropna().
            price_data = df_engineered.loc[valid_predictions.index]['close'].copy()
            price_data.index = pd.to_datetime(price_data.index)
            
            # 2. Generate the sparse signals based on predictions.
            #    Their index will match price_data's index for now.
            sparse_entries = (valid_predictions == 1)
            sparse_exits = (valid_predictions == -1)
            
            # 3. Explicitly align the signals to the price_data index.
            #    This is the crucial step. It ensures entries, exits, and price_data
            #    are perfectly identical in shape and index. fill_value=False means
            #    any timestamp without a signal is explicitly marked as "no signal".
            entries = sparse_entries.reindex(price_data.index, fill_value=False)
            exits = sparse_exits.reindex(price_data.index, fill_value=False)
            
            portfolio = vbt.Portfolio.from_signals(
                price_data, 
                entries, 
                exits, 
                freq=config['data_source']['timeframe'], 
                init_cash=config['backtestSettings']['capital'],
                fees=config['backtestSettings']['commissionBps'] / 10000,  # Convert bps to decimal
                slippage=config['backtestSettings']['slippageBps'] / 10000, # Convert bps to decimal
                # trade_on_close=config['backtestSettings']['tradeOnClose']
            )
            stats = portfolio.stats()
            
            # --- Step 7: Assemble the Final JSON Payload (REVISED) ---
            await send_log("Assembling final report...")
            
            metrics_data = {
                "Net_Profit": stats.get('Total Return [%]', 0),
                "Total_Trades": stats.get('Total Trades', 0),
                "Winrate": stats.get('Win Rate [%]', 0),
                "Sharpe_Ratio": stats.get('Sharpe Ratio', 0),
                "Max_Drawdown": stats.get('Max Drawdown [%]', 0),
                "Profit_Factor": stats.get('Profit Factor', 0),
                "Calmar_Ratio": stats.get('Calmar_Ratio', 0), # Corrected key from previous version
            }
            
            equity_df = portfolio.value().reset_index()
            equity_df.columns = ['timestamp_ns', 'equity']
            equity_df['timestamp_ms'] = (equity_df['timestamp_ns'].astype(np.int64) // 1_000_000)
            equity_curve_data = equity_df[['timestamp_ms', 'equity']].values.tolist()
            
            trades_data = []
            if portfolio.trades.count() > 0:
                trades_df = pd.DataFrame(portfolio.trades.records)
                
                # `price_index` is now guaranteed to be a DatetimeIndex
                price_index = price_data.index
                
                # Selecting with integers from a DatetimeIndex returns a new DatetimeIndex
                entry_timestamps = price_index[trades_df['entry_idx']]
                exit_timestamps = price_index[trades_df['exit_idx']]
                
                # A DatetimeIndex object has a vectorized .strftime() method. This is the cleanest way.
                trades_df['entry_date'] = entry_timestamps.strftime('%Y-%m-%dT%H:%M:%SZ')
                trades_df['exit_date'] = exit_timestamps.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                trades_data = trades_df.to_dict(orient='records')
            else:
                await send_log("⚠️ No trades were executed in this backtest.")
            
            # Generate the standard report from scikit-learn
            class_report_dict = classification_report(
                true_labels_for_preds, 
                valid_predictions,
                labels=all_possible_labels,
                output_dict=True, 
                zero_division=0
            )
            
            # Create a new, formatted dictionary
            formatted_report = {}
            for key, value in class_report_dict.items():
                try:
                    # Check if the key is a number-like string (e.g., "-1", "0", "1.0")
                    float(key)
                    # If it is, format it with the "class_" prefix
                    formatted_report[f"class_{key}"] = value
                except ValueError:
                    # If it's not a number-like string (e.g., "accuracy", "macro avg"), keep it as is
                    formatted_report[key] = value
            
            model_analysis_data = {
                
                "feature_importance": pd.DataFrame({
                    'feature': X.columns, 
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15).to_dict(orient='records'),
                
                "classification_report": formatted_report, 
                
                "confusion_matrix": confusion_matrix(
                    true_labels_for_preds, 
                    valid_predictions,
                    labels=all_possible_labels
                    ).tolist()
            }
            
            # Add the run_config data
            run_config_data = {
                "model": config['model']['name'],
                "features": [f['name'] for f in config['features']],
                "labeling_method": config['problem_definition']['template_key'],
                "symbol": config['data_source']['symbol'],
                "timeframe": config['data_source']['timeframe']
            }
            
            final_payload = {
                "strategy_name": "ML Model",
                "metrics": metrics_data,
                "equity_curve": equity_curve_data,
                "trades": trades_data, # Use the correctly generated trades data
                "monthly_returns": [], 
                "model_analysis": model_analysis_data,
                "run_config": run_config_data
            }
            
            sanitized_payload = sanitize_for_json(final_payload)
            
            # --- Step 8: Send the result ---
            await send_log("Sending final report to the frontend...")
            
            # Send the clean, sanitized payload
            await queue.put((batch_id, {"type": "strategy_result", "payload": sanitized_payload}))
            
            await send_log("✅ ML Pipeline finished successfully.", level='SUCCESS')
        
        except Exception as e:
            error_message = f"An error occurred in the ML pipeline: {e}"
            print(error_message)
            if "labeling logic" not in str(e):
                print(traceback.format_exc())
            # --- Use the updated send_log for fatal errors ---
            await send_log(f"FATAL ERROR: {error_message}", level='ERROR')
        finally:
            # This signals the end of the stream
            await queue.put((batch_id, {"type": "batch_complete", "payload": {}}))


    def sanitize_for_json(data):
        """
        Recursively traverses a data structure and converts any non-JSON-serializable
        types (like NumPy types) into their JSON-safe Python equivalents.
        """
        if isinstance(data, dict):
            # If it's a dict, sanitize each value
            return {key: sanitize_for_json(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            # If it's a list or tuple, sanitize each item
            return [sanitize_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.int64)):
            # If it's a NumPy integer, convert to a standard Python int
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            # If it's a NumPy float, handle NaN/inf and convert to a standard Python float
            if np.isnan(data) or np.isinf(data):
                return None  # Convert NaN/inf to null in JSON
            return float(data)
        elif isinstance(data, pd.Timestamp):
            # Convert Pandas Timestamps to ISO 8601 strings
            return data.isoformat()
        # If the type is already JSON-safe (int, float, str, bool, None), return it as is
        return data


    asyncio.run_coroutine_threadsafe(main_logic(), loop)