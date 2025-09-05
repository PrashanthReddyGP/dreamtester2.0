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

def transform_features_for_backend(features_config: list, timeframe: str) -> list:
    """
    Converts the feature config from the frontend into the tuple format
    required by the `calculate_indicators` function.
    """
    backend_format_indicators = []
    for feature in features_config:
        name = feature['name']
        
        if name in INDICATOR_REGISTRY:
            param_order = INDICATOR_REGISTRY[name]['params']
            params_list = [feature['params'][p_name] for p_name in param_order]
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
    X = df_cleaned.drop(columns=['label', 'timestamp']) 
    
    return X, y, df_cleaned.index

def run_ml_pipeline_manager(batch_id: str, config: dict, manager, queue, loop):
    """
    Manages the entire ML pipeline from data prep to backtesting.
    This is the main orchestrator.
    """
    async def send_log(message: str, level: str = 'INFO'):
        # Clean up the message from any stray newline characters
        for line in message.strip().split('\n'):
            if line.strip():
                # --- Wrap the message in a payload object ---
                payload = {"level": level, "message": line.strip()}
                await queue.put((batch_id, {"type": "log", "payload": payload}))

    async def main_logic():
        try:
            await send_log("✅ ML Pipeline job started.")
            await send_log(f"Configuration received for model: {config['model']['name']}")

            # --- Step 1: Data Loading ---
            data_source = config['data_source']
            
            # TODO: The UI needs Start/End date pickers. For now, we use defaults if not provided.
            # We'll set a default of the last 3 years.
            end_date_default = datetime.now().strftime('%Y-%m-%d')
            start_date_default = (datetime.now() - timedelta(days=365 * 50)).strftime('%Y-%m-%d')
            
            start_date = data_source.get('startDate', start_date_default)
            end_date = data_source.get('endDate', end_date_default)

            await send_log(f"Fetching data for {data_source['symbol']} ({data_source['timeframe']}) from {start_date} to {end_date}...")
            
            # Call the clean, dedicated data loading function
            df_raw = load_data_for_ml(data_source['symbol'], data_source['timeframe'], start_date, end_date)
            
            if df_raw.empty:
                await send_log("❌ ERROR: No data returned for the specified symbol and date range. Stopping pipeline.")
                return # Stop execution
            
            await send_log(f"Data loaded successfully. Shape: {df_raw.shape}")

            # --- Step 2: Feature Engineering ---
            await send_log(f"Generating {len(config['features'])} features...")

            # 2a. Transform the frontend feature config to the backend's expected format
            indicator_tuples = transform_features_for_backend(
                config['features'], 
                data_source['timeframe']
            )
            
            # 2b. Create a mock 'strategy' object. SimpleNamespace is perfect for this,
            #     as it allows us to set attributes dynamically (e.g., .indicators).
            mock_strategy = SimpleNamespace(indicators=indicator_tuples)

            # 2c. Call your refactored indicator calculation function
            df_features = calculate_indicators(mock_strategy, df_raw.copy())
            
            await send_log("Features generated successfully.")

            # --- Step 3: Label Generation ---
            await send_log("Generating labels from user-defined logic...")
            user_code = config['problem_definition']['custom_code']
            
            # This will be the environment for our executed code.
            execution_scope = {}
            labels = None
            
            try:
                # Pass the `execution_scope` dictionary as BOTH the globals and locals.
                # This ensures that any function defined inside the user's code (like our Numba helper)
                # is available to any other function in that same code block.
                exec(user_code, execution_scope)

                labeling_func = execution_scope.get('generate_labels')
                
                if not callable(labeling_func):
                    raise ValueError("'generate_labels' function not found or is not callable in the provided code.")
                
                # Now, call the main function. It will be able to find `_calculate_labels_nb`
                # within the `execution_scope` which it now treats as its global scope.
                labels = labeling_func(df_features.copy())
                
                if labels is None or not isinstance(labels, pd.Series):
                    raise ValueError("'generate_labels' function did not return a valid Pandas Series.")

                df_features['label'] = labels
                await send_log(f"Labels generated successfully. Label distribution:\n{labels.value_counts().to_string(na_rep='NaNs')}")
                
            except Exception as e:
                error_msg = f"Error executing user's labeling logic: {e}"
                await send_log(error_msg, level='ERROR') 
                print(traceback.format_exc()) 
                raise

            # --- Step 4: Final Data Preparation for ML ---
            # This code will now only run if the label generation was successful.
            await send_log("Preparing data for model training (handling NaNs, separating X/y)...")
            X, y, valid_indices = prepare_data_for_ml(df_features)

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
            n_splits = 5 
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

                predictions_for_fold = np.argmax(pred_probas, axis=1)
                fold_predictions = pd.Series(predictions_for_fold, index=y_test.index)
                all_predictions.update(fold_predictions)

            await send_log("All training folds completed.")
            await send_log(f"Generated {all_predictions.count()} predictions out of {len(y)} possible.")

            # --- Step 6: Backtesting and Performance Analysis ---
            await send_log("Running backtest and generating performance metrics...")
            
            # Align the true labels with the predictions that were actually made
            valid_predictions = all_predictions.dropna()
            true_labels_for_preds = y.loc[valid_predictions.index]
            price_data = df_features.loc[valid_predictions.index]['close']

            # Create entry/exit signals for vectorbt
            # Assuming labels are: 1 for Long, -1 for Short, 0 for Hold
            entries = valid_predictions == 1
            exits = valid_predictions == -1

            # Run the backtest with vectorbt
            portfolio = vbt.Portfolio.from_signals(
                price_data, 
                entries, 
                exits, 
                freq=config['data_source']['timeframe'], 
                init_cash=100000
            )
            stats = portfolio.stats()

            # --- Step 7: Assemble the Final JSON Payload ---
            await send_log("Assembling final report...")

            # 7a. Convert vectorbt stats to your `StrategyMetrics` format
            metrics_data = {
                "Net_Profit": stats.get('Total Return [%]'),
                "Total_Trades": stats.get('Total Trades'),
                "Winrate": stats.get('Win Rate [%]'),
                "Sharpe_Ratio": stats.get('Sharpe Ratio'),
                "Max_Drawdown": stats.get('Max Drawdown [%]'),
                "Profit_Factor": stats.get('Profit Factor'),
                "Calmar_Ratio": stats.get('Calmar Ratio'),
                # Add mappings for all other metrics you need from `stats`
                # ...
            }

            # 7b. Format the Equity Curve
            equity_df = portfolio.value().reset_index()
            # The frontend expects [timestamp_ms, value]
            equity_curve_data = [
                (int(row.iloc[0].timestamp() * 1000), row.iloc[1]) 
                for _, row in equity_df.iterrows()
            ]

            # 7c. Format the Trades Log
            trades_df = portfolio.trades.records_df
            # Convert the trades DataFrame to the format your frontend expects
            trades_data = trades_df.reset_index().to_dict(orient='records')

            # 7d. Generate the `model_analysis` block
            # Get feature importances from the last trained model
            feature_importance_data = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15).to_dict(orient='records')

            # Get classification report and confusion matrix
            class_report_dict = classification_report(true_labels_for_preds, valid_predictions, output_dict=True)
            confusion_matrix_list = confusion_matrix(true_labels_for_preds, valid_predictions).tolist()

            model_analysis_data = {
                "feature_importance": feature_importance_data,
                "classification_report": class_report_dict,
                "confusion_matrix": confusion_matrix_list
            }

            # 7e. Assemble the final payload matching the `MLResult` interface
            final_payload = {
                "strategy_name": "ML Model",
                "metrics": metrics_data,
                "equity_curve": equity_curve_data,
                "trades": trades_data,
                "monthly_returns": [], # Placeholder, vectorbt can generate this too if needed
                "model_analysis": model_analysis_data,
                "run_config": {
                    "model": config['model']['name'],
                    "features": [f['name'] for f in config['features']],
                    "labeling_method": config['problem_definition']['template_key'],
                    "symbol": config['data_source']['symbol'],
                    "timeframe": config['data_source']['timeframe']
                }
            }

            # --- Step 8: Send the result to the frontend ---
            await send_log("Sending final report to the frontend...")
            
            # Use the type 'strategy_result' which your WebSocketManager already handles
            await queue.put((batch_id, {"type": "strategy_result", "payload": final_payload}))

            await send_log("✅ ML Pipeline finished successfully.")
            
        except Exception as e:
            # This is the main catch-all for the entire pipeline.
            error_message = f"An error occurred in the ML pipeline: {e}"
            print(error_message)
            # We already printed the detailed traceback if the error was in labeling.
            # If it's a new error, print its traceback.
            if "labeling logic" not in str(e):
                print(traceback.format_exc())
            await send_log(f"FATAL ERROR: {error_message}", level='ERROR')
        finally:
            # This block will always run, signaling completion to the frontend.
            await queue.put((batch_id, {"type": "batch_complete", "payload": {}}))

    asyncio.run_coroutine_threadsafe(main_logic(), loop)