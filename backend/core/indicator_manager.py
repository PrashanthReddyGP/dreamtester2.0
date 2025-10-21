import pandas as pd
from core.indicators import Indicators

class IndicatorManager:
    def __init__(self):
        self.idk = Indicators()
        # This is the dispatcher map.
        # To add a new indicator, just add a line here.
        self._indicator_map = {
            'SMA': self.idk.calculate_sma,
            'EMA': self.idk.calculate_ema,
            'WMA': self.idk.calculate_wma,
            'HMA': self.idk.calculate_hma,
            'RSI': self.idk.calculate_rsi,
            'STOCHASTIC RSI': self.idk.calculate_stocastic, 
            'MACD': self.idk.calculate_macd,
            'BOLLINGER BANDS': self.idk.calculate_bollinger_bands,
            'ATR': self.idk.calculate_stop_loss, 
            'ADX': self.idk.calculate_adx,
            'SUPERTREND': self.idk.supertrend,
            'CCI': self.idk.calculate_cci,
            'AROON': self.idk.calculate_aroon,
            'WILLIAMS RANGE': self.idk.williams_range,
            'HL OSCILLATOR': self.idk.calculate_hl_oscillator,
            'SMA SLOPE': self.idk.calculate_sma_slope,
            'SMA SLOPE NORMALIZED': self.idk.calculate_sma_slope_normalized,
            'CANDLESTICK PATTERNS': self.idk.identify_candlestick_patterns,
            'HH LL': self.idk.hh_ll,
            'CUSTOM TREND FILTER': self.idk.custom_trend_filter,
            'PIVOT POINTS': self.idk.pivot_points,
            'SUPPORT RESISTANCE': self.idk.calculate_support_resistance,
            'ZSCORE': self.idk.calculate_zscore,
            'ZLEMA': self.idk.calculate_zero_lag_trend_signals, # Zero Lag EMA
            'ZIGZAG': self.idk.calculate_zigzag,
            'DONCHAIN CHANNEL': self.idk.donchian_channels,
            'ATR FIRST TOUCH': self.idk.calculate_atr_first_touch,
            # Add all other indicators here...
        }
    
    def calculate_indicators(self, strategy_instance, dataframes: dict):
        """
        Calculates all indicators specified in a strategy and merges them onto the main dataframe.
        
        Args:
            strategy_instance: An instance of a strategy class.
                                Must have `indicators` list and `timeframe` string.
            dataframes (dict): A dictionary where keys are timeframe strings (e.g., '15m', '1d')
                                and values are the corresponding pandas DataFrames.
        """
        main_timeframe = strategy_instance.timeframe
        
        if main_timeframe not in dataframes:
            raise ValueError(f"Main timeframe '{main_timeframe}' not found in dataframes dictionary.")
        
        main_df = dataframes[main_timeframe].copy()
        
        print(f"--- Indicators DataFrame has {main_df.shape[0]} rows before indicator calculations ---")
        
        indicators_to_calc = strategy_instance.indicators
        print(f"--- Calculating {len(indicators_to_calc)} indicators ---")
        
        for indicator_config in indicators_to_calc:
            name, timeframe, params = indicator_config
            
            # 1. Validate Indicator and Timeframe
            if name not in self._indicator_map:
                print(f"Warning: Indicator '{name}' is not supported. Skipping.")
                continue
            
            if timeframe not in dataframes:
                print(f"Warning: Data for timeframe '{timeframe}' not available. Skipping {name}.")
                continue
            
            # 2. Select the correct dataframe and calculation function
            source_df = dataframes[timeframe]
            calc_function = self._indicator_map[name]
            
            # 3. Calculate the indicator
            try:
                # 4. Process based on timeframe
                if timeframe == main_timeframe:
                    print(f"Calculating '{name}' for main timeframe '{main_timeframe}' with params {params}")
                    # Use the current main_df as the source and update it in place.
                    # This ensures we accumulate indicators.
                    main_df = calc_function(main_df, timeframe, *params)
                else:
                    print(f"Calculating '{name}' for timeframe '{timeframe}' to merge into '{main_timeframe}'")
                    # This is for higher/other timeframes. The logic here was already correct.
                    source_df = dataframes[timeframe]
                    
                    # Calculate on a copy to not modify the original HTF dataframe
                    calculated_df = calc_function(source_df.copy(), timeframe, *params)
                    
                    # Identify newly added columns
                    new_columns = calculated_df.columns.difference(source_df.columns).tolist()
                    
                    if not new_columns:
                        print(f"Warning: No new columns found after calculating {name} on {timeframe}. Skipping merge.")
                        continue
                    
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # !!! CRITICAL FIX FOR LOOKAHEAD BIAS: Shift the new columns by 1 !!!
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # This makes the indicator value for the bar ending at e.g., 10:00
                    # available starting at the 10:00 candle on the lower timeframe.
                    calculated_df[new_columns] = calculated_df[new_columns].shift(1)
                    
                    print(f"Merging '{name}' from timeframe '{timeframe}' into '{main_timeframe}'")
                    
                    merge_cols = ['timestamp'] + new_columns
                    df_to_merge = calculated_df[merge_cols]
                    
                    # Use merge_asof for robustness with potentially misaligned timestamps
                    # Ensure both dataframes are sorted by timestamp
                    main_df = main_df.sort_values('timestamp')
                    df_to_merge = df_to_merge.sort_values('timestamp').dropna(subset=new_columns)
                    
                    main_df = pd.merge_asof(
                        main_df, 
                        df_to_merge, 
                        on='timestamp', 
                        direction='backward' # 'backward' is equivalent to ffill
                    )
            
            except TypeError as e:
                print(f"Error calculating {name} on {timeframe} with params {params}. Check parameter count: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred during calculation of {name}: {e}")
                continue
        
        # Drop any initial rows with NaNs from merged indicators
        # main_df.dropna(inplace=True)
        main_df.reset_index(drop=True, inplace=True)
        
        print(f"--- Indicators DataFrame has {main_df.shape[0]} rows after indicator calculations ---")
        print(f"--- Indicators DataFrame has {main_df.shape[1]} columns after indicator calculations ---")
        
        print("Final columns:", main_df.columns.tolist())
        print("Indicators processed successfully")
        
        return main_df