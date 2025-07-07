from core.indicators import Indicators

idk = Indicators()

def calculate_indicators(strategy_instance, df):
    
    indicators = strategy_instance.indicators

    print(f"--- Calculating {indicators} indicators ---")

    for indicator in indicators:
        
        name, timeframe, params = indicator
                    
        if name == 'EMA':
            
            # Ensure you have only 1 value
            if len(params) == 1:
                # Convert the first three params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                df = idk.calculate_ema(df, length)
                
            else:
                print(f"EMA takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'WMA':
            
            # Ensure you have only 1 value
            if len(params) == 1:
                # Convert the first three params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                df = idk.calculate_wma(df, length)
                
            else:
                print(f"WMA takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'SMA':

            # Ensure you have only 1 value
            if len(params) == 1:
                
                # Convert the first three params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                
                if timeframe == '1m':
                    df = idk.calculate_sma(df, length)
                elif timeframe == '3m':
                    df_3m = idk.calculate_sma(df_3m, timeframe, length)
                    df = df.merge(df_3m[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                    df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                elif timeframe == '5m':
                    df_5m = idk.calculate_sma(df_5m, length)
                    df = df.merge(df_5m[['timestamp', f'sma_{length}']], how='left', on='timestamp')
                    df[f'sma_{length}'] = df[f'sma_{length}'].ffill()
                elif timeframe == '15m':
                    df_15m = idk.calculate_sma(df_15m, length)
                    df = df.merge(df_15m[['timestamp', f'sma_{length}']], how='left', on='timestamp')
                    df[f'sma_{length}'] = df[f'sma_{length}'].ffill()
                elif timeframe == '1h':
                    df_1h = idk.calculate_sma(df_1H, timeframe, length)
                    df = df.merge(df_1h[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                    df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                elif timeframe == '1d':
                    df_1d = idk.calculate_sma(df_1D, timeframe, length)
                    df = df.merge(df_1d[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                    df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                    
            else:
                print(f"SMA takes 1 Input Value, But {len(params)} were given...")
            
        if name == 'RSI':
            
            # Ensure you have only 1 value
            if len(params) == 1:
                # Convert the first three params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                df = idk.calculate_rsi(df, length)
                
            else:
                
                print(f"RSI takes 1 Input Value, But {len(params)} were given...")
            
        if name == 'STOCASTIC RSI':
            
            # Ensure you have only 1 value
            if len(params) == 2:
                # Convert the first three params to integers (or float, as needed)
                k = int(params[0])  # or float(params[0]) if you need float
                d = int(params[1])
                df = idk.calculate_stocastic(df, k, d)
                
            else:
                
                print(f"STOCASTIC RSI takes 2 Input Value, But {len(params)} were given...")
            
        if name == 'MACD':
            
            # Ensure you have at least 3 params
            if len(params) == 3:
                # Convert the first three params to integers (or float, as needed)
                short_window = int(params[0])  # or float(params[0]) if you need float
                long_window = int(params[1])  # or float(params[1]) if you need float
                signal_window = int(params[2])  # or float(params[2]) if you need float
                
                df = idk.calculate_macd(df, short_window, long_window, signal_window)
                                    
            else:
                
                print(f"MACD takes 3 Input params, But {len(params)} were given...")
                
        if name == 'BOLLINDER BANDS':
            
            # Ensure you have only 1 value
            if len(params) == 2:
                
                # Convert the first three params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                deviation = int(params[1])
                
                if timeframe == '1m':
                    df = idk.calculate_bollinger_bands(df, timeframe, length, deviation)
                # elif timeframe == '3m':
                #     df_3m = idk.calculate_bollinger_bands(df_3m, timeframe, length, deviation)
                #     df = df.merge(df_3m[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                #     df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                # elif timeframe == '5m':
                #     df_5m = idk.calculate_bollinger_bands(df_5m, timeframe, length, deviation)
                #     df = df.merge(df_5m[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                #     df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                # elif timeframe == '15m':
                #     df_15m = idk.calculate_bollinger_bands(df_15m, timeframe, length, deviation)
                #     df = df.merge(df_15m[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                #     df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                # elif timeframe == '1h':
                #     df_1h = idk.calculate_bollinger_bands(df_1H, timeframe, length, deviation)
                #     df = df.merge(df_1h[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                #     df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                # elif timeframe == '1d':
                #     df_1d = idk.calculate_bollinger_bands(df_1D, timeframe, length, deviation)
                #     df = df.merge(df_1d[['timestamp', f'{timeframe}_SMA_{length}']], how='left', on='timestamp')
                #     df[f'{timeframe}_SMA_{length}'] = df[f'{timeframe}_SMA_{length}'].ffill()
                    
            else:
                print(f"BOLLINGER BANDS takes 2 Input params, But {len(params)} were given...")
        
        if name == 'ATR':
            
            # Ensure you have at least 2 params
            if len(params) == 2:
                # Convert the first two params to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                multiplier = float(params[1])  # or float(params[1]) if you need float
                
                df = idk.calculate_stop_loss(df, length, multiplier)
            
            else:
                
                print(f"ATR takes 2 Input params, But {len(params)} were given...")
                
        if name == 'SUPERTREND':
            
            # Ensure you have at least 2 params
            if len(params) == 2:
                # Convert the first two params to integers (or float, as needed)
                period = int(params[0])  # or float(params[0]) if you need float
                multiplier = float(params[1])  # or float(params[1]) if you need float
                
                df = idk.supertrend(df, period, multiplier)
            
            else:
                
                print(f"SUPERTREND takes 2 Input params, But {len(params)} were given...")
                
        if name == 'CCI':
            
            if len(params) == 1:
                
                length = int(params[0])
                
                df = idk.calculate_cci(df, length)
                
            else:
                print(f"CCI takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'AROON':
            
            if len(params) == 1:
                
                period = int(params[0])
                
                df = idk.calculate_aroon(df, period)
                
            else:
                print(f"AROON takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'WILLIAMS RANGE':
            
            if len(params) == 1:
                
                period = int(params[0])
                
                df = idk.williams_range(df, period)
                
            else:
                print(f"WILLIAMS RANGE takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'HL OSCILLATOR':
            
            if len(params) == 1:
                
                period = int(params[0])
                
                df = idk.calculate_hl_oscillator(df, period)
                
            else:
                print(f"HL OSCILLATOR takes 1 Input Value, But {len(params)} were given...")
        
        if name == 'SMA SLOPE':
            
            # Ensure you have at least 3 params
            if len(params) == 3:
                # Convert the first value to integers (or float, as needed)
                length = int(params[0])  # or float(params[0]) if you need float
                smoothBars = int(params[1])  # or float(params[0]) if you need float
                neutralZoneHeight = int(params[2])  # or float(params[0]) if you need float
                
                df = idk.calculate_sma_slope(df, length, smoothBars, neutralZoneHeight)
                
            else:
                
                print(f"EMA Slope takes 3 Input params, But {len(params)} were given...")
        
        if name == 'CANDLESTICK PATTERNS':
            
            df = idk.identify_candlestick_patterns(df)
        
        if name == 'HH LL':
            
            if len(params) == 1:
                period = int(params[0])
                df = idk.hh_ll(df, period)
            else:
                print(f"HH LL takes 1 Input Value, But {len(params)} were given...")
                
        if name == 'MARKET REGIME':
            
            df = idk.calculate_market_regime(df)
            
        if name == 'PIVOT POINTS':
            
            df = idk.pivot_points(df)
            
        if name == 'LAG OHLCV':
            
            if len(params) == 1:
                lag = int(params[0])
                df = idk.lag_ohlcv(df, lag)
            else:
                print(f'LAG OHLCV takes 1 input value, But {len(params)} were given')
                
        if name == 'DATETIME SPLIT':
            
            df = idk.datetime_split(df)
        
        if name == 'CANDLE LENGTH':
            
            df = idk.candle_lengths(df)
        
        if name == 'PRICE MOVEMENT':
            
            df = idk.price_movement(df)
    
    print("Indicators processed successfully")
    
    return df