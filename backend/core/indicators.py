import pandas as pd
import numpy as np

import ta
from tqdm import tqdm
import pandas_ta as pta

from numba import njit
from scipy.signal import find_peaks

import pandas_datareader.data as web


@njit
def _find_zigzag_points_numba(high: np.ndarray, low: np.ndarray, deviation_pct: float):
    """
    Numba-optimized function to find ZigZag pivot points.
    Returns an array where: 1 = Swing High, -1 = Swing Low, 0 = No pivot.
    """
    if len(high) == 0:
        return np.zeros(0, dtype=np.int8)

    pivots = np.zeros(len(high), dtype=np.int8)
    
    # State variables
    last_pivot_price = high[0]
    last_pivot_idx = 0
    trend = 0  # 0 = undecided, 1 = up, -1 = down

    # Find the first real trend
    for i in range(1, len(high)):
        change_up = (high[i] - low[0]) / low[0] * 100
        change_down = (low[i] - high[0]) / high[0] * 100
        
        if change_up >= deviation_pct:
            trend = 1
            last_pivot_price = high[i]
            last_pivot_idx = i
            pivots[0] = -1 # The first point was a low
            break
        elif change_down <= -deviation_pct:
            trend = -1
            last_pivot_price = low[i]
            last_pivot_idx = i
            pivots[0] = 1 # The first point was a high
            break

    # Main loop to find subsequent pivots
    for i in range(last_pivot_idx + 1, len(high)):
        if trend == 1:  # Looking for a high
            if high[i] >= last_pivot_price:
                last_pivot_price = high[i]
                last_pivot_idx = i
            # A reversal is confirmed if price drops by deviation % from the last high
            elif low[i] <= last_pivot_price * (1 - deviation_pct / 100.0):
                pivots[last_pivot_idx] = 1  # Confirmed high
                trend = -1
                last_pivot_price = low[i]
                last_pivot_idx = i
        elif trend == -1:  # Looking for a low
            if low[i] <= last_pivot_price:
                last_pivot_price = low[i]
                last_pivot_idx = i
            # A reversal is confirmed if price rises by deviation % from the last low
            elif high[i] >= last_pivot_price * (1 + deviation_pct / 100.0):
                pivots[last_pivot_idx] = -1  # Confirmed low
                trend = 1
                last_pivot_price = high[i]
                last_pivot_idx = i
                
    return pivots


@njit
def _predict_atr_touch_numba(high: np.ndarray, low: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized function to predict which ATR band (+/- 1 multiplier) is hit first,
    searching indefinitely until the end of the data.
    
    Returns: 
        1. Array of directions (1=Upper, -1=Lower, 0=None/Ambiguous).
        2. Array of bars elapsed until the touch (0=None/Ambiguous).
    """
    
    N = len(high)
    result_direction = np.zeros(N, dtype=np.int8)
    # Use float or int32 for bars elapsed, assuming standard datasets won't exceed int32 limits
    result_bars = np.zeros(N, dtype=np.int32) 
    
    for i in range(N):
        ub_i = upper_band[i]
        lb_i = lower_band[i]
        
        # Skip if bands are NaN (due to rolling ATR calculation start) or bands are invalid
        if np.isnan(ub_i) or np.isnan(lb_i) or ub_i <= lb_i:
            continue
            
        # Start search from the next bar (j = i + 1) until the end of the data (N)
        for j in range(i + 1, N): 
            
            # Check if high hits or crosses the upper band
            hit_upper = high[j] >= ub_i
            # Check if low hits or crosses the lower band
            hit_lower = low[j] <= lb_i
            
            if hit_upper and hit_lower:
                # Ambiguous hit in one bar. Direction and count remain 0.
                break
            
            elif hit_upper:
                # Upper band hit first
                result_direction[i] = 1
                result_bars[i] = j - i
                break
            
            elif hit_lower:
                # Lower band hit first
                result_direction[i] = -1
                result_bars[i] = j - i
                break
            
            # If the inner loop finishes without a hit, results remain 0.
    
    return result_direction, result_bars



# INDICATORS

class Indicators(object):

    # HULL MOVING AVERAGE (HMA)
    def calculate_hma(self, df, timeframe, length = 200):
        wma1 = df['close'].rolling(window=int(length / 2)).mean()  # WMA of half the period
        wma2 = df['close'].rolling(window=length).mean()           # WMA of the full period
        df[f'hma_{length}'] = (2 * wma1 - wma2).rolling(window=int(np.sqrt(length))).mean()  # WMA of the difference
        return df

    # Exponential Moving Average (EMA)
    def calculate_ema(self, df, timeframe, length = 200):
        df[f'ema_{length}'] = ta.trend.ema_indicator(df['close'], length)
        return df

    # Simple Moving Average (SMA)
    def calculate_sma(self, df, timeframe, length = 200):
        df[f'sma_{int(length)}_{timeframe}'] = ta.trend.sma_indicator(df['close'], int(length))
        return df

    # Weighted Moving Average (WMA)
    def calculate_wma(self, df, timeframe, length = 200):
        df[f'wma_{length}'] = ta.trend.wma_indicator(df['close'], length)
        return df

    # # Adaptive Moving Average (AMA)
    # def calculate_ama(self, df, timeframe, length = 200):
    #     df[f'ama_{length}'] = ta.trend.wma_indicator(df['close'], length)
    #     return df

    # RSI
    def calculate_rsi(self, df, timeframe, length = 14):
        
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=length).rsi()
        df['rsi'] = pd.to_numeric(df['rsi'].round(0).fillna(0), errors='coerce').astype(int)
        return df

    # MACD
    def calculate_macd(self, df, timeframe, short_window = 12, long_window = 26, signal_window = 9):
        
        # Calculate the short-term EMA (12-period by default)
        emaS = df['close'].ewm(span=short_window, adjust=False).mean()
        
        # Calculate the long-term EMA (26-period by default)
        emaL = df['close'].ewm(span=long_window, adjust=False).mean()
        
        # MACD Line: (12-period EMA - 26-period EMA)
        df['macd_line'] = emaS - emaL
        
        # Signal Line: 9-period EMA of the MACD Line
        df['signal_line'] = df['macd_line'].ewm(span=signal_window, adjust=False).mean()
        
        # MACD Histogram: MACD Line - Signal Line
        df['macd_histogram'] = df['macd_line'] - df['signal_line']
        
        return df

    #STOCASTIC
    def calculate_stocastic(self, df, timeframe, k_length = 14, d_length = 3):
        
        stochastic  = pta.stoch(df['high'], df['low'], df['close'], k_length, d_length)
        
        # Rename the columns
        stochastic.columns = ['stocastic_%k', 'stocastic_%d']
        
        df = pd.concat([df, stochastic], axis=1)
        df['stocastic_%k'] = pd.to_numeric(df['stocastic_%k'].round(0).fillna(0), errors='coerce').astype(int)
        df['stocastic_%d'] = pd.to_numeric(df['stocastic_%d'].round(0).fillna(0), errors='coerce').astype(int)
        return df 

    # Main distance function
    def ma_distance(self, df, timeframe, length = 200):
        
        df = self.calculate_ema(df, timeframe, length)
        df = self.calculate_sma(df, timeframe, length)
        # df = self.calculate_wma(df, length)
        df = self.calculate_hma(df, timeframe, length)
        
        df[f'ema_{length}_distance'] =np.where(df['close'] >= df[f'ema_{length}'],
                                                df['close'] - df[f'ema_{length}'],
                                                df[f'ema_{length}'] - df['close'])
        
        df[f'sma_{length}_distance'] =np.where(df['close'] >= df[f'sma_{length}_{timeframe}'],
                                                df['close'] - df[f'sma_{length}_{timeframe}'],
                                                df[f'sma_{length}_{timeframe}'] - df['close'])
        
        df[f'hma_{length}_distance'] =np.where(df['close'] >= df[f'hma_{length}'],
                                                df['close'] - df[f'hma_{length}'],
                                                df[f'hma_{length}'] - df['close'])
        
        return df

    # ADX
    def calculate_adx(self, df, timeframe, length = 14):
        
        # Calculate the ADX with the default period of 14
        adx = pta.adx(df['high'], df['low'], df['close'], length)
        
        adx.columns = ['adx', 'dmp', 'dmn']
        
        # Add the ADX components to the original DataFrame
        df = pd.concat([df, adx], axis=1)
        
        df['adx'] = pd.to_numeric(df['adx'].round(0).fillna(0), errors='coerce').astype(int)
        df['dmp'] = pd.to_numeric(df['dmp'].round(0).fillna(0), errors='coerce').astype(int)
        df['dmn'] = pd.to_numeric(df['dmn'].round(0).fillna(0), errors='coerce').astype(int)
        df['dmi_spread'] = df['dmn'] - df['dmp']
        
        return df
    
    def calculate_aroon(self, df, timeframe, length=14):
        """
        Calculates the Aroon Indicator and Oscillator using the optimized pandas-ta library.

        Args:
            df (pd.DataFrame): Input DataFrame.
            length (int): The lookback period for Aroon.

        Returns:
            pd.DataFrame: DataFrame with 'aroon_up', 'aroon_down', and 'aroon_osc' columns.
        """
        # pandas-ta's aroon function is highly optimized.
        aroon_df = pta.aroon(df['high'], df['low'], length=length)
        
        # The columns are named like 'AROONU_14', 'AROOND_14', 'AROONOSC_14'.
        # We can rename them for consistency with your code.
        aroon_df.rename(columns={
            f'AROONU_{length}': 'aroon_up',
            f'AROOND_{length}': 'aroon_down',
            f'AROONOSC_{length}': 'aroon_osc'
        }, inplace=True)
        
        # Join the results back to the original DataFrame
        df = pd.concat([df, aroon_df], axis=1)
        return df

    def calculate_aroon_vectorized(self, df, timeframe, period=14):
        """
        Calculates the Aroon Indicator using a fast, vectorized method.

        This version avoids slow .apply() loops by using rolling().idxmax(), which is
        significantly more performant.

        Args:
            df (pd.DataFrame): Input DataFrame with 'high' and 'low' columns.
            period (int): The lookback period.

        Returns:
            pd.DataFrame: The DataFrame with 'aroon_up' and 'aroon_down' columns added.
        """
        # Create a series representing the integer position of each row
        row_idx = pd.Series(range(len(df)), index=df.index)

        # Find the index LABEL of the highest high in the rolling window
        high_idx_labels = df['high'].rolling(window=period, min_periods=1).idxmax()
        # Find the index LABEL of the lowest low in the rolling window
        low_idx_labels = df['low'].rolling(window=period, min_periods=1).idxmin()

        # Map the index LABELS back to their integer POSITIONS for calculation
        # .searchsorted is a fast way to do this mapping.
        periods_since_high = row_idx - df.index.searchsorted(high_idx_labels)
        periods_since_low = row_idx - df.index.searchsorted(low_idx_labels)

        # Calculate Aroon Up and Aroon Down using the standard formula
        df['aroon_up'] = ((period - periods_since_high) / period) * 100
        df['aroon_down'] = ((period - periods_since_low) / period) * 100
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        return df

    # Defaulting to 15m Timeframe
    def calculate_standard_deviation(self, df, timeframe):
        """
        Calculates the annualized historical volatility.
        
        This is the statistical standard deviation of log returns over a given period,
        scaled by an annualizing factor.
        
        Args:
            df (pd.DataFrame): Input DataFrame with a 'close' column.
            period (int): The lookback window for calculating the standard deviation.
            annualizing_factor (int): The factor to scale volatility to an annual figure.
                                    Common values:
                                    - 252 for daily data (trading days)
                                    - 365 for daily data (calendar days)
                                    - 252 * 6.5 for hourly US stock market data
                                    - 365 * 24 for hourly crypto data

        For Daily timeframe: 20, 365
        For 15m timeframe: 96, 35040
        
        Returns:
            pd.DataFrame: DataFrame with the 'hist_vol' column added.
        """
                
        if timeframe == 15:
            period = 96
            annualizing_factor = 35040
        
        if timeframe == 60:
            period = 24
            annualizing_factor = 8760
        
        elif timeframe == 24:
            period = 20
            annualizing_factor = 365
        
        else:
            print("STD Timeframe ISSUE")
        
        # Calculate logarithmic returns, which are standard for volatility calculations
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate the rolling standard deviation of the log returns
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize the volatility
        df['std_hist_volatility'] = rolling_std * np.sqrt(annualizing_factor)
        
        return df

    def williams_range(self, df, timeframe, period = 14):
        
        # Calculate the rolling highest high and lowest low over the period
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        # Calculate Williams %R
        df['williams_range'] = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        
        return df

    # CCI
    def calculate_cci(self, df, timeframe, length = 20):
        
        # Add the CCI column to the original DataFrame
        df['cci'] = pta.cci(df['high'], df['low'], df['close'], length)
        df['cci'] = pd.to_numeric(df['cci'].round(0).fillna(0), errors='coerce').astype(int)

        return df

    # VWAP
    def calculate_VWAP(self, df, timeframe):
        
        # Calculate typical price (which is (High + Low + Close) / 3)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative price * volume and cumulative volume
        df['cum_price_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cum_price_vol'] / df['cum_vol']
        
        return df

    # BOLLINGER BANDS
    def calculate_bollinger_bands(self, df, timeframe, length = 20, deviation = 2):
        
        # Calculate the Simple Moving Average (SMA)
        temp_df = self.calculate_sma(df, timeframe, length)
        
        # Calculate the rolling standard deviation
        temp_df['std'] = df['close'].rolling(window=length).std()
        
        # Calculate Upper and Lower Bands
        df['bollinger_upperband'] = temp_df[f'sma_{length}_{timeframe}'] + (deviation * temp_df['std'])
        df['bollinger_lowerband'] = temp_df[f'sma_{length}_{timeframe}'] - (deviation * temp_df['std'])
        df['bollinger_middleband'] = temp_df[f'sma_{length}_{timeframe}']
        
        df['bb_width'] = (df['bollinger_upperband'] - df['bollinger_lowerband']) / temp_df[f'sma_{length}_{timeframe}']
        
        # Normalized Bollinger Band Width
        df['bb_width_pct'] = (df['bollinger_upperband'] - df['bollinger_lowerband']) / df['close'] * 100

        return df

    # MOMENTUM
    def calculate_momentum(self, df, timeframe, lookback = 10):
        
        df['momentum'] = df['close'] / df['close'].shift(lookback) * 100
        df['momentum'] = pd.to_numeric(df['momentum'].round(0).fillna(0), errors='coerce').astype(int)

        return df

    # OBV
    def calculate_obv(self, df, timeframe):
        
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df

    # ROC
    def calculate_roc(self, df, timeframe, lookback = 9):
        
        df['roc'] = ((df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback)) * 100
        df['roc'] = pd.to_numeric(df['roc'].round(0).fillna(0), errors='coerce').astype(int)

        return df

    def calculate_signed_extreme_change(
        self,
        df: pd.DataFrame,
        timeframe,
        lookback: int = 9,
        as_percentage: bool = True
    ) -> pd.DataFrame:
        """
        Calculates a signed indicator representing the dominant "stretch" of the market.

        This indicator measures whether the primary price move was an upward stretch
        (current high vs. lookback low) or a downward stretch (current low vs.
        lookback high) and returns a signed value representing that dominant move.

        - A positive value means the upward stretch was greater.
        - A negative value means the downward stretch was greater.

        Args:
            df (pd.DataFrame): Input DataFrame with 'high' and 'low' columns.
            lookback (int, optional): The number of preceding periods to look back. Defaults to 9.
            as_percentage (bool, optional): If True, returns the change as a percentage
                                            of the lookback period's midpoint price.
                                            Defaults to False (absolute price change).

        Returns:
            pd.DataFrame: A new DataFrame with the signed indicator column added.
        """
        
        if not all(col in df.columns for col in ['high', 'low']):
            raise ValueError("Input DataFrame must contain 'high' and 'low' columns.")
        
        df_out = df.copy()

        # 1. Get the highest high and lowest low of the PRECEDING 'lookback' periods
        lookback_high = df_out['high'].rolling(window=lookback).max().shift(1)
        lookback_low = df_out['low'].rolling(window=lookback).min().shift(1)

        # 2. Calculate the signed stretches
        upward_stretch = df_out['high'] - lookback_low
        downward_stretch = df_out['low'] - lookback_high # This will be negative

        # 3. Determine the dominant stretch using np.where
        # The condition checks which stretch has a larger absolute magnitude.
        # `upward_stretch > -downward_stretch` is equivalent to `abs(upward_stretch) > abs(downward_stretch)`
        # since `downward_stretch` is negative.
        dominant_stretch = np.where(
            upward_stretch > -downward_stretch,
            upward_stretch,      # Value if True
            downward_stretch     # Value if False
        )
        
        # 4. Handle results based on user preference
        if as_percentage:
            lookback_midpoint = (lookback_high + lookback_low) / 2
            # Use the signed dominant_stretch value for the calculation
            pct_change = (dominant_stretch / lookback_midpoint).replace([np.inf, -np.inf], 0) * 100
            df_out[f'roec_{lookback}'] = pct_change.round(2).fillna(0)
        else:
            # The result is already a float, assign it directly
            df_out[f'roec_{lookback}'] = pd.Series(dominant_stretch, index=df_out.index).round(4).fillna(0)
        
        return df_out

    # ICHIMOKU
    def calculate_ichimoku(self, df, timeframe, conversion_line = 9, base_line = 26, lagging_span_b = 52, lagging_span = 26):
        
        # Tenkan-sen (Conversion Line)
        df['conversion_line'] = (df['high'].rolling(window=conversion_line).max() + df['low'].rolling(window=conversion_line).min()) / 2
        
        # Kijun-sen (Base Line)
        df['base_line'] = (df['high'].rolling(window=base_line).max() + df['low'].rolling(window=base_line).min()) / 2
        
        # Senkou Span A (Leading Span A)
        df['leading_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(lagging_span)
        
        # Senkou Span B (Leading Span B)
        df['leading_span_b'] = ((df['high'].rolling(window=lagging_span_b).max() + df['low'].rolling(window=lagging_span_b).min()) / 2).shift(lagging_span)
        
        # Chikou Span (Lagging Span)
        df['lagging_span'] = df['close'].shift(-lagging_span)
        
        return df

    # PSAR
    def calculate_psar(self, df, timeframe, af_initial = 0.02, af_step = 0.02, af_max = 0.20):
        
        """
        Calculate the Parabolic SAR for a given DataFrame of OHLC data.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        af_initial (float): The initial acceleration factor (default is 0.02).
        af_step (float): The step by which AF increases (default is 0.02).
        af_max (float): The maximum acceleration factor (default is 0.20).
        
        Returns:
        pd.DataFrame: DataFrame with the PSAR values.
        """
        
        # Initialize variables
        psar = df['low'][0]  # Starting PSAR value
        af = af_initial  # Initial acceleration factor
        high_point = df['high'][0]  # Highest point in uptrend
        low_point = df['low'][0]  # Lowest point in downtrend
        long_position = True  # Start assuming a long position
        
        psar_values = [psar]  # Initialize the PSAR values with the first value
        
        # Loop through the DataFrame and calculate PSAR for each row
        for i in range(1, len(df)):
            previous_psar = psar
            if long_position:
                # Uptrend case
                psar = previous_psar + af * (high_point - previous_psar)
                psar = min(psar, df['low'][i-1], df['low'][i])
                if df['high'][i] > high_point:
                    high_point = df['high'][i]
                    af = min(af + af_step, af_max)
                if df['low'][i] < psar:
                    long_position = False
                    psar = high_point
                    low_point = df['low'][i]
                    af = af_initial
            else:
                # Downtrend case
                psar = previous_psar + af * (low_point - previous_psar)
                psar = max(psar, df['high'][i-1], df['high'][i])
                if df['low'][i] < low_point:
                    low_point = df['low'][i]
                    af = min(af + af_step, af_max)
                if df['high'][i] > psar:
                    long_position = True
                    psar = low_point
                    high_point = df['high'][i]
                    af = af_initial
            
            psar_values.append(psar)
        
        # Append the PSAR values to the dataframe
        df['psar'] = pd.Series(psar_values, index=df.index)
        
        return df

    # SUPERTREND
    
    # ATR for Supertrend
    def atr(self, df, period):
        
        df['candle_length'] = df['high'] - df['low']
        df['pulldown_length'] = abs(df['high'] - df['close'].shift(1))
        df['pullup_length'] = abs(df['low'] - df['close'].shift(1))
        df['tr_st'] = df[['candle_length', 'pulldown_length', 'pullup_length']].max(axis=1)
        df['atr_st'] = df['tr_st'].rolling(window=period).mean()
        
        return df

    # Function to calculate SuperTrend
    def supertrend(self, df, timeframe, period=10, multiplier=3.0):
        
        df = self.atr(df, period)  # Calculate ATR for the specified period
        
        # Initialize arrays for upper and lower SuperTrend bands and trend
        upper_band = ((df['high'] + df['low']) / 2) + (multiplier * df['atr_st'])
        lower_band = ((df['high'] + df['low']) / 2) - (multiplier * df['atr_st'])
        
        supertrend = np.zeros(len(df))  # Array to hold SuperTrend values
        prev_upper_band = upper_band[0]  # Initial previous upper band
        prev_lower_band = lower_band[0]  # Initial previous lower band
        prev_trend = 0  # Initialize previous trend to neutral (0)
        
        for i in range(1, len(df)):
            # Determine current trend (bullish or bearish)
            if df['close'][i] > prev_upper_band:
                current_trend = 1  # Bullish
            elif df['close'][i] < prev_lower_band:
                current_trend = -1  # Bearish
            else:
                current_trend = prev_trend  # Continue previous trend
            
            # Adjust upper and lower bands based on trend
            if current_trend == 1:  # Bullish trend
                if lower_band[i] < prev_lower_band:
                    lower_band[i] = prev_lower_band  # Maintain previous lower band if new one is lower
            elif current_trend == -1:  # Bearish trend
                if upper_band[i] > prev_upper_band:
                    upper_band[i] = prev_upper_band  # Maintain previous upper band if new one is higher
            
            # Store the current SuperTrend value and update previous bands and trend
            supertrend[i] = current_trend
            prev_upper_band = upper_band[i]
            prev_lower_band = lower_band[i]
            prev_trend = current_trend
        
        # Assign the SuperTrend to the DataFrame
        df['supertrend'] = supertrend
        df['supertrend_upperband'] = upper_band
        df['supertrend_lowerband'] = lower_band
        
        return df

    # DONCHAIN CHANNELS
    def donchian_channels(self, df, timeframe, window = 20):
        
        # Upper channel (max over the window)
        df['donchain_upperband'] = df['high'].rolling(window=window).max()
        
        # Lower channel (min over the window)
        df['donchain_lowerband'] = df['low'].rolling(window=window).min()
        
        # Middle channel (average of upper and lower)
        df['donchain_midline'] = (df['donchain_upperband'] + df['donchain_lowerband']) / 2
        
        # 2. Calculate the raw width of the channel
        donchian_width = df['donchain_upperband'] - df['donchain_lowerband']
        
        # 3. Normalize the width by the closing price to get a percentage
        # Use .replace(0, np.nan) to avoid division by zero errors
        close_safe = df['close'].replace(0, np.nan)
        df[f'donchian_width_norm'] = (donchian_width / close_safe) * 100
        
        return df

    # # PIVOT POINTS
    # def pivot_points(self, df, timeframe, pivot_type = 'Fibonacci'):
        
    #     df = df.copy()
    #     df.set_index('timestamp', inplace=True)
        
    #     # Assumes the DataFrame index is a datetime index
    #     daily = df.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
        
    #     # Calculate the basic daily pivot point (PP)
    #     daily['pivot'] = (daily['high'] + daily['low'] + daily['close']) / 3.0
        
    #     # Calculate the daily range
    #     daily['range'] = daily['high'] - daily['low']
        
    #     # Calculate Fibonacci-based support and resistance levels
    #     daily['r1'] = daily['pivot'] + 0.382 * daily['range']
    #     daily['s1'] = daily['pivot'] - 0.382 * daily['range']
    #     daily['r2'] = daily['pivot'] + 0.618 * daily['range']
    #     daily['s2'] = daily['pivot'] - 0.618 * daily['range']
    #     daily['r3'] = daily['pivot'] + 1.000 * daily['range']
    #     daily['s3'] = daily['pivot'] - 1.000 * daily['range']
        
    #     # Reindex the daily pivot data to the lower timeframe index using forward fill.
    #     # This assigns each lower timeframe period the pivot values from the corresponding day.
    #     daily_pivots = daily.reindex(df.index, method='ffill')
        
    #     # Join the pivot columns back to the original 15m DataFrame
    #     df = df.join(daily_pivots[['pivot', 'range', 'r1', 's1', 'r2', 's2', 'r3', 's3']])
    #     df = df.reset_index()
        
    #     return df
    
    # PIVOT POINTS
    def pivot_points(self, df, timeframe, pivot_type='Fibonacci'):
        """
        Calculates pivot points for each row based on that row's OHLC data.
        
        Note: This is a non-standard, row-wise calculation. Traditional pivot points
        use the previous period's (e.g., daily) HLC to calculate levels for the
        current period. This function calculates a "pivot" for each individual candle.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close' columns.
            timeframe: The timeframe of the data (not used in this specific calculation).
            pivot_type (str): The type of pivot point to calculate (currently only 'Fibonacci').
        
        Returns:
            pd.DataFrame: The DataFrame with pivot point levels added for each row.
        """
        df_out = df.copy()
        
        # Get the high, low, and close from the entire period's data
        period_high = df_out['high']
        period_low = df_out['low']
        period_close = df_out['close']
        
        # Calculate the basic pivot point (PP)
        pivot = (period_high + period_low + period_close) / 3.0
        
        # Calculate the range
        period_range = period_high - period_low
        
        # Calculate Fibonacci-based support and resistance levels
        r1 = pivot + 0.382 * period_range
        s1 = pivot - 0.382 * period_range
        r2 = pivot + 0.618 * period_range
        s2 = pivot - 0.618 * period_range
        r3 = pivot + 1.000 * period_range
        s3 = pivot - 1.000 * period_range
        
        # Add the calculated pivot levels to the DataFrame.
        # Pandas will automatically broadcast these single values to all rows.
        df_out['pivot'] = pivot
        df_out['range'] = period_range
        df_out['r1'] = r1
        df_out['s1'] = s1
        df_out['r2'] = r2
        df_out['s2'] = s2
        df_out['r3'] = r3
        df_out['s3'] = s3
        
        return df_out
    
    # WILLIAMS %R

    # CHAIKIN MONEY FLOW

    # KELTNER CHANNELS

    # MONEY FLOW INDEX

    # # HEIKIN-ASHI CANDLES
    # def heikin_ashi(self, df, timeframe):
        
    #     df = df.copy()
        
    #     # Initialize Heikin Ashi columns
    #     df['heikin_ashi_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    #     df['heikin_ashi_open'] = 0.0
    #     df['heikin_ashi_high'] = 0.0
    #     df['heikin_ashi_low'] = 0.0
        
    #     # Set the first row's HA_Open as the original open price
    #     df.loc[0, 'heikin_ashi_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2
        
    #     # Calculate the Heikin Ashi values
    #     for i in range(1, len(df)):
    #         df.loc[i, 'heikin_ashi_open'] = (df.loc[i-1, 'heikin_ashi_open'] + df.loc[i-1, 'heikin_ashi_close']) / 2
    #         df.loc[i, 'heikin_ashi_high'] = max(df.loc[i, 'high'], df.loc[i, 'heikin_ashi_open'], df.loc[i, 'heikin_ashi_close'])
    #         df.loc[i, 'heikin_ashi_low'] = min(df.loc[i, 'low'], df.loc[i, 'heikin_ashi_open'], df.loc[i, 'heikin_ashi_close'])
        
    #     return df

    # CANDLESTICK PATTERNS
    def identify_candlestick_patterns(self, df, timeframe):
        
        # Initialize pattern columns
        df['bullish_engulfing'] = 0
        df['bearish_engulfing'] = 0
        df['hammer'] = 0
        df['shooting_star'] = 0
        df['doji'] = 0
        
        # Calculate common terms
        open_price = df['open']
        high_price = df['high']
        low_price = df['low']
        close_price = df['close']
        
        prev_open = open_price.shift(1)
        prev_close = close_price.shift(1)
        
        # data['Candlestick_Pattern'] = 0 # 0 for none, 1 for Bullish Engulfing 
        
        # Bullish Engulfing
        df['bullish_engulfing'] = (
            (close_price > open_price) &
            (prev_close < prev_open) &
            (open_price <= prev_close) &
            (close_price > prev_open)
        ).astype(int)
        
        # Bearish Engulfing
        df['bearish_engulfing'] = (
            (close_price < open_price) &
            (prev_close > prev_open) &
            (open_price >= prev_close) &
            (close_price < prev_open)
        ).astype(int)
        
        # Hammer (Bullish and Bearish)
        body_size = abs(close_price - open_price)
        lower_shadow = open_price - low_price
        upper_shadow = high_price - close_price
        
        df['hammer'] = (
            ((close_price > open_price) & (lower_shadow > 2 * body_size) & (upper_shadow < body_size)) |
            ((close_price < open_price) & (close_price - low_price > 2 * body_size) & (high_price - open_price < body_size))
        ).astype(int)
        
        # Shooting Star (Bullish and Bearish)
        upper_shadow_bullish = high_price - close_price
        upper_shadow_bearish = high_price - open_price
        
        df['shooting_star'] = (
            ((close_price > open_price) & (upper_shadow_bullish > 2 * body_size) & (low_price >= open_price)) |
            ((close_price < open_price) & (upper_shadow_bearish > 2 * body_size) & (low_price >= close_price))
        ).astype(int)
        
        # Doji
        df['doji'] = (
            abs(close_price - open_price) <= 0.01 * (high_price - low_price)
        ).astype(int)
        
        return df
    
    # ATR
    def calculate_atr(self, df, timeframe, length, smoothing='sma'):
        
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        
        if smoothing == 'rma':
            df['atr'] = df['tr'].ewm(span=length, adjust=False).mean()
        elif smoothing == 'sma':
            df['atr'] = df['tr'].rolling(window=length).mean()
            df['atr_norm'] = df['atr'] / df['close'] * 100
        elif smoothing == 'ema':
            df['atr'] = df['tr'].ewm(span=length, adjust=False).mean()
        
        return df

    # CALCULATE STOP LOSS
    def calculate_stop_loss(self, df, timeframe, length = 14, multiplier = 8, index=1):
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        
        df = self.calculate_atr(df, timeframe, length)
        df['atr_multiplier'] = df['atr'] * multiplier
        df[f'stoploss_long_{index}'] = df['low'] - df['atr_multiplier']
        df[f'stoploss_short_{index}'] = df['high'] + df['atr_multiplier']
        
        return df
    
    # # EMA SLOPE
    # def calculate_ema_slope(self, df, timeframe, length):
        
    #     df[f'EMA_Slope_{length}'] = df[f'EMA_{length}'].diff()
        
    #     return df
    
    def calculate_sma_slope(self, df, timeframe, length = 200, smoothbars = 3, hlineheight = 10):
    
        df = self.calculate_sma(df, timeframe, length)
        
        # Calculate the difference between current MA and MA from `smoothBars` ago
        df['ma_df'] = df[f'sma_{length}_{timeframe}'] - df[f'sma_{length}_{timeframe}'].shift(smoothbars)
        
        # MA Slope Scaling
        def fmaDf(ma, maDF):
            maMax = maDF.rolling(window=500).max()
            maMin = maDF.rolling(window=500).min()
            ma_range = maMax - maMin
            maDf = 100 * maDF / ma_range
            return maDf
        
        # Calculate the normalized slope (scaled slope)
        df['ma_slope'] = fmaDf(df[f'sma_{length}'], df['ma_df'])
        
        # Calculate acceleration of the MA slope
        df['ma_acceleration'] = np.abs(df['ma_slope'] - df['ma_slope'].shift(1)) * smoothbars * 2
        
        # Normalize acceleration by the highest value over the last 200 periods
        df['ma_highest_acceleration'] = df['ma_acceleration'].rolling(window=length).max()
        df['ma_acceleration_normalized'] = 50 * df['ma_acceleration'] / df['ma_highest_acceleration']
        
        # Determine the trend based on MA slope
        df['ma_trend'] = np.where((df['ma_slope'] < -hlineheight), -1,  # Down Trend
                    np.where((df['ma_slope'] > hlineheight), 1,    # Up Trend
                                0))                              # Sideways
        
        # # Determine if the MA slope is in the "Neutral Zone" channel
        # df['inChannel'] = (df['maDf'] < hLineHeight) & (df['maDf'] > -hLineHeight) & \
        #                 ((df['maDf'].shift(1) >= hLineHeight) | (df['maDf'].shift(1) <= -hLineHeight))
        df['ma_slope'] = pd.to_numeric(df['ma_slope'].round(0).fillna(0), errors='coerce').astype(int)
        df['ma_acceleration'] = pd.to_numeric(df['ma_acceleration'].round(0).fillna(0), errors='coerce').astype(int)
        df['ma_highest_acceleration'] = pd.to_numeric(df['ma_highest_acceleration'].round(0).fillna(0), errors='coerce').astype(int)
        df['ma_acceleration_normalized'] = pd.to_numeric(df['ma_acceleration_normalized'].round(0).fillna(0), errors='coerce').astype(int)

        return df
    
    def calculate_sma_slope_normalized(self, df, timeframe, ma_period=50, slope_period=5, atr_period=14):
        """
        Calculates the slope of a Simple Moving Average, normalized by the ATR.

        This provides a robust measure of trend momentum by quantifying the MA's
        movement in terms of the asset's recent volatility (ATR).

        Args:
            df (pd.DataFrame): Input DataFrame.
            ma_period (int): The lookback period for the SMA.
            slope_period (int): The number of bars over which to measure the slope.
            atr_period (int): The lookback period for the ATR calculation.

        Returns:
            pd.DataFrame: DataFrame with 'sma_slope_norm' column added.
        """
        # 1. Calculate the necessary base indicators
        # Using pandas-ta's ATR for simplicity and correctness
        tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        
        atr = tr.rolling(window=atr_period).mean()
        
        # Calculate the SMA
        sma_series = pta.sma(df['close'], length=ma_period)

        # 2. Calculate the change in the SMA over the slope_period
        sma_diff = sma_series.diff(slope_period)
        
        # 3. Normalize the change by the ATR
        # We use .shift(1) on ATR to avoid using today's volatility to measure today's slope
        atr_val = atr.shift(1) 
        
        # Avoid division by zero
        atr_val = atr_val.replace(0, np.nan) 
        
        df[f'sma_slope_norm'] = sma_diff / atr_val
        
        return df
    
    # DATETIME SPLIT
    def datetime_split(self, df, timeframe):
        ts = df['timestamp']
        
        # First day of each month
        month_start = ts.dt.to_period('M').dt.start_time
        
        # Week of month (ISO style)
        week_of_month_iso = ts.dt.isocalendar().week - month_start.dt.isocalendar().week + 1
        
        new_columns = pd.DataFrame({
            'year': ts.dt.year,
            'month': ts.dt.month,
            'day': ts.dt.day,
            'week_of_month_def': (ts.dt.day - 1) // 7 + 1,
            'week_of_month_iso': week_of_month_iso,
            'day_of_year': ts.dt.dayofyear,
            'day_of_week': ts.dt.dayofweek,
            'hour': ts.dt.hour,
            'minute': ts.dt.minute,
            'week_of_year': ts.dt.isocalendar().week,
            'days_in_month': ts.dt.days_in_month,
            'quarter': ts.dt.quarter,
            'is_month_start': ts.dt.is_month_start,
            'is_month_end': ts.dt.is_month_end,
            'is_quarter_start': ts.dt.is_quarter_start,
            'is_quarter_end': ts.dt.is_quarter_end,
            'is_year_start': ts.dt.is_year_start,
            'is_year_end': ts.dt.is_year_end,
            'is_leap_year': ts.dt.is_leap_year
        }, index=df.index)
        
        df = pd.concat([df, new_columns], axis=1)
        
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_weekday'] = df['day_of_week'].isin([0, 1, 2, 3, 4])
        df['is_week_start'] = df['day_of_week'] == 0
        df['is_week_end'] = df['day_of_week'] == 6
        
        return df
    
    def hh_ll(self, df, timeframe, period=14):
        df['hh'] = df['high'].rolling(window=period).max()
        df['hl'] = df['low'].rolling(window=period).max()
        df['hc'] = df['close'].rolling(window=period).max()
        df['ho'] = df['open'].rolling(window=period).max()
        df['lh'] = df['high'].rolling(window=period).min()
        df['ll'] = df['low'].rolling(window=period).min()
        df['lc'] = df['close'].rolling(window=period).min()
        df['lo'] = df['open'].rolling(window=period).min()
        return df
    
    def calculate_hl_oscillator(self, df, timeframe, period=14):
        
        # Calculate higher high and lower low
        df['higher_high'] = df['high'].rolling(window=period).max()
        df['lower_low'] = df['low'].rolling(window=period).min()

        # Shift higher_high and lower_low to compare with the previous period
        df['higher_high_prev'] = df['higher_high'].shift(1)
        df['lower_low_prev'] = df['lower_low'].shift(1)

        # Check if current candle forms a new HH or LL
        df['hl_oscillator'] = np.where(df['close'] > df['higher_high_prev'], 1, 
                                np.where(df['close'] < df['lower_low_prev'], -1, 0))

        # Drop intermediate columns (optional, if you want to keep them, remove this line)
        df.drop(columns=['higher_high', 'lower_low', 'higher_high_prev', 'lower_low_prev'], inplace=True)

        return df
    
    def calculate_vix(self, df, timeframe, period=14):
        
        # Calculate the VIX
        df['vix'] = 100 * df['close'].pct_change().rolling(window=period).std() * np.sqrt(252)
        df['vix'] = pd.to_numeric(df['vix'].round(0).fillna(0), errors='coerce').astype(int)

        return df
    
    def market_regime(self, df, timeframe):
        """
        Calculate market regime type and score using vectorized operations.
        """
        # Step 1: Precompute all indicators for the entire DataFrame
        temp_df = self.calculate_sma(df, timeframe, 200)
        temp_df = self.calculate_ema(temp_df, timeframe, 200)
        temp_df = self.calculate_macd(temp_df, timeframe, 12, 26, 9)
        temp_df = self.calculate_adx(temp_df, timeframe, 14)
        temp_df = self.calculate_rsi(temp_df, timeframe, 14)
        temp_df = self.calculate_atr(temp_df, timeframe, 14)
        temp_df = self.calculate_bollinger_bands(temp_df, timeframe, 20, 2)
        temp_df = self.calculate_obv(temp_df, timeframe)
        temp_df = self.calculate_vix(temp_df, timeframe, 14)
        
        # Step 2: Precompute means for efficiency
        atr_mean = temp_df['atr'].mean()
        obv_mean = temp_df['obv'].mean()
        volume_mean = df['volume'].mean()
        
        # Step 3: Define regime type and score using vectorized operations
        conditions = [
            (temp_df['adx'] > 25) & (df['close'] > temp_df['sma_200']) & (df['close'] > temp_df['ema_200']) & (temp_df['macd_line'] > temp_df['signal_line']),  # Uptrend
            (temp_df['adx'] > 25) & (df['close'] < temp_df['sma_200']) & (df['close'] < temp_df['ema_200']) & (temp_df['macd_line'] < temp_df['signal_line']),  # Downtrend
            (temp_df['bb_width'] < 0.03) & (temp_df['rsi'].between(40, 60)),  # Consolidation
            (temp_df['atr'] > atr_mean * 1.5) | (temp_df['vix'] > 20)  # Volatile
        ]
        
        choices = ["Uptrend", "Downtrend", "Consolidation", "Volatile"]
        df['regime_type'] = np.select(conditions, choices, default="Unclear")
        
        # Step 4: Calculate regime score
        df['regime_score'] = 0.0
        df.loc[df['regime_type'] == "Uptrend", 'regime_score'] = (
            (df['close'] / temp_df['sma_200'] - 1) * 100 +
            (temp_df['macd_line'] - temp_df['signal_line']) +
            temp_df['adx'] / 100
        )
        df.loc[df['regime_type'] == "Downtrend", 'regime_score'] = (
            (df['close'] / temp_df['sma_200'] - 1) * 100 +
            (temp_df['signal_line'] - temp_df['macd_line']) -
            temp_df['adx'] / 100
        )
        df.loc[df['regime_type'] == "Consolidation", 'regime_score'] = -temp_df['bb_width'] * 100
        df.loc[df['regime_type'] == "Volatile", 'regime_score'] = (
            temp_df['atr'] / atr_mean +
            (temp_df['vix'] / 20).fillna(0)
        )
        
        # Step 5: Adjust scores based on RSI and OBV
        df['regime_score'] += np.where(temp_df['rsi'] > 70, (temp_df['rsi'] - 70) / 10, 0)
        df['regime_score'] -= np.where(temp_df['rsi'] < 30, (30 - temp_df['rsi']) / 10, 0)
        df['regime_score'] += np.where((temp_df['obv'] > obv_mean) & (df['volume'] > volume_mean), 0.5, 0)
        df['regime_score'] -= np.where((temp_df['obv'] < obv_mean) & (df['volume'] > volume_mean), 0.5, 0)
        
        df['regime_score'] = df['regime_score'].round(2)
        
        return df
    
    def calculate_market_regime(self, df, timeframe):
        
        temp_df = self.calculate_sma(df, timeframe, 9)
        temp_df = self.calculate_sma(temp_df, timeframe, 200)
        
        temp_df['ma_distance'] = temp_df['sma_9'] - temp_df['sma_200']
        
        temp_df = self.calculate_sma_slope(temp_df, timeframe, 200)
        
        conditions = [
            # Bullish: Close above both MAs and short MA above long MA
            (df['close'] > temp_df['sma_9']) & (df['close'] > temp_df['sma_200']) & (temp_df['sma_9'] > temp_df['sma_200']),
            
            # Bearish: Close below both MAs and short MA below long MA
            (df['close'] < temp_df['sma_9']) & (df['close'] < temp_df['sma_200']) & (temp_df['sma_9'] < temp_df['sma_200']),
            
            # Consolidation: Close between MAs OR short MA crossing long MA
            ((df['close'] > temp_df['sma_200']) & (df['close'] < temp_df['sma_9'])) |
            ((df['close'] < temp_df['sma_200']) & (df['close'] > temp_df['sma_9'])) |
            ((temp_df['sma_9'] > temp_df['sma_200']) & (df['close'] < temp_df['sma_200'])) |
            ((temp_df['sma_9'] < temp_df['sma_200']) & (df['close'] > temp_df['sma_200']))
        ]
        
        choices = ["BULLISH", "BEARISH", "CONSOLIDATION"]
        df['regime_type'] = np.select(conditions, choices, default="CONSOLIDATION")
        
        df['regime'] = np.where(df['regime_type'] == "BULLISH", 1, np.where(df['regime_type'] == "BEARISH", -1, 0))
        
        # Calculate relative_regime based on the last 14 rows
        def calculate_relative_regime(regime_series):
            
            # Count occurrences of each regime
            count_bullish = (regime_series == 1).sum()
            count_bearish = (regime_series == -1).sum()
            count_consolidation = (regime_series == 0).sum()
            
            # Determine the dominant regime
            max_count = max(count_bullish, count_bearish, count_consolidation)
            
            # Check for ties: if multiple regimes have the same max_count
            dominant_regimes = []
            if count_bullish == max_count:
                dominant_regimes.append(1)  # BULLISH
            if count_bearish == max_count:
                dominant_regimes.append(-1)  # BEARISH
            if count_consolidation == max_count:
                dominant_regimes.append(0)  # CONSOLIDATION
            
            # If there's a tie, prioritize the most recent regime
            if len(dominant_regimes) > 1:
                return regime_series.iloc[-1]  # Most recent regime breaks the tie
            
            # Otherwise, return the single dominant regime
            return dominant_regimes[0]
        
        # Apply rolling window to calculate relative_regime
        df['relative_regime'] = df['regime'].rolling(window=14, min_periods=1).apply(calculate_relative_regime, raw=False)
        
        # Normalize MA distance and slope for scoring
        ma_distance_norm = temp_df['ma_distance'] / temp_df['sma_200'].abs().rolling(window=200).max() # Relative distance
        slope_norm = temp_df['ma_slope'] / temp_df['ma_slope'].abs().rolling(window=20).max()  # Scale slope for better interpretability
        
        # Combine normalized distance and slope into a single score
        df['regime_score'] = (
            ma_distance_norm * 50 +  # Weighted contribution from MA distance
            slope_norm * 50          # Contribution from slope
        )
        
        # df['regime_score'] = df['regime_score'].clip(lower=-100, upper=100)
        
        # Round regime score to 2 decimal places
        df['regime_score'] = pd.to_numeric(df['regime_score'].round(0).fillna(0), errors='coerce').astype(int)
        
        return df
    
    def lag_ohlcv(self, df, timeframe, lag = 5):
        
        for i in range(lag):
            df[f'open_lag{i}'] = df['open'].shift(i)
            df[f'high_lag{i}'] = df['high'].shift(i)
            df[f'low_lag{i}'] = df['low'].shift(i)
            df[f'close_lag{i}'] = df['close'].shift(i)
            df[f'volume_lag{i}'] = df['volume'].shift(i)
        
        return df
    
    # in percentage format
    def candle_lengths(self, df, timeframe):
        # Calculate the total price range once and avoid division by zero by replacing zeros with NaN
        total_range = df['high'] - df['low']
        total_range = total_range.replace(0, np.nan)
        df['candle'] = total_range
        df['candle_pct_open'] = (df['candle'] / df['open']) * 100
        
        # Condition for bullish candles
        is_bull = df['close'] > df['open']
        df['is_bull'] = is_bull
        df['is_bear'] = df['open'] > df['close']
        df['is_indecisive'] = df['open'] == df['close']

        # Compute all candle components as percentages of total range
        df['upper_wick_pct'] = np.where(is_bull, 
                                        (df['high'] - df['close']) / total_range * 100, 
                                        (df['high'] - df['open']) / total_range * 100)
        df['lower_wick_pct'] = np.where(is_bull, 
                                        (df['open'] - df['low']) / total_range * 100, 
                                        (df['close'] - df['low']) / total_range * 100)
        df['body_pct'] = np.where(is_bull, 
                                  (df['close'] - df['open']) / total_range * 100, 
                                  (df['open'] - df['close']) / total_range * 100)
        df['body_upper_wick_pct'] = np.where(is_bull, 
                                             (df['high'] - df['open']) / total_range * 100, 
                                             (df['high'] - df['close']) / total_range * 100)
        df['body_lower_wick_pct'] = np.where(is_bull, 
                                             (df['close'] - df['low']) / total_range * 100, 
                                             (df['open'] - df['low']) / total_range * 100)
        df['wick_pct'] = np.where(is_bull,
                                  ((df['high'] - df['close']) + (df['open'] - df['low'])) / total_range * 100,
                                  ((df['high'] - df['open']) + (df['close'] - df['low'])) / total_range * 100)
        
        prev_close = df['close'].shift(1)
        
        # Compute the absolute percentage gap relative to the previous close.
        df['gap'] = np.where(prev_close != 0,
                    ((df['open'] - prev_close) / prev_close).abs() * 100,0)
        
        prev_candle = df['candle'].shift(1)
        df['candle_size_change_pct'] = (df['candle'] / prev_candle * 100).fillna(0)
        
        return df
    
    def price_movement(self, df, timeframe):
        
        df['prev_candle_pct_change'] = df['close'].pct_change().shift(1) * 100
        df['open_to_close_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['open_to_high_pct'] = (df['high'] - df['open']) / df['open'] * 100
        df['open_to_low_pct'] = (df['low'] - df['open']) / df['open'] * 100

        # Calculate the momentum of the candle move (current reaction compared to the previous candle)
        df['price_momentum'] = df['open_to_close_pct'] - df['open_to_close_pct'].shift(1)

        # Additional indicator: the candle's intraday range as a percentage of the open
        df['intracandle_range_pct'] = (df['high'] - df['low']) / df['open'] * 100

        # Optionally, smooth the momentum with a simple moving average over 5 periods
        df['momentum_sma'] = df['price_momentum'].rolling(window=5, min_periods=1).mean()
        
        return df
        
    def economic_data(self, df, timeframe):
        
        # Ensure the 'timestamp' column is datetime and sort the DataFrame by it
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        start, end = df['timestamp'].iloc[0], df['timestamp'].iloc[-1]

        def process_fred_series(symbol, new_col, start, end, resample_freq=None):
            series_df = web.DataReader(symbol, 'fred', start, end)

            # Reset the index and extract the 'DATE' index as a 'timestamp' column
            series_df.reset_index(inplace=True)
                        
            # Rename the date column to 'timestamp' if it's named differently
            if 'DATE' in series_df.columns:  # FRED often uses 'DATE' for the date column
                series_df.rename(columns={'DATE': 'timestamp'}, inplace=True)
                
            # Convert 'timestamp' column to datetime if not already
            series_df['timestamp'] = pd.to_datetime(series_df['timestamp'])

            if resample_freq:
                # Set 'timestamp' as the index before resampling
                series_df.set_index('timestamp', inplace=True)
                # Resample and forward-fill
                series_df = series_df.resample(resample_freq).ffill().reset_index()
            
            # Rename the data column to the desired new column name.
            series_df.rename(columns={symbol: new_col}, inplace=True)
            return series_df

        # Process different FRED series with desired resample frequencies
        econ_series = {
            'GDP': {'col': 'us_gdp', 'freq': 'QS'},   # GDP is typically quarterly
            'FF': {'col': 'ff_rate', 'freq': None},    # FF rate is available daily; no resample needed
            'CPIAUCSL': {'col': 'cpi', 'freq': 'ME'}     # CPI is typically monthly
        }

        df.set_index('timestamp', inplace=True)

        # Merge all processed economic series into the main DataFrame
        for symbol, params in econ_series.items():
            series_df = process_fred_series(symbol, params['col'], start, end, params['freq'])
            
            series_df.set_index('timestamp', inplace=True)
            
            df = pd.merge_asof(df, series_df, on='timestamp', direction='backward')
        
        return df
    
    # REGIME FILTERS
    # def calculate_regime_filters(self, df, timeframe):
        
        # TREND FILTER (Using MA, MA SLOPE, SUPERTREND, PSAR, PIVOT POINTS, MACD, ADX)
        # LOOKBACK FILTER (Using HH, LL, HC, LC, Percentage change, Candle Lengths)
        # VOLATILITY FILTER (Using ATR, Bollinger Bands, Keltner Channels)
        # VOLUME FILTER (Using OBV, CMF, MFI, VOLUME)
        # MOMENTUM FILTER (Using RSI, STOCH, CCI, WILLIAMS %R)
        # STRENGTH FILTER (Using VIX, VIX FIX, VIX EMA, MAJORITY OF INDICATORS)
        # CORRELATION FILTER (Using BTC, ETH)
        # CYCLE FILTER (Using Bitcoin Halving, 4 Year Cycle)
        # TIME OF DAY FILTER (Using Hour, Day of Week, Month)
        
        # """
        # Calculates a comprehensive set of regime filters based on various technical indicators.

        # This function enriches the DataFrame with detailed columns for different market regimes,
        # including Trend, Volatility, Momentum, and Volume. For each indicator used, it provides
        # both a qualitative state (e.g., 'BULLISH', 'LOW') and a quantitative measure of
        # confidence or strength (e.g., a percentage or normalized score).

        # Args:
        #     df (pd.DataFrame): The input DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns.

        # Returns:
        #     pd.DataFrame: The DataFrame with added regime filter columns.
        # """
        # # --- 1. Pre-calculation of all necessary base indicators ---
        # # This ensures all required columns are available before calculating filters.
        # df = self.calculate_sma(df, length=200)
        # df = self.calculate_sma_slope(df, length=200, smoothbars=3, hlineheight=10)
        # df = self.supertrend(df, period=10, multiplier=3.0)
        # df = self.calculate_psar(df)
        # df = self.calculate_macd(df)
        # df = self.calculate_adx(df, length=14)
        # df = self.calculate_atr(df, length=14, smoothing='sma')
        # df = self.calculate_bollinger_bands(df, length=20)
        # df = self.calculate_rsi(df, length=14)
        # df = self.calculate_stocastic(df, k_length=14, d_length=3)
        # df = self.calculate_cci(df, length=20)
        # df = self.williams_range(df, period=14)
        # df = self.calculate_obv(df)
        
        # # --- 2. TREND FILTERS ---
        
        # # SMA Trend
        # df['sma_trend_direction'] = np.where(df['close'] > df['sma_200'], 'BULLISH', 'BEARISH')
        # df['sma_trend_strength'] = abs(df['close'] - df['sma_200']) / df['sma_200'] * 100
        
        # # MA Slope Trend
        # slope_conditions = [df['ma_trend'] == 1, df['ma_trend'] == -1]
        # slope_choices = ['BULLISH', 'BEARISH']
        # df['slope_trend_direction'] = np.select(slope_conditions, slope_choices, default='SIDEWAYS')
        # df['slope_trend_strength'] = abs(df['ma_slope'])

        # # Supertrend
        # st_conditions = [df['supertrend'] == 1, df['supertrend'] == -1]
        # st_choices = ['BULLISH', 'BEARISH']
        # df['supertrend_direction'] = np.select(st_conditions, st_choices, default='SIDEWAYS')
        # st_distance = np.where(df['supertrend_direction'] == 'BULLISH', 
        #                         df['close'] - df['supertrend_lowerband'], 
        #                         df['supertrend_upperband'] - df['close'])
        # df['supertrend_strength'] = (st_distance / df['atr']).clip(0, 10) * 10 # Scaled strength

        # # PSAR Trend
        # df['psar_trend_direction'] = np.where(df['close'] > df['psar'], 'BULLISH', 'BEARISH')
        # psar_distance = abs(df['close'] - df['psar'])
        # df['psar_trend_strength'] = (psar_distance / df['atr']).clip(0, 10) * 10 # Scaled strength

        # # MACD Trend
        # df['macd_trend_direction'] = np.where(df['macd_histogram'] > 0, 'BULLISH', 'BEARISH')
        # # Normalize histogram by its rolling max absolute value over 100 periods for a confidence score
        # rolling_max_hist = df['macd_histogram'].abs().rolling(window=100, min_periods=1).max()
        # df['macd_trend_confidence'] = (df['macd_histogram'].abs() / rolling_max_hist * 100).fillna(0)
        
        # # ADX Trend Strength (ADX is non-directional, it measures strength)
        # adx_conditions = [df['adx'] > 25, df['adx'] < 20]
        # adx_choices = ['TRENDING', 'WEAK/RANGING']
        # df['adx_trend_status'] = np.select(adx_conditions, adx_choices, default='DEVELOPING')
        # df['adx_trend_strength'] = df['adx']
        
        # # --- 3. VOLATILITY FILTERS ---
        
        # # ATR Volatility
        # atr_q75 = df['atr_pct'].rolling(window=200).quantile(0.75)
        # atr_q25 = df['atr_pct'].rolling(window=200).quantile(0.25)
        # atr_conditions = [df['atr_pct'] > atr_q75, df['atr_pct'] < atr_q25]
        # atr_choices = ['HIGH', 'LOW']
        # df['atr_volatility_level'] = np.select(atr_conditions, atr_choices, default='MODERATE')
        # df['atr_volatility_value'] = df['atr_pct'] # ATR as a percentage of price
        
        # # Bollinger Bands Volatility (Squeeze/Expansion)
        # bb_width_q75 = df['bb_width'].rolling(window=200).quantile(0.75)
        # bb_width_q25 = df['bb_width'].rolling(window=200).quantile(0.25)
        # bb_conditions = [df['bb_width'] > bb_width_q75, df['bb_width'] < bb_width_q25]
        # bb_choices = ['EXPANSION', 'SQUEEZE'] # High width = Expansion, Low width = Squeeze
        # df['bb_volatility_level'] = np.select(bb_conditions, bb_choices, default='NORMAL')
        # df['bb_volatility_value'] = df['bb_width'] * 100 # BB width as percentage

        # # --- 4. MOMENTUM FILTERS ---
        
        # # RSI Momentum
        # rsi_conditions = [df['rsi'] > 70, df['rsi'] < 30]
        # rsi_choices = ['OVERBOUGHT', 'OVERSOLD']
        # df['rsi_momentum_level'] = np.select(rsi_conditions, rsi_choices, default='NEUTRAL')
        # df['rsi_momentum_value'] = df['rsi']
        
        # # Stochastic Momentum
        # stoch_conditions = [df['stocastic_%k'] > 80, df['stocastic_%k'] < 20]
        # stoch_choices = ['OVERBOUGHT', 'OVERSOLD']
        # df['stoch_momentum_level'] = np.select(stoch_conditions, stoch_choices, default='NEUTRAL')
        # df['stoch_momentum_value'] = df['stocastic_%k']

        # # CCI Momentum
        # cci_conditions = [df['cci'] > 100, df['cci'] < -100]
        # cci_choices = ['OVERBOUGHT', 'OVERSOLD']
        # df['cci_momentum_level'] = np.select(cci_conditions, cci_choices, default='NEUTRAL')
        # df['cci_momentum_value'] = df['cci']

        # # Williams %R Momentum
        # wr_conditions = [df['williams_range'] > -20, df['williams_range'] < -80]
        # wr_choices = ['OVERBOUGHT', 'OVERSOLD']
        # df['wr_momentum_level'] = np.select(wr_conditions, wr_choices, default='NEUTRAL')
        # df['wr_momentum_value'] = df['williams_range']

        # # --- 5. VOLUME FILTERS ---

        # # Volume Spike/Lull
        # vol_sma_50 = df['volume'].rolling(window=50, min_periods=1).mean()
        # vol_conditions = [df['volume'] > (vol_sma_50 * 2), df['volume'] < (vol_sma_50 * 0.5)]
        # vol_choices = ['HIGH', 'LOW']
        # df['volume_level'] = np.select(vol_conditions, vol_choices, default='NORMAL')
        # df['volume_ratio_to_avg'] = (df['volume'] / vol_sma_50).fillna(1.0) # Ratio to 50-period average

        # # On-Balance Volume (OBV)
        # obv_sma_20 = df['obv'].rolling(window=20, min_periods=1).mean()
        # df['obv_direction'] = np.where(df['obv'] > obv_sma_20, 'BULLISH', 'BEARISH')
        # # Use slope of OBV as a measure of strength/confidence
        # obv_slope = df['obv'].diff()
        # rolling_std_obv = obv_slope.abs().rolling(window=50, min_periods=1).mean()
        # df['obv_strength'] = (obv_slope / rolling_std_obv * 50).clip(-100, 100).fillna(0)


        # return df
    
    # REGIME FILTERS
    def _normalize_min_max(self, series, window):
        """Helper to normalize a series to 0-100 using rolling min-max."""
        roll_min = series.rolling(window=window, min_periods=1).min()
        roll_max = series.rolling(window=window, min_periods=1).max()
        # Avoid division by zero
        denominator = roll_max - roll_min
        normalized = (series - roll_min) / denominator.replace(0, np.nan) * 100
        return normalized.fillna(50) # Fill NaNs with a neutral 50

    def _normalize_quantile(self, series, window):
        """Helper to normalize a series to 0-100 using rolling quantile rank."""
        return series.rolling(window=window, min_periods=1).rank(pct=True) * 100

    def calculate_regime_filters(self, df, timeframe):
        """
        Calculates a comprehensive set of regime filters with normalized scores (0-100).

        This function enriches the DataFrame with detailed columns for different market regimes,
        including Trend, Volatility, Momentum, and Volume. For each indicator, it provides:
        1. A qualitative state (e.g., 'BULLISH', 'HIGH').
        2. A quantitative, normalized score from 0-100 for easy interpretation and comparison.

        Args:
            df (pd.DataFrame): The input DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns.

        Returns:
            pd.DataFrame: The DataFrame with added regime filter columns.
        """
        norm_window = 200 # Lookback period for normalization

        # --- 1. Pre-calculation of all necessary base indicators ---
        df = self.calculate_sma(df, timeframe, length=200)
        df = self.calculate_sma_slope(df, timeframe, length=200, smoothbars=3, hlineheight=10)
        df = self.supertrend(df, timeframe, period=10, multiplier=3.0)
        df = self.calculate_psar(df, timeframe)
        df = self.calculate_macd(df, timeframe)
        df = self.calculate_adx(df, timeframe, length=14)
        df = self.calculate_atr(df, timeframe, length=14, smoothing='sma')
        df = self.calculate_bollinger_bands(df, timeframe, length=20)
        df = self.calculate_rsi(df, timeframe, length=14)
        df = self.calculate_stocastic(df, timeframe, k_length=14, d_length=3)
        df = self.calculate_cci(df, timeframe, length=20)
        df = self.williams_range(df, timeframe, period=14)
        df = self.calculate_obv(df, timeframe)

        # --- 2. TREND FILTERS ---

        # SMA Trend
        df['sma_trend_direction'] = np.where(df['close'] > df['sma_200'], 'BULLISH', 'BEARISH')
        # Prevent division by zero if sma_200 is 0
        sma_200_safe = df['sma_200'].replace(0, np.nan)
        sma_distance = abs(df['close'] - sma_200_safe) / sma_200_safe * 100
        df['sma_trend_score'] = self._normalize_quantile(sma_distance, norm_window)

        # MA Slope Trend
        slope_conditions = [df['ma_trend'] == 1, df['ma_trend'] == -1]
        slope_choices = ['BULLISH', 'BEARISH']
        df['slope_trend_direction'] = np.select(slope_conditions, slope_choices, default='SIDEWAYS')
        df['slope_trend_score'] = self._normalize_min_max(abs(df['ma_slope']), norm_window)

        # Supertrend
        st_conditions = [df['supertrend'] == 1, df['supertrend'] == -1]
        st_choices = ['BULLISH', 'BEARISH']
        df['supertrend_direction'] = np.select(st_conditions, st_choices, default='SIDEWAYS')
        st_distance_np = np.where(df['supertrend_direction'] == 'BULLISH',
                                df['close'] - df['supertrend_lowerband'],
                                df['supertrend_upperband'] - df['close'])
        st_distance = pd.Series(st_distance_np, index=df.index)
        df['supertrend_score'] = self._normalize_min_max(st_distance, norm_window)

        # PSAR Trend
        df['psar_trend_direction'] = np.where(df['close'] > df['psar'], 'BULLISH', 'BEARISH')
        psar_distance = abs(df['close'] - df['psar'])
        df['psar_trend_score'] = self._normalize_min_max(psar_distance, norm_window)

        # MACD Trend
        df['macd_trend_direction'] = np.where(df['macd_histogram'] > 0, 'BULLISH', 'BEARISH')
        df['macd_trend_score'] = self._normalize_min_max(abs(df['macd_histogram']), norm_window)

        # ADX Trend Strength
        adx_conditions = [df['adx'] > 25, df['adx'] < 20]
        adx_choices = ['TRENDING', 'WEAK/RANGING']
        df['adx_trend_status'] = np.select(adx_conditions, adx_choices, default='DEVELOPING')
        df['adx_trend_score'] = df['adx']

        # --- 3. VOLATILITY FILTERS ---

        # ATR Volatility
        atr_q75 = df['atr_norm'].rolling(window=norm_window).quantile(0.75)
        atr_q25 = df['atr_norm'].rolling(window=norm_window).quantile(0.25)
        atr_conditions = [df['atr_norm'] > atr_q75, df['atr_norm'] < atr_q25]
        atr_choices = ['HIGH', 'LOW']
        df['atr_volatility_level'] = np.select(atr_conditions, atr_choices, default='MODERATE')
        df['atr_volatility_score'] = self._normalize_quantile(df['atr_norm'], norm_window)

        # Bollinger Bands Volatility
        bb_width_q75 = df['bb_width'].rolling(window=norm_window).quantile(0.75)
        bb_width_q25 = df['bb_width'].rolling(window=norm_window).quantile(0.25)
        bb_conditions = [df['bb_width'] > bb_width_q75, df['bb_width'] < bb_width_q25]
        bb_choices = ['EXPANSION', 'SQUEEZE']
        df['bb_volatility_level'] = np.select(bb_conditions, bb_choices, default='NORMAL')
        df['bb_volatility_score'] = self._normalize_quantile(df['bb_width'], norm_window)

        # --- 4. MOMENTUM FILTERS ---

        # RSI Momentum
        rsi_conditions = [df['rsi'] > 70, df['rsi'] < 30]
        rsi_choices = ['OVERBOUGHT', 'OVERSOLD']
        df['rsi_momentum_level'] = np.select(rsi_conditions, rsi_choices, default='NEUTRAL')
        df['rsi_momentum_score'] = df['rsi']

        # Stochastic Momentum
        stoch_conditions = [df['stocastic_%k'] > 80, df['stocastic_%k'] < 20]
        stoch_choices = ['OVERBOUGHT', 'OVERSOLD']
        df['stoch_momentum_level'] = np.select(stoch_conditions, stoch_choices, default='NEUTRAL')
        df['stoch_momentum_score'] = df['stocastic_%k']

        # CCI Momentum
        cci_conditions = [df['cci'] > 100, df['cci'] < -100]
        cci_choices = ['OVERBOUGHT', 'OVERSOLD']
        df['cci_momentum_level'] = np.select(cci_conditions, cci_choices, default='NEUTRAL')
        df['cci_momentum_score'] = self._normalize_min_max(abs(df['cci']), norm_window)

        # Williams %R Momentum
        wr_conditions = [df['williams_range'] > -20, df['williams_range'] < -80]
        wr_choices = ['OVERBOUGHT', 'OVERSOLD']
        df['wr_momentum_level'] = np.select(wr_conditions, wr_choices, default='NEUTRAL')
        df['wr_momentum_score'] = abs(df['williams_range'])

        # --- 5. VOLUME FILTERS ---

        # Volume Spike/Lull
        vol_sma_50 = df['volume'].rolling(window=50, min_periods=1).mean()
        vol_conditions = [df['volume'] > (vol_sma_50 * 2), df['volume'] < (vol_sma_50 * 0.5)]
        vol_choices = ['HIGH', 'LOW']
        df['volume_level'] = np.select(vol_conditions, vol_choices, default='NORMAL')
        df['volume_score'] = self._normalize_quantile(df['volume'], norm_window)

        # On-Balance Volume (OBV)
        obv_sma_20 = df['obv'].rolling(window=20, min_periods=1).mean()
        df['obv_direction'] = np.where(df['obv'] > obv_sma_20, 'BULLISH', 'BEARISH')
        obv_slope = df['obv'].diff().fillna(0)
        df['obv_score'] = self._normalize_min_max(abs(obv_slope), norm_window)
        
        # --- FIX: Final robust cleanup before casting to integer ---
        score_cols = [col for col in df.columns if col.endswith('_score')]
        
        # Step 1: Replace infinite values with NaN
        df[score_cols] = df[score_cols].replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Fill any and all remaining NaNs with a neutral value (50)
        df[score_cols] = df[score_cols].fillna(50)
        
        # Step 3: Now it's safe to round and cast to integer
        df[score_cols] = df[score_cols].round(0).astype(int)
        
        df['Composite_Regime_Score'] = df["sma_trend_score"] + df["macd_trend_score"] + df["rsi_momentum_score"] + df["volume_score"]
        
        return df
    
    # ACTIVITY FILTER
    # 0 = Low Activity, 1 = High Activity
    def activity_filter(self, df, timeframe, window = 50, thresh=0.05):
        
        df = self.calculate_adx(df, timeframe, length=14)
        df = self.calculate_atr(df, timeframe, length=14, smoothing='sma')
        df = self.calculate_bollinger_bands(df, timeframe, length=20)
        
        adx_mean = df['adx'].rolling(window=window, min_periods=1).mean()
        volume_mean = df['volume'].rolling(window=window, min_periods=1).mean()
        
        # Store the rolling mean values in the DataFrame
        df['adx_rolling_mean'] = adx_mean
        df['volume_rolling_mean'] = volume_mean
        
        adx_mean_thresh = (1 + thresh) * adx_mean   # require ADX to be 5% above mean
        vol_mean_thresh = (1 + thresh) * volume_mean
        
        df['Activity_Mean_Thresh'] = np.where(
            (df['adx'] < adx_mean_thresh) & (df['volume'] < vol_mean_thresh), 0, 1
        ).astype(np.int64)
        
        df['Activity_Mean'] = np.where((df['adx'] < adx_mean) & (df['volume'] < volume_mean), 0, 1).astype(np.int64)
        
        df['Activity_Mean_Smoothed'] = (
            df['Activity_Mean'].rolling(window=5, min_periods=1).mean().round().astype(int)
        )
        
        adx_mid = df['adx'].rolling(window=window, min_periods=1).median()
        volume_mid = df['volume'].rolling(window=window, min_periods=1).median()
        
        # Store the rolling median values in the DataFrame
        df['adx_rolling_median'] = adx_mid
        df['volume_rolling_median'] = volume_mid
        
        adx_mid_thresh = (1 + thresh) * adx_mid   # require ADX to be 5% above mean
        vol_mid_thresh = (1 + thresh) * volume_mid
        
        df['Activity_Median_Thresh'] = np.where(
            (df['adx'] < adx_mid_thresh) & (df['volume'] < vol_mid_thresh), 0, 1
        ).astype(np.int64)
        
        df['Activity_Median'] = np.where((df['adx'] < adx_mid) & (df['volume'] < volume_mid), 0, 1).astype(np.int64)
        
        df['Activity_Median_Smoothed'] = (
            df['Activity_Median'].rolling(window=5, min_periods=1).mean().round().astype(int)
        )
        
        adx_z = (df['adx'] - adx_mean) / (df['adx'].rolling(window).std(ddof=0) + 1e-9)
        vol_z = (df['volume'] - volume_mean) / (df['volume'].rolling(window).std(ddof=0) + 1e-9)
        
        df['Activity_Z'] = (adx_z + vol_z) / 2
        df['Activity_Binary'] = np.where(df['Activity_Z'] < 0, 0, 1)
        
        df['Meta_Activity'] = (
            df['Activity_Median'] + 
            (df['atr'] > df['atr'].rolling(50).mean()).astype(int) +
            (df['bb_width'] > df['bb_width'].rolling(50).mean()).astype(int)
        ) / 3
        
        return df
    
    def calculate_spike(self, df, timeframe, lookback = 25):
        """
        Calculates both an absolute and a directional price spike over a lookback period.

        - 'spike_abs': The absolute range (highest high - lowest low) over the
                        lookback period, as a percentage of the current close. This
                        measures volatility.
        - 'spike_directional': The same range, but signed based on the net
                                price movement (close vs. open of the lookback window).
                                Positive indicates an upward trend, negative a downward trend.

        Args:
            df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close', 'open'.
            lookback (int): The number of periods to look back.

        Returns:
            pd.DataFrame: DataFrame with 'spike_abs' and 'spike_directional' columns.
        """
        # Find the highest high and lowest low over the lookback window.
        rolling_window_high = df['high'].rolling(window=lookback)
        rolling_window_low = df['low'].rolling(window=lookback)

        # Get the values
        higher_high = rolling_window_high.max()
        lower_low = rolling_window_low.min()

        # Get the index labels of the max/min
        # Note: idxmax/idxmin return the *first* occurrence in case of a tie.
        # Using .apply() for compatibility with older pandas versions.
        high_idx_labels = rolling_window_high.apply(lambda x: x.idxmax(), raw=False)
        low_idx_labels = rolling_window_low.apply(lambda x: x.idxmin(), raw=False)

        # Create a series of integer positions for fast lookup
        row_idx = pd.Series(range(len(df)), index=df.index)

        # Map the index labels back to integer positions using searchsorted
        # Fill NA values that can result from .apply() on initial windows
        high_pos = df.index.searchsorted(high_idx_labels.ffill())
        low_pos = df.index.searchsorted(low_idx_labels.ffill())

        # Calculate how many periods ago the high/low occurred
        # This is the difference between the current row's position and the extreme's position
        df['spike_high_index'] = row_idx - high_pos
        df['spike_low_index'] = row_idx - low_pos

        # Store the high and low values
        df['spike_low'] = lower_low
        df['spike_high'] = higher_high
        
        # Calculate the absolute range of the spike.
        spike_range = higher_high - lower_low

        # Avoid division by zero by replacing 0 with NaN before division.
        close_safe = df['close'].replace(0, np.nan)

        # 1. Calculate the absolute spike (volatility measure).
        df['spike_abs'] = (spike_range / close_safe) * 100

        # 2. Calculate the directional spike.
        # Determine the direction by the net change from the start of the window to the current close.
        open_of_window = df['open'].shift(lookback - 1)
        net_change = df['close'] - open_of_window
        
        # Get the sign of the net change (-1 for down, 1 for up, 0 for no change).
        direction = np.sign(net_change)

        # Apply the direction to the absolute spike value.
        df['spike_directional'] = direction * df['spike_abs']

        return df

    def calculate_consolidation(self, df, timeframe, change=5):
        """
        Calculates the consolidation period for each bar in a DataFrame.
        
        For each bar, it looks back to find the longest period (in number of bars)
        where the price has stayed within a specified percentage range ('change').
        It adds the consolidation period length, and the high and low of that
        period to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'High' and 'Low' columns.
            change (float): The maximum allowed percentage change for a
                            consolation period. E.g., 5 means a 5% range.

        Returns:
            pd.DataFrame: The original DataFrame with three new columns:
                            'consolidation_period', 'consolidation_low',
                            and 'consolidation_high'.
        """
        # Ensure we're working with a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Initialize lists to store the results for each bar
        consolidation_periods = []
        consolidation_lows = []
        consolidation_highs = []

        # Convert columns to numpy arrays for faster access in the loop
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        
        # Iterate over each bar (row) in the DataFrame
        for i in range(len(df)):
            # These will store the results for the current bar `i`
            # A single bar is always in a "consolidation" of period 1 with itself.
            longest_period = 0
            period_low = np.nan
            period_high = np.nan

            # Look backwards from the current bar `i` to the first bar `0`
            for j in range(i, -1, -1):
                # Define the current lookback window
                window_highs = highs[j:i+1]
                window_lows = lows[j:i+1]

                # Find the max high and min low in the window
                current_max_high = np.max(window_highs)
                current_min_low = np.min(window_lows)
                
                # Avoid division by zero if low is 0
                if current_min_low == 0:
                    break

                # Check if the price range is within the allowed percentage change
                if ((current_max_high - current_min_low) / current_min_low * 100) <= change:
                    # If it is, this is a valid consolidation period.
                    # We store its details and continue the loop to check for a longer one.
                    longest_period = i - j + 1
                    period_low = current_min_low
                    period_high = current_max_high
                else:
                    # If the range is too wide, the consolidation is broken.
                    # We stop looking back for this bar `i`.
                    break
            
            # Append the results for the longest valid period found for bar `i`
            consolidation_periods.append(longest_period)
            consolidation_lows.append(period_low)
            consolidation_highs.append(period_high)
        
        # Add the results as new columns to the DataFrame
        # The prompt used 'consolidation_period' so we will match that
        df['consolidation_period'] = consolidation_periods
        df['consolidation_low'] = consolidation_lows
        df['consolidation_high'] = consolidation_highs
        
        return df
    
    def calculate_zero_lag_trend_signals(self, df, timeframe, length=70, mult=1.2):
        """
        Calculates the Zero Lag Trend Signals indicator values and signals.
        
        This Zero Lag Trend Signals focusing on the calculation and signal generation aspects.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close' columns.
            length (int): The look-back window for ZLEMA calculations.
            mult (float): The multiplier for the volatility bands.
        
        Returns:
            pd.DataFrame: The DataFrame with the following columns added:
                - zlema: The Zero-Lag Exponential Moving Average.
                - upper_band: The upper volatility band.
                - lower_band: The lower volatility band.
                - trend: The main trend state (1 for Bullish, -1 for Bearish).
                - trend_reversal: Signals trend changes (1 for bull, -1 for bear, 0 otherwise).
                - entry_signal: Signals continuation entries (1 for bull, -1 for bear, 0 otherwise).
        """
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'high', 'low', and 'close' columns.")
        
        # 1. Core Calculations (ZLEMA and Volatility)
        lag = (length - 1) // 2
        
        # Calculate the modified source for the ZLEMA
        zlema_src = df['close'] + (df['close'] - df['close'].shift(lag))
        
        # Calculate the ZLEMA using pandas-ta
        df[f'zlema_{length}'] = pta.ema(zlema_src, length=length)
        
        # Calculate ATR for volatility
        atr = pta.atr(df['high'], df['low'], df['close'], length=length)
        
        # Calculate volatility as the highest ATR in the lookback window, scaled by the multiplier
        volatility = atr.rolling(window=length * 3).max() * mult
        
        # Calculate the upper and lower bands (the zones)
        df[f'zlema_upper_band_{length}'] = df[f'zlema_{length}'] + volatility
        df[f'zlema_lower_band_{length}'] = df[f'zlema_{length}'] - volatility
        
        # 2. Trend Detection
        # Identify the exact bar where the price crosses the bands
        bull_cross = (df['close'] > df[f'zlema_upper_band_{length}']) & (df['close'].shift(1) <= df[f'zlema_upper_band_{length}'].shift(1))
        bear_cross = (df['close'] < df[f'zlema_lower_band_{length}']) & (df['close'].shift(1) >= df[f'zlema_lower_band_{length}'].shift(1))
        
        # Create a series that is 1 on a bullish cross, -1 on a bearish cross, and NaN otherwise
        trend_signal = np.select([bull_cross, bear_cross], [1, -1], default=np.nan)
        
        # Forward-fill the signal to maintain the trend state, then fill initial NaNs with 0
        df[f'zlema_trend_{length}'] = pd.Series(trend_signal, index=df.index).ffill().fillna(0).astype(int)
        
        # 3. Signal Generation
        
        # Trend Reversal Signals (Big Arrows)
        # Bullish reversal: trend changes from -1 to 1
        bull_reversal = (df[f'zlema_trend_{length}'] == 1) & (df[f'zlema_trend_{length}'].shift(1) == -1)
        # Bearish reversal: trend changes from 1 to -1
        bear_reversal = (df[f'zlema_trend_{length}'] == -1) & (df[f'zlema_trend_{length}'].shift(1) == 1)
        
        df[f'zlema_trend_reversal_{length}'] = np.select([bull_reversal, bear_reversal], [1, -1], default=0).astype(int)
        
        # Entry Signals (Small Arrows)
        # Crossover/under of the zlema line itself
        bull_entry_cross = (df['close'] > df[f'zlema_{length}']) & (df['close'].shift(1) <= df[f'zlema_{length}'].shift(1))
        bear_entry_cross = (df['close'] < df[f'zlema_{length}']) & (df['close'].shift(1) >= df[f'zlema_{length}'].shift(1))
        
        # Condition: cross must happen while the trend is already established for two bars
        is_bull_trend_established = (df[f'zlema_trend_{length}'] == 1) & (df[f'zlema_trend_{length}'].shift(1) == 1)
        is_bear_trend_established = (df[f'zlema_trend_{length}'] == -1) & (df[f'zlema_trend_{length}'].shift(1) == -1)
        
        bull_entry_condition = bull_entry_cross & is_bull_trend_established
        bear_entry_condition = bear_entry_cross & is_bear_trend_established
        
        df[f'zlema_entry_signal_{length}'] = np.select([bull_entry_condition, bear_entry_condition], [1, -1], default=0).astype(int)
        
        return df
    
    def calculate_vortex_indicator(self, df, timeframe, length=14):
        """
        Calculates the Vortex Indicator (VI) and its signals.
        
        The Vortex Indicator consists of two lines, VI+ and VI-, which help identify
        the start of a new trend or the continuation of an existing trend. This implementation
        also generates buy and sell signals based on crossovers of these lines.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close' columns.
            length (int): The look-back period for the VI calculation.
        
        Returns:
            pd.DataFrame: The DataFrame with the following columns added:
                - vi_plus: The VI+ line.
                - vi_minus: The VI- line.
                - vi_trend: The main trend state (1 for Bullish, -1 for Bearish, 0 for Neutral).
                - vi_signal: Signals trend changes (1 for buy, -1 for sell, 0 otherwise).
        """
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'high', 'low', and 'close' columns.")
        
        # Calculate True Range (TR)
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Positive and Negative Vortex Movement
        vm_plus = (df['high'] - df['high'].shift(1)).abs()
        vm_minus = (df['low'].shift(1) - df['low']).abs()
        
        # Sum TR, VM+ and VM- over the specified length
        tr_sum = tr.rolling(window=length).sum()
        vm_plus_sum = vm_plus.rolling(window=length).sum()
        vm_minus_sum = vm_minus.rolling(window=length).sum()
        
        # Calculate VI+ and VI-
        df['vi_plus'] = vm_plus_sum / tr_sum
        df['vi_minus'] = vm_minus_sum / tr_sum
        
        # 2. Trend Detection
        # Identify crossovers
        bull_cross = (df['vi_plus'] > df['vi_minus']) & (df['vi_plus'].shift(1) <= df['vi_minus'].shift(1))
        bear_cross = (df['vi_plus'] < df['vi_minus']) & (df['vi_plus'].shift(1) >= df['vi_minus'].shift(1))
        
        # Create a series that is 1 on a bullish cross, -1 on a bearish cross, and NaN otherwise
        trend_signal = np.select([bull_cross, bear_cross], [1, -1], default=np.nan)
        
        # Forward-fill the signal to maintain the trend state, then fill initial NaNs with 0
        df['vi_trend'] = pd.Series(trend_signal, index=df.index).ffill().fillna(0).astype(int)
        
        # 3. Signal Generation
        
        # Trend Change Signals
        bull_reversal = (df['vi_trend'] == 1) & (df['vi_trend'].shift(1) == -1)
        bear_reversal = (df['vi_trend'] == -1) & (df['vi_trend'].shift(1) == 1)
        df['vi_signal'] = np.select([bull_reversal, bear_reversal], [1, -1], default=0).astype(int)
        
        return df
    
    def custom_trend_filter(self, df, timeframe):
        """
        A custom trend filter combining multiple indicators to determine overall trend direction.
        
        This function uses SMA, MACD, SuperTrend, Ichimoku, Bollinger Bands and ADX to generate a composite trend signal.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'open', 'high', 'low', 'close', 'volume'
        
        Returns:
            pd.DataFrame: The DataFrame with the following columns added:
                - trend_filter: The overall trend direction (1 for Bullish, -1 for Bearish, 0 for Neutral).
        """
        
        # Ensure all necessary columns are present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain the following columns: {required_cols}")
        
        # Calculate necessary indicators
        df = self.calculate_sma(df, timeframe, length=100)
        df = self.calculate_macd(df, timeframe)
        df = self.supertrend(df, timeframe)
        df = self.calculate_ichimoku(df, timeframe)
        df = self.calculate_bollinger_bands(df, timeframe)
        df = self.calculate_adx(df, timeframe)
        
        # Initialize trend score
        df['trend_score'] = 0
        
        # 1. SMA Trend
        df['sma_trend'] = np.where(df['close'] > df[f'sma_100_{timeframe}'], 1, -1)
        df['trend_score'] += df['sma_trend']
        
        # 2. MACD Trend
        df['macd_trend'] = np.where(df['macd_line'] > 0, 1, -1)
        df['signal_trend'] = np.where(df['signal_line'] > 0, 1, -1)
        
        df['trend_score'] += df['macd_trend']
        df['trend_score'] += df['signal_trend']
        
        # 3. SuperTrend
        df['supertrend_trend'] = np.where(df['supertrend'] == 1, 1, -1)
        df['trend_score'] += df['supertrend_trend']
        
        # 4. Ichimoku Trend
        ichimoku_bullish = (df['close'] > df['leading_span_a']) & (df['close'] > df['leading_span_b'])
        ichimoku_bearish = (df['close'] < df['leading_span_a']) & (df['close'] < df['leading_span_b'])
        df['ichimoku_trend'] = np.select([ichimoku_bullish, ichimoku_bearish], [1, -1], default=0)
        df['trend_score'] += df['ichimoku_trend']
        
        # 5. Bollinger Bands Trend
        # Initialize bb_trend column
        df['bb_trend'] = np.nan
        
        # Set trend to 1 when close crosses above the upper band
        df.loc[df['close'] > df['bollinger_upperband'], 'bb_trend'] = 1
        # Set trend to -1 when close crosses below the lower band
        df.loc[df['close'] < df['bollinger_lowerband'], 'bb_trend'] = -1
        # Forward-fill the trend to maintain the state until it's flipped
        df['bb_trend'] = df['bb_trend'].ffill().fillna(0).astype(int)
        
        df['trend_score'] += df['bb_trend']
        
        # 6. ADX Trend Strength
        adx_trending = df['adx'] >= 20
        adx_weak = df['adx'] < 20
        df['adx_trend'] = np.select([adx_trending, adx_weak], [1, 0], default=0)
        df['trend_score'] += df['adx_trend']
        
        # Final Trend Filter
        df[f'trend_filter_{timeframe}'] = np.select(
            [df['trend_score'] == 7, df['trend_score'] == -5],
            [1, -1],
            default=0
        ).astype(int)
        
        return df
    
    # SUPPORTS & RESISTANCES BASED ON PIVOTS AND POWER
    def calculate_support_resistance(self, df):
        """
        Identifies key support and resistance levels based on pivot points and their "power".
        
        The function finds local highs and lows (pivots) in the price data, groups them into zones,
        scores these zones by how many pivots they contain (their "power"), and then identifies
        the closest powerful support and resistance levels relative to the current price.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'high', 'low', 'close' columns.
        Returns:
            df (pd.DataFrame): The original DataFrame with additional columns for support and resistance levels.
        """
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'high', 'low', and 'close' columns.")
        
        # --- Step 1 & 2: Find Pivots and Score Every Zone by Power ---
        peaks, _ = find_peaks(df['high'], distance=10, width=5)
        troughs, _ = find_peaks(-df['low'], distance=10, width=5)
        
        resistance_levels = df['high'].iloc[peaks]
        support_levels = df['low'].iloc[troughs]
        
        rounding_value = 0.0005 
        resistance_zones = (resistance_levels / rounding_value).round() * rounding_value
        support_zones = (support_levels / rounding_value).round() * rounding_value
        
        resistance_scores = resistance_zones.value_counts()
        support_scores = support_zones.value_counts()
        
        # --- Step 3: Define Current Price and Relevance Window ---
        current_price = df['close'].iloc[-1]
        relevance_window_pct = 0.20 # Look for levels within 20% of the current price
        
        price_floor = current_price * (1 - relevance_window_pct)
        price_ceiling = current_price * (1 + relevance_window_pct)
        
        # Find Closest Powerful Supports
        # First, try to find supports within the relevance window
        supports_in_window = support_scores[(support_scores.index < current_price) & 
                                            (support_scores.index > price_floor)]
        
        if not supports_in_window.empty:
            print("Found powerful supports within the relevance window.")
            # If we find some, sort them by power and take the top 3 closest
            closest_powerful_supports = supports_in_window.sort_values(ascending=False).head(3).sort_index(ascending=False)
        else:
            print("No supports in relevance window. Falling back to globally most powerful.")
            # Fallback logic: find the most powerful supports globally if the window is empty
            potential_supports = support_scores[support_scores.index < current_price]
            closest_powerful_supports = potential_supports.sort_values(ascending=False).head(3).sort_index(ascending=False)
        
        # --- Step 5: Find Closest Powerful Resistances ---
        # First, try to find resistances within the relevance window
        resistances_in_window = resistance_scores[(resistance_scores.index > current_price) & 
                                                (resistance_scores.index < price_ceiling)]
        
        if not resistances_in_window.empty:
            print("\nFound powerful resistances within the relevance window.")
            closest_powerful_resistances = resistances_in_window.sort_values(ascending=False).head(3).sort_index(ascending=True)
        else:
            print("\nNo resistances in relevance window. Falling back to globally most powerful.")
            potential_resistances = resistance_scores[resistance_scores.index > current_price]
            closest_powerful_resistances = potential_resistances.sort_values(ascending=False).head(3).sort_index(ascending=True)
        
        # --- Step 6: Print the Final Results ---
        print("\nClosest 3 POWERFUL Resistance Zones:")
        print(closest_powerful_resistances)
        
        print("\nClosest 3 POWERFUL Support Zones:")
        print(closest_powerful_supports)
        
        return df
    
    def calculate_zscore(self, df, timeframe, period=20):
        """
        Calculates the Z-Score of the closing prices over a specified rolling window.
        The Z-Score indicates how many standard deviations a data point is from the mean.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'close' column.
            period (int): The rolling window size for mean and std deviation. Default is 20.
        
        Returns:
            pd.DataFrame: The original DataFrame with an additional 'zscore' column.
        """
        
        if 'close' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['zscore'] = (df['close'] - rolling_mean) / (rolling_std + 1e-9)
        df['zscore'] = df['zscore'].fillna(0)  # Handle NaN values for initial periods
        
        return df
    
    def calculate_zigzag(self, df, timeframe, deviation_pct=5.0):
        """
        Calculates ZigZag swing points to identify significant market highs and lows.
        
        This method identifies pivots (swing points) where the market has reversed
        by at least a certain percentage (`deviation_pct`). It is highly useful for
        market structure analysis, such as calculating run/pullback ratios.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'high' and 'low' columns.
            timeframe: The timeframe of the data (for API consistency).
            deviation_pct (float): The minimum percentage change required to form a new pivot.
        
        Returns:
            pd.DataFrame: DataFrame with a 'zigzag' column.
                        - 'zigzag' = 1 for a swing high.
                        - 'zigzag' = -1 for a swing low.
                        - 'zigzag' = 0 otherwise.
        """
        
        if not all(col in df.columns for col in ['high', 'low']):
            raise ValueError("Input DataFrame must contain 'high' and 'low' columns.")
        
        # Convert to numpy arrays for Numba compatibility
        high_np = df['high'].to_numpy()
        low_np = df['low'].to_numpy()
        
        # Call the high-performance Numba function
        zigzag_points = _find_zigzag_points_numba(high_np, low_np, deviation_pct)
        
        df['zigzag'] = zigzag_points
        
        return df
    
    
    def calculate_atr_first_touch(self, df, timeframe, length=14, multiplier=1.0):
        """
        Predicts whether the price will hit the 1x ATR upper band or lower band first,
        and reports the number of bars elapsed until the touch.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            timeframe: Current timeframe.
            length (int): ATR lookback period. Default 14.

        Returns:
            pd.DataFrame: DataFrame with the 'atr_first_touch_dir' and 
                            'atr_first_touch_bars' columns added.
        """
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Input DataFrame must contain 'high', 'low', and 'close' columns.")
        
        # 1. Calculate ATR (using RMA smoothing)
        df = self.calculate_atr(df, timeframe, length=length)
        
        # 2. Define the ATR bands (+/- 1 multiplier from close)
        upper_band = df['close'] + df['atr'] * multiplier
        lower_band = df['close'] - df['atr'] * multiplier
        
        # 3. Prepare NumPy arrays for Numba
        high_np = df['high'].to_numpy()
        low_np = df['low'].to_numpy()
        upper_band_np = upper_band.to_numpy()
        lower_band_np = lower_band.to_numpy()
        
        # 4. Call Numba function
        atr_touch_dir, atr_touch_bars = _predict_atr_touch_numba(
            high_np, 
            low_np, 
            upper_band_np, 
            lower_band_np
        )
        
        # 5. Assign results back to DataFrame
        df['A_atr_first_touch_dir'] = atr_touch_dir
        df['A_atr_first_touch_bars'] = atr_touch_bars
        
        return df