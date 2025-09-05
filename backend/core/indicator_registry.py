from core.indicators import Indicators

# A single, shared instance of your indicator calculation class
idk = Indicators()

# ==============================================================================
# The Indicator Registry - SINGLE SOURCE OF TRUTH
# ==============================================================================
# This dictionary maps a unique key (for the frontend) to:
#   - 'function': The actual calculation method to call from the Indicators class.
#   - 'params': A list of parameter names IN THE EXACT ORDER the function expects.
#   - 'display_name': A user-friendly name for the UI.
# ==============================================================================

INDICATOR_REGISTRY = {
    # --- Moving Averages ---
    'SMA': {
        'function': idk.calculate_sma,
        'params': ['length'],
        'display_name': 'Simple Moving Average (SMA)'
    },
    'EMA': {
        'function': idk.calculate_ema,
        'params': ['length'],
        'display_name': 'Exponential Moving Average (EMA)'
    },
    'WMA': {
        'function': idk.calculate_wma,
        'params': ['length'],
        'display_name': 'Weighted Moving Average (WMA)'
    },
    'HMA': {
        'function': idk.calculate_hma,
        'params': ['length'],
        'display_name': 'Hull Moving Average (HMA)'
    },
    'SMA_SLOPE': {
        'function': idk.calculate_sma_slope,
        'params': ['length', 'smoothbars', 'hlineheight'],
        'display_name': 'SMA Slope'
    },
    'VWAP': {
        'function': idk.calculate_VWAP,
        'params': [],
        'display_name': 'Volume-Weighted Average Price (VWAP)'
    },

    # --- Oscillators & Momentum ---
    'RSI': {
        'function': idk.calculate_rsi,
        'params': ['length'],
        'display_name': 'Relative Strength Index (RSI)'
    },
    'STOCH': {
        'function': idk.calculate_stocastic,
        'params': ['k_length', 'd_length'],
        'display_name': 'Stochastic Oscillator'
    },
    'MACD': {
        'function': idk.calculate_macd,
        'params': ['short_window', 'long_window', 'signal_window'],
        'display_name': 'Moving Average Convergence Divergence (MACD)'
    },
    'CCI': {
        'function': idk.calculate_cci,
        'params': ['length'],
        'display_name': 'Commodity Channel Index (CCI)'
    },
    'AROON': {
        'function': idk.calculate_aroon,
        'params': ['period'],
        'display_name': 'Aroon Indicator'
    },
    'WILLIAMS_R': {
        'function': idk.williams_range,
        'params': ['period'],
        'display_name': 'Williams %R'
    },
    'MOMENTUM': {
        'function': idk.calculate_momentum,
        'params': ['lookback'],
        'display_name': 'Momentum'
    },
    'ROC': {
        'function': idk.calculate_roc,
        'params': ['lookback'],
        'display_name': 'Rate of Change (ROC)'
    },
    'ROEC': {
        'function': idk.calculate_signed_extreme_change,
        'params': ['lookback', 'as_percentage'],
        'display_name': 'Rate of Extreme Change (ROEC)'
    },
    'HL_OSCILLATOR': {
        'function': idk.calculate_hl_oscillator,
        'params': ['period'],
        'display_name': 'Higher High / Lower Low Oscillator'
    },

    # --- Volatility & Trend Strength ---
    'ADX': {
        'function': idk.calculate_adx,
        'params': ['length'],
        'display_name': 'Average Directional Index (ADX)'
    },
    'ATR': {
        'function': idk.calculate_atr,
        'params': ['length', 'smoothing'],
        'display_name': 'Average True Range (ATR)'
    },
    'BBANDS': {
        'function': idk.calculate_bollinger_bands,
        'params': ['length', 'deviation'],
        'display_name': 'Bollinger Bands'
    },
    'SUPERTREND': {
        'function': idk.supertrend,
        'params': ['period', 'multiplier'],
        'display_name': 'Supertrend'
    },
    'PSAR': {
        'function': idk.calculate_psar,
        'params': ['af_initial', 'af_step', 'af_max'],
        'display_name': 'Parabolic SAR (PSAR)'
    },
    'DONCHIAN': {
        'function': idk.donchian_channels,
        'params': ['window'],
        'display_name': 'Donchian Channels'
    },
    'ICHIMOKU': {
        'function': idk.calculate_ichimoku,
        'params': ['conversion_line', 'base_line', 'lagging_span_b', 'lagging_span'],
        'display_name': 'Ichimoku Cloud'
    },

    # --- Volume ---
    'OBV': {
        'function': idk.calculate_obv,
        'params': [],
        'display_name': 'On-Balance Volume (OBV)'
    },

    # --- Price Action & Structure ---
    'PIVOT_POINTS': {
        'function': idk.pivot_points,
        'params': ['pivot_type'],
        'display_name': 'Pivot Points'
    },
    'CANDLE_PATTERNS': {
        'function': idk.identify_candlestick_patterns,
        'params': [],
        'display_name': 'Candlestick Patterns'
    },
    'HH_LL': {
        'function': idk.hh_ll,
        'params': ['period'],
        'display_name': 'Highest High / Lowest Low'
    },
    'LAG_OHLCV': {
        'function': idk.lag_ohlcv,
        'params': ['lag'],
        'display_name': 'Lagged OHLCV Features'
    },
    'DATETIME_SPLIT': {
        'function': idk.datetime_split,
        'params': [],
        'display_name': 'Datetime Features'
    },
    'CANDLE_LENGTHS': {
        'function': idk.candle_lengths,
        'params': [],
        'display_name': 'Candle Properties (Wicks, Body %)'
    },
    'PRICE_MOVEMENT': {
        'function': idk.price_movement,
        'params': [],
        'display_name': 'Intra-Candle Price Movement'
    },

    # --- Composite / Regime Indicators ---
    'MARKET_REGIME': {
        'function': idk.calculate_market_regime,
        'params': [],
        'display_name': 'Market Regime (Simple)'
    },
    'REGIME_FILTERS': {
        'function': idk.calculate_regime_filters,
        'params': [],
        'display_name': 'Comprehensive Regime Filters'
    },

    # --- Application-Specific (Use with caution as general features) ---
    'STOP_LOSS_CALC': {
        'function': idk.calculate_stop_loss,
        'params': ['length', 'multiplier', 'index'],
        'display_name': 'ATR Stop Loss Calculator'
    }
}

# NOTE: economic_data is intentionally excluded as it makes external API calls
# and should be handled separately from standard technical indicator calculations.


def get_indicator_schema():
    """
    Creates a JSON-serializable schema for the frontend.
    This function formats the registry and provides sensible default values
    for the UI to consume.
    """
    # Helper dictionary for common default values
    default_values = {
        'length': 14, 'period': 14, 'window': 14, 'lookback': 9, 'lag': 5,
        'short_window': 12, 'long_window': 26, 'signal_window': 9,
        'k_length': 14, 'd_length': 3,
        'deviation': 2, 'multiplier': 3.0, 'index': 1,
        'smoothbars': 3, 'hlineheight': 10,
        'af_initial': 0.02, 'af_step': 0.02, 'af_max': 0.2,
        'conversion_line': 9, 'base_line': 26, 'lagging_span_b': 52, 'lagging_span': 26,
    }

    schema = {}
    for key, details in INDICATOR_REGISTRY.items():
        params_schema = []
        for param_name in details['params']:
            # Handle special cases for non-numeric parameters
            if param_name == 'smoothing':
                param_info = {"name": param_name, "type": "string", "defaultValue": "sma", "options": ["sma", "ema", "rma"]}
            elif param_name == 'pivot_type':
                param_info = {"name": param_name, "type": "string", "defaultValue": "Fibonacci", "options": ["Fibonacci", "Standard"]}
            elif param_name == 'as_percentage':
                param_info = {"name": param_name, "type": "boolean", "defaultValue": True}
            else:
                # Default case for numeric parameters
                param_info = {
                    "name": param_name,
                    "type": "number",
                    "defaultValue": default_values.get(param_name, 10) # Default to 10 if not found
                }
            params_schema.append(param_info)

        schema[key] = {
            "name": details['display_name'],
            "params": params_schema
        }
    return schema