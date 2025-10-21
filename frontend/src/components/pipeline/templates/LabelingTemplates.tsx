import type { LabelingTemplate } from '../../../context/PipelineContext';

export const INITIAL_LABELING_TEMPLATES: { [key: string]: LabelingTemplate } = {
    market_regime: {
        name: 'Market Regime Detection (Classification)',
        description: 'Classify the market as Uptrend, Downtrend, or Sideways.',
        code: 
`import pandas as pd

# This function receives a DataFrame with OHLCV data
# and must return a Pandas Series with the labels.
# Labels can be strings or integers (e.g., 1 for Uptrend, -1 for Downtrend, 0 for Sideways).
def generate_labels(df: pd.DataFrame) -> pd.Series:
    # Use the ADX indicator to define the regime
    adx = df.ta.adx(length=14)
    plus_di = adx['DMP_14']
    minus_di = adx['DMN_14']

    # Conditions for labeling
    conditions = [
        (adx['ADX_14'] > 25) & (plus_di > minus_di),  # Uptrend
        (adx['ADX_14'] > 25) & (minus_di > plus_di),  # Downtrend
    ]
    choices = ['Uptrend', 'Downtrend']
    
    # Use numpy.select for conditional labeling
    import numpy as np
    labels = np.select(conditions, choices, default='Sideways')
    
    return pd.Series(labels, index=df.index)
`,
    isDeletable: false,
    },
    triple_barrier: {
        name: 'Trade Entry Signal (Triple Barrier Classification)',
        description: 'Predict if a position should be Long, Short, or Hold.',
        code: 
`import pandas as pd

# This function must return a Pandas Series with labels (e.g., 1 for Long, -1 for Short, 0 for Hold).
def generate_labels(df: pd.DataFrame, atr_mult_profit=2.0, atr_mult_loss=1.0, look_forward_candles=10) -> pd.Series:
    """
    Generates labels using the Triple Barrier Method.
    - 1: Profit target hit first (Long signal)
    - -1: Stop loss hit first (Short signal)
    - 0: Time limit hit first (Hold signal)
    """
    try:
        df['ATR'] = df.ta.atr(length=14)
        labels = pd.Series(0, index=df.index)

        for i in range(len(df) - look_forward_candles):
            entry_price = df['close'].iloc[i]
            profit_target = entry_price + (df['ATR'].iloc[i] * atr_mult_profit)
            stop_loss = entry_price - (df['ATR'].iloc[i] * atr_mult_loss)

            for j in range(1, look_forward_candles + 1):
                future_high = df['high'].iloc[i + j]
                future_low = df['low'].iloc[i + j]

                if future_high >= profit_target:
                    labels.iloc[i] = 1  # Profit target hit
                    break
                if future_low <= stop_loss:
                    labels.iloc[i] = -1  # Stop loss hit
                    break
            # If loop finishes without break, label remains 0 (time barrier)
            
        return labels
        
    except Exception as e:
        print(f"Error in label generation: {e}")
        return pd.Series(0, index=df.index)
        `,
    isDeletable: false,
},
    stop_loss_placement: {
        name: 'Stop Loss Placement (Regression)',
        description: 'Predict the expected price drop (Max Adverse Excursion) for a long trade.',
        code: 
`import pandas as pd

# This function must return a Pandas Series of continuous values (the target for regression).
def generate_labels(df: pd.DataFrame, look_forward_candles=10) -> pd.Series:
    """
    Calculates the Maximum Adverse Excursion (MAE) for potential long trades.
    The label is the maximum drop from the entry price over the next N candles.
    """
    labels = pd.Series(0.0, index=df.index)

    for i in range(len(df) - look_forward_candles):
        entry_price = df['close'].iloc[i]
        future_lows = df['low'].iloc[i+1 : i+1+look_forward_candles]
        max_drawdown = (entry_price - future_lows.min()) / entry_price
        labels.iloc[i] = max_drawdown if max_drawdown > 0 else 0.0

    return labels
`,
    isDeletable: false,
    },
};