export interface FETemplate {
    name: string;
    description: string;
    code: string;
    isDeletable: boolean;
}


export const INITIAL_FEATURE_ENGINEERING_CODE: { [key: string]: FETemplate } = {
    guide: {
        name: 'Guide',
        description: 'Default Guide.',
        code:`import pandas as pd
import numpy as np

# This function receives the DataFrame after technical indicators have been added.
# It must return a new DataFrame with your custom features.
# The original 'open', 'high', 'low', 'close', 'volume' columns will be dropped
# by the backend after this script runs, so ensure any features you want to keep
# are derived and have new names.

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply custom feature engineering.
    
    Args:
        df: DataFrame with OHLCV data and any pre-calculated indicators.
        
    Returns:
        DataFrame with the newly engineered features.
    """
    
    # --- Example 1: Create simple interaction features ---
    # Create a feature for the daily price range.
    df['feat_day_range'] = df['high'] - df['low']
    
    # --- Example 2: Create lagged features ---
    # Create a feature for the previous candle's closing price change.
    df['feat_close_change_1'] = df['close'].diff()
    
    # Create a feature for the close price 5 periods ago
    df['feat_close_lag_5'] = df['close'].shift(5)
    
    # --- Example 3: Volatility and Momentum ---
    # Calculate returns
    df['feat_returns'] = df['close'].pct_change()
    
    # Calculate volatility (rolling standard deviation of returns)
    df['feat_volatility_10'] = df['feat_returns'].rolling(window=10).std()
    
    # --- Example 4: Clean up ---
    # The script should handle NaNs that it creates.
    # We can fill them or drop the rows.
    df.fillna(method='bfill', inplace=True) # Backfill to avoid losing first rows
    df.dropna(inplace=True) # Drop any remaining NaNs
    
    print("Custom feature engineering script executed successfully.")
    
    # IMPORTANT: Return the modified DataFrame
    return df
`,
    isDeletable: false,
    },
    none: {
        name: 'None',
        description: 'Default.',
        code:`import pandas as pd
import numpy as np

def transform_features(df: pd.DataFrame) -> pd.DataFrame:

    print("Custom feature engineering script executed successfully.")
    
    return df
`,
    isDeletable: false,
    },
};