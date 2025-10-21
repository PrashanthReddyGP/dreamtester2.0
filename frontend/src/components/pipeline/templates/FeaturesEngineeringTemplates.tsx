import type { FETemplate } from '../../../context/PipelineContext';

export const INITIAL_FEATURE_ENGINEERING_CODE: { [key: string]: FETemplate } = {
    guide: {
        name: 'Guide',
        description: 'Default Guide.',
        code: `import pandas as pd
import numpy as np

# This function receives a DataFrame from the previous node.
# It must be named 'process' and accept one argument.
# It must return a new DataFrame.

def process(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply custom feature engineering.
    
    Args:
        data: DataFrame with OHLCV data and any pre-calculated indicators.
        
    Returns:
        DataFrame with the newly engineered features.
    """
    
    # --- Always work on a copy to avoid unexpected behavior ---
    data = data.copy()

    # --- Example 1: Create simple interaction features ---
    data['feat_day_range'] = data['high'] - data['low']
    
    # --- Example 2: Create lagged features ---
    data['feat_close_change_1'] = data['close'].diff()
    data['feat_close_lag_5'] = data['close'].shift(5)
    
    # --- Example 3: Volatility and Momentum ---
    data['feat_returns'] = data['close'].pct_change()
    data['feat_volatility_10'] = data['feat_returns'].rolling(window=10).std()
    
    # --- Example 4: Clean up ---
    # The script should handle NaNs that it creates.
    data.fillna(method='bfill', inplace=True) # Backfill to avoid losing first rows
    data.dropna(inplace=True) # Drop any remaining NaNs
    
    print("Custom feature engineering script executed successfully.")
    
    # IMPORTANT: Return the modified DataFrame
    return data
`,
        isDeletable: false,
    },
    none: {
        name: 'None',
        description: 'Does nothing, just passes data through.',
        code: `import pandas as pd
import numpy as np

def process(data: pd.DataFrame) -> pd.DataFrame:
    # This template does nothing, it's a pass-through.
    print("Pass-through node executed.")
    return data
`,
        isDeletable: false,
    },
};