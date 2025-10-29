import pandas as pd
from typing import Dict, Any

from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

class MergeExecutor(BaseNodeExecutor):
    """
    Executor for the 'merge' node. Merges two DataFrames based on their timestamp index.
    """
    
    @staticmethod
    def _to_pandas_freq(timeframe_str: str) -> str:
        """Converts a common timeframe string like '15m' to a pandas-compatible one like '15T'."""
        if not timeframe_str:
            return None
        return timeframe_str.upper().replace('M', 'T')

    @staticmethod
    def _determine_timeframe(df: pd.DataFrame) -> str:
        """
        Infers the timeframe of a DataFrame by calculating the median difference
        between consecutive timestamps.
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            raise ValueError("Cannot determine timeframe from DataFrame with < 2 rows or no 'timestamp' column.")

        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        freq = timestamps.diff().median()

        if pd.isna(freq):
            raise ValueError("Could not calculate a valid frequency from timestamps (result was NaN).")

        total_seconds = freq.total_seconds()

        if total_seconds <= 0:
            raise ValueError(f"Calculated frequency '{freq}' is not a positive duration.")

        if total_seconds >= 86400 and total_seconds % 86400 == 0:
            return f"{int(total_seconds // 86400)}d"
        elif total_seconds >= 3600 and total_seconds % 3600 == 0:
            return f"{int(total_seconds // 3600)}h"
        elif total_seconds >= 60 and total_seconds % 60 == 0:
            return f"{int(total_seconds // 60)}m"
        else:
            raise ValueError(f"Unable to map calculated frequency '{freq}' to a standard timeframe string ('m', 'h', 'd').")
    
    def execute(
        self, 
        node: Any, 
        inputs: ExecutorInput, 
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Merge Node: {node.data.get('label', node.id)}")
        
        # --- 1. Validate Inputs ---
        if 'a' not in inputs or 'b' not in inputs:
            raise ValueError("Merge node requires inputs on both 'a' and 'b' handles.")
        
        main_handle = node.data.get('mainInputHandle', 'a')
        sub_handle = 'b' if main_handle == 'a' else 'a'

        df_main = inputs[main_handle]['data'].copy()
        df_sub = inputs[sub_handle]['data'].copy()

        # --- 2. Pre-processing and Timeframe Inference ---
        if df_main.empty or df_sub.empty:
            print("  -> One of the DataFrames is empty. Returning the main DataFrame.")
            return {"default": df_main}

        for df, name in [(df_main, "Main"), (df_sub, "Sub")]:
            if 'timestamp' not in df.columns:
                raise KeyError(f"{name} DataFrame is missing the required 'timestamp' column.")
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df.sort_values('timestamp', inplace=True, ignore_index=True)

        try:
            main_timeframe = self._determine_timeframe(df_main)
            sub_timeframe = self._determine_timeframe(df_sub)
        except ValueError as e:
            raise ValueError(f"Failed to process inputs for merge node: {e}")

        print(f"  -> Inferred Main TF: '{main_timeframe}' (handle '{main_handle}'), Shape: {df_main.shape}")
        print(f"  -> Inferred Sub TF: '{sub_timeframe}' (handle '{sub_handle}'), Shape: {df_sub.shape}")

        # --- 3. Identify Common Columns for Suffixing ---
        main_cols = set(df_main.columns)
        sub_cols = set(df_sub.columns)
        common_cols = list(main_cols.intersection(sub_cols))
        if 'timestamp' in common_cols:
            common_cols.remove('timestamp')

        suffix = f"_{sub_timeframe}"
        rename_map = {col: f"{col}{suffix}" for col in common_cols}
        
        print(f"  -> Identified common columns to be suffixed: {common_cols}")

        # --- 4. Determine Frequencies and Merge Strategy ---
        main_df_freq = df_main['timestamp'].diff().median()
        sub_df_freq = df_sub['timestamp'].diff().median()
        merge_info = { "main_timeframe": main_timeframe, "sub_timeframe": sub_timeframe }

        # --- 5. Perform the Merge ---
        # SCENARIO 1: Sub-dataframe is lower frequency
        if sub_df_freq > main_df_freq:
            print(f"  -> Scenario: Merging lower freq sub_df ({sub_timeframe}) into higher freq main_df ({main_timeframe}).")
            merge_info["scenario"] = "low_freq_sub_to_high_freq_main"

            data_cols_to_shift = [col for col in df_sub.columns if col != 'timestamp']
            df_sub[data_cols_to_shift] = df_sub[data_cols_to_shift].shift(1)

            df_sub_renamed = df_sub.rename(columns=rename_map)

            # Create a list of the *new* column names to check for NaNs.
            # Use rename_map.get(col, col) to get the new name if it was renamed, or keep the old name if not.
            subset_for_dropna = [rename_map.get(col, col) for col in data_cols_to_shift]
            df_sub_renamed.dropna(subset=subset_for_dropna, how='all', inplace=True)

            df_merged = pd.merge_asof(df_main, df_sub_renamed, on='timestamp', direction='backward')
            
            sub_cols_in_merged_df = [rename_map.get(c, c) for c in sub_cols if c != 'timestamp']
            df_merged[sub_cols_in_merged_df] = df_merged[sub_cols_in_merged_df].ffill()

        # SCENARIO 2: Sub-dataframe is higher or equal frequency
        else:
            print(f"  -> Scenario: Merging higher freq sub_df ({sub_timeframe}) into lower freq main_df ({main_timeframe}).")
            merge_info["scenario"] = "high_freq_sub_to_low_freq_main"

            resample_freq_str = self._to_pandas_freq(main_timeframe)
            agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            applicable_agg_rules = {k: v for k, v in agg_rules.items() if k in df_sub.columns}

            if not applicable_agg_rules:
                raise ValueError(f"Sub-dataframe from '{sub_timeframe}' has no standard OHLCV columns to aggregate.")

            df_sub_agg = df_sub.set_index('timestamp').resample(resample_freq_str).agg(
                applicable_agg_rules
            ).dropna(how='all').reset_index()

            df_sub_agg_renamed = df_sub_agg.rename(columns=rename_map)

            df_merged = pd.merge(df_main, df_sub_agg_renamed, on='timestamp', how='left')

        print(f"  -> Merged DF shape: {df_merged.shape}")

        # --- 6. Store Metadata ---
        context.node_metadata[node.id] = {
            "merge_info": merge_info,
            "main_shape_before": list(df_main.shape),
            "sub_shape_before": list(df_sub.shape),
            "output_shape": list(df_merged.shape),
            "suffixed_columns": list(rename_map.values())
        }

        return {"default": df_merged}