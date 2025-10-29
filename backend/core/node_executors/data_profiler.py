# executors/data_profiler.py

# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Any
import pandas as pd
import numpy as np

# ==============================================================================
# Local Application/Library Specific Imports
# ==============================================================================
from .base import BaseNodeExecutor, ExecutionContext, ExecutorOutput, ExecutorInput

# ==============================================================================
# The Main Executor Class
# ==============================================================================

class DataProfilerExecutor(BaseNodeExecutor):
    """
    Analyzes a specific feature in a dataset for distribution, outliers,
    and provides a recommendation for a scaling method.
    This is an informational node; it passes the input DataFrame through unchanged
    and puts all analysis into the node_metadata context.
    """

    def execute(
        self,
        node: Any,
        inputs: ExecutorInput,
        context: ExecutionContext
    ) -> ExecutorOutput:
        print(f"Executing Data Profiler Node: {node.data.get('label', node.id)}")

        # --- 1. Input Validation and Preparation ---
        if not inputs:
            raise ValueError("DataProfiler node requires an input connection.")
        
        # NOTE: We use the key 'default' here because profiler has one input handle.
        # If the input key is different (e.g., edge.id), this logic needs to be robust.
        # list(inputs.values())[0] is a safe fallback.
        df_input = list(inputs.values())[0]['data']
        df = df_input.copy() # Make a copy to avoid modifying the original in-place
        
        non_feature_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]

        if not feature_cols:
            raise ValueError("No feature columns found to profile.")

        selected_feature = node.data.get('selectedFeature')

        if not selected_feature or selected_feature not in feature_cols:
            selected_feature = feature_cols[0]

        # --- 2. Perform Analysis on the Selected Feature ---
        feature_series = df[selected_feature].dropna()
        
        stats = feature_series.describe().to_dict()
        stats = {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in stats.items()}
        
        skewness = round(feature_series.skew(), 4)
        kurtosis = round(feature_series.kurt(), 4)

        q1 = feature_series.quantile(0.25)
        q3 = feature_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = feature_series[(feature_series < lower_bound) | (feature_series > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = round((outlier_count / len(feature_series)) * 100, 2) if len(feature_series) > 0 else 0

        counts, bin_edges = np.histogram(feature_series, bins=50)
        
        # --- 3. Generate Scaling Recommendation ---
        suggestion = {}
        if abs(skewness) > 0.8 or outlier_percentage > 1.5:
            suggestion = {
                "scaler": "RobustScaler",
                "reason": f"The data is significantly skewed ({skewness}) and/or has a high percentage of outliers ({outlier_percentage}%). RobustScaler uses the median and quartile range, making it resilient to extreme values."
            }
        else:
            suggestion = {
                "scaler": "StandardScaler",
                "reason": f"The data has low skewness ({skewness}) and few outliers ({outlier_percentage}%). StandardScaler is a great default choice for data that is relatively symmetrical."
            }
        
        # --- 4. Assemble the METADATA Payload ---
        # THIS IS THE CORRECT PATTERN: Put all display info into the context.
        context.node_metadata[node.id] = {
            "feature_list": feature_cols,
            "profile": {
                "feature_name": selected_feature,
                "stats": stats,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "outliers": {
                    "count": outlier_count,
                    "percentage": outlier_percentage
                },
                "distribution": {
                    "bins": bin_edges.tolist(),
                    "counts": counts.tolist()
                },
                "suggestion": suggestion
            }
        }

        # --- 5. Return the UNCHANGED DataFrame on the "data highway" ---
        # The key 'default' corresponds to the single, unnamed output handle.
        return {
            "default": df_input 
        }