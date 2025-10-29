# core/json_encoder.py
import json
import numpy as np
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles special data types from NumPy and Pandas.
    """
    def default(self, obj):
        # If the object is a NumPy integer, convert it to a standard Python int
        if isinstance(obj, np.integer):
            return int(obj)
        # If the object is a NumPy float, convert it to a standard Python float
        elif isinstance(obj, np.floating):
            return float(obj)
        # If the object is a NumPy array, convert it to a list
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # If the object is a Pandas Timestamp, convert it to an ISO string
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Let the base class default method handle other types
        return super(CustomJSONEncoder, self).default(obj)