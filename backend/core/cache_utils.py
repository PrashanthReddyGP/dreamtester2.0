import os
import json
import hashlib
import pandas as pd
import pickle
from typing import Dict, Any, List

# Define the directory where cache files will be stored.
CACHE_DIR = ".pipeline_cache"

def setup_cache_directory():
    """Ensures the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def generate_node_hash(node_data: Dict[str, Any], parent_hashes: List[str]) -> str:
    """
    Generates a unique SHA256 hash for a node's state.

    The hash depends on the node's own configuration and the hashes of its parents,
    ensuring that any change upstream propagates down.
    """
    # Use json.dumps with sort_keys=True to ensure consistent serialization
    node_data_str = json.dumps(node_data, sort_keys=True)
    
    # Combine the node's data string with the sorted hashes of its parents
    combined_string = node_data_str + "".join(sorted(parent_hashes))
    
    # Return the hex digest of the SHA256 hash
    return hashlib.sha256(combined_string.encode()).hexdigest()

def save_to_cache(hash_key: str, data: Any):
    """Saves node output (dataframe, metadata, model) to a pickle file."""
    filepath = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"CACHE SAVE: Saved result for hash {hash_key[:10]}...")

def load_from_cache(hash_key: str) -> Any:
    """Loads node output from a pickle file if it exists."""
    filepath = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
    if os.path.exists(filepath):
        print(f"CACHE HIT: Loading result for hash {hash_key[:10]}...")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    print(f"CACHE MISS: No result found for hash {hash_key[:10]}...")
    return None

def clear_cache():
    """Deletes all files in the cache directory."""
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, filename))
        print("CACHE CLEARED.")