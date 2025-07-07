from binance.um_futures import UMFutures
from fastapi import HTTPException

from database import get_api_key

def get_client(exchange_name):
    
    keys = get_api_key(exchange=exchange_name)

    if not keys:
        raise HTTPException(
            status_code=404, 
            detail=f"API keys not found for exchange: {exchange_name}"
        )

    api = keys['api_key']
    secret = keys['api_secret']
    
    client = connect_to_exchange(api, secret)
    
    return client

def connect_to_exchange(api, secret):
    
    # Add if conditions for additional brockerage connections
    client = UMFutures(api, secret)
    print("Connected to the Exchange")
    
    return client