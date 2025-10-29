from datetime import datetime
import pandas as pd

from core.data_manager import get_ohlcv

def fetch_candlestick_data(client, strategy_instance, sub_timeframe=''):
    
    # Access the symbol set in the strategy's __init__ method
    symbol = strategy_instance.symbol 
    
    if sub_timeframe == '':
        timeframe = strategy_instance.timeframe
    else:
        timeframe = sub_timeframe
    
    startdate = datetime.strptime(strategy_instance.start_date, '%Y-%m-%d') # Convert String to Datetime format
    enddate = datetime.strptime(strategy_instance.end_date, '%Y-%m-%d')

    print(f"--- Fetching OHLCV data for Asset: {symbol}, Timeframe: {timeframe}, StartDate: {startdate}, EndDate: {enddate} ---")

    df = get_ohlcv(client, symbol, timeframe, startdate, enddate)

    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')

    return symbol, df