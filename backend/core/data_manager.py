# backend/core/data_manager.py
import threading
import pandas as pd
from datetime import datetime
from sqlalchemy import text

from database import engine, SessionLocal 

DATABASE_PATH = 'path/to/your/database.db'

# All threads will have to wait to acquire this lock before fetching.
exchange_api_lock = threading.Lock()
db_write_lock = threading.Lock()

def get_ohlcv(client, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Manages fetching, caching, and retrieving OHLCV data from the SQLite database.
    """
        
    table_name = f"data_{symbol}_{timeframe}"
    
    # --- 2. Use the imported engine to connect ---
    # `engine.connect()` uses the same connection pool as the rest of your app.
    with db_write_lock:
        
        print(f"  -> Acquired DB lock for {symbol}")

        with engine.connect() as conn:
            
            create_table_sql = text(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL, high REAL, low REAL, close REAL, volume REAL
                )
            ''')
            
            # Create the table if it doesn't exist
            conn.execute(create_table_sql)
            conn.commit() # Make sure the table is definitely created

            # Check existing data range
            select_range_sql = text(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}")
            res = conn.execute(select_range_sql).fetchone()
            min_ts, max_ts = res if res and res[0] is not None else (None, None)

            start_ts_ms = int(start_dt.timestamp() * 1000)
            end_ts_ms = int(end_dt.timestamp() * 1000)

            # 3. Determine and fetch missing data
            # This list will hold all newly fetched data from all chunks.
            all_new_data = []

            # --- REVISED, SURGICAL FETCHING LOGIC ---

            # Case 1: The database is completely empty. Fetch the entire requested range.
            if min_ts is None:
                print(f"Database table '{table_name}' is empty. Fetching initial data...")
                all_new_data.extend(fetch_ohlcv_paginated(client, symbol, timeframe, start_ts_ms, end_ts_ms))
            
            else:
                # Case 2: Fetch data *before* our cached range, if needed.
                if start_ts_ms < min_ts:
                    # We need data from the user's start date up to the beginning of our cache.
                    # The `-1` prevents fetching the first candle we already have.
                    print(f"Fetching older data from {datetime.fromtimestamp(start_ts_ms/1000)} to {datetime.fromtimestamp((min_ts-1)/1000)}...")
                    data_before = fetch_ohlcv_paginated(client, symbol, timeframe, start_ts_ms, min_ts - 1)
                    all_new_data.extend(data_before)

                # Case 3: Fetch data *after* our cached range, if needed.
                if end_ts_ms > max_ts:
                    # We need data from the end of our cache up to the user's end date.
                    # The `+1` prevents fetching the last candle we already have.
                    print(f"Fetching recent data from {datetime.fromtimestamp((max_ts+1)/1000)} to {datetime.fromtimestamp(end_ts_ms/1000)}...")
                    data_after = fetch_ohlcv_paginated(client, symbol, timeframe, max_ts + 1, end_ts_ms)
                    all_new_data.extend(data_after)
            
            # --- Now, save any new data we collected ---
            if all_new_data:
                print(f"Found {len(all_new_data)} new candles to add to the database.")
                to_insert_dicts = [{'ts': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4], 'v': r[5]} for r in all_new_data]
                
                insert_sql = text(f'''
                    INSERT OR IGNORE INTO {table_name} (timestamp, open, high, low, close, volume) 
                    VALUES (:ts, :o, :h, :l, :c, :v)
                ''')
                # Using INSERT OR IGNORE is safer here than REPLACE. It will not overwrite
                # existing data if there's an overlap, which is good.
                
                conn.execute(insert_sql, to_insert_dicts)
                conn.commit()
                
                print("New data successfully saved to the database.")

            # 5. Final data retrieval (also within the same transaction)
            query = text(f"SELECT * FROM {table_name} WHERE timestamp BETWEEN :start AND :end ORDER BY timestamp")
            
            # Use a new connection for read_sql_query to avoid transaction state issues
            # OR better, read the data first then close the write connection. Let's do that.
            df = pd.read_sql_query(query, conn, params={"start": start_ts_ms, "end": end_ts_ms})
    
    print(f"  <- Released DB lock for {symbol}")

    # Convert timestamp back to datetime for use in pandas
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/Vancouver').dt.tz_localize(None)
    
    # print(df.tail(3))
    
    return df

def fetch_ohlcv_paginated(client, symbol, timeframe, fetch_start, fetch_end, limit=1000):
    
    all_ohlcv = []
    
    with exchange_api_lock:
        
        print(f"  -> Acquired API lock for {symbol} {timeframe}")

        while True:
            try:
                ohlcv = client.klines(symbol, timeframe, startTime=fetch_start, endTime=fetch_end, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                fetch_start = ohlcv[-1][0] + 1  # Increment to the next millisecond
                if len(ohlcv) < limit:
                    break
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
            
    print(f"  <- Released API lock for {symbol} {timeframe}")
    
    return all_ohlcv