# backend/core/data_manager.py
import os
import threading
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from functools import lru_cache

from database import engine, SessionLocal 

# All threads will have to wait to acquire this lock before fetching.
exchange_api_lock = threading.Lock()
db_write_lock = threading.Lock()

def detect_delimiter(file_path):
    """Detects the delimiter of the input file."""
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        return None
    with open(file_path, 'r') as f:
        f.readline() # Skip header
        line = f.readline()
        if '\t' in line: return '\t'
        if ',' in line: return ','
        if ';' in line: return ';'
    return None

############## FIND OHLCV DATA FROM BINANCE, METATRADER, KAGGLE, INVESTING.COM*, FOREXSOFTWARE ######################

def convert_metatrader_to_standard_csv(input_path, volume_column='tick_volume'):
    """
    Reads Metatrader data and saves it as a standard, comma-separated CSV with a header.
    """
    
    try:
        column_names = [
            'date', 'time', 'open', 'high', 'low', 'close',
            'tick_volume', 'real_volume', 'spread'
        ]
        
        source_delimiter = detect_delimiter(input_path)
        
        # Step 1: Read the tab-separated input file
        df = pd.read_csv(
            input_path,
            delimiter=source_delimiter,
            header=None, skiprows=1, names=column_names,
            engine='python', skipinitialspace=True
        )
        
        # Step 2: Process the data
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
        df['timestamp'] = (df['datetime'].astype('int64') // 1_000_000)
        
        final_df = df[['timestamp', 'open', 'high', 'low', 'close', volume_column]]
        final_df = final_df.rename(columns={volume_column: 'volume'})
        
        print(f"\n Conversion successful!")
        print("Preview of the data that was saved:")
        print(final_df.head())
        
        return final_df
    
    except Exception as e:
        print(f"\n An error occurred during conversion: {e}")

def convert_kaggle_to_standard_csv(input_path, volume_column='Volume'):
    """
    Reads Kaggle-style data and converts it to the standard format.
    Assumes a single datetime column.
    """
    
    try:
        source_delimiter = detect_delimiter(input_path)
        if source_delimiter is None:
            print("Could not detect delimiter. Assuming ';'.")
            source_delimiter = ';'
            
        # Step 1: Read the source file. It has a header.
        df = pd.read_csv(
            input_path,
            delimiter=source_delimiter,
            header=0, # The first row is the header
            engine='python',
            skipinitialspace=True
        )
        
        # Step 2: Process the data
        # The column names from the file are likely capitalized, e.g., 'Date', 'Open'.
        # Standardize them to lowercase for easier processing.
        df.columns = [col.lower() for col in df.columns]
        
        # Convert the 'date' column (which contains date and time) to datetime objects.
        # The format from the example is 'YYYY.MM.DD HH:MM'.
        df['datetime'] = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M')
        
        # Convert datetime to Unix timestamp in milliseconds.
        df['timestamp'] = (df['datetime'].astype('int64') // 1_000_000)
        
        # Select and rename columns to the standard format.
        # The original volume column name is 'volume' after lowercasing.
        final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"\n Conversion successful!")
        print("Preview of the data that was prepared for import:")
        print(final_df.head())
        
        return final_df
    
    except Exception as e:
        print(f"\n An error occurred during conversion: {e}")
        return None


def convert_non_header_to_standard_csv(input_path, volume_column=None):
    """
    Reads CSV data that has no header and converts it to the standard format.
    Assumes the following column order: datetime, open, high, low, close, volume.
    """
    
    try:
        # Define the column names since the file has no header.
        column_names = ['datetime_str', 'open', 'high', 'low', 'close', 'volume']
        
        source_delimiter = detect_delimiter(input_path)
        if source_delimiter is None:
            print("Could not detect delimiter. Assuming ','.")
            source_delimiter = ','
            
        # Step 1: Read the source file. It has no header.
        df = pd.read_csv(
            input_path,
            delimiter=source_delimiter,
            header=None, # No header row
            names=column_names, # Assign column names
            engine='python',
            skipinitialspace=True
        )
        
        # Step 2: Process the data
        # Convert the datetime string column to datetime objects.
        # The format from the example is 'YYYY-MM-DD HH:MM'.
        df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M')
        
        # Convert datetime to Unix timestamp in milliseconds.
        df['timestamp'] = (df['datetime'].astype('int64') // 1_000_000)
        
        # Select and rename columns to the standard format.
        final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"\n Conversion successful!")
        print("Preview of the data that was prepared for import:")
        print(final_df.head())
        
        return final_df
    
    except Exception as e:
        print(f"\n An error occurred during conversion: {e}")
        return None

def import_csv_data(csv_path: str, symbol: str, timeframe: str, source: str, chunksize: int = 50000):
    """
    Imports historical OHLCV data from a prepared CSV file into the database.

    The CSV file is expected to have the following columns:
    'timestamp', 'open', 'high', 'low', 'close', 'volume'
    where 'timestamp' is a Unix timestamp in milliseconds.

    Args:
        csv_path (str): The full path to the CSV file.
        symbol (str): The symbol identifier (e.g., 'EURUSD').
        timeframe (str): The timeframe identifier (e.g., '1h').
        chunksize (int): The number of rows to process and insert at a time.
                        Useful for very large files to manage memory usage.
    """
    table_name = f"data_{symbol}_{timeframe}"
    print(f"--- Starting import for {symbol} ({timeframe}) from {csv_path} ---")
    
    if source == 'MetaTrader':
        df = convert_metatrader_to_standard_csv(csv_path)
    elif source == 'Kaggle':
        df = convert_kaggle_to_standard_csv(csv_path)
    else:
        df = convert_non_header_to_standard_csv(csv_path)
    
    if df is None or df.empty:
        print("No data to import. Exiting.")
        return
    
    # 1. Acquire the same database lock to prevent race conditions
    with db_write_lock:
        print(f"  -> Acquired DB lock for CSV import of {symbol}")
        
        with engine.connect() as conn:
            # 2. Ensure the target table exists, same as in get_ohlcv
            create_table_sql = text(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL, high REAL, low REAL, close REAL, volume REAL
                )
            ''')
            conn.execute(create_table_sql)
            conn.commit()

            # 3. Read the CSV in chunks and insert into the database
            try:
                total_rows_inserted = 0
                # Use a chunked reader for memory efficiency with large files
                # Since the entire DataFrame is already in memory, we can iterate over it in chunks
                # for better control and to fit the existing loop structure.
                # This avoids re-reading the file.
                for i in range(0, len(df), chunksize):
                    
                    chunk_df = df.iloc[i:i + chunksize]
                    
                    # Ensure column names match the database schema
                    # Your pre-processing script already does this, but it's good practice.
                    chunk_df = chunk_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    print(f"  -> Processing chunk of {len(chunk_df)} rows...")
                    
                    # Use pandas' to_sql for a highly efficient bulk insert.
                    # 'append' adds the data, and 'index=False' prevents writing the pandas index.
                    # We use INSERT OR IGNORE to prevent errors on duplicate primary keys (timestamps).
                    # This makes the import process idempotent (re-runnable).
                    chunk_df.to_sql(
                        name=table_name,
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi', # Efficiently inserts multiple rows
                        chunksize=1000, # How many rows to insert per SQL statement
                    )
                    
                    total_rows_inserted += len(chunk_df)
                
                # SQLite does not have a native 'INSERT OR IGNORE' for pandas' to_sql.
                # The primary key constraint on 'timestamp' will cause an error on duplicates.
                # A more robust approach for idempotent imports is to first delete overlapping data,
                # or manually handle it. However, for a one-time seed, the simple append is usually fine.
                # If you need to re-run it, you might clear the table first.
                
                conn.commit()
                print(f"\n Successfully imported {total_rows_inserted} rows into '{table_name}'.")
            
            except Exception as e:
                print(f"\n An error occurred during the import process: {e}")
                print(" Rolling back transaction.")
                conn.rollback()
    
    print(f"  <- Released DB lock for {symbol}")
    print(f"--- Import for {symbol} finished ---")




# This helper function calls the API and is cached. 
# It will only run ONCE for a given client instance.
@lru_cache(maxsize=1)
def _get_valid_api_symbols(client) -> set:
    """
    Fetches all symbol information from the exchange and returns a set of symbol names.
    The result is cached to prevent repeated API calls.
    """
    print("  -> Fetching exchange information to get all valid symbols (this will run only once)...")
    try:
        info = client.exchange_info()
        # The response contains a 'symbols' key with a list of dictionaries.
        # We create a set of the 'symbol' value from each dictionary for fast lookups.
        symbols = {s['symbol'] for s in info['symbols']}
        print(f"  -> Found {len(symbols)} valid symbols.")
        return symbols
    except Exception as e:
        print(f"  -> ERROR: Could not fetch exchange information: {e}")
        return set() # Return an empty set on failure

# This is the function we call from get_ohlcv. It's now extremely fast after the first run.
def _is_valid_api_symbol(client, symbol: str) -> bool:
    """
    Checks if a symbol is valid by looking it up in the cached set of all exchange symbols.
    """
    if not client:
        return False
    
    # Get the set of symbols (this will be instant after the first call)
    valid_symbols = _get_valid_api_symbols(client)
    
    is_valid = symbol in valid_symbols
    
    if not is_valid:
        # This log is still useful for non-API symbols like 'EURUSD'
        print(f"  -> INFO: Symbol '{symbol}' is not a valid API symbol. Will rely on DB only.")
        
    return is_valid

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
            res = conn.execute(text(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}")).fetchone()
            
            min_ts, max_ts = res if res and res[0] is not None else (None, None)

            start_ts_ms = int(start_dt.timestamp() * 1000)
            end_ts_ms = int(end_dt.timestamp() * 1000)

            # 3. Determine and fetch missing data
            # This list will hold all newly fetched data from all chunks.
            all_new_data = []

            # --- REVISED, SURGICAL FETCHING LOGIC ---
            if _is_valid_api_symbol(client, symbol):
                print(f"Symbol '{symbol}' is valid for the API. Proceeding with data fetch if needed.")
                
                # Case 1: The database is completely empty. Fetch the entire requested range.
                if min_ts is None:
                    print(f"Database table '{table_name}' is empty. Fetching initial data...")
                    all_new_data.extend(fetch_ohlcv_paginated(client, symbol, timeframe, start_ts_ms, end_ts_ms))
                
                else:
                    # Case 2: Fetch data *before* our cached range, if needed.
                    if start_ts_ms < min_ts:
                        # We need data from the user's start date up to the beginning of our cache.
                        # The `-1` prevents fetching the first candle we already have.
                        print(f"Fetching older data up to {datetime.fromtimestamp(min_ts / 1000)}...")
                        data_before = fetch_ohlcv_paginated(client, symbol, timeframe, start_ts_ms, min_ts - 1)
                        all_new_data.extend(data_before)

                    # Case 3: Fetch data *after* our cached range, if needed.
                    if end_ts_ms > max_ts:
                        # We need data from the end of our cache up to the user's end date.
                        # The `+1` prevents fetching the last candle we already have.
                        fetch_since = max_ts
                        
                        print(f"Fetching recent data since {datetime.fromtimestamp(fetch_since / 1000)}...")
                        
                        data_after = fetch_ohlcv_paginated(client, symbol, timeframe, fetch_since, end_ts_ms)
                        
                        # When we save, we will use INSERT OR REPLACE to update the last candle
                        # and insert the new ones.
                        if data_after:
                            print(f"Found {len(data_after)} new/updated candles to save.")
                            # Use a different save method for recent data
                            to_replace_dicts = [{'ts': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4], 'v': r[5]} for r in data_after]
                            replace_sql = text(f'''
                                INSERT OR REPLACE INTO {table_name} (timestamp, open, high, low, close, volume) 
                                VALUES (:ts, :o, :h, :l, :c, :v)
                            ''')
                            conn.execute(replace_sql, to_replace_dicts)
                            conn.commit()
                            print("Recent data successfully updated in the database.")
                
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
            else:
                print(f"Symbol '{symbol}' is NOT valid for the API. Skipping data fetch. Relying on existing DB data only.")
            
            # 5. Final data retrieval (also within the same transaction)
            query = text(f"SELECT * FROM {table_name} WHERE timestamp BETWEEN :start AND :end ORDER BY timestamp")
            
            # Use a new connection for read_sql_query to avoid transaction state issues
            # OR better, read the data first then close the write connection. Let's do that.
            df = pd.read_sql_query(query, conn, params={"start": start_ts_ms, "end": end_ts_ms})
    
    print(f"  <- Released DB lock for {symbol}")

    # Convert timestamp back to datetime for use in pandas
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
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