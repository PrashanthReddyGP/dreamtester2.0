import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseStrategy:
    def __init__(self, df = None, capital = 1000, cash = 10, risk_percent = 1, optim_number = 1,\
        symbol='ADAUSDT', timeframe='1d', start_date='2000-01-01', end_date='2100-12-31'):
        
        self.optim_number = optim_number
        
        self.sub_timeframes = []
        self.dataframes = {}
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        
        self.exit_type = 'TP and SL'
        
        self.initial_capital = capital
        self.capital = capital
        self.cash = cash
        self.equity = [capital]  # Track equity over time
        self.portfolio_equity = [capital]
        
        self.rr = 2
        self.risk_percent = risk_percent
        self.commission = 0.05
        
        self.realized_commission = 0
        self.no_fee_equity = 0
        
        if df is not None and not df.empty:
            
            self.open = df['open'].values
            self.high = df['high'].values
            self.low = df['low'].values
            self.close = df['close'].values
            self.volume = df['volume'].values
            
            df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6 
            self.timestamp = df['timestamp'].values
            
            self.trade_closed = np.zeros(len(self.close))
            self.result = np.zeros(len(self.close))
            self.exit_time = np.zeros(len(self.close))
            self.open_trades = np.zeros(len(self.close))
            self.signals = np.zeros(len(self.close))
    
    def _get_indicator_column_name(self, indicator_tuple):
        """
        Generates the DataFrame column name based on an indicator tuple.
        This MUST match the naming convention used in your process_indicators.py.
        """
        name, timeframe, params = indicator_tuple
        
        name = name.lower() # Standardize to lowercase
        
        if name != 'sma':
            return name.replace(' ', '_')
        else:
            # Assumes your process_indicators creates columns like 'sma_50', 'sma_200'
            length = int(params[0]) 
            return f'sma_{length}_{timeframe}'
    
    def _get_indicator_args(self):
        """
        Iterates through self.indicators and builds the list of numpy arrays
        to be passed to the JIT function. This method handles all special cases.
        """
        args = []
        for indicator_tuple in self.indicators:
            indicator_name = indicator_tuple[0].lower()
            
            # --- HANDLE SPECIAL CASES CENTRALLY ---
            if indicator_name != 'sma':
                pass
            else:
                column_name = self._get_indicator_column_name(indicator_tuple)
                args.append(self.df[column_name].values)
                
        return args
    
    def optimized_run(self):
        pass
    
    def optimized_exit(self):
        pass
    
    def optimized_hedge(self):
        pass
    
    def run(self):
        
        df = self.df
        
        if self.exit_type == 'TP and SL':
            self.optimized_run()
        elif self.exit_type == 'EXIT':
            self.optimized_exit()
        elif self.exit_type == 'HEDGE':
            self.optimized_hedge()
        
        # This "shuffles" the equity values to their correct exit timestamps.
        corrected_equity, corrected_no_fee_equity, realized_commission = self.correct_equity_timelines()
        
        # This DataFrame is indexed by the ENTRY timestamp, which is what we want for the UI trade list.
        df_full = self.df.copy()
        df_full['timestamp'] = pd.to_datetime(self.timestamp, unit='ms')
        df_full['Entry'] = self.entry
        df_full['Take Profit'] = self.takeprofit
        df_full['Stop Loss'] = self.stoploss
        df_full['Trade_Closed'] = self.trade_closed
        df_full['Result'] = self.result
        df_full['Exit_Time'] = pd.to_datetime(np.where(self.exit_time == 0.0, pd.NaT, self.exit_time), unit='ms')
        df_full['Open_Trades'] = self.open_trades
        df_full['Signal'] = self.signals
        df_full['RR'] = self.rr
        df_full['Risk'] = self.risk_percent
        df_full['Returns'] = self.returns
        df_full['Commissioned Returns'] = self.commissioned_returns
        
        # 1. Perform the calculation. The result is a NumPy array.
        reduction_array = np.where(
            df_full['Returns'].values == 0,
            0,
            abs((df_full['Returns'].values - df_full['Commissioned Returns'].values) * 100 / (df_full['Returns'].values + 1e-9))
        ).round(2)
        
        # 2. Assign the array to the DataFrame column
        df_full['Reduction'] = reduction_array
        
        # 3. Now, call .fillna() on the pandas Series (the column itself). This is the safe way.
        df_full['Reduction'] = df_full['Reduction'].fillna(0) # Assign back, don't use inplace=True
        
        df_full['Max Drawdown(%)'] = self.max_drawdown
        
        df_full['Drawdown Duration'] = pd.to_datetime(np.where(self.drawdown_duration == 0.0, pd.NaT, self.drawdown_duration), unit='ms').time
        df_full['Max Pull Up(%)'] = self.max_pull_up
        df_full['Pull Up Duration'] = pd.to_datetime(np.where(self.pull_up_duration == 0.0, pd.NaT, self.pull_up_duration), unit='ms').time
        df_full['Avg Volatility(%)'] = self.avg_volatility
        
        df_full['Symbol'] = self.symbol
        df_full['Timeframe'] = self.timeframe
        
        # Reorder columns
        cols = df_full.columns.tolist()
        new_order = [cols[0]] + ['Symbol', 'Timeframe'] + cols[1:-2]
        df_full = df_full[new_order]
        
        dfSignals = df_full[df_full['Signal'] != 0].copy()
        
        # This is the new, simpler, and more robust method you suggested.
        
        # 1. Start with a direct copy of all trade signals and their data.
        trade_events_df = dfSignals.copy()
        
        # Drop any trades that never closed
        trade_events_df.dropna(subset=['Exit_Time'], inplace=True)
        
        return corrected_equity, dfSignals, df_full, realized_commission, corrected_no_fee_equity, trade_events_df
    
    # You should also create this helper method in BaseStrategy for cleanliness
    def correct_equity_timelines(self):
        trades_df = pd.DataFrame({
            'entry_timestamp': pd.to_datetime(self.timestamp, unit='ms'),
            'exit_timestamp': pd.to_datetime(np.where(self.exit_time == 0.0, pd.NaT, self.exit_time), unit='ms'),
            'equity_at_entry': self.equity,
            'no_fee_equity_at_entry': self.no_fee_equity
        }).dropna(subset=['exit_timestamp'])
        
        full_timeline_df = pd.DataFrame(index=pd.to_datetime(self.timestamp, unit='ms'))
        full_timeline_df['equity'] = np.nan
        full_timeline_df['no_fee_equity'] = np.nan
        full_timeline_df.iloc[0] = self.initial_capital
        
        for _, trade in trades_df.iterrows():
            full_timeline_df.loc[trade['exit_timestamp'], 'equity'] = trade['equity_at_entry']
            full_timeline_df.loc[trade['exit_timestamp'], 'no_fee_equity'] = trade['no_fee_equity_at_entry']
        
        full_timeline_df['equity'] = full_timeline_df['equity'].ffill()
        full_timeline_df['no_fee_equity'] = full_timeline_df['no_fee_equity'].ffill()
        
        corrected_equity_array = full_timeline_df['equity'].values
        corrected_no_fee_equity_array = full_timeline_df['no_fee_equity'].values
        
        corrected_no_fee_gain = corrected_no_fee_equity_array[-1] - corrected_no_fee_equity_array[0]
        corrected_net_gain = corrected_equity_array[-1] - corrected_equity_array[0]
        
        if corrected_no_fee_gain > 0:
            realized_commission = round(((corrected_no_fee_gain - corrected_net_gain) * 100) / corrected_no_fee_gain, 2)
        else:
            realized_commission = 0.0
        
        return corrected_equity_array, corrected_no_fee_equity_array, realized_commission
    
    def portfolio(self,
                timestamps,
                entry_trade_ids_list,
                entry_directions_list,
                entry_prices_list,
                entry_sls_list,
                entry_tps_list,
                entry_risks_list,
                exit_trade_ids_list,
                exit_results_list):
        """
        A realistic, event-driven portfolio simulator that now also returns a
        log of closed trades and enforces portfolio-level risk limits.
        
        Portfolio Risk:
        - max_longs: The maximum number of long positions allowed to be open at any time.
        - max_shorts: The maximum number of short positions allowed to be open at any time.
        """
        capital = self.initial_capital
        portfolio_equity = np.zeros(len(timestamps), dtype=np.float64)
        open_trade_count = np.zeros(len(timestamps), dtype=np.int32)
        
        # --- Portfolio Risk Parameters ---
        max_longs = 50
        max_shorts = 50
        fee_rate = 0.001
        
        if len(timestamps) > 0:
            portfolio_equity[0] = capital
        
        open_trades = {}
        closed_trades_log = []
        
        for i in range(1, len(timestamps)):
            capital = portfolio_equity[i-1]
            
            # --- Step 1: Process Entries (Modified for Portfolio Risk) ---
            if isinstance(entry_trade_ids_list[i], list):
                
                # <<< NEW: Count existing open positions before processing new entries for this bar >>>
                current_longs = sum(1 for trade in open_trades.values() if trade['direction'] == 1)
                current_shorts = len(open_trades) - current_longs

                for idx, entry_id in enumerate(entry_trade_ids_list[i]):
                    
                    direction = entry_directions_list[i][idx]

                    # <<< NEW: Check portfolio risk limits before entering a trade >>>
                    if direction == 1: # Proposed trade is a Long
                        if current_longs >= max_longs:
                            continue # Skip this trade, max longs reached for this bar
                    else: # Proposed trade is a Short
                        if current_shorts >= max_shorts:
                            continue # Skip this trade, max shorts reached for this bar

                    # If the code reaches here, the trade is allowed by portfolio risk rules.
                    risk_percent_for_trade = entry_risks_list[i][idx]
                    
                    entry_price = entry_prices_list[i][idx]
                    stop_loss = entry_sls_list[i][idx]
                    risk_per_unit = abs(entry_price - stop_loss)
                    
                    if risk_per_unit > 1e-9:
                        cash_at_risk = capital * (risk_percent_for_trade / 100)
                        position_size = cash_at_risk / risk_per_unit
                        open_trades[int(entry_id)] = {
                            'direction': direction,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': entry_tps_list[i][idx],
                            'position_size': position_size
                        }
                        
                        # <<< NEW: Increment the relevant counter for this bar after opening the trade >>>
                        # This ensures subsequent signals in the SAME bar are checked against the new count.
                        if direction == 1:
                            current_longs += 1
                        else:
                            current_shorts += 1

            # --- Step 2: Process Exits (No changes here) ---
            total_return_this_bar = 0.0
            if isinstance(exit_trade_ids_list[i], list):
                for idx, exit_id in enumerate(exit_trade_ids_list[i]):
                    if int(exit_id) in open_trades:
                        trade = open_trades[int(exit_id)]
                        result = exit_results_list[i][idx]
                        exit_price = trade['take_profit'] if result == 1 else trade['stop_loss']
                        
                        if trade['direction'] == 1: # Long
                            gross_return = (exit_price - trade['entry_price']) * trade['position_size']
                        else: # Short
                            gross_return = (trade['entry_price'] - exit_price) * trade['position_size']
                            
                        entry_value = trade['entry_price'] * trade['position_size']
                        # The fee for closing the trade is on the exit_value, but for simplicity
                        # and common practice, many just double the entry fee. Let's assume a
                        # simple fee on the entry notional value for this example.
                        total_fee = (entry_value) * fee_rate 
                        net_return = gross_return - total_fee
                        
                        total_return_this_bar += net_return
                        
                        closed_trades_log.append({
                            'trade_id': int(exit_id),
                            'final_gross_return': gross_return,
                            'final_net_return': net_return
                        })
                        
                        del open_trades[int(exit_id)]
            
            # --- Step 3: Apply net P/L and store state (No changes here) ---
            capital += total_return_this_bar
            portfolio_equity[i] = capital
            open_trade_count[i] = len(open_trades)
        
        return portfolio_equity, open_trade_count, closed_trades_log