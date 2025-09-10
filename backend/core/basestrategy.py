import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseStrategy:
    def __init__(self, df = None, capital = 1000, cash = 10, risk_percent = 1, optim_number = 1,\
        symbol='ADAUSDT', timeframe='1d', start_date='2000-01-01', end_date='2100-12-31'):
        
        self.optim_number = optim_number
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date

        # Indicators as a dictionary, specifying the indicator type and parameters
        self.indicators = [
            ('ATR', '1m', (100, 8))
        ]
        
        self.exit_type = 'TP and SL'
        
        self.initial_capital = capital
        self.capital = capital
        self.cash = cash
        self.equity = [capital]  # Track equity over time
        self.portfolio_equity = [capital]
        
        self.rr = 2
        self.risk_percent = risk_percent

        self.commission = 0
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
            
            self.df = df
            
    # def entry_condition(self, i):
    #     """Override this method in specific strategy class to define exit conditions."""
    #     raise NotImplementedError
    
    # def takeprofit_condition(self, i, j):
    #     """Override this method in specific strategy class to define exit conditions."""
    #     raise NotImplementedError
    
    # def stoploss_condition(self, i, j):
    #     """Override this method in specific strategy class to define exit conditions."""
    #     raise NotImplementedError
    
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
            return f'sma_{length}'
    
    # --- THIS IS THE NEW, CENTRALIZED LOGIC ---
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
        corrected_equity, corrected_no_fee_equity, commission = self.correct_equity_timelines()
        
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
        
        return corrected_equity, dfSignals, df_full, commission, corrected_no_fee_equity, trade_events_df
    
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
            commission = round(((corrected_no_fee_gain - corrected_net_gain) * 100) / corrected_no_fee_gain, 2)
        else:
            commission = 0.0

        return corrected_equity_array, corrected_no_fee_equity_array, commission
    
    # def portfolio(self, timestamps, results, rrs, reductions, commissioned_returns_combined):

    #     capital = self.initial_capital
    #     cash = (capital * self.risk_percent) / 100
        
    #     # Initialize the equity array with the starting capital.
    #     # It must have the same length as the number of timestamps.
    #     portfolio_equity = np.zeros(len(timestamps), dtype=np.float64)
    #     portfolio_equity[0] = capital
        
    #     for i in range(1, len(timestamps)):
            
    #         result = results[i]
    #         rr = rrs[i]
    #         reduction = reductions[i] / 100
            
    #         # First, carry forward the previous capital value
    #         capital = portfolio_equity[i-1]
            
    #         if result == 1:
                
    #             net_cash = cash * (1 - reduction)
    #             capital  += (net_cash * rr)
    #             cash = (capital * self.risk_percent) / 100
            
    #         elif result == -1:
                
    #             net_cash = cash * (1 - reduction)
    #             capital  -= net_cash
    #             cash = (capital * self.risk_percent) / 100
            
    #         # Store the new capital value for this timestamp
    #         portfolio_equity[i] = capital
            
    #         # # If the current timestamp is different from the previous one, append the equity once
    #         # if current_timestamp != previous_timestamp:
    #         #     capital = temp_capital  # Update the main capital with temp changes
    #         #     self.portfolio_equity.append(capital)
        
    #     # # The print statement is great for debugging, let's keep it but make it clearer
    #     # print(f"DEBUG: Input timestamps length: {len(timestamps)}, Output portfolio_equity length: {len(portfolio_equity)}")
        
    #     return portfolio_equity
    
    # def portfolio(self,
    #             timestamps,
    #             entry_trade_ids_list, # Renamed to reflect it's a list of lists
    #             entry_rrs_list,
    #             entry_reductions_list,
    #             exit_trade_ids_list,
    #             exit_results_list):
    #     """
    #     A realistic, event-driven portfolio simulator that correctly models
    #     shared capital dynamics, including multiple concurrent events at the same timestamp.
    #     """
    #     capital = self.initial_capital
    #     portfolio_equity = np.zeros(len(timestamps), dtype=np.float64)
    #     open_trade_count = np.zeros(len(timestamps), dtype=np.int32)
        
    #     if len(timestamps) > 0:
    #         portfolio_equity[0] = capital
        
    #     open_trades = {}
        
    #     for i in range(1, len(timestamps)):
            
    #         capital = portfolio_equity[i-1]
            
    #         # --- Step 1: Process ALL NEW ENTRIES for this timestamp ---
    #         # Check if the entry data is a list (not NaN)
    #         if isinstance(entry_trade_ids_list[i], list):
    #             # Loop through all entry events that happened at this exact time
    #             for idx, entry_id in enumerate(entry_trade_ids_list[i]):
    #                 cash_at_risk = capital * (self.risk_percent / 100)
                    
    #                 open_trades[int(entry_id)] = {
    #                     'cash_at_risk': cash_at_risk,
    #                     'rr': entry_rrs_list[i][idx],
    #                     'reduction': entry_reductions_list[i][idx] / 100.0
    #                 }
            
    #         # --- Step 2: Process ALL EXITS for this timestamp ---
    #         # NOTE: Capital is NOT updated between exits on the same bar.
    #         # All exits are settled against the capital from the START of the bar.
    #         total_return_this_bar = 0.0
            
    #         if isinstance(exit_trade_ids_list[i], list):
    #             for idx, exit_id in enumerate(exit_trade_ids_list[i]):
    #                 if int(exit_id) in open_trades:
    #                     trade = open_trades[int(exit_id)]
    #                     result = exit_results_list[i][idx]
                        
    #                     trade_return = 0
    #                     net_cash = trade['cash_at_risk'] * (1 - trade['reduction'])
                        
    #                     if result == 1: trade_return = net_cash * trade['rr']
    #                     elif result == -1: trade_return = -net_cash
                        
    #                     # Accumulate the returns for this bar
    #                     total_return_this_bar += trade_return
                        
    #                     # Remove the trade from open positions
    #                     del open_trades[int(exit_id)]
            
    #         # --- Step 3: Apply net P/L and store final state ---
    #         capital += total_return_this_bar
    #         portfolio_equity[i] = capital
    #         open_trade_count[i] = len(open_trades)
        
    #     return portfolio_equity, open_trade_count


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
        log of closed trades with their final, accurate P/L.
        """
        capital = self.initial_capital
        portfolio_equity = np.zeros(len(timestamps), dtype=np.float64)
        open_trade_count = np.zeros(len(timestamps), dtype=np.int32)
        
        risk_percent = 1
        fee_rate = 0.001
        
        if len(timestamps) > 0:
            portfolio_equity[0] = capital
        
        open_trades = {}
        closed_trades_log = []
        
        for i in range(1, len(timestamps)):
            capital = portfolio_equity[i-1]
            
            # --- Step 1: Process Entries (No changes here) ---
            if isinstance(entry_trade_ids_list[i], list):
                
                for idx, entry_id in enumerate(entry_trade_ids_list[i]):
                    
                    risk_percent_for_trade = entry_risks_list[i][idx]
                    
                    entry_price = entry_prices_list[i][idx]
                    stop_loss = entry_sls_list[i][idx]
                    risk_per_unit = abs(entry_price - stop_loss)
                    
                    if risk_per_unit > 1e-9:
                        cash_at_risk = capital * (risk_percent_for_trade / 100)
                        position_size = cash_at_risk / risk_per_unit
                        open_trades[int(entry_id)] = {
                            'direction': entry_directions_list[i][idx],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': entry_tps_list[i][idx],
                            'position_size': position_size
                        }

            # --- Step 2: Process Exits ---
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
                        exit_value = exit_price * trade['position_size']
                        total_fee = (entry_value) * fee_rate
                        net_return = gross_return - total_fee
                        
                        total_return_this_bar += net_return
                        
                        # <<< Log the closed trade details >>>
                        closed_trades_log.append({
                            'trade_id': int(exit_id),
                            'final_gross_return': gross_return,
                            'final_net_return': net_return
                        })
                        
                        del open_trades[int(exit_id)]
            
            # --- Step 3: Apply net P/L and store state ---
            capital += total_return_this_bar
            portfolio_equity[i] = capital
            open_trade_count[i] = len(open_trades)
        
        return portfolio_equity, open_trade_count, closed_trades_log