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
        # for i in range(1, len(self.close)):
            
        #     if self.entry_condition(i): # Replace the ENTRY CONDITION    
                
        #         self.signals[i] = 1
                
        #         for j in range(i, len(self.close)):
                    
        #             self.open_trades[j] += 1
                    
        #             if self.takeprofit_condition(i, j):
                        
        #                 self.trade_closed[i] = 1
        #                 self.result[i] = 1
        #                 self.exit_time[i] = self.timestamp[j]
        #                 self.capital += (self.cash * self.rr)
        #                 self.cash = ((self.capital * self.risk_percent) / 100)
        #                 break
                    
        #             elif self.stoploss_condition(i, j):
                        
        #                 self.trade_closed[i] = 1
        #                 self.result[i] = -1
        #                 self.exit_time[i] = self.timestamp[j]                        
        #                 self.capital -= self.cash
        #                 self.cash = ((self.capital * self.risk_percent) / 100)
        #                 break
                    
        #     self.equity.append(self.capital)
    
    def optimized_exit(self):
        pass
    
    def run(self):
        
        df = self.df
        
        if self.exit_type == 'TP and SL':
            # self.optimized_run()
            
            # profiler = LineProfiler()
            # profiler.add_function(self.optimized_run)
            
            # profiler.enable_by_count()
            
            self.optimized_run()
            
            # profiler.disable_by_count()
            
            # profiler.print_stats()  # Print the detailed report
            
        elif self.exit_type == 'EXIT':
            self.optimized_exit()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['Entry'] = self.entry
        df['Take Profit'] = self.takeprofit
        df['Stop Loss'] = self.stoploss
        df['Trade_Closed'] = self.trade_closed
        df['Result'] = self.result        
        df['Exit_Time'] = pd.to_datetime(np.where(self.exit_time == 0.0, pd.NaT, self.exit_time), unit='ms')
        df['Open_Trades'] = self.open_trades
        df['Signal'] = self.signals
        df['RR'] = self.rr
        df['Returns'] = self.returns
        df['Commissioned Returns'] = self.commissioned_returns
        # If 'Returns' is 0, the reduction is not mathematically defined. We can set it to 0 or 100.
        # Setting to 0 is safer to prevent errors, as it implies no reduction from a non-existent profit.
        df['Reduction'] = np.where(
            df['Returns'] == 0,
            0, # Value to use if Returns is 0
            round(abs((df['Returns'] - df['Commissioned Returns']) * 100 / df['Returns']), 2) # Original calculation
        )
        # Make sure to handle potential new NaNs if Commissioned Returns can be NaN when Returns is 0
        df['Reduction'] = df['Reduction'].fillna(0)
        
        df['Max Drawdown(%)'] = self.max_drawdown
        
        df['Drawdown Duration'] = pd.to_datetime(np.where(self.drawdown_duration == 0.0, pd.NaT, self.drawdown_duration), unit='ms').time
        df['Max Pull Up(%)'] = self.max_pull_up
        df['Pull Up Duration'] = pd.to_datetime(np.where(self.pull_up_duration == 0.0, pd.NaT, self.pull_up_duration), unit='ms').time
        df['Avg Volatility(%)'] = self.avg_volatility
        
        df['Symbol'] = self.symbol
        df['Timeframe'] = self.timeframe
        
        # Reorder columns
        cols = df.columns.tolist()
        new_order = [cols[0]] + ['Symbol', 'Timeframe'] + cols[1:-2]
        df = df[new_order]
        
        dfSignals = df[df['Signal'] != 0]
        
        return self.equity, dfSignals, df, self.commission, self.no_fee_equity
    
    def portfolio(self, timestamps, results, rrs, reductions, commissioned_returns_combined):

        capital = self.initial_capital
        cash = (capital * self.risk_percent) / 100
        
        previous_timestamp = timestamps[0]  # To track the previous timestamp
        temp_capital = capital     # To handle multiple entries for the same timestamp
        
        for i in range(1, len(timestamps)):
            
            current_timestamp = timestamps[i]
            result = results[i]
            rr = rrs[i]
            reduction = reductions[i] / 100
            
            if result == 1:
                
                net_cash = cash * (1 - reduction)
                temp_capital += (net_cash * rr)
                cash = (temp_capital * self.risk_percent) / 100
            
            elif result == -1:
                
                net_cash = cash * (1 - reduction)
                temp_capital -= net_cash
                cash = (temp_capital * self.risk_percent) / 100
                
            # If the current timestamp is different from the previous one, append the equity once
            if current_timestamp != previous_timestamp:
                capital = temp_capital  # Update the main capital with temp changes
                self.portfolio_equity.append(capital)
            
            # Update the previous timestamp to the current one for the next iteration
            previous_timestamp = current_timestamp

        return self.portfolio_equity