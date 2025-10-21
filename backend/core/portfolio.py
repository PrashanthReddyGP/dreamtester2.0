import numpy as np

class Portfolio:
    """Base class for all portfolio management strategies."""

    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital

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
        This is the default portfolio logic. It should be overridden by custom classes.
        This version is a simple pass-through with basic risk management.
        """
        capital = self.initial_capital
        portfolio_equity = np.zeros(len(timestamps), dtype=np.float64)
        open_trade_count = np.zeros(len(timestamps), dtype=np.int32)
        
        # --- Default Portfolio Risk Parameters ---
        max_longs = 50
        max_shorts = 50
        fee_rate = 0.001
        
        if len(timestamps) > 0:
            portfolio_equity[0] = capital
        
        open_trades = {}
        closed_trades_log = []
        
        for i in range(1, len(timestamps)):
            capital = portfolio_equity[i-1]
            
            # --- Step 1: Process Entries ---
            if isinstance(entry_trade_ids_list[i], list):
                current_longs = sum(1 for t in open_trades.values() if t['direction'] == 1)
                current_shorts = len(open_trades) - current_longs

                for idx, entry_id in enumerate(entry_trade_ids_list[i]):
                    direction = entry_directions_list[i][idx]
                    if direction == 1 and current_longs >= max_longs:
                        continue
                    if direction != 1 and current_shorts >= max_shorts:
                        continue
                        
                    risk_percent = entry_risks_list[i][idx]
                    entry_price = entry_prices_list[i][idx]
                    stop_loss = entry_sls_list[i][idx]
                    risk_per_unit = abs(entry_price - stop_loss)
                    
                    if risk_per_unit > 1e-9:
                        cash_at_risk = capital * (risk_percent / 100)
                        position_size = cash_at_risk / risk_per_unit
                        open_trades[int(entry_id)] = {
                            'direction': direction,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': entry_tps_list[i][idx],
                            'position_size': position_size
                        }
                        if direction == 1: current_longs += 1
                        else: current_shorts += 1

            # --- Step 2: Process Exits ---
            total_return_this_bar = 0.0
            if isinstance(exit_trade_ids_list[i], list):
                for idx, exit_id in enumerate(exit_trade_ids_list[i]):
                    if int(exit_id) in open_trades:
                        trade = open_trades[int(exit_id)]
                        result = exit_results_list[i][idx]
                        exit_price = trade['take_profit'] if result == 1 else trade['stop_loss']
                        
                        gross_return = (exit_price - trade['entry_price']) * trade['position_size'] if trade['direction'] == 1 else (trade['entry_price'] - exit_price) * trade['position_size']
                        
                        entry_value = trade['entry_price'] * trade['position_size']
                        total_fee = entry_value * fee_rate
                        net_return = gross_return - total_fee
                        
                        total_return_this_bar += net_return
                        
                        closed_trades_log.append({
                            'trade_id': int(exit_id),
                            'final_gross_return': gross_return,
                            'final_net_return': net_return
                        })
                        del open_trades[int(exit_id)]
            
            # --- Step 3: Apply net P/L ---
            capital += total_return_this_bar
            portfolio_equity[i] = capital
            open_trade_count[i] = len(open_trades)
        
        return portfolio_equity, open_trade_count, closed_trades_log