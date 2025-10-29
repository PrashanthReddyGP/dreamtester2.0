import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime

def calculate_metrics(strategy_data, initialCapital, commission):
    
    print("  - Calculating metrics (Sharpe, Drawdown)...")

    strategy_metrics = {}
    
    strategy = strategy_data
    
    equity = np.array(strategy['equity'])
    
    dfSignals = strategy['signals'].copy()
    df = strategy['ohlcv'].copy()
    
    if dfSignals.empty:
        # If there are no trades, return a default/empty metrics dictionary
        print("Strategy produced zero trades!!!!!!")
        return {'Sharpe_Ratio': 0, 'Max_Drawdown': 0, 'Total_Trades': 0, 'Winrate': 0}, [] # Return empty monthly returns too
    
    # # Calculate how many rows are missing from the start
    # len_diff = len(df) - len(equity)
    
    # # Create a new pandas Series for the equity.
    # # We pad the beginning with the initial_capital value for the rows
    # # that were part of the indicator lookback period.
    # aligned_equity = pd.Series(
    #     [initialCapital] * len_diff + equity.tolist(), 
    #     index=df.index
    # )
    
    monthly_returns = calculate_monthly_returns(
        df, 
        equity
    )

    avg_monthly_returns = 0.0
    if monthly_returns: # Check if the list is not empty
        # Create a list of just the percentage return values
        all_returns = [row['Returns (%)'] for row in monthly_returns]
        # Calculate the average (mean)
        avg_monthly_returns = np.mean(all_returns)

    def max_drawdown(equity):
        peaks = np.maximum.accumulate(equity)  # Track peaks
        drawdowns = (equity - peaks) / peaks  # Calculate drawdowns
        return np.min(drawdowns)*100, drawdowns*100  # Return the largest (worst) drawdown

    maxDrawdown, drawdowns = max_drawdown(equity)
    
    # if type == 'backtest':
    
    netProfit = equity[-1] - equity[0]
    grossProfit = equity[-1]
    profitPercentage = (netProfit / equity[0]) * 100
    totalTrades = len(dfSignals)
    openTrades = len(dfSignals[dfSignals['Trade_Closed'] == 0])
    closedTrades = len(dfSignals[dfSignals['Trade_Closed'] == 1])
    maxDrawdown = maxDrawdown
    avgDrawdown = np.mean(drawdowns)
    profitFactor = 0
    sharpeRatio = 0
    totalWins = len(dfSignals[dfSignals['Result'] == 1])
    totalLosses = len(dfSignals[dfSignals['Result'] == -1])
    consecutiveWins = 0
    consecutiveLosses = 0
    largestWin = 0
    largestLoss = 0
    avgWin = 0
    avgLoss = 0
    avgTrade = 0
    avgTradeTime = 0
    avgWinTime = 0
    avgLossTime = 0
    maxRunup = 0
    avgRunup = 0
    winrate = (totalWins/totalTrades) * 100 if totalTrades > 0 else 0
    rr = 0
    
    open_trades_series = dfSignals['Open_Trades']
    
    if not open_trades_series.empty:
        maxOpenTrades = int(open_trades_series.fillna(0).max())
        avgOpenTrades = 1 if dfSignals['Open_Trades'].mean() < 1 else round(dfSignals['Open_Trades'].mean())
    else:
        maxOpenTrades = 0
        avgOpenTrades = 0
    
    # Profit Factor Calculation
    def calculate_profit_factor(equity):
        gains = np.diff(equity)[np.diff(equity) > 0]  # Positive returns (gains)
        losses = np.abs(np.diff(equity)[np.diff(equity) < 0])  # Negative returns (losses)
        gross_profit = np.sum(gains)
        gross_loss = np.sum(losses)
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    profitFactor = calculate_profit_factor(equity)

    # # Sharpe Ratio Calculation
    # def calculate_sharpe_ratio(equity):
        
    #     minute_returns = np.diff(equity) / equity[:-1]
    #     avg_minute_return = np.mean(minute_returns)
    #     minute_std_dev = np.std(minute_returns)
        
    #     annual_return = (1 + avg_minute_return) ** 525600 - 1
        
    #     annual_std_dev = minute_std_dev * np.sqrt(525600)
        
    #     risk_free_rate_annual = 0.02 # 2%
        
    #     return (annual_return - risk_free_rate_annual) / annual_std_dev if annual_std_dev > 0 else 0
    
    # sharpeRatio = calculate_sharpe_ratio(equity)
    
    # def calculate_sharpe_ratio(equity, timeframe='15m', risk_free_rate_annual=0.02):
    #     """
    #     Calculate annualized Sharpe ratio dynamically for any crypto timeframe.
        
    #     Parameters:
    #         equity (np.ndarray): equity curve array
    #         timeframe (str): timeframe string e.g. '1m', '5m', '15m', '1h', '4h', '1d'
    #         risk_free_rate_annual (float): annual risk-free rate (default 2%)
    #     """
    #     if len(equity) < 2:
    #         return 0.0
        
    #     # Simple returns per bar
    #     returns = np.diff(equity) / equity[:-1]
    #     mean_ret = np.mean(returns)
    #     std_ret = np.std(returns)
        
    #     # Annualization factors (24/7 crypto)
    #     periods_per_year = {
    #         '1m': 525600,
    #         '3m': 175200,
    #         '5m': 105120,
    #         '15m': 35040,
    #         '30m': 17520,
    #         '1h': 8760,
    #         '4h': 2190,
    #         '6h': 1460,
    #         '12h': 730,
    #         '1d': 365,
    #         '1w': 52,
    #         '1mo': 12
    #     }
        
    #     N = periods_per_year.get(timeframe.lower(), 525600)  # default 1-minute
        
    #     # Convert annual risk-free rate to per-bar equivalent
    #     rf_per_bar = (1 + risk_free_rate_annual)**(1 / N) - 1
        
    #     # Annualized Sharpe
    #     sharpe = (mean_ret - rf_per_bar) / std_ret * np.sqrt(N) if std_ret > 0 else 0
        
    #     return sharpe
    
    # sharpeRatio = calculate_sharpe_ratio(equity, timeframe=timeframe)
        
    def calculate_sharpe_ratio(
        equity_series,  # Expects a pandas Series with a DatetimeIndex
        risk_free_rate_annual=0.02
    ):
        """
        Calculates the annualized Sharpe ratio from a pandas Series of equity values.
        The standard and correct method is to use daily returns.
        """
        if len(equity_series) < 2:
            return 0.0

        # 1. Resample to daily frequency, getting the last value of each day.
        daily_equity = equity_series.resample('D').last()

        # 2. Calculate daily returns.
        # The .pct_change() method is perfect for this.
        daily_returns = daily_equity.pct_change().dropna()
        
        if len(daily_returns) < 2:
            return 0.0

        # 3. Calculate mean and std of DAILY returns.
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        if std_daily_return == 0:
            return 0.0

        # 4. Annualize. The annualization factor for daily returns is sqrt(365) for crypto.
        # We also need to calculate the daily risk-free rate.
        # Note: For daily returns, it's often acceptable to simplify and use (annual_return - rfr) / annual_stdev
        # but we will stick to the more precise method.
        
        daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/365) - 1
        
        excess_return = mean_daily_return - daily_risk_free_rate
        
        # The annualization factor is sqrt(number of periods in a year)
        sharpe_ratio = (excess_return / std_daily_return) * np.sqrt(365)
        
        return sharpe_ratio
    
    sharpeRatio = calculate_sharpe_ratio(equity_series=pd.Series(equity, index=df['timestamp']), risk_free_rate_annual=0.02)
    
    @njit
    def calculate_consecutive_wins_losses(results):
        consecutiveWins = 0
        consecutiveLosses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for i in range(len(results)):
            if results[i] == 1:
                current_consecutive_wins += 1
                current_consecutive_losses = 0  # Reset losses if there's a win
                consecutiveWins = max(consecutiveWins, current_consecutive_wins)
            elif results[i] == -1:
                current_consecutive_losses += 1
                current_consecutive_wins = 0  # Reset wins if there's a loss
                consecutiveLosses = max(consecutiveLosses, current_consecutive_losses)
            else:
                current_consecutive_wins = 0
                current_consecutive_losses = 0  # Reset both if result is neither win nor loss

        return consecutiveWins, consecutiveLosses


    @njit
    def calculate_trade_stats(equity):
        largestWin = -np.inf
        largestLoss = np.inf
        
        winning_trades = []
        losing_trades = []
        total_trades = []
        
        for i in range(1, len(equity)):
            diff = equity[i] - equity[i - 1]
            
            # Update largest win and largest loss
            largestWin = max(largestWin, diff)
            largestLoss = min(largestLoss, diff)
            
            if diff > 0:
                winning_trades.append(diff)
            elif diff < 0:
                losing_trades.append(diff)
            
            total_trades.append(diff)

        # Convert lists to numpy arrays for mean calculation
        winning_trades = np.array(winning_trades)
        losing_trades = np.array(losing_trades)
        total_trades = np.array(total_trades)
        
        # Use Numba-supported operations on arrays
        avgWin = np.mean(winning_trades) if winning_trades.size > 0 else 0
        avgLoss = np.mean(losing_trades) if losing_trades.size > 0 else 0
        avgTrade = np.mean(total_trades) if total_trades.size > 0 else 0
        
        return largestWin, largestLoss, avgWin, avgLoss, avgTrade
    
    # Max Run-up Calculation
    troughs = np.minimum.accumulate(equity)  # Track troughs
    runups = np.where(troughs != 0, (equity - troughs) / troughs, 0)  # Avoid division by zero
    maxRunup = np.max(runups) * 100
    avgRunup = np.mean(runups * 100)


    # Step 1: Calculate consecutive wins/losses
    consecutiveWins, consecutiveLosses = calculate_consecutive_wins_losses(dfSignals['Result'].to_numpy())

    # Step 2: Calculate largest win, loss, and average trade statistics
    equity_np = np.array(equity)
    largestWin, largestLoss, avgWin, avgLoss, avgTrade = calculate_trade_stats(equity_np)
    
    # Calculate Risk to Reward Ratio
    def calculate_RR(avg_profit_per_win, avg_loss_per_loss):
        
        RR_solution = abs(avg_profit_per_win) / abs(avg_loss_per_loss) if abs(avg_loss_per_loss) > 0 else 0
        
        return RR_solution
    
    def format_RR_as_ratio(RR):
        
        if RR > 0:
            risk = 1
            reward = round(RR, 1)
        else:        
            risk = round(RR, 1)
            reward = 1
        
        return f"{risk} : {reward}"
    
    rr_decimal = calculate_RR(avgWin, avgLoss)
    rr = format_RR_as_ratio(rr_decimal)
    
    dfSignals.loc[:, 'Trade_Duration'] = pd.NaT
    
    # Convert columns to datetime (if not already in datetime format)
    dfSignals.loc[:, 'timestamp'] = pd.to_datetime(dfSignals['timestamp'])
    dfSignals.loc[:, 'Exit_Time'] = pd.to_datetime(dfSignals['Exit_Time'])

    dfSignals['Trade_Duration'] = dfSignals[dfSignals['Result'] != 0]['Exit_Time'] - dfSignals[dfSignals['Result'] != 0]['timestamp']

    avgTradeTime = dfSignals[dfSignals['Result'] != 0]['Trade_Duration'].mean()
    
    avgWinTime = dfSignals[dfSignals['Result'] == 1]['Trade_Duration'].mean()
    avgLossTime = dfSignals[dfSignals['Result'] == -1]['Trade_Duration'].mean()
    
    start_date = dfSignals['timestamp'].iloc[0]
    end_date = dfSignals['timestamp'].iloc[-1]
    
    time_period_years = round((end_date - start_date).days / 365.25)  # Convert days to years
    
    total_return = round(((grossProfit - initialCapital) / initialCapital), 2)
    
    annualized_return = (1 + total_return) ** (1 / time_period_years) - 1 if time_period_years > 0 else 0
    
    calmar_ratio = round(annualized_return, 4) / round(abs(maxDrawdown / 100), 4) if abs(maxDrawdown) != 0 else 0
    
    peaks = np.maximum.accumulate(equity)  # Track peaks
    df['Peak'] = peaks
    df['Drawdown Phase'] = (equity < df['Peak']).astype(int)
    df['Drawdown Group'] = (df['Drawdown Phase'].diff() == 1).cumsum()
    drawdown_durations = df.groupby('Drawdown Group')['timestamp'].agg(['first', 'last'])
    drawdown_durations['duration'] = (drawdown_durations['last'] - drawdown_durations['first']).dt.days
    max_drawdown_duration = drawdown_durations['duration'].max() if not drawdown_durations.empty else 0
    
    trade_strength = np.log(totalTrades + 1) if totalTrades > 0 else 0
    profit_target = 100
    if profitPercentage > 0:
        profit_component = (profitPercentage / profit_target) * trade_strength
        risk_penalty = (1 / (1 + abs(maxDrawdown))) * (1 / (1 + max_drawdown_duration))
        eer = profit_component * risk_penalty
    else:
        loss_component = (abs(profitPercentage) / profit_target) * trade_strength
        eer = -loss_component
    eer = eer * 1000 # For readbility

    def get_eer_color(eer):
        if eer < 0.1:
            return 'Poor'  # Poor
        elif 0.1 <= eer < 0.5:
            return 'Bad'  # Bad
        elif 0.5 <= eer < 1:
            return 'Okay'  # Okayish
        elif 1 <= eer < 2:
            return 'Good'  # Good
        else:
            return 'Excellent'  # Excellent

    quality = get_eer_color(round(eer, 2))   
    
    ## END CALCULATIONS
    
    ## UPDATE

    strategy_metrics = {
        "Net_Profit": netProfit,
        "Gross_Profit": grossProfit,
        "Profit_Percentage": round(profitPercentage, 2),
        "Annual_Return": round(annualized_return * 100, 2),
        "Avg_Monthly_Return": round(avg_monthly_returns, 2),
        "Total_Trades": round(totalTrades),
        "Open_Trades": round(openTrades),
        "Closed_Trades": round(closedTrades),
        "Max_Drawdown": round(abs(maxDrawdown), 2),
        "Avg_Drawdown": round(abs(avgDrawdown), 2),
        "Profit_Factor": round(profitFactor, 2),
        "Sharpe_Ratio": round(sharpeRatio, 2),
        "Calmar_Ratio": round(calmar_ratio, 2),
        "Equity_Efficiency_Rate": round(eer, 2),
        "Strategy_Quality": quality,
        "Max_Drawdown_Duration_days": max_drawdown_duration,
        "Total_Wins": round(totalWins),
        "Total_Losses": round(totalLosses),
        "Consecutive_Wins": round(consecutiveWins),
        "Consecutive_Losses": round(consecutiveLosses),
        "Largest_Win": largestWin,
        "Largest_Loss": largestLoss,
        "Avg_Win": avgWin,
        "Avg_Loss": avgLoss,
        "Avg_Trade": avgTrade,
        "Avg_Trade_Time": str(avgTradeTime).split('.')[0],
        "Avg_Win_Time": str(avgWinTime).split('.')[0],
        "Avg_Loss_Time": str(avgLossTime).split('.')[0],
        "Max_Runup": round(abs(maxRunup), 2),
        "Avg_Runup": round(abs(avgRunup), 2),
        "Winrate": round(winrate, 2),
        "RR": rr,
        "Max_Open_Trades": maxOpenTrades,
        "Avg_Open_Trades": avgOpenTrades,
        "Commission": commission
    }

    # if netProfit < 0:
    #     self.ui.Net_Profit_Output.setStyleSheet("color: red;")  # Set text color to red
    #     self.ui.Net_Profit_Output.setText(f'-${abs(netProfit):,.0f}')
    # else:
    #     self.ui.Net_Profit_Output.setStyleSheet("color: lime;")  # Set text color to green
    #     self.ui.Net_Profit_Output.setText(f'${netProfit:,.0f}')
    
    # self.ui.Gross_Profit_Output.setText(f'${grossProfit:,.0f}')
    # self.ui.Profit_Percentage_Output.setText(f'{round(profitPercentage, 2)}%')
    # self.ui.Annual_Return_Output.setText(f'{round((annualized_return * 100), 2)}%')
    # self.ui.Total_Trades_Output.setText(f'{round(totalTrades)}')
    # self.ui.Open_Trades_Output.setText(f'{round(openTrades)}')
    # self.ui.Closed_Trades_Output.setText(f'{round(closedTrades)}')
    # self.ui.Max_Drawdown_Output.setText(f'{round(abs(maxDrawdown), 2)}%')
    # self.ui.Avg_Drawdown_Output.setText(f'{round(abs(avgDrawdown), 2)}%')
    # self.ui.Profit_Factor_Output.setText(f'{round(profitFactor, 2)}')
    # self.ui.Sharpe_Ratio_Output.setText(f'{round(sharpeRatio, 2)}')
                    
    # self.ui.Calmar_Ratio_Output.setText(f'{calmar_ratio:.2f}')                
    
    # # Function to determine the color based on Calmar Ratio
    # def get_eer_color(eer):
    #     if eer < 0.1:
    #         return "red", 'Poor'  # Poor
    #     elif 0.1 <= eer < 0.5:
    #         return "orange", 'Bad'  # Bad
    #     elif 0.5 <= eer < 1:
    #         return "yellow", 'Okay'  # Okayish
    #     elif 1 <= eer < 2:
    #         return "lightgreen", 'Good'  # Good
    #     else:
    #         return "lime", 'Excellent'  # Excellent
    
    # eer_color, quality = get_eer_color(round(eer, 2))
    
    # self.ui.Equity_Efficiency_Rate_Output.setText(f'<span style="color:{eer_color};">{eer:.2f}</span>')
    # self.ui.Strategy_Quality_Output.setText(f'<span style="color:{eer_color};">{quality}</span>')
    # self.ui.Max_Drawdown_Duration_Output.setText(f'{max_drawdown_duration} days')
    # self.ui.Total_Wins_Output.setText(f'{round(totalWins)}')
    # self.ui.Total_Losses_Output.setText(f'{round(totalLosses)}')
    # self.ui.Consecutive_Wins_Output.setText(f'{round(consecutiveWins)}')
    # self.ui.Consecutive_Losses_Output.setText(f'{round(consecutiveLosses)}')
    # self.ui.Largest_Win_Output.setText(f'${largestWin:,.0f}')
    # self.ui.Largest_Loss_Output.setText(f'-${abs(largestLoss):,.0f}')
    # self.ui.Avg_Win_Output.setText(f'${avgWin:,.0f}')
    # self.ui.Avg_Loss_Output.setText(f'-${abs(avgLoss):,.0f}')
    # self.ui.Avg_Trade_Output.setText(f'${avgTrade:,.0f}')
    # self.ui.Avg_Trade_Time_Output.setText(str(avgTradeTime).split('.')[0])
    # self.ui.Avg_Win_Time_Output.setText(str(avgWinTime).split('.')[0])
    # self.ui.Avg_Loss_Time_Output.setText(str(avgLossTime).split('.')[0])
    # self.ui.Max_Runup_Output.setText(f'{round(abs(maxRunup), 2)}%')
    # self.ui.Avg_Runup_Output.setText(f'{round(abs(avgRunup), 2)}%')
    # self.ui.Winrate_Output.setText(f'{round(winrate, 2)}%')
    # self.ui.RR_Output.setText(f'{rr}')
    # self.ui.Max_Open_Trades_Output.setText(str(maxOpenTrades))
    # self.ui.Avg_Open_Trades_Output.setText(str(avgOpenTrades))
        
    return strategy_metrics, monthly_returns
    
    # elif type == 'optimization':
        
    #     totalTrades = len(dfSignals)
    #     totalWins = len(dfSignals[dfSignals['Result'] == 1])
    #     winrate = (totalWins/totalTrades) * 100 if totalTrades > 0 else 0
        
    #     self.ui.tableWidget.setItem(row, 0, QTableWidgetItem('INCREMENT'))
    #     self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(str(threshold)))
    #     self.ui.tableWidget.setItem(row, 2, QTableWidgetItem('TRADES'))
    #     self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(str(totalTrades)))
    #     self.ui.tableWidget.setItem(row, 4, QTableWidgetItem('WIN RATE'))
    #     self.ui.tableWidget.setItem(row, 5, QTableWidgetItem(f'{round(winrate, 2)}%'))
    #     self.ui.tableWidget.setItem(row, 6, QTableWidgetItem('MAX DRAWDOWN'))
    #     self.ui.tableWidget.setItem(row, 7, QTableWidgetItem(f'{round(abs(maxDrawdown), 2)}%'))
    #     self.ui.tableWidget.setItem(row, 8, QTableWidgetItem('NET PROFIT'))
    #     value = equity[-1] - equity[0]
    #     formatted_value = f'${abs(value):,.0f}' if value >= 0 else f'-${abs(value):,.0f}'
    #     self.ui.tableWidget.setItem(row, 9, QTableWidgetItem(formatted_value))
        
    #     if winrate <= 15:
    #         above_thresholds.append(str(threshold))
            
    #     threshold_string = ', '.join(above_thresholds)
    #     self.ui.tableWidget.setItem(row, 10, QTableWidgetItem(threshold_string))
        
    #     return above_thresholds
    

def calculate_monthly_returns(ohlcv_df: pd.DataFrame, equity_array: np.ndarray) -> list[dict]:
    """
    Calculates the percentage and absolute return for each month.
    """
    if ohlcv_df.empty or len(equity_array) < 2:
        return []

    df = ohlcv_df.copy()
        
    df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') 
    
    if len(df) > len(equity_array):
        df = df.tail(len(equity_array)).reset_index(drop=True)
    
    df['equity'] = equity_array

    # These lines will now work because `df['timestamp']` is a datetime type
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    monthly_returns_data = []
    
    for (year, month), group in df.groupby(['year', 'month']):
        if group.empty:
            continue
        
        start_equity = group['equity'].iloc[0]
        end_equity = group['equity'].iloc[-1]
        
        if start_equity > 0:
            profit_dollars = end_equity - start_equity
            return_percent = (profit_dollars / start_equity) * 100
            month_name = datetime(year, month, 1).strftime('%b')

            monthly_returns_data.append({
                "Month": f"{month_name} {year}",
                "Profit ($)": profit_dollars,
                "Returns (%)": return_percent,
            })
            
    return monthly_returns_data
