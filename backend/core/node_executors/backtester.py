from typing import Any, List
import pandas as pd
from .base import BaseNodeExecutor, ExecutionContext, ExecutorInput, ExecutorOutput
from pipeline import run_single_node_backtest, prepare_strategy_payload, convert_to_json_serializable
from core.connect_to_brokerage import get_client
import traceback

# Dummy objects for headless execution
class DummyManager:
    async def send_json_to_batch(self, *args, **kwargs): pass
class DummyQueue:
    def put_nowait(self, *args, **kwargs): pass
class DummyLoop:
    def call_soon_threadsafe(self, *args, **kwargs): pass


def assemble_strategy_code(node_data: dict, indicator_columns: List[str], symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
    """
    Assembles a runnable, Numba-jitted Python strategy class string by injecting
    user-defined logic snippets into pre-defined backtest loop templates.
    """
    config = node_data.get('config', {})
    code_blocks = node_data.get('codeBlocks', {})
    
    # --- 1. Extract Config and Code Snippets ---
    initial_capital = config.get('initialCapital', 1000)
    risk_percent = config.get('riskPercent', 1.0)
    rr = config.get('rr', 1.0)
    commission = config.get('commission', 0.1)
    exit_type = config.get('exitType', 'tp_sl')
    
    trade_direction = config.get('tradeDirection', 'long')
    
    # Logic Snippets with defaults. They should be single-line expressions.
    # The user's code will now return a signal: 1 for long, -1 for short, 0 for none.
    entry_logic = code_blocks.get('entryLogic', '0')
    sl_logic = code_blocks.get('stopLossLogic', '0.0')
    sizing_logic = code_blocks.get('positionSizingLogic', '0.0')
    exit_logic = code_blocks.get('customExitLogic', 'False')
    
    # --- 2. Generate Dynamic Code Parts from Indicator Columns ---
    init_assignments = "\n        ".join([f"self.{col} = self.df['{col}'].values" for col in indicator_columns])
    jit_params_str = ", ".join(indicator_columns)
    jit_call_args_str = ", ".join([f"self.{col}" for col in indicator_columns]) if indicator_columns else ""
    
    # --- 3. Define the Numba Loop Templates ---
    # Template for strategies using Take Profit & Stop Loss
    tp_sl_loop_template = f"""
    @staticmethod
    @jit(nopython=True)
    def optimizer_TP_SL(open, high, low, close, volume, timestamp, initial_capital, risk_percent, rr, commission, {jit_params_str}):
        n = len(close)
        
        capital = initial_capital
        no_fee_capital = initial_capital
        
        cash = (capital * risk_percent) / 100
        
        entry = np.zeros(n)
        takeprofit = np.zeros(n)
        stoploss = np.zeros(n)
        
        trade_closed = np.zeros(n)
        result = np.zeros(n)
        exit_time = np.zeros(n)
        open_trades = np.zeros(n)
        signals = np.zeros(n)
        
        # Pre-allocate NumPy array
        equity = np.zeros(n)
        equity[0] = initial_capital
        
        no_fee_equity = np.zeros(n)
        no_fee_equity[0] = initial_capital
        
        returns = np.zeros(n)
        commissioned_returns = np.zeros(n)
        
        max_drawdown = np.zeros(n)
        drawdown_duration = np.zeros(n)
        max_pull_up = np.zeros(n)
        pull_up_duration = np.zeros(n)
        avg_volatility = np.zeros(n)
        
        last_trade = 0
        
        for i in range(1, n):
            
            if {entry_logic}:
                signals[i] = {'1' if trade_direction == 'long' else '-1'} # Assuming short for this example; needs long/short logic
                
                # --- USER-DEFINED STOP LOSS LOGIC ---
                initial_sl = {sl_logic}
                
                if initial_sl == 0.0: 
                    print("Initial SL is 0")
                    continue # Skip trade if SL is not set
                
                entry_price = close[i]
                
                # Calculate stop distance and add a guard clause
                stop_distance = {'entry_price - initial_sl' if trade_direction == 'long' else 'initial_sl - entry_price'}
                
                # Use a small epsilon for float comparison to avoid precision issues
                if stop_distance < 1e-9: 
                    print("Stop Distance is Smaller")
                    continue # Skip this trade, the risk is zero/undefined
                
                initial_tp = {'entry_price + (stop_distance * rr)' if trade_direction == 'long' else 'entry_price - (stop_distance * rr)'}
                
                # --- USER-DEFINED POSITION SIZING LOGIC ---
                # We pass relevant variables to the user's logic snippet
                capital = equity[i-1]
                cash = (capital * risk_percent) / 100.0
                current_position_size = cash / stop_distance
                
                if current_position_size <= 0: 
                    print("Position Size is Smaller")
                    continue
                
                # --- Standard TP/SL Trade Management Loop ---
                # (This part is boilerplate and not edited by the user)
                entry[i], takeprofit[i], stoploss[i] = entry_price, initial_tp, initial_sl
                
                fee = {commission/100} * (entry_price * current_position_size)
                
                max_price = entry_price
                min_price = entry_price
                stop_loss_hit = False
                atr_values_during_trade = []
                
                for j in range(i + 1, n):
                    
                    max_price = max(max_price, high[j])
                    min_price = min(min_price, low[j])
                    atr_values_during_trade.append(atr[j])
                    
                    open_trades[j] += 1
                    
                    if {'high[j] >= initial_tp' if trade_direction == 'long' else 'low[j] <= initial_tp'}:
                        trade_closed[i] = 1
                        trade_return = {'(initial_tp - entry_price) * current_position_size' if trade_direction == 'long' else '(entry_price - initial_tp) * current_position_size'}
                        net_trade_return = trade_return - fee
                        capital += net_trade_return
                        no_fee_capital += trade_return
                        result[i] = 1
                        last_trade = 1
                        cash = (capital * risk_percent) / 100
                        exit_time[i] = timestamp[j]
                        returns[i] = round(trade_return, 2)
                        commissioned_returns[i] = round(net_trade_return, 2)
                        break
                    elif {'low[j] <= initial_sl' if trade_direction == 'long' else 'high[j] >= initial_sl'}:
                        stop_loss_hit = True
                        trade_closed[i] = 1
                        exit_time[i] = timestamp[j]
                        result[i] = -1
                        last_trade = -1
                        trade_return = {'(initial_sl - entry_price) * current_position_size' if trade_direction == 'long' else '(entry_price - initial_sl) * current_position_size'}
                        net_trade_return = trade_return - fee  # Fee for half position
                        capital += net_trade_return
                        cash = (capital * risk_percent) / 100
                        no_fee_capital += trade_return
                        returns[i] = round(trade_return, 2)
                        commissioned_returns[i] = round(net_trade_return, 2)
                        break
                
                if stop_loss_hit:
                    tp_distance = entry_price - initial_tp
                    
                    if abs(tp_distance) > 1e-9:
                        max_pull_up[i] = round(((entry_price - min_price) / tp_distance) * 100) if min_price < entry_price else 0
                    else:
                        max_pull_up[i] = 0
                    
                    pull_up_duration[i] = timestamp[j] - timestamp[i]
                
                else:
                    sl_distance = initial_sl - entry_price
                    
                    if abs(sl_distance) > 1e-9:
                        max_drawdown[i] = round(((max_price - entry_price) / sl_distance) * 100) if max_price > entry_price else 0
                    else:
                        max_drawdown[i] = 0
                    
                    drawdown_duration[i] = timestamp[j] - timestamp[i]
                
                if len(atr_values_during_trade) > 0:
                    if entry_price > 1e-9:
                        atr_array = np.array(atr_values_during_trade)
                        atr_avg = np.mean(atr_array)
                        avg_volatility[i] = round((atr_avg / entry_price) * 100, 2)
                    else:
                        avg_volatility[i] = 0 # Or np.nan, if you handle it later
            
            no_fee_equity[i] = no_fee_capital
            equity[i] = capital
        
        gross_pnl = no_fee_equity[-1] - no_fee_equity[0]
        
        if abs(gross_pnl) > 1e-9:
            realized_commission = round(
                (((no_fee_equity[-1] - no_fee_equity[0]) - (equity[-1] - equity[0])) * 100) / gross_pnl,
                2
            )
        else:
            realized_commission = 0.0
        
        return realized_commission, no_fee_equity, trade_closed, result, exit_time, open_trades, signals, equity, returns, commissioned_returns, entry, takeprofit, stoploss, max_drawdown, drawdown_duration, max_pull_up, pull_up_duration, avg_volatility
    """
    
    # Template for strategies using a Custom Exit Condition
    custom_exit_loop_template = f"""
    # (This would follow a similar pattern, injecting 'entry_logic' and 'custom_exit_logic' into a different loop structure)
    """
    
    # --- 4. Assemble the Final Class String ---
    chosen_loop = tp_sl_loop_template if exit_type == 'tp_sl' else custom_exit_loop_template
    
    strategy_template = f"""
from core.basestrategy_copy import BaseStrategy
from numba import jit
import numpy as np
import pandas as pd

class DynamicStrategy(BaseStrategy):
    def __init__(self, df=None, capital=1000, cash=10, risk_percent=1, optim_number=1,\
        symbol='ADAUSDT', timeframe='1d', start_date='2000-01-01', end_date='2100-12-31'):
        
        super().__init__(df, capital, cash, risk_percent, optim_number, symbol, timeframe, start_date, end_date)
        
        self.symbol = '{symbol}'
        self.timeframe = '{timeframe}'
        self.start_date = '{start_date}'
        self.end_date = '{end_date}'
        
        self.capital = {initial_capital}
        self.cash = cash
        self.equity = [{initial_capital}]
        self.portfolio_equity = [{initial_capital}]
        
        self.initial_capital = {initial_capital}
        self.risk_percent = {risk_percent}
        self.commission = {commission}
        self.params = {{'rr': {rr}}}
        self.rr = self.params['rr']
        self.exit_type = "{'EXIT' if exit_type == 'single_condition' else 'TP and SL'}"
        
        self.realized_commission = 0
        self.no_fee_equity = 0
    
    def optimized_run(self): # For TP/SL Exits
        super().optimized_run() # Call the parent class's optimized_run
        
        {init_assignments}
        
        if "{exit_type}" != "tp_sl":
            self.equity = np.full(len(self.df), self.initial_capital)
            return
        
        (self.realized_commission, no_fee, self.trade_closed, self.result, self.exit_time, self.open_trades, self.signals, self.equity, \\
            self.returns, self.commissioned_returns, self.entry, self.takeprofit, self.stoploss, \\
            self.max_drawdown, self.drawdown_duration, self.max_pull_up, self.pull_up_duration, self.avg_volatility) = self.optimizer_TP_SL(
            self.open, self.high, self.low, self.close, self.volume, self.timestamp, self.initial_capital, self.risk_percent, self.rr, self.commission,
            {jit_call_args_str}
        )
        
        self.no_fee_equity = no_fee[-1] - no_fee[0] if len(no_fee) > 1 else 0
    
    def optimized_exit(self): # Corresponds to optimizer_CUSTOM_EXIT
        super().optimized_exit() # Call the parent class's optimized_exit
        
        # Run the optimized backtest
        (self.trade_closed, self.result, self.exit_time, self.open_trades, self.signals, self.equity, \
            self.returns, self.commissioned_returns, self.entry, self.takeprofit, self.stoploss, \
                self.max_drawdown, self.drawdown_duration, self.max_pull_up, self.pull_up_duration, self.avg_volatility) = self.optimizer_OPEN(
            self.open, self.high, self.low, self.close, self.timestamp, self.initial_capital, self.risk_percent, self.rr,
            
            # ADD THE INDICATORS AS VALUES BELOW
            self.df['hl_oscillator'].values,
            self.df['hh'].values,
            self.df['hc'].values,
            self.df['sma_9'].values,
            self.df['sma_50'].values,
            self.df['sma_100'].values,
            self.df['stoploss_short'].values,
            self.df['atr'].values,
            self.optim_number
        )
    
    {chosen_loop}
"""
    return strategy_template


class BacktesterNodeExecutor(BaseNodeExecutor):
    """
    Executor for the 'backtester' node.
    It assembles a strategy from structured data and runs it.
    """
    def execute(self, node: Any, inputs: ExecutorInput, context: ExecutionContext) -> ExecutorOutput:
        print(f"Executing Backtester Node: {node.data.get('label', node.id)}")
        
        if not inputs:
            raise ValueError("BacktesterNode requires an upstream data source.")
        
        # Get symbol and timeframe from the shared context
        symbol = context.pipeline_params.get('symbol', 'UNKNOWN_SYMBOL')
        timeframe = context.pipeline_params.get('timeframe', 'UNKNOWN_TIMEFRAME')
        start_date = context.pipeline_params.get('start_date', 'UNKNOWN_START_DATE')
        end_date = context.pipeline_params.get('end_date', 'UNKNOWN_END_DATE')
        
        df_input = list(inputs.values())[0]['data'].copy()
        
        standard_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        indicator_columns = [col for col in df_input.columns if col not in standard_cols]
        print(f"Identified Indicator Columns: {indicator_columns}")
        
        # Pass the identified columns to the assembler
        strategy_code = assemble_strategy_code(node.data, indicator_columns, symbol, timeframe, start_date, end_date)
        
        try:
            (
                strategy_data, _, strategy_metrics, monthly_returns, _, _
            ) = run_single_node_backtest(
                batch_id='pipeline_exec', manager=DummyManager(), job_id=node.id,
                file_name=node.data.get('label', 'Backtest'),
                strategy_code=strategy_code,
                initialCapital=node.data.get('config', {}).get('initialCapital', 1000),
                client=get_client('binance'), queue=DummyQueue(), loop=DummyLoop(),
                df_input=df_input
            )
            
            if monthly_returns == []:
                print("##### ZERO TRADES FOUND #####")
            
            backtest_payload = prepare_strategy_payload(strategy_data, strategy_metrics, monthly_returns)
            
            df_output = strategy_data.get('ohlcv', pd.DataFrame())
            df_signals = strategy_data.get('signals', pd.DataFrame())
            
            if not df_signals.empty and 'timestamp' in df_signals.columns:
                # Ensure the timestamp column is in datetime format
                df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
                
                # Assume UTC if timezone is naive, then convert to Pacific Time (PDT/PST)
                if df_signals['timestamp'].dt.tz is None:
                    df_signals['timestamp'] = df_signals['timestamp'].dt.tz_localize('UTC')
                
                df_signals['timestamp'] = df_signals['timestamp'].dt.tz_convert('US/Pacific')
                
                # Remove timezone information to make the datetime naive
                df_signals['timestamp'] = df_signals['timestamp'].dt.tz_localize(None)
            
            # pd.set_option('display.max_columns', None)
            # print(df_signals.tail(5))
            
            final_payload = { 
                "backtest_info": {
                    "Total Trades": strategy_metrics.get("Total_Trades"), 
                    "Annual Return": strategy_metrics.get("Annual_Return"), 
                    "Max Drawdown": strategy_metrics.get("Max_Drawdown"), 
                    "Sharpe Ratio": strategy_metrics.get("Sharpe_Ratio"), 
                    "Profit Factor": strategy_metrics.get("Profit_Factor"), 
                    "Winrate": strategy_metrics.get("Winrate"), 
                    "RR": strategy_metrics.get("RR"), 
                },
                "backtest_result": backtest_payload # Use the original, un-sanitized payload here
            }
            
            context.node_metadata[node.id] = convert_to_json_serializable(final_payload)
            
            return {"default": df_output}
        
        except Exception as e:
            print(f"ERROR executing BacktesterNode {node.id}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to run backtest in node '{node.data.get('label')}': {e}")