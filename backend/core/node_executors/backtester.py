from typing import Any
import pandas as pd
from .base import BaseNodeExecutor, ExecutionContext, ExecutorInput, ExecutorOutput
from pipeline import run_single_backtest
from core.connect_to_brokerage import get_client
import json
import traceback

# Dummy objects for headless execution
class DummyManager:
    async def send_json_to_batch(self, *args, **kwargs): pass
class DummyQueue:
    def put_nowait(self, *args, **kwargs): pass
class DummyLoop:
    def call_soon_threadsafe(self, *args, **kwargs): pass

def assemble_strategy_code(node_data: dict) -> str:
    """
    Assembles a runnable Python strategy class string from the structured
    node data provided by the frontend.
    """
    config = node_data.get('config', {})
    code_blocks = node_data.get('codeBlocks', {})

    # Extract values with defaults
    initial_capital = config.get('initialCapital', 1000)
    risk_percent = config.get('riskPercent', 1.0)
    rr = config.get('rr', 2.0)
    trade_direction = config.get('tradeDirection', 'hedge') # long, short, hedge
    exit_type = config.get('exitType', 'tp_sl') # tp_sl, single_condition, time_based

    # Extract code snippets
    indicators_code = code_blocks.get('indicators', '[]')
    entry_logic_code = code_blocks.get('entryLogic', 'return 0')
    exit_logic_code = code_blocks.get('exitLogic', 'return False')

    # Indent user-provided code correctly
    indented_entry_logic = "\n".join([" " * 8 + line for line in entry_logic_code.split('\n')])
    indented_exit_logic = "\n".join([" " * 8 + line for line in exit_logic_code.split('\n')])

    # Determine which entry/exit methods to generate
    long_entry_enabled = trade_direction in ['long', 'hedge']
    short_entry_enabled = trade_direction in ['short', 'hedge']

    # Base entry_condition always exists
    entry_method = f"""
    def entry_condition(self, i):
        # Long Entry Logic
        if {long_entry_enabled} and self.long_entry(i):
            return 1
        # Short Entry Logic
        if {short_entry_enabled} and self.short_entry(i):
            return -1
        return 0

    def long_entry(self, i):
{indented_entry_logic}

    def short_entry(self, i):
        # To enable short entries, change 'tradeDirection' to 'short' or 'hedge'
        # and implement your short logic here.
        return False
"""
    # Conditionally generate exit_condition
    exit_method = ""
    if exit_type == 'single_condition':
        exit_method = f"""
    def exit_condition(self, i, j):
{indented_exit_logic}
"""
    
    # Assemble the final class using an f-string template
    strategy_template = f"""
from core.basestrategy import BaseStrategy
import numpy as np
import pandas as pd

class DynamicStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- Configuration from Node UI ---
        self.initial_capital = {initial_capital}
        self.risk_percent = {risk_percent}
        self.params = {{'rr': {rr}}}
        self.exit_type = "{'EXIT' if exit_type == 'single_condition' else 'TP and SL'}"
        
        # --- Indicator Definitions from Node UI ---
        self.indicators = {indicators_code}

    # --- Entry Logic from Node UI ---
{entry_method}

    # --- Exit Logic from Node UI ---
{exit_method}
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

        input_data = list(inputs.values())[0]
        df_input = input_data['data'].copy()
        
        # Assemble the full strategy code from the node's structured data
        strategy_code = assemble_strategy_code(node.data)
        
        try:
            (
                strategy_data, _, strategy_metrics, _, _, _
            ) = run_single_backtest(
                batch_id='pipeline_exec', manager=DummyManager(), job_id=node.id,
                file_name=node.data.get('label', 'Backtest'),
                strategy_code=strategy_code,
                initialCapital=node.data.get('config', {}).get('initialCapital', 1000),
                client=get_client('binance'), queue=DummyQueue(), loop=DummyLoop(),
                ohlcv_df=df_input
            )

            df_output = strategy_data.get('ohlcv', pd.DataFrame())
            context.node_metadata[node.id] = { "backtest_metrics": strategy_metrics }
            return {"default": df_output}

        except Exception as e:
            print(f"ERROR executing BacktesterNode {node.id}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to run backtest in node '{node.data.get('label')}': {e}")