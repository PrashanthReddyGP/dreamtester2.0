import traceback

try:
    import ast
    import sys
    import time
    import uuid
    import uvicorn
    import asyncio
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone
    from typing import List, Literal, Union, Annotated
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Literal, Union, Any
    from sqlalchemy.orm import Session
    import json
    from core.json_encoder import CustomJSONEncoder # Import our new encoder
    from types import SimpleNamespace

    # Import the new database functions
    from database import (
        BacktestJob,
        SessionLocal,
        save_api_key, 
        get_api_key,
        get_strategies_tree,
        create_strategy_item,
        update_strategy_item,
        delete_strategy_item,
        move_strategy_item,
        clear_all_strategies,
        create_multiple_strategy_items,
        create_backtest_job,
        get_backtest_job,
        clear_ohlcv_tables,
        get_all_labeling_templates,
        save_labeling_template,
        delete_labeling_template,
        get_all_fe_templates,
        save_fe_template,
        delete_fe_template
    )

    from pipeline import run_unified_test_manager, run_optimization_manager, run_asset_screening_manager, run_batch_manager, run_local_backtest_manager, run_hedge_optimization_manager
    from core.machinelearning import run_ml_pipeline_manager
    
    from core.indicator_registry import get_indicator_schema
    from core.connect_to_brokerage import get_client
    from core.data_manager import get_ohlcv
    from core.machinelearning import load_data_for_ml, transform_features_for_backend
    from core.robust_idk_processor import calculate_indicators
    from core.state import WORKFLOW_CACHE

    ##########################################

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    class BacktestSubmitConfig(BaseModel):
        strategyCode: str
        asset: str
        timeframe: str
        startDate: Optional[str] = None
        endDate: Optional[str] = None

    class StrategyFileModel(BaseModel):
        id: str
        name: str
        content: str

    class BatchSubmitPayload(BaseModel):
        strategies: List[StrategyFileModel]
        use_training_set: bool

    #########################################

    class ApiKeyBody(BaseModel):
        """Defines the expected JSON body for the save_keys endpoint."""
        apiKey: str
        apiSecret: str

    class StrategyItemCreate(BaseModel):
        id: str
        name: str
        type: str
        content: Optional[str] = None
        
        # This tells Pydantic:
        # - The Python attribute is `parent_id` (snake_case).
        # - When reading from JSON, look for a field named `parentId` (camelCase).
        parent_id: Optional[str] = Field(default=None, alias='parentId')

    class StrategyItemUpdate(BaseModel):
        name: Optional[str] = None
        content: Optional[str] = None

    class StrategyItemMove(BaseModel):
        newParentId: Optional[str] = None

    #######################
    class LocalBacktestConfig(BaseModel):
        strategy_code: str
        strategy_name: str
        csv_data: str

    ########################
    class OptimizationParam(BaseModel):
        indicatorIndex: int
        paramIndex: int
        name: str
        start: float
        end: float
        step: float
        
    class OptimizableParameterModel(BaseModel):
        id: str
        type: str
        name: str
        value: float
        enabled: bool
        mode: str
        start: Optional[float] = None
        end: Optional[float] = None
        step: Optional[float] = None
        list_values: Optional[str] = None
        indicatorIndex: Optional[int] = None 
        paramIndex: Optional[int] = None


    class OptimizationSubmitBody(BaseModel):
        strategy_code: str
        parameters_to_optimize: List[OptimizableParameterModel]

    class CombinationRuleModel(BaseModel):
        param1: str
        operator: str
        param2: str

    class SuperOptimizationConfig(BaseModel):
        strategy_code: str
        parameters_to_optimize: List[OptimizableParameterModel]
        symbols_to_screen: List[str]
        combination_rules: Optional[List[CombinationRuleModel]] = []

    class AssetScreeningBody(BaseModel):
        strategy_code: str
        symbols_to_screen: List[str]


    # --- Pydantic models for the final analysis settings ---
    class FinalAnalysisNoneModel(BaseModel):
        """Represents the case where no final analysis is selected."""
        type: Literal['none']

    class FinalDataSegmentationModel(BaseModel):
        type: Literal['data_segmentation']
        training_pct: int
        validation_pct: int

    class FinalWalkForwardModel(BaseModel):
        type: Literal['walk_forward']
        training_period_length: int
        training_period_unit: str
        testing_period_length: int
        testing_period_unit: str
        step_forward_size_pct: int
        # You might want to add is_anchored here too for walk-forward

    class SingleStrategyHedgeModel(BaseModel):
        strategy_code: str
        parameters_to_optimize: List[OptimizableParameterModel]
        combination_rules: List[CombinationRuleModel]

    class HedgeOptimizationConfigModel(BaseModel):
        test_type: Literal['hedge_optimization']
        strategy_a: SingleStrategyHedgeModel
        strategy_b: SingleStrategyHedgeModel
        symbols_to_screen: List[str]
        top_n_candidates: int
        portfolio_metric: str
        num_results_to_return: int
        final_analysis: Annotated[
            Union[
                FinalAnalysisNoneModel,
                FinalDataSegmentationModel,
                FinalWalkForwardModel
            ],
            Field(discriminator='type')
        ]


    # The base model with all common fields
    class BaseDurabilityConfig(BaseModel):
        strategy_code: str
        parameters_to_optimize: List[OptimizableParameterModel]
        symbols_to_screen: List[str]
        combination_rules: List[CombinationRuleModel]

    # The specific model for Data Segmentation tests
    class DataSegmentationConfig(BaseDurabilityConfig):
        # This is the "discriminator" field. It MUST be a Literal.
        test_type: Literal['data_segmentation']
        
        # Fields specific to this test type
        training_pct: int
        validation_pct: int
        testing_pct: int
        optimization_metric: str
        top_n_sets: int

    # You can add other test configurations here as you build them
    # class MonteCarloConfig(BaseDurabilityConfig):
    #     test_type: Literal['monte_carlo']
    #     num_runs: int
    #     # ... other MC specific fields

    # The Union type that combines all possible configurations.
    # Pydantic will use the `test_type` field to figure out which model to use.
    DurabilitySubmissionConfig = Annotated[
        Union[
            DataSegmentationConfig,
            # MonteCarloConfig, # Add other configs here later
        ],
        Field(discriminator="test_type"),
    ]


    class ProblemDefinitionConfig(BaseModel):
        type: Literal['template', 'custom']
        # Use Field(alias=...) to map from frontend's camelCase to Python's snake_case
        template_key: str = Field(alias='templateKey')
        custom_code: str = Field(alias='customCode')


    class ModelConfig(BaseModel):
        name: str
        hyperparameters: Dict[str, Any] = Field(
            default_factory=dict,
            description="A dictionary of hyperparameter names and their values."
        )

    class ValidationConfig(BaseModel):
        method: Literal['train_test_split', 'walk_forward']
        train_split: int = Field(alias='trainSplit')
        walk_forward_train_window: int = Field(alias='walkForwardTrainWindow')
        walk_forward_test_window: int = Field(alias='walkForwardTestWindow')

    class LabelingTemplateModel(BaseModel):
        key: str
        name: str
        description: str
        code: str
        
    class FETemplateModel(BaseModel):
        key: str
        name: str
        description: str
        code: str

    class DataSourceConfig(BaseModel):
        symbol: str
        timeframe: str
        startDate: str
        endDate: str

    class IndicatorConfig(BaseModel):
        id: str
        name: str
        params: Dict[str, Any]

    class PreprocessingConfig(BaseModel):
        scaler: str
        removeCorrelated: bool
        correlationThreshold: float
        usePCA: bool
        pcaComponents: int
        customFeatureCode: str
    
    class MLBacktestConfig(BaseModel):
        capital: float
        risk: float
        commissionBps: float
        slippageBps: float
        tradeOnClose: bool
    
    class FeaturesCalculationRequest(BaseModel):
        dataSource: DataSourceConfig
        features: List[IndicatorConfig] = Field(default_factory=list)

    class FeaturesEngineeringRequest(BaseModel):
        dataSource: DataSourceConfig
        features: List[IndicatorConfig] = Field(default_factory=list)
        preprocessing: PreprocessingConfig    
        
    class DataValidationRequest(BaseModel):
        dataSource: DataSourceConfig
        features: List[IndicatorConfig] = Field(default_factory=list)
        preprocessing: PreprocessingConfig
        validation: ValidationConfig
        problemDefinition: ProblemDefinitionConfig

    class MLPipelineConfig(BaseModel):
        """The main model that captures the entire configuration from the frontend."""
        problem_definition: ProblemDefinitionConfig = Field(alias='problemDefinition')
        data_source: DataSourceConfig = Field(alias='dataSource')
        features: List[IndicatorConfig]
        model: ModelConfig
        validation: ValidationConfig
        preprocessing: PreprocessingConfig
        backtestSettings: MLBacktestConfig
    
    # --- Helper function to generate the detailed info dictionary ---

    def format_bytes(size_bytes):
        """Converts bytes to a human-readable string."""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(np.floor(np.log(size_bytes) / np.log(1024))) 
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def generate_df_info(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Analyzes a DataFrame and returns a dictionary of metadata."""
        if df.empty:
            return { "message": "No data available to generate information." }

        # Work on a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        if 'timestamp' not in df_copy.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column to generate info.")

        # The incoming timestamp is UTC. Convert it to US/Pacific (PDT/PST).
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
        df_copy['timestamp'] = df_copy['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
        df_copy.set_index('timestamp', inplace=True)
        
        # General Info
        memory_usage_bytes = df_copy.memory_usage(deep=True).sum()
        
        # Data Quality
        missing_values = df_copy.isnull().sum()
        total_missing = int(missing_values.sum())
        missing_by_column = missing_values[missing_values > 0].to_dict()
        
        # Structure and Stats
        # Step 1: Build the data_types dictionary, including the index.
        index_name = df_copy.index.name if df_copy.index.name is not None else 'timestamp'
        data_types = {index_name: str(df_copy.index.dtype)}
        data_types.update({col: str(dtype) for col, dtype in df_copy.dtypes.items()})

        # Step 2: Derive the column count from the dictionary for accuracy.
        total_columns = len(data_types)
        
        # Structure and Stats (only for numeric columns)
        descriptive_stats = df_copy.describe(include=np.number).round(4).to_dict()
        
        info = {
            # General Info
            "Symbol": symbol,
            "Timeframe": timeframe,
            "Data Points": f"{df_copy.shape[0]} rows x {total_columns} columns",
            "Start Date": df_copy.index.min().strftime('%Y-%m-%d %H:%M:%S'),
            "End Date": df_copy.index.max().strftime('%Y-%m-%d %H:%M:%S'),
            "Memory Usage": format_bytes(memory_usage_bytes),
            
            # Data Quality Analysis
            "Total Missing Values": total_missing,
            "Missing Values by Column": missing_by_column,
            
            # Data Structure
            "Data Types": data_types,
            "Descriptive Statistics": descriptive_stats        }
        return info

    # You will need this helper function for the preprocessing step
    def apply_standard_preprocessing(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
        """
        Applies standard preprocessing steps like scaling after custom code has run.
        NOTE: This is a placeholder for your future logic.
        """
        print("Applying standard preprocessing steps (scaling, PCA, etc)...")
        
        # In the future, you would implement your scaler/PCA logic here.
        # For now, it just returns the dataframe.
        # Example:
        # if config.scaler != 'none':
        #     scaler = get_scaler(config.scaler)
        #     numeric_cols = df.select_dtypes(include=np.number).columns
        #     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        print("Standard preprocessing complete.")
        return df

    class ConnectionManager:
        def __init__(self):
            # A dictionary to hold active connections for each batch_id
            # Format: { "batch_id_1": [websocket1, websocket2], "batch_id_2": [websocket3] }
            self.active_connections: Dict[str, List[WebSocket]] = {}

        async def connect(self, websocket: WebSocket, batch_id: str):
            await websocket.accept()
            if batch_id not in self.active_connections:
                self.active_connections[batch_id] = []
            self.active_connections[batch_id].append(websocket)
            print(f"Client connected, now listening for updates on batch_id: {batch_id}")

        def disconnect(self, websocket: WebSocket, batch_id: str):
            if batch_id in self.active_connections:
                self.active_connections[batch_id].remove(websocket)
                if not self.active_connections[batch_id]: # Clean up if no listeners are left
                    del self.active_connections[batch_id]
            print(f"Client disconnected from batch_id: {batch_id}")

        async def send_json_to_batch(self, batch_id: str, message: dict):
            # 1. Manually serialize the message dictionary to a JSON string
            #    using our custom encoder.
            json_string = json.dumps(message, cls=CustomJSONEncoder)
            
            # 2. Send the resulting string as a text message.
            # The frontend's `JSON.parse()` will handle this perfectly.
            tasks = [connection.send_text(json_string) for connection in self.active_connections[batch_id]]
            
            await asyncio.gather(*tasks)


    websocket_queue = asyncio.Queue()

    # --- 6. CREATE A SINGLE, GLOBAL INSTANCE OF THE MANAGER ---
    manager = ConnectionManager()


                
    async def websocket_sender(queue: asyncio.Queue, manager: ConnectionManager):
        """
        This coroutine runs forever, waiting for messages to appear in the queue
        and sending them to the correct clients.
        """
        while True:
            batch_id, message = await queue.get()
            
            # Before attempting to send, check if there are any active connections
            # for this batch_id. This is a thread-safe way to prevent the race condition.
            if batch_id in manager.active_connections and manager.active_connections[batch_id]:
                try:
                    await manager.send_json_to_batch(batch_id, message)
                except Exception as e:
                    # Log any errors during the actual sending, but don't crash the sender.
                    print(f"ERROR sending message to batch {batch_id}: {e}")
            else:
                # This is not an error, it just means the client disconnected before
                # all messages could be sent. It's safe to just discard the message.
                print(f"INFO: No active connections for batch {batch_id}. Discarding message.")
            
            queue.task_done()





    # Create the FastAPI application instance
    app = FastAPI()

    origins = ["http://localhost:5173"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"], # Allows all methods (GET, POST, OPTIONS, etc.)
        allow_headers=["*"], # Allows all headers
    )

    # It creates a new database session for each request that needs one,
    # and ensures the session is closed afterward, even if an error occurs.
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


    symbol_cache = {}
    CACHE_DURATION_SECONDS = 3600  # Cache for 1 hour



    @app.on_event("startup")
    async def startup_event():
        # This dictionary will map batch_id -> asyncio.Event
        app.state.connection_events = {}
        # Make sure you also have your websocket_queue defined
        app.state.websocket_queue = asyncio.Queue()
        # Start your websocket sender task
        asyncio.create_task(websocket_sender(app.state.websocket_queue, manager))
        # Start the heartbeat task
        asyncio.create_task(heartbeat_sender(manager))

    async def heartbeat_sender(manager: ConnectionManager):
        """
        Sends a simple data message every 15 seconds to all connected clients
        to keep the connection alive and bypass network intermediaries.
        """
        while True:
            await asyncio.sleep(15) # Send a heartbeat every 15 seconds
            # Create a list of all clients across all batches
            all_clients = [client for clients in manager.active_connections.values() for client in clients]
            if all_clients:
                print("--- Sending application-level heartbeat to all clients ---")
                # Send a simple JSON message
                await asyncio.gather(*[client.send_json({"type": "heartbeat"}) for client in all_clients], return_exceptions=True)



    @app.get("/api/health")
    def health_check():
        return {"status": "ok", "message": "Python API is running!"}

    @app.get("/api/keys/{exchange_name}")
    def load_keys_endpoint(exchange_name: str):
        keys = get_api_key(exchange=exchange_name)
        if not keys:
            raise HTTPException(
                status_code=404, 
                detail=f"API keys not found for exchange: {exchange_name}"
            )
        return keys

    @app.post("/api/keys/{exchange_name}")
    def save_keys_endpoint(exchange_name: str, keys: ApiKeyBody):
        result = save_api_key(
            exchange=exchange_name, 
            key=keys.apiKey, 
            secret=keys.apiSecret
        )
        return result

    @app.get("/api/strategies")
    def get_strategies_endpoint():
        """Fetches the entire strategy file tree."""
        return get_strategies_tree()

    @app.post("/api/strategies")
    def create_strategy_endpoint(item: StrategyItemCreate):
        """Creates a new file or folder."""
        new_item = create_strategy_item(
            item_id=item.id,
            name=item.name,
            type=item.type,
            parent_id=item.parent_id,
            content=item.content
        )
        if not new_item:
            raise HTTPException(status_code=500, detail="Failed to create item.")
        return new_item

    @app.put("/api/strategies/{item_id}")
    def update_strategy_endpoint(item_id: str, item_update: StrategyItemUpdate):
        """Updates a file's name or content."""
        updated_item = update_strategy_item(
            item_id=item_id, 
            name=item_update.name, 
            content=item_update.content
        )
        if updated_item is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        return updated_item
        
    @app.delete("/api/strategies/{item_id}")
    def delete_strategy_endpoint(item_id: str):
        """Deletes a file or folder (and its children)."""
        result = delete_strategy_item(item_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        return result

    @app.put("/api/strategies/{item_id}/move")
    def move_strategy_endpoint(item_id: str, move_data: StrategyItemMove):
        """Moves an item to a new parent folder."""
        result = move_strategy_item(item_id=item_id, new_parent_id=move_data.newParentId)
        if result is None:
            raise HTTPException(status_code=404, detail="Item not found or invalid move.")
        return result

    @app.delete("/api/strategies")
    def clear_all_strategies_endpoint():
        """Deletes all strategy files and folders."""
        result = clear_all_strategies()
        if result is None:
            raise HTTPException(status_code=500, detail="An error occurred while clearing strategies.")
        return result

    @app.post("/api/strategies/bulk")
    def create_multiple_strategies_endpoint(items: list[StrategyItemCreate]):
        """Creates multiple strategy files from a list."""
        items_as_dicts = [item.model_dump() for item in items]
        
        result = create_multiple_strategy_items(items=items_as_dicts)
        if result is None:
            raise HTTPException(status_code=500, detail="An error occurred during bulk import.")
        return result
    
    # Using the DELETE HTTP method is semantically correct for a deletion action.
    @app.delete("/api/data/cache")
    async def clear_data_cache_endpoint():
        """
        Clears all cached OHLCV data from the database.
        """
        try:
            result = clear_ohlcv_tables()
            return result
        except Exception as e:
            print(f"ERROR clearing OHLCV cache: {e}")
            raise HTTPException(status_code=500, detail="An error occurred while clearing the data cache.")

    #########################################################################################

    @app.post("/api/strategies/parse-parameters")
    async def parse_strategy_parameters(request: Request):
        """
        Parses a strategy file for both `self.params` (like rr) and `self.indicators`.
        """
        code_bytes = await request.body()
        code_str = code_bytes.decode('utf-8')
        
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Python syntax error in strategy file: {e}")
        
        parsed_data = {
            "settings": {}, # Start with an empty settings dict
            "optimizable_params": []
        }

        # ast.walk traverses all nodes in the tree
        for node in ast.walk(tree):
            # Find the class definition that inherits from BaseStrategy
            if isinstance(node, ast.ClassDef):
                is_strategy_class = False
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseStrategy':
                        is_strategy_class = True
                        break
                
                if not is_strategy_class:
                    continue

                # Once inside the correct class, look for the __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        
                        for arg in item.args.defaults:
                            try:
                                val = ast.literal_eval(arg)
                                # This part is tricky, we need to match arg to name.
                                # For simplicity, we'll let the assignments below override this.
                            except (ValueError, SyntaxError):
                                pass
                        
                        # Now look for assignments within __init__
                        for sub_node in item.body:
                            
                            # We only care about `self.variable = ...` assignments
                            if isinstance(sub_node, ast.Assign) and isinstance(sub_node.targets[0], ast.Attribute):
                                target_attr = sub_node.targets[0].attr

                                # --- 1. Parse `self.params` dictionary ---
                                if target_attr == 'params' and isinstance(sub_node.value, ast.Dict):
                                    for i, key_node in enumerate(sub_node.value.keys):
                                        param_name = ast.literal_eval(key_node)
                                        default_value = ast.literal_eval(sub_node.value.values[i])
                                        
                                        parsed_data["optimizable_params"].append({
                                            "type": "strategy_param", # New type
                                            "name": param_name,
                                            "value": default_value,
                                            "id": f"sp_{param_name}"
                                        })
                                
                                # --- 2. Parse `self.indicators` list ---
                                elif target_attr == 'indicators' and isinstance(sub_node.value, ast.List):
                                    for i, element_tuple in enumerate(ast.literal_eval(sub_node.value)):
                                        indicator_name, _, params = element_tuple
                                        
                                        # For each numeric parameter in the indicator...
                                        for j, p_val in enumerate(params):
                                            parsed_data["optimizable_params"].append({
                                                "type": "indicator_param", # New type
                                                "name": f"{indicator_name} #{j+1}",
                                                "value": p_val,
                                                "id": f"ip_{i}_{j}",
                                                "indicatorIndex": i,
                                                "paramIndex": j
                                            })
                                            
                                else:
                                    try:
                                        value = ast.literal_eval(sub_node.value)
                                        if target_attr == 'symbol':
                                            parsed_data["settings"]["symbol"] = value
                                        elif target_attr == 'timeframe':
                                            parsed_data["settings"]["timeframe"] = value
                                        elif target_attr in ['start_date', 'startDate']:
                                            parsed_data["settings"]["startDate"] = value
                                        elif target_attr in ['end_date', 'endDate']:
                                            parsed_data["settings"]["endDate"] = value
                                    except (ValueError, SyntaxError):
                                        # Ignore assignments that aren't simple literals
                                        # e.g., self.x = some_function()
                                        pass

                break # Stop after finding __init__
        return parsed_data

    @app.post("/api/screen/submit")
    async def submit_asset_screening(body: AssetScreeningBody, background_tasks: BackgroundTasks, request: Request):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id)
        
        config_dict = body.model_dump()
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()
        
        # Call the new manager function
        background_tasks.add_task(
            run_asset_screening_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop
        )
        return {"message": "Asset screening job started.", "batch_id": batch_id}

    @app.get("/api/exchange/symbols/{exchange_name}", response_model=List[str])
    async def get_exchange_symbols(exchange_name: str, response: Response):
        """
        Fetches all available trading symbols (pairs) for a given exchange.
        Results are cached for 1 hour to reduce API calls to the exchange.
        """
        current_time = time.time()
        
        # Check if a valid cache exists
        if exchange_name in symbol_cache:
            cache_time, cached_symbols = symbol_cache[exchange_name]
            if current_time - cache_time < CACHE_DURATION_SECONDS:
                print(f"Serving symbols for '{exchange_name}' from cache.")
                return cached_symbols
                
        try:
            print(f"Fetching fresh symbols for '{exchange_name}' from the exchange.")
            client = get_client(exchange_name)
            
            # Fetch all symbols from Binance USDⓈ-M Futures
            futures_exchange_info = client.exchange_info()
            symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols'] if symbol['status'] == 'TRADING']
            
            # Sort symbols alphabetically
            sorted_symbols = sorted(symbols)
            
            return sorted_symbols
            
        except Exception as e:
            print(f"ERROR: Could not fetch symbols for {exchange_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch symbols for {exchange_name}.")

    @app.post("/api/backtest/local-submit")
    async def submit_local_backtest(body: LocalBacktestConfig, request: Request):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id)
        
        config_dict = body.model_dump()
        queue = request.app.state.websocket_queue
        loop = asyncio.get_running_loop()
        
        # We run this in an executor because CSV parsing can be slow
        loop.run_in_executor(
            None, # Use the default thread pool
            run_local_backtest_manager,
            batch_id, config_dict, manager, queue, loop
        )
        
        return {"message": "Local backtest job accepted.", "batch_id": batch_id}

    ###########################################################################################

    @app.post("/api/backtest/submit")
    def submit_backtest_endpoint(config: BacktestSubmitConfig):
        """
        Accepts a backtest job, adds it to a queue (conceptually), 
        and returns a job ID for status polling.
        """
        job_id = str(uuid.uuid4())
        
        # In a real implementation, you would add this job to a Celery/BackgroundTask queue here.
        # The worker would then pick it up and run the long process.
        print(f"Received backtest job {job_id} for asset {config.asset}")
        print(f"Strategy Code:\n{config.strategyCode[:100]}...") # Print first 100 chars
        
        # For now, we immediately return the job_id as if it were accepted.
        return {"job_id": job_id}

    @app.post("/api/backtest/batch-submit")
    async def submit_batch_backtest_endpoint(payload: BatchSubmitPayload, background_tasks: BackgroundTasks, request: Request):
        """
        Accepts a batch of strategies, creates a job record, queues a background task,
        and returns a unique batch_id for the client to connect to via WebSocket.
        """
        print(f"API: Received a batch of {len(payload.strategies)} strategies.")
        print(f"API: Use Training Set flag is: {payload.use_training_set}") # You can log the new value
        
        batch_id = str(uuid.uuid4())
        
        # Immediately create the job record in the database so the frontend can query its status
        create_backtest_job(batch_id)
        
        connection_event = asyncio.Event()
        request.app.state.connection_events[batch_id] = connection_event
        
        # Extract the list of files from the payload
        files_data = [file.model_dump() for file in payload.strategies]
        
        # Get the queue and pass it
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop() # Get the loop

        background_tasks.add_task(
            run_batch_manager, 
            batch_id=batch_id, 
            files_data=files_data,
            use_training_set=payload.use_training_set,
            manager=manager,
            queue=queue,
            loop=main_loop,
            connection_event=connection_event
        )
        
        return {"message": "Batch processing started.", "batch_id": batch_id}

    @app.get("/api/backtest/results/{batch_id}")
    def get_job_results_endpoint(batch_id: str):
        """
        Retrieves the status, logs, and final results for a specific backtest batch.
        This is the endpoint the frontend will poll.
        """
        job_data = get_backtest_job(batch_id)
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job with ID '{batch_id}' not found.")
        return job_data

    @app.post("/api/optimize/submit")
    async def submit_optimization(body: SuperOptimizationConfig, background_tasks: BackgroundTasks, request: Request):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id) # Reuse the same job table for tracking
        
        config_dict = body.model_dump()
        # Get the queue from the app state and pass it to the background task
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()
        connection_event = asyncio.Event()
        request.app.state.connection_events[batch_id] = connection_event

        background_tasks.add_task(
            run_unified_test_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop,
            connection_event=connection_event
        )
        return {"message": "Optimization job started.", "batch_id": batch_id}

    @app.post("/api/durability/submit")
    async def submit_durability_test(
        body: DurabilitySubmissionConfig,
        background_tasks: BackgroundTasks, 
        request: Request
    ):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id)
        
        # Create a new event for this batch_id
        connection_event = asyncio.Event()
        # Store it in our global dictionary
        request.app.state.connection_events[batch_id] = connection_event

        config_dict = body.model_dump()
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()

        background_tasks.add_task(
            run_unified_test_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop,
            # Pass the event object to the background task
            connection_event=connection_event 
        )
        return {"message": "Durability test job started.", "batch_id": batch_id}


    @app.post("/api/optimize/hedge")
    async def submit_hedge_optimization(
        body: HedgeOptimizationConfigModel,
        background_tasks: BackgroundTasks,
        request: Request
    ):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id)
        connection_event = asyncio.Event()
        request.app.state.connection_events[batch_id] = connection_event
        
        config_dict = body.model_dump()
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()

        background_tasks.add_task(
            run_hedge_optimization_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop,
            connection_event=connection_event
        )
        return {"message": "Hedge optimization job started.", "batch_id": batch_id}


    @app.get("/api/ml/indicators")
    def get_ml_indicators_schema():
        """
        Provides a schema of all available indicators and their parameters
        for the frontend to dynamically build its UI.
        """
        try:
            return get_indicator_schema()
        except Exception as e:
            print(f"Error generating indicator schema: {e}")
            raise HTTPException(status_code=500, detail="Could not generate indicator schema.")

    # --- Endpoint to Start a Workflow ---
    @app.get("/api/ml/workflow/start")
    def start_workflow():
        """Generates a unique ID for a new ML workflow session."""
        workflow_id = str(uuid.uuid4())
        WORKFLOW_CACHE[workflow_id] = {} # Initialize an empty dictionary for this workflow
        print(f"Started new ML workflow with ID: {workflow_id}")
        return {"workflow_id": workflow_id}
    
    @app.post("/api/ml/workflow/{workflow_id}/fetch-data")
    async def fetch_ohlcv_data_for_workflow(workflow_id: str, body: DataSourceConfig):
        """
        Acts as a controller that calls the data manager to get OHLCV data,
        then formats it for the frontend.
        """
        
        if workflow_id not in WORKFLOW_CACHE:
            raise HTTPException(status_code=404, detail="Workflow ID not found. Please start a new workflow.")
        
        print(f"Received request to fetch data for: {body.symbol} ({body.timeframe}) from {body.startDate} to {body.endDate}")
        
        try:
            # --- Step 1: Data Loading ---
            df = load_data_for_ml(
                symbol=body.symbol,
                timeframe=body.timeframe,
                start_date_str=body.startDate,
                end_date_str=body.endDate
            )
            
            # Handle the case where no data is returned
            if df.empty:
                return {"data": [], "info": {"message": "No data found for the selected criteria."}}
            
            # Store the raw DataFrame in the cache for the next step
            WORKFLOW_CACHE[workflow_id]['raw_df'] = df
            
            # # Sort the DataFrame by the 'timestamp' column in descending order
            # # The data manager already returns a datetime column, which sorts correctly.
            # df.sort_values(by='timestamp', ascending=False, inplace=True)
            
            # Generate the info dictionary from the DataFrame when it still has the correct dtypes.
            info = generate_df_info(df.copy(), body.symbol, body.timeframe)
            
            # Ensure correct numeric types after fetching
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # --- Format DataFrame for JSON response ---
            df['timestamp'] = (df['timestamp'].astype('int64') // 10**6)
            
            data_as_dicts = df.to_dict(orient='records')
            
            return {"data": data_as_dicts, "info": info}
        except Exception as e:
            print(f"ERROR in fetch_ohlcv_data endpoint: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred while processing the data request: {str(e)}"
            )
    
    @app.post("/api/ml/workflow/{workflow_id}/calculate-features")
    async def calculate_features_for_workflow(workflow_id: str, body: FeaturesCalculationRequest):
        """
        This endpoint fetches raw data, calculates features/indicators based on the provided
        configuration, and returns the resulting DataFrame along with its metadata.
        """
        
        if workflow_id not in WORKFLOW_CACHE or 'raw_df' not in WORKFLOW_CACHE[workflow_id]:
            raise HTTPException(status_code=400, detail="Raw data not found. Please fetch data first.")
        
        try:
            # STEP 1: Retrieve the DataFrame from the previous step (the cache)
            df_raw = WORKFLOW_CACHE[workflow_id]['raw_df']
            
            # --- Step 2: Feature Engineering ---
            # 2a. Transform the frontend feature config to the backend's expected format
            indicator_tuples = transform_features_for_backend(
                body.features, 
                body.dataSource.timeframe
            )
            
            # If no indicators are selected, just return the raw data
            if not indicator_tuples:
                print("No indicators specified. Returning raw data.")
                df_final = df_raw.copy()
            else:
                print(f"Calculating {len(indicator_tuples)} indicators...")
                # 2b. Create a mock 'strategy' object
                mock_strategy = SimpleNamespace(indicators=indicator_tuples)
                
                # 2c. Call the indicator calculation function
                df_final = calculate_indicators(mock_strategy, df_raw.copy())
                print(f"Features calculated. Final shape: {df_final.shape}")
            
            # STEP 3: Save the result of THIS step to the cache for the NEXT step
            WORKFLOW_CACHE[workflow_id]['features_df'] = df_final
            
            # --- Step 3: Generate Info BEFORE Sanitization ---
            # Generate the info dictionary from the DataFrame when it still has the correct dtypes.
            info = generate_df_info(df_final.copy(), body.dataSource.symbol, body.dataSource.timeframe)
            
            # --- Step 4: Sanitize a copy of the data for JSON response ---
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_for_json = df_final.astype(object).where(pd.notnull(df_final), None)
            
            # --- Step 5: Format the Sanitized Data for the Frontend ---
            # df_reset = df_for_json.reset_index()
            
            # if 'timestamp' in df_for_json.columns:
            #     df_for_json['timestamp'] = pd.to_datetime(df_for_json['timestamp']).astype('int64') // 10**6
            
            data_as_dicts = df_for_json.to_dict(orient='records')
            
            print(f"Total Columns: {df_for_json.columns}")
            
            return {"data": data_as_dicts, "info": info}
        
        except Exception as e:
            print(f"ERROR in calculate_features endpoint: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred during feature calculation: {str(e)}"
            )

    @app.post("/api/ml/workflow/{workflow_id}/feature-engineering")
    async def engineer_features_for_workflow(workflow_id: str, body: FeaturesEngineeringRequest):
        """
        This single, unified endpoint handles the entire feature generation process:
        1. Loads raw OHLCV data.
        2. Calculates technical indicators.
        3. Executes the user's custom Python code for feature engineering.
        4. Applies standard preprocessing (scaling, etc.).
        5. Returns the final, transformed DataFrame and its metadata.
        """
        
        if workflow_id not in WORKFLOW_CACHE or 'features_df' not in WORKFLOW_CACHE[workflow_id]:
            raise HTTPException(status_code=400, detail="Features data not found. Please calculate indicators first.")
        
        try:
            df_features = WORKFLOW_CACHE[workflow_id]['features_df']
            
            # --- Step 3: Execute Custom Feature Engineering Code ---
            df_transformed = df_features.copy()
            
            custom_code = body.preprocessing.customFeatureCode
            
            if custom_code and custom_code.strip():
                print("Executing custom feature engineering script...")
                local_namespace = {}
                # WARNING: exec is powerful. Run in a sandboxed/isolated environment in production.
                exec(custom_code, globals(), local_namespace)

                if 'transform_features' in local_namespace:
                    transform_func = local_namespace['transform_features']
                    
                    # FIX: Pass the correct DataFrame into the function and update it with the result.
                    df_transformed = transform_func(df_transformed.copy())
                    
                    if not isinstance(df_transformed, pd.DataFrame):
                        raise TypeError("Custom script's 'transform_features' function must return a pandas DataFrame.")
                    print(f"Custom script executed successfully. Shape after transform: {df_transformed.shape}")
                else:
                    raise NameError("Custom script must contain a function named 'transform_features'.")
            else:
                print("No custom feature code provided. Skipping custom script execution.")


            # --- Step 4: Apply Standard Preprocessing (Scaling, PCA, etc.) ---
            # The user's custom code runs BEFORE standard scaling.
            df_final = apply_standard_preprocessing(df_transformed, body.preprocessing)

            # --- Step 5: Generate Final Info and Format for Frontend ---
            print("Generating final metadata and formatting for response...")
            
            # STEP 3: Save the final engineered DataFrame to the cache for the final run
            WORKFLOW_CACHE[workflow_id]['engineered_df'] = df_final
            
            # FIX: Generate info from the FINAL DataFrame to get accurate stats.
            info = generate_df_info(df_final, body.dataSource.symbol, body.dataSource.timeframe)

            # Sanitize a copy of the final data for JSON response
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_for_json = df_final.astype(object).where(pd.notnull(df_final), None)
            
            # Format the data for the frontend (reset index, handle timestamp)
            # df_reset = df_for_json.reset_index()
            # if 'timestamp' in df_for_json.columns:
            #     # Ensure timestamp column exists before trying to convert
            #     df_for_json['timestamp'] = pd.to_datetime(df_for_json['timestamp']).astype('int64') // 10**6
            
            # Convert the FINAL processed data to dicts.
            data_as_dicts = df_for_json.to_dict(orient='records')
            
            print(f"Processing complete. Returning {len(data_as_dicts)} rows.")
            
            # 6. Prepare and return the response
            return {"data": data_as_dicts, "info": info}
        
        except Exception as e:
            print(f"ERROR in feature_engineering endpoint: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred during feature engineering: {str(e)}"
            )
    
    @app.post("/api/ml/workflow/{workflow_id}/data-validation")
    async def validate_data_for_workflow(workflow_id: str, body: DataValidationRequest):
        if workflow_id not in WORKFLOW_CACHE or 'engineered_df' not in WORKFLOW_CACHE[workflow_id]:
            raise HTTPException(status_code=400, detail="Engineered data not found. Please engineer the features first.")

        try:
            # STEP 1: Retrieve the engineered DataFrame from the cache
            df_engineered = WORKFLOW_CACHE[workflow_id]['engineered_df'].copy()
            print(f"Data validation started on DataFrame with shape: {df_engineered.shape}")

            # STEP 2: Generate Labels using the provided custom code
            user_code = body.problemDefinition.custom_code
            execution_scope = {}
            try:
                exec(user_code, execution_scope)
                labeling_func = execution_scope.get('generate_labels')
                if not callable(labeling_func):
                    raise ValueError("'generate_labels' function not found in custom code.")
                
                labels = labeling_func(df_engineered.copy())
                if not isinstance(labels, pd.Series):
                    raise TypeError("'generate_labels' function must return a Pandas Series.")
                
                df_engineered['label'] = labels
                # Drop rows where a label could not be generated (e.g., look-forward period)
                df_labeled = df_engineered.dropna(subset=['label'])
                print(f"Labels generated. Shape after dropping NaN labels: {df_labeled.shape}")

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error in labeling logic: {str(e)}")

            # STEP 3: Calculate Label Distribution (Support)
            label_counts = df_labeled['label'].value_counts()
            # Convert to a JSON-friendly format (string keys)
            label_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}
            print(f"Label Distribution: {label_distribution}")

            # STEP 4: Calculate Data Split Info
            split_info = {}
            total_samples = len(df_labeled)

            if body.validation.method == 'train_test_split':
                train_pct = body.validation.train_split / 100.0
                split_idx = int(total_samples * train_pct)
                train_count = split_idx
                test_count = total_samples - split_idx
                split_info = {
                    "method": "Train/Test Split",
                    "train_samples": train_count,
                    "test_samples": test_count,
                    "total_samples": total_samples
                }
            elif body.validation.method == 'walk_forward':
                train_window = body.validation.walk_forward_train_window
                test_window = body.validation.walk_forward_test_window
                # Calculate the number of folds possible with these settings
                if total_samples > train_window and test_window > 0:
                    num_folds = 1 + (total_samples - train_window) // test_window
                else:
                    num_folds = 0
                
                split_info = {
                    "method": "Walk-Forward",
                    "train_window_size": train_window,
                    "test_window_size": test_window,
                    "approximate_folds": num_folds,
                    "total_samples": total_samples
                }
            print(f"Split Info: {split_info}")

            # STEP 5: Assemble the final response
            # We don't need to re-cache anything, this is just for information
            
            # Get the base info (shape, columns, etc.)
            info = generate_df_info(df_labeled.copy(), body.dataSource.symbol, body.dataSource.timeframe)
            
            # Add our new validation section to the info object
            info['validation_info'] = {
                "label_distribution": label_distribution,
                "split_info": split_info
            }

            # Format the full, labeled DataFrame for the frontend to display
            df_json = df_labeled.copy()
            df_json.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_for_json = df_json.astype(object).where(pd.notnull(df_json), None)
            
            # if 'timestamp' in df_for_json.columns:
            #     df_for_json['timestamp'] = pd.to_datetime(df_for_json['timestamp']).astype('int64') // 10**6
            
            data_as_dicts = df_for_json.to_dict(orient='records')
            
            return {"data": data_as_dicts, "info": info}

        except Exception as e:
            print(f"ERROR in data_validation endpoint: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred during data validation: {str(e)}"
            )

    @app.post("/api/ml/workflow/{workflow_id}/run")
    async def run_ml_pipeline_for_workflow(
        workflow_id: str,
        body: MLPipelineConfig, # We still send the full config for logging and final settings
        background_tasks: BackgroundTasks,
        request: Request
    ):
        """
        Triggers the final ML pipeline run for a given workflow.
        It uses the cached, engineered DataFrame and the provided final configuration.
        """
        if workflow_id not in WORKFLOW_CACHE or 'engineered_df' not in WORKFLOW_CACHE[workflow_id]:
            raise HTTPException(status_code=400, detail="Engineered data not found in workflow cache. Please complete previous steps.")

        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id)

        config_dict = body.model_dump()
        
        # --- Add the workflow_id to the config for the manager ---
        config_dict['workflow_id'] = workflow_id
        
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()

        background_tasks.add_task(
            run_ml_pipeline_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop
        )

        return {"message": "Machine Learning pipeline job started.", "batch_id": batch_id}


    @app.post("/api/ml/run")
    async def submit_ml_pipeline(
        body: MLPipelineConfig,
        background_tasks: BackgroundTasks,
        request: Request
    ):
        """
        Accepts an ML pipeline job, starts it as a background task,
        and returns a batch_id for WebSocket connection.
        """
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id) # Log the job in the database

        # Pydantic v2's model_dump converts the model back to a dict.
        # by_alias=True ensures it uses the original camelCase keys if needed,
        # but our runner will expect snake_case, so we can omit it.
        config_dict = body.model_dump()
        
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop()

        # Add the long-running ML pipeline manager to the background tasks
        background_tasks.add_task(
            run_ml_pipeline_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop
        )

        return {"message": "Machine Learning pipeline job started.", "batch_id": batch_id}

    @app.get("/api/ml/templates")
    def get_labeling_templates_endpoint(db: Session = Depends(get_db)):
        """Fetches all custom labeling templates stored in the database."""
        try:
            return get_all_labeling_templates(db)
        except Exception as e:
            print(f"Error fetching templates: {e}")
            raise HTTPException(status_code=500, detail="Could not fetch labeling templates.")

    @app.post("/api/ml/templates")
    def save_labeling_template_endpoint(template: LabelingTemplateModel, db: Session = Depends(get_db)):
        """Saves a new or updates an existing labeling template."""
        try:
            saved = save_labeling_template(
                db=db,
                key=template.key,
                name=template.name,
                description=template.description,
                code=template.code
            )
            return {"status": "success", "template": saved}
        except Exception as e:
            print(f"Error saving template: {e}")
            raise HTTPException(status_code=500, detail="Could not save labeling template.")

    @app.delete("/api/ml/templates/{template_key}")
    def delete_labeling_template_endpoint(template_key: str, db: Session = Depends(get_db)):
        """Deletes a custom labeling template."""
        try:
            result = delete_labeling_template(db=db, key=template_key)
            if result is None:
                raise HTTPException(status_code=404, detail="Template not found.")
            return result
        except Exception as e:
            print(f"Error deleting template: {e}")
            raise HTTPException(status_code=500, detail="Could not delete labeling template.")


    @app.get("/api/ml/fe-templates")
    def get_fe_templates_endpoint(db: Session = Depends(get_db)):
        """Fetches all custom feature engineering templates stored in the database."""
        try:
            return get_all_fe_templates(db)
        except Exception as e:
            print(f"Error fetching FE templates: {e}")
            raise HTTPException(status_code=500, detail="Could not fetch feature engineering templates.")

    @app.post("/api/ml/fe-templates")
    def save_fe_template_endpoint(template: FETemplateModel, db: Session = Depends(get_db)):
        """Saves a new or updates an existing feature engineering template."""
        try:
            saved = save_fe_template(
                db=db,
                key=template.key,
                name=template.name,
                description=template.description,
                code=template.code
            )
            return {"status": "success", "template": saved}
        except Exception as e:
            print(f"Error saving FE template: {e}")
            raise HTTPException(status_code=500, detail="Could not save feature engineering template.")

    @app.delete("/api/ml/fe-templates/{template_key}")
    def delete_fe_template_endpoint(template_key: str, db: Session = Depends(get_db)):
        """Deletes a custom feature engineering template."""
        try:
            result = delete_fe_template(db=db, key=template_key)
            if result is None:
                raise HTTPException(status_code=404, detail="FE Template not found.")
            return result
        except Exception as e:
            print(f"Error deleting FE template: {e}")
            raise HTTPException(status_code=500, detail="Could not delete feature engineering template.")


    ####################################

    @app.websocket("/ws/results/{batch_id}")
    async def websocket_endpoint(websocket: WebSocket, batch_id: str):
        # Manually check the origin of the WebSocket request
        origin = websocket.headers.get('origin')
        
        if origin not in origins and origin is not None:
            # If the origin is not in our allowed list, close the connection
            print(f"--- WebSocket connection from untrusted origin '{origin}' rejected. ---")
            await websocket.close(code=1008) # 1008 = Policy Violation
            return
        
        # If the origin is valid, proceed as before
        await manager.connect(websocket, batch_id)
        
        # When a client connects, find the corresponding event and set it.
        connection_event = app.state.connection_events.get(batch_id)
        if connection_event:
            print(f"Client connected for batch {batch_id}. Setting connection event.")
            connection_event.set()

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket, batch_id)
            # Clean up the event from the dictionary to save memory
            if batch_id in app.state.connection_events:
                del app.state.connection_events[batch_id]
            print(f"Client disconnected from batch {batch_id}")

    if __name__ == "__main__":
        # Make sure to pass the 'app' object you created earlier
        uvicorn.run(app, host="127.0.0.1", port=8000, ws_per_message_deflate=False, ws_ping_interval=20, ws_ping_timeout=60, ws_max_size=16777216)
        
except Exception as e:
    # If any error occurs, print it and wait for user input
    print("An unhandled exception occurred!")
    print(traceback.format_exc())
    input("\nPress Enter to exit...")
