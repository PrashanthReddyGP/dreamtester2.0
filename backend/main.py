import traceback

try:
    import os
    import shutil
    import ast
    from ast import unparse, fix_missing_locations
    import sys
    import inspect
    from pathlib import Path
    import time
    import uuid
    import uvicorn
    import asyncio
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone
    from typing import List, Literal, Union, Annotated
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response, Depends, APIRouter, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Literal, Union, Any
    from sqlalchemy.orm import Session
    import json
    from core.json_encoder import CustomJSONEncoder # Import our new encoder
    from types import SimpleNamespace
    from collections import deque
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import classification_report, confusion_matrix
    from starlette.concurrency import run_in_threadpool # Ensure this is imported
            

    from core.ml_models import get_model, UNSUPERVISED_MODELS # Import your model factory
    
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
        delete_fe_template,
        get_all_backtest_templates,
        save_backtest_template,
        delete_backtest_template
    )

    from pipeline import run_unified_test_manager, get_charting_data, run_asset_screening_manager, run_batch_manager, run_local_backtest_manager, download_dataframe #, run_hedge_optimization_manager
    from hedge_pipeline import run_hedge_optimization_manager
    from core.machinelearning import run_ml_pipeline_manager
    
    from core.indicator_registry import get_indicator_schema
    from core.connect_to_brokerage import get_client
    from core.data_manager import get_ohlcv, import_csv_data
    from core.machinelearning import load_data_for_ml, transform_features_for_backend
    from core.robust_idk_processor import calculate_indicators
    from core.state import WORKFLOW_CACHE
    from core.node_executors import get_executor
    from core.node_executors.base import ExecutionContext
    
    from core.types import IndicatorConfig
    
    from core.cache_utils import (
        setup_cache_directory,
        generate_node_hash,
        save_to_cache,
        load_from_cache,
        clear_cache,
    )
    
    ##########################################

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # This defines the expected JSON structure for the frontend.
    class StrategySettingsResponse(BaseModel):
        symbol: Optional[str] = None
        timeframe: Optional[str] = None
        startDate: Optional[str] = None
        endDate: Optional[str] = None

    class DataSourceConfig(BaseModel):
        symbol: str
        timeframe: str
        startDate: str
        endDate: str
    
    class ChartingRequest(BaseModel):
        code: str
        config: DataSourceConfig

    class IndicatorDataPoint(BaseModel):
        time: int  # Millisecond timestamp
        value: float

    class ChartDataPayload(BaseModel):
        ohlcv: List[List[Union[int, float]]] # [timestamp_ms, o, h, l, c, v]
        indicators: Dict[str, List[IndicatorDataPoint]]
        strategy_name: str

    class UpdateCodeRequest(BaseModel):
        code: str
        # Use Field alias to map from the frontend's camelCase
        new_settings: DataSourceConfig = Field(..., alias='newSettings')

    class UpdateCodeResponse(BaseModel):
        updated_code: str = Field(..., alias='updatedCode')

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
        portfolio_code: Optional[str] = None

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
    
    # The specific model for Walk-Forward tests
    class WalkForwardConfig(BaseDurabilityConfig):
        # This is the "discriminator" field.
        test_type: Literal['walk_forward']

        # Fields specific to this test type (matching the frontend)
        training_period_length: int
        training_period_unit: str  # or Literal['days', 'weeks', 'months'] for stricter validation
        testing_period_length: int
        testing_period_unit: str   # or Literal['days', 'weeks', 'months']
        is_anchored: bool
        step_forward_pct: int
        optimization_metric: str

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
            WalkForwardConfig,
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
    
    class BacktestTemplateModel(BaseModel):
        key: str
        name: str
        description: str
        code: str
    
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

    class NodeModel(BaseModel):
        id: str
        type: str
        data: Dict[str, Any]

    class EdgeModel(BaseModel):
        id: str
        source: str
        target: str
        source_handle: Optional[str] = Field(None, alias='sourceHandle')
        target_handle: Optional[str] = Field(None, alias='targetHandle')

    class PipelineExecutionRequest(BaseModel):
        nodes: List[NodeModel]
        edges: List[EdgeModel]
        target_node_id: str = Field(alias='targetNodeId')


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
        expose_headers=["Content-Disposition"]
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

    @app.post("/api/data/import-csv", tags=["Data Management"])
    async def import_ohlcv_from_csv(
        symbol: str = Form(...),
        timeframe: str = Form(...),
        source: str = Form(...),
        file: UploadFile = File(...)
    ):
        """
        Endpoint to upload a CSV file and import its OHLCV data into the database.
        Receives symbol, timeframe, source, and the file as multipart/form-data.
        """
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .csv files are accepted.")
        
        # Define a temporary directory for file uploads
        # Ensure this is defined at the top level or passed correctly
        TEMP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'ohlcv_data')
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        # Create a unique temporary path to save the file
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        
        try:
            # Save the uploaded file to the temporary location
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"File '{file.filename}' temporarily saved to '{temp_file_path}'")
            
            # Now, call your existing synchronous data manager function.
            # Running it in a threadpool prevents blocking the FastAPI event loop.
            await run_in_threadpool(
                import_csv_data,
                csv_path=temp_file_path,
                symbol=symbol,
                timeframe=timeframe,
                source=source
            )
            
            return {"message": f"Successfully imported data for {symbol} ({timeframe})."}
        
        except Exception as e:
            # Catch potential errors from the import process or file handling
            print(f"Error during CSV import: {e}")
            traceback.print_exc() # Print full traceback for better debugging
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during the import process: {str(e)}"
            )
        finally:
            # CRITICAL: Clean up the temporary file whether the import succeeded or failed
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Temporary file '{temp_file_path}' deleted.")


    #########################################################################################

    @app.post(
        "/api/strategies/parse-settings-from-code",
        response_model=StrategySettingsResponse,
        tags=["Strategy Config"]
    )
    async def parse_settings_from_code(request: Request):
        """
        Safely parses a string of Python strategy code to extract key settings
        like symbol, timeframe, start date, and end date.

        This endpoint reads the raw request body as text/plain.
        """
        try:
            # 1. Read the raw Python code from the request body
            code_bytes = await request.body()
            code_str = code_bytes.decode('utf-8')

            if not code_str.strip():
                return StrategySettingsResponse() # Return empty if no code

            # 2. Parse the code into an Abstract Syntax Tree (AST)
            tree = ast.parse(code_str)

        except SyntaxError as e:
            # If the code is not valid Python, return a 400 error
            raise HTTPException(status_code=400, detail=f"Python syntax error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred during parsing: {e}")

        # 3. Walk the tree to find the strategy settings
        settings = {}
        
        # We define all possible attribute names we want to look for
        # and map them to the camelCase key the frontend expects.
        TARGET_ATTRIBUTES = {
            'symbol': 'symbol',
            'timeframe': 'timeframe',
            'start_date': 'startDate',
            'startDate': 'startDate', # Accept camelCase from code too
            'end_date': 'endDate',
            'endDate': 'endDate',     # Accept camelCase from code too
        }

        # ast.walk traverses all nodes in the tree recursively
        for node in ast.walk(tree):
            # We are looking for assignment statements, e.g., `self.symbol = "BTC/USDT"`
            if isinstance(node, ast.Assign):
                # The target of the assignment must be an attribute, e.g., `self.symbol`
                # We check the first target, node.targets[0]
                if isinstance(node.targets[0], ast.Attribute):
                    target = node.targets[0]
                    
                    # The attribute should be on an object named 'self'
                    if isinstance(target.value, ast.Name) and target.value.id == 'self':
                        
                        # Check if the attribute name is one we care about
                        if target.attr in TARGET_ATTRIBUTES:
                            try:
                                # Safely evaluate the value (e.g., "BTC/USDT")
                                # ast.literal_eval is SAFE. It only handles simple Python
                                # literals (strings, numbers, lists, etc.) and will
                                # raise an error for anything else, preventing code execution.
                                value = ast.literal_eval(node.value)
                                
                                # Get the correct camelCase key for our response dict
                                response_key = TARGET_ATTRIBUTES[target.attr]
                                settings[response_key] = value

                            except (ValueError, SyntaxError):
                                # Could not evaluate the value, just ignore it
                                pass

        return settings

    @app.post(
        "/api/strategies/update-code-from-settings",
        response_model=UpdateCodeResponse,
        tags=["Strategy Config"]
    )
    async def update_code_from_settings(body: UpdateCodeRequest):
        """
        Parses Python strategy code, updates the settings assignments (symbol,
        timeframe, etc.) within the __init__ method, and returns the modified code.
        """
        try:
            tree = ast.parse(body.code)
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Python syntax in provided code: {e}")

        # Map the desired UI settings to the Python attribute names
        settings_to_update = {
            'symbol': body.new_settings.symbol,
            'timeframe': body.new_settings.timeframe,
            'startDate': body.new_settings.startDate,
            'endDate': body.new_settings.endDate
        }
        
        # We use snake_case for Python attributes in the code
        PYTHON_ATTR_MAP = {
            'symbol': 'symbol',
            'timeframe': 'timeframe',
            'startDate': 'start_date',
            'endDate': 'end_date'
        }

        class CodeTransformer(ast.NodeTransformer):
            def __init__(self):
                self.updated_attrs = set()

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                # Check if this class inherits from BaseStrategy (or a similar name)
                is_strategy_class = any(
                    isinstance(base, ast.Name) and 'Strategy' in base.id
                    for base in node.bases
                )
                
                if not is_strategy_class:
                    return node # Don't modify non-strategy classes

                # Find the __init__ method and modify it
                for i, item in enumerate(node.body):
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        node.body[i] = self.update_init_method(item)
                        break # Assume only one __init__

                return node

            def update_init_method(self, init_node: ast.FunctionDef) -> ast.FunctionDef:
                # First, iterate through existing assignments and update them
                for stmt in init_node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Attribute):
                        target_attr = stmt.targets[0]
                        if isinstance(target_attr.value, ast.Name) and target_attr.value.id == 'self':
                            # Find which setting this line corresponds to
                            frontend_key = next((k for k, v in PYTHON_ATTR_MAP.items() if v == target_attr.attr), None)
                            
                            if frontend_key and frontend_key in settings_to_update:
                                # Update the value of this assignment
                                new_value = settings_to_update[frontend_key]
                                stmt.value = ast.Constant(value=new_value)
                                self.updated_attrs.add(frontend_key)
                
                # Now, add any settings that weren't found and updated
                attrs_to_add = set(settings_to_update.keys()) - self.updated_attrs
                
                new_statements = []
                for key in attrs_to_add:
                    python_attr = PYTHON_ATTR_MAP[key]
                    value = settings_to_update[key]
                    
                    # Create a new assignment statement: `self.symbol = "VALUE"`
                    new_stmt = ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id='self', ctx=ast.Load()),
                                attr=python_attr,
                                ctx=ast.Store()
                            )
                        ],
                        value=ast.Constant(value=value)
                    )
                    new_statements.append(new_stmt)

                # Insert new statements at the top of the __init__ method for consistency
                # (after the super().__init__() call, if it exists)
                insert_pos = 1 if (len(init_node.body) > 0 and 'super' in ast.dump(init_node.body[0])) else 0
                init_node.body[insert_pos:insert_pos] = new_statements

                return init_node

        transformer = CodeTransformer()
        modified_tree = transformer.visit(tree)
        
        fix_missing_locations(modified_tree)
        
        # Convert the modified AST back into Python code
        updated_code = unparse(modified_tree)

        return {"updatedCode": updated_code}


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
            if isinstance(node, ast.ClassDef):
                is_strategy_class = any(
                    isinstance(base, ast.Name) and base.id == 'BaseStrategy'
                    for base in node.bases
                )
                
                if not is_strategy_class:
                    continue

                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        
                        indicator_counts = {}

                        for sub_node in item.body:
                            if isinstance(sub_node, ast.Assign) and isinstance(sub_node.targets[0], ast.Attribute):
                                target_attr = sub_node.targets[0].attr

                                if target_attr == 'params' and isinstance(sub_node.value, ast.Dict):
                                    for i, key_node in enumerate(sub_node.value.keys):
                                        param_name = ast.literal_eval(key_node)
                                        default_value = ast.literal_eval(sub_node.value.values[i])
                                        
                                        parsed_data["optimizable_params"].append({
                                            "type": "strategy_param",
                                            "name": param_name,
                                            "value": default_value,
                                            "id": f"sp_{param_name}"
                                        })
                                
                                elif target_attr == 'indicators' and isinstance(sub_node.value, ast.List):
                                    for i, element_tuple in enumerate(ast.literal_eval(sub_node.value)):
                                        indicator_name, _, params = element_tuple
                                        
                                        # Increment the count for this specific indicator name
                                        indicator_counts[indicator_name] = indicator_counts.get(indicator_name, 0) + 1
                                        instance_num = indicator_counts[indicator_name]

                                        for j, p_val in enumerate(params):
                                            
                                            # Example: "SMA 1 #1", "SMA 2 #1", "ATR 1 #1", "ATR 1 #2"
                                            unique_param_name = f"{indicator_name} {instance_num} #{j+1}"

                                            parsed_data["optimizable_params"].append({
                                                "type": "indicator_param",
                                                "name": unique_param_name, # Use the new unique name
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
                                        pass
                        break
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
            
            # Manually add some popular symbols to ensure they are present,
            # in case the API filter misses them.
            manual_symbols_to_add = ["EURUSD", "MSFT", "NVDA", "XAUUSD"]
            
            # Use a set for efficient checking and adding to avoid duplicates
            symbols_set = set(symbols)
            symbols_set.update(manual_symbols_to_add)
            symbols = list(symbols_set)
            
            # Update the cache with the fresh, combined list
            sorted_symbols = sorted(symbols)
            symbol_cache[exchange_name] = (current_time, sorted_symbols)
            
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


    @app.post("/api/charting/prepare-data", response_model=ChartDataPayload, tags=["Charting"])
    async def prepare_charting_data(body: ChartingRequest):
        """
        Fetches OHLCV data and calculates indicators based on a strategy's code
        to prepare a payload for the advanced charting view.
        """
        try:
            
            ohlcv_payload, indicator_payload, strategy_name = await run_in_threadpool(
                get_charting_data,
                code=body.code,
                symbol=body.config.symbol,
                timeframe=body.config.timeframe
            )            
            
            # 4c. Assemble the final payload
            final_payload = ChartDataPayload(
                ohlcv=ohlcv_payload,
                indicators=indicator_payload,
                strategy_name=strategy_name
            )
            
            return final_payload
        
        except Exception as e:
            print(f"ERROR preparing chart data: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while preparing chart data: {str(e)}"
            )
            
    @app.post("/api/download-csv")
    async def download_data_csv(body: ChartingRequest):
        """
        Fetches data and prepares a CSV for download based on the request body.
        """
        try:
            # Use the specific symbol and timeframe from the request for the filename
            filename = f"{body.config.symbol.replace('/', '-')}_{body.config.timeframe}.csv"
            
            csv_data = await run_in_threadpool(
                download_dataframe,
                code=body.code,
                symbol=body.config.symbol,
                timeframe=body.config.timeframe
            )
            
            return Response(
                            content=csv_data,
                            media_type="text/csv",  # Set the MIME type to text/csv
                            headers={ "Content-Disposition": f"attachment; filename={filename}" }
                            )
        
        except Exception as e:
            print(f"ERROR preparing data: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while preparing data: {str(e)}"
            )

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
            portfolio_code=payload.portfolio_code,
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


#######################
    # Create a directory for saved pipelines if it doesn't exist
    SAVED_PIPELINES_DIR = Path("pipelines/saved")
    SAVED_PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    
    @app.post("/api/pipelines/save")
    async def save_pipeline_workflow(request: Request):
        """
        Saves the current state of a pipeline workflow (nodes and edges).
        """
        try:
            data = await request.json()
            name = data.get("name")
            workflow_data = data.get("workflow")

            if not name or not workflow_data:
                return JSONResponse(status_code=400, content={"detail": "Missing workflow name or data."})

            # Sanitize filename
            safe_filename = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            filepath = SAVED_PIPELINES_DIR / f"{safe_filename}.json"

            with open(filepath, "w") as f:
                json.dump(workflow_data, f, indent=4)

            return {"message": f"Workflow '{name}' saved successfully."}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.get("/api/pipelines/list")
    async def list_pipeline_workflows():
        """
        Returns a list of all saved pipeline workflow names.
        """
        try:
            files = [f.stem for f in SAVED_PIPELINES_DIR.glob("*.json")]
            return {"workflows": files}
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.get("/api/pipelines/load/{workflow_name}")
    async def load_pipeline_workflow(workflow_name: str):
        """
        Loads a specific pipeline workflow by its name.
        """
        try:
            # Sanitize filename to match how it was saved
            safe_filename = "".join(c for c in workflow_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            filepath = SAVED_PIPELINES_DIR / f"{safe_filename}.json"

            if not filepath.exists():
                return JSONResponse(status_code=404, content={"detail": "Workflow not found."})

            with open(filepath, "r") as f:
                workflow_data = json.load(f)

            return workflow_data
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

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
        
    # --- NEW: BACKTEST STRATEGY TEMPLATE ENDPOINTS ---

    @app.get("/api/ml/backtest-templates")
    def get_backtest_templates_endpoint(db: Session = Depends(get_db)):
        """Fetches all custom backtest strategy templates stored in the database."""
        try:
            # This assumes you've created a get_all_backtest_templates function
            # in your database.py, similar to the one for FE templates.
            return get_all_backtest_templates(db)
        except Exception as e:
            print(f"Error fetching backtest templates: {e}")
            raise HTTPException(status_code=500, detail="Could not fetch backtest templates.")

    @app.post("/api/ml/backtest-templates")
    def save_backtest_template_endpoint(template: BacktestTemplateModel, db: Session = Depends(get_db)):
        """Saves a new or updates an existing backtest strategy template."""
        try:
            saved = save_backtest_template(
                db=db,
                key=template.key,
                name=template.name,
                description=template.description,
                code=template.code
            )
            return {"status": "success", "template": saved}
        except Exception as e:
            print(f"Error saving backtest template: {e}")
            raise HTTPException(status_code=500, detail="Could not save backtest template.")

    @app.delete("/api/ml/backtest-templates/{template_key}")
    def delete_backtest_template_endpoint(template_key: str, db: Session = Depends(get_db)):
        """Deletes a custom backtest strategy template."""
        try:
            result = delete_backtest_template(db=db, key=template_key)
            if result is None:
                raise HTTPException(status_code=404, detail="Backtest template not found.")
            return result
        except Exception as e:
            print(f"Error deleting backtest template: {e}")
            raise HTTPException(status_code=500, detail="Could not delete backtest template.")

    # An endpoint to clear the cache
    @app.post("/api/ml/workflow/clear_cache")
    async def clear_pipeline_cache():
        """Clears all cached results on the backend."""
        clear_cache()
        return {"message": "Backend cache cleared successfully."}

    @app.post("/api/ml/workflow/{workflow_id}/execute")
    async def execute_pipeline_up_to_node(workflow_id: str, body: PipelineExecutionRequest):
        """
        Executes the pipeline defined by the graph up to a specific target node.
        """
        if workflow_id not in WORKFLOW_CACHE:
            raise HTTPException(status_code=404, detail="Workflow ID not found.")
        
        nodes = {node.id: node for node in body.nodes}
        edges = body.edges
        target_node_id = body.target_node_id
        
        # --- 1. Determine Execution Order (Topological Sort) ---
        def calculate_execution_order(nodes, edges, target_node_id):
            
            # Build adjacency list and in-degree count for the graph
            adj = {node_id: [] for node_id in nodes}
            in_degree = {node_id: 0 for node_id in nodes}
            for edge in edges:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1
            
            # Find all nodes that are ancestors of the target node
            q = deque([target_node_id])
            ancestors = {target_node_id}
            visited_for_ancestors = {target_node_id}
            
            # Reverse edges to traverse backwards from target
            reverse_adj = {node_id: [] for node_id in nodes}
            for edge in edges:
                reverse_adj[edge.target].append(edge.source)
            
            while q:
                curr = q.popleft()
                for parent in reverse_adj.get(curr, []):
                    if parent not in visited_for_ancestors:
                        visited_for_ancestors.add(parent)
                        ancestors.add(parent)
                        q.append(parent)
            
            # Perform topological sort ONLY on the ancestors subgraph
            queue = deque([node_id for node_id in ancestors if in_degree[node_id] == 0])
            exec_order = []
            while queue:
                node_id = queue.popleft()
                exec_order.append(node_id)
                for neighbor in adj.get(node_id, []):
                    if neighbor in ancestors:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
            
            if len(exec_order) != len(ancestors):
                raise ValueError("Cycle detected in the pipeline graph.")
            
            return exec_order
        
        try:
            exec_order = calculate_execution_order(nodes, edges, target_node_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid pipeline structure: {e}")
        
        # --- 2. Execute Nodes in Order WITH CACHING ---
        setup_cache_directory() # Ensure the .pipeline_cache folder exists
        context = ExecutionContext()
        node_hashes: Dict[str, str] = {} # To store the calculated hash for each node
        
        for node_id in exec_order:
            node = nodes[node_id]
            print(f"--- Processing Node: {node.data.get('label', node.type)} (ID: {node_id}) ---")
            
            # --- A. Assemble Parent Inputs and Hashes ---
            parent_edges = [edge for edge in edges if edge.target == node_id]
            
            parent_hashes = []
            parent_inputs = {} 
            
            for edge in parent_edges:
                parent_id = edge.source
                parent_node = nodes[parent_id] # Get the full parent node object
                
                parent_output_dict = context.node_outputs.get(parent_id)
                if not parent_output_dict:
                    raise ConnectionError(f"Parent node {parent_id} did not produce any output.")
                
                df_to_pass = None
                source_handle_to_use = edge.source_handle
                
                if source_handle_to_use:
                    # Case 1: The edge explicitly defines the source handle. Use it.
                    df_to_pass = parent_output_dict.get(source_handle_to_use)
                elif len(parent_output_dict) == 1:
                    # Case 2: Edge is implicit, but parent has only ONE output. Use that output.
                    # This is the key fix for single-output nodes.
                    source_handle_to_use = list(parent_output_dict.keys())[0]
                    df_to_pass = parent_output_dict[source_handle_to_use]
                else:
                    # Case 3 (Fallback): Edge is implicit, parent has multiple outputs.
                    # Try 'default' as a last resort.
                    source_handle_to_use = 'default'
                    df_to_pass = parent_output_dict.get(source_handle_to_use)
                
                if df_to_pass is None:
                    # Give a much more helpful error message
                    raise ConnectionError(
                        f"Could not connect nodes. Parent '{parent_node.data.get('label', parent_id)}' "
                        f"did not produce an output on the required handle '{source_handle_to_use}'. "
                        f"Available handles: {list(parent_output_dict.keys())}"
                    )
                
                target_handle_to_use = edge.target_handle or 'default'
                
                parent_hashes.append(node_hashes[parent_id])
                
                input_key = target_handle_to_use if node.type in ['modelTrainer', 'merge'] else edge.id

                parent_inputs[input_key] = {
                    "data": df_to_pass,
                    "source_node_id": parent_id,
                    "source_node_type": parent_node.type,
                    "source_handle": source_handle_to_use, 
                }
            
            current_hash = generate_node_hash(node.data, sorted(parent_hashes)) 
            node_hashes[node_id] = current_hash
            
            # --- B. Check the Cache ---
            cached_result = load_from_cache(current_hash)
            
            if cached_result:
                # CACHE HIT: Load results directly into the context
                context.node_outputs[node_id] = cached_result['output']
                if cached_result.get('metadata'):
                    context.node_metadata[node_id] = cached_result['metadata']
                if cached_result.get('model'):
                    context.trained_models[node_id] = cached_result['model']
            else:
                # CACHE MISS: Execute the node normally
                try:
                    executor = get_executor(node.type)
                    
                    # The executor now receives the rich parent_inputs object
                    output_dict  = executor.execute(node, parent_inputs, context)
                    
                    # Store the direct dataframe output for immediate use by children
                    context.node_outputs[node_id] = output_dict
                    
                    # --- C. Save the new result to the cache ---
                    result_to_cache = {
                        'output': output_dict,
                        'metadata': context.node_metadata.get(node_id),
                        'model': context.trained_models.get(node_id)
                    }
                    save_to_cache(current_hash, result_to_cache)
                
                except Exception as e:
                    print(f"ERROR executing node {node_id}: {e}")
                    traceback.print_exc()
                    error_info = {
                        "error": f"Error in node '{node.data.get('label', node.type)}': {str(e)}",
                        "nodeId": node_id
                    }
                    # Consider returning a more structured error response
                    raise HTTPException(status_code=500, detail=error_info)
        
        # Add this constant near the top of your file, perhaps after the imports.
        # This makes it easy to change the preview size later.
        ROW_LIMIT_FOR_DISPLAY = 100
        
        # --- 3. Format and Return ALL Node Outputs (WITH TRUNCATION) ---
        all_node_results = {}
        
        for node_id in exec_order:
            node_output_dict = context.node_outputs.get(node_id, {})
            
            # Intelligently select the primary DataFrame to display
            display_df = None
            if 'default' in node_output_dict:
                display_df = node_output_dict['default']
            elif 'train' in node_output_dict:
                display_df = node_output_dict['train']
            elif node_output_dict:
                display_df = list(node_output_dict.values())[0]
                
            data_as_dicts = []
            # Start with a default info message
            node_info = {"message": f"Node '{nodes[node_id].data.get('label', node_id)}' executed successfully."}

            # Check if display_df is a valid, non-empty DataFrame
            if display_df is not None and isinstance(display_df, pd.DataFrame) and not display_df.empty:
                
                # --- IMPORTANT: Generate info from the FULL, original DataFrame first! ---
                first_node_id = exec_order[0]
                first_node = nodes.get(first_node_id, None)
                if first_node:
                    symbol = first_node.data.get('symbol', 'N/A')
                    timeframe = first_node.data.get('timeframe', 'N/A')
                    # This function now gets the full, untruncated row count
                    node_info = generate_df_info(display_df.copy(), symbol, timeframe)
                
                # --- NEW: Add truncation info if necessary ---
                total_rows = len(display_df)
                if total_rows > ROW_LIMIT_FOR_DISPLAY:
                    node_info['truncation_info'] = {
                        "message": f"Displaying first {ROW_LIMIT_FOR_DISPLAY} of {total_rows} total rows.",
                        "is_truncated": True,
                        "displaying_rows": ROW_LIMIT_FOR_DISPLAY,
                        "total_rows": total_rows
                    }

                # --- NEW: Create the truncated DataFrame for serialization ---
                df_for_serialization = display_df.head(ROW_LIMIT_FOR_DISPLAY).copy()
                
                # --- DataFrame Serialization Logic (now runs on the smaller df) ---
                df_for_serialization.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_sanitized = df_for_serialization.astype(object).where(pd.notnull(df_for_serialization), None)
                
                if 'timestamp' in df_sanitized.columns:
                    df_sanitized['timestamp'] = pd.to_datetime(df_sanitized['timestamp']).astype('int64') // 10**6
                
                data_as_dicts = df_sanitized.to_dict(orient='records')
            
            # Merge any special metadata (like model performance) stored in the context
            if node_id in context.node_metadata:
                # Using deepcopy can prevent nested dicts from being overwritten, but update is usually fine
                node_info.update(context.node_metadata[node_id])
                
            all_node_results[node_id] = {
                "data": data_as_dicts,
                "info": node_info
            }

        return all_node_results




    # @app.post("/api/ml/workflow/{workflow_id}/execute")
    # async def execute_pipeline_up_to_node(workflow_id: str, body: PipelineExecutionRequest):
    #     """
    #     Executes the pipeline defined by the graph up to a specific target node.
    #     """
    #     if workflow_id not in WORKFLOW_CACHE:
    #         raise HTTPException(status_code=404, detail="Workflow ID not found.")

    #     nodes = {node.id: node for node in body.nodes}
    #     edges = body.edges
    #     target_node_id = body.target_node_id

    #     # --- 1. Determine Execution Order (Topological Sort) ---
    #     try:
    #         # Build adjacency list and in-degree count for the graph
    #         adj = {node_id: [] for node_id in nodes}
    #         in_degree = {node_id: 0 for node_id in nodes}
    #         for edge in edges:
    #             adj[edge.source].append(edge.target)
    #             in_degree[edge.target] += 1
            
    #         # Find all nodes that are ancestors of the target node
    #         q = deque([target_node_id])
    #         ancestors = {target_node_id}
    #         visited_for_ancestors = {target_node_id}
            
    #         # Reverse edges to traverse backwards from target
    #         reverse_adj = {node_id: [] for node_id in nodes}
    #         for edge in edges:
    #             reverse_adj[edge.target].append(edge.source)

    #         while q:
    #             curr = q.popleft()
    #             for parent in reverse_adj.get(curr, []):
    #                 if parent not in visited_for_ancestors:
    #                     visited_for_ancestors.add(parent)
    #                     ancestors.add(parent)
    #                     q.append(parent)

    #         # Perform topological sort ONLY on the ancestors subgraph
    #         queue = deque([node_id for node_id in ancestors if in_degree[node_id] == 0])
    #         exec_order = []
    #         while queue:
    #             node_id = queue.popleft()
    #             exec_order.append(node_id)
    #             for neighbor in adj.get(node_id, []):
    #                 if neighbor in ancestors:
    #                     in_degree[neighbor] -= 1
    #                     if in_degree[neighbor] == 0:
    #                         queue.append(neighbor)
            
    #         if len(exec_order) != len(ancestors):
    #             raise ValueError("Cycle detected in the pipeline graph.")
            
    #     except Exception as e:
    #         raise HTTPException(status_code=400, detail=f"Invalid pipeline structure: {e}")

    #     # --- 2. Execute Nodes in Order ---
    #     node_outputs: Dict[str, pd.DataFrame] = {} # Cache for intermediate dataframes
    #     node_metadata: Dict[str, Dict[str, Any]] = {}

    #     for node_id in exec_order:
    #         node = nodes[node_id]
    #         print(f"Executing Node: {node.data.get('label', node.type)} (ID: {node_id})")

    #         try:
    #             # Find parent nodes for the current node
    #             parent_ids = [edge.source for edge in edges if edge.target == node_id]
                
    #             # This is a critical step: get the output from the parent node(s)
    #             # For simplicity, we assume single-input nodes for now.
    #             # We will handle multi-input nodes like 'processIndicators' specially.
    #             df_input = node_outputs[parent_ids[0]].copy() if parent_ids else pd.DataFrame()

    #             # --- Logic for each node type ---
    #             if node.type == 'dataSource':
    #                 ds_config = DataSourceConfig(**node.data)
    #                 df_output = load_data_for_ml(
    #                     symbol=ds_config.symbol,
    #                     timeframe=ds_config.timeframe,
    #                     start_date_str=ds_config.startDate,
    #                     end_date_str=ds_config.endDate
    #                 )

    #             elif node.type == 'feature':
    #                 # Reformat feature data for the backend function
    #                 indicator_config = [IndicatorConfig(id=node.id, name=node.data['indicator'], params=node.data['params'])]
    #                 indicator_tuples = transform_features_for_backend(indicator_config, nodes[parent_ids[0]].data['timeframe'])
    #                 mock_strategy = SimpleNamespace(indicators=indicator_tuples)
    #                 df_output = calculate_indicators(mock_strategy, df_input)

    #             elif node.type == 'processIndicators':
    #                 # This node is special: it combines multiple inputs
    #                 selected_ids = [pid for pid, is_selected in node.data.get('selectedIndicators', {}).items() if is_selected]
                    
    #                 # Ensure we only process inputs that are actually connected
    #                 connected_and_selected_ids = list(set(parent_ids) & set(selected_ids))

    #                 if not connected_and_selected_ids:
    #                     # If nothing is selected, pass through the first parent's data as a default
    #                     df_output = df_input
    #                 else:
    #                     # Start with the base dataframe (e.g., from dataSource)
    #                     base_df_parent_id = next((pid for pid in parent_ids if nodes[pid].type not in ['feature']), None)
    #                     if base_df_parent_id:
    #                         df_output = node_outputs[base_df_parent_id].copy()
    #                     else: # Fallback if no non-feature parent is found
    #                         df_output = node_outputs[parent_ids[0]].copy()

    #                     # Merge selected feature columns
    #                     for feature_node_id in connected_and_selected_ids:
    #                         feature_df = node_outputs[feature_node_id]
    #                         # Find the new column(s) the feature node added
    #                         new_cols = feature_df.columns.difference(df_output.columns)
    #                         df_output = pd.merge(df_output, feature_df[new_cols.tolist() + [feature_df.index.name if feature_df.index.name else 'timestamp']], on=feature_df.index.name if feature_df.index.name else 'timestamp', how='left')
                    
    #             elif node.type == 'customCode':
    #                 custom_code = node.data.get('code', '')
    #                 sub_type = node.data.get('subType', 'feature_engineering')

    #                 if not custom_code.strip():
    #                     df_output = df_input
    #                 else:
    #                     local_namespace = {}
    #                     # We pass a copy of globals() so the exec doesn't pollute our main global scope
    #                     exec_globals = globals().copy() 
    #                     exec(custom_code, exec_globals, local_namespace)
                        
    #                     if sub_type == 'labeling':
    #                         print("Executing custom node as 'Labeling'")
    #                         labeling_func = local_namespace.get('generate_labels')
    #                         if not callable(labeling_func):
    #                             raise NameError("Labeling script must have a 'generate_labels(data)' function.")
                            
    #                         # By updating the function's globals with our local_namespace,
    #                         # it can now "see" the other functions defined in the same script.
    #                         labeling_func.__globals__.update(local_namespace)

    #                         labels_series = labeling_func(df_input.copy())
    #                         if not isinstance(labels_series, pd.Series):
    #                             raise TypeError("'generate_labels' function must return a pandas Series.")
                            
    #                         df_output = df_input.copy()
    #                         df_output['label'] = labels_series
                            
    #                         # Drop rows where the label might be NaN after creation
    #                         df_labeled = df_output.dropna(subset=['label'])
                            
    #                         label_counts = df_labeled['label'].value_counts()
    #                         label_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}
                            
    #                         print(f"Node {node_id} Label Distribution: {label_distribution}")
                            
    #                         # Store this special metadata against the node's ID
    #                         node_metadata[node_id] = {
    #                             "label_distribution": label_distribution
    #                         }

    #                     else: # 'feature_engineering'
    #                         print("Executing custom node as 'Feature Engineering'")
    #                         process_func = local_namespace.get('process')
    #                         if not callable(process_func):
    #                             raise NameError("Feature Engineering script must have a 'process(data)' function.")
                            
    #                         process_func.__globals__.update(local_namespace)

    #                         df_output = process_func(df_input.copy())
    #                         if not isinstance(df_output, pd.DataFrame):
    #                             raise TypeError("'process' function must return a pandas DataFrame.")
                
    #             elif node.type == 'dataScaling':
    #                 print("Executing Data Scaling & Preprocessing Node")
    #                 df_processed = df_input.copy()
                    
    #                 # Store original columns to select only feature columns for scaling/PCA
    #                 # Exclude OHLCV, timestamp, and the label if it exists
    #                 original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'label']
    #                 feature_cols = [col for col in df_processed.columns if col not in original_cols]
                    
    #                 if not feature_cols:
    #                     print("Warning: No feature columns found to scale.")
    #                     df_output = df_processed # Pass through if no features
    #                 else:
    #                     features_df = df_processed[feature_cols]

    #                     # --- A. Remove Correlated Features ---
    #                     if node.data.get('removeCorrelated'):
    #                         threshold = node.data.get('correlationThreshold', 0.9)
    #                         print(f"Removing features with correlation > {threshold}")
    #                         corr_matrix = features_df.corr().abs()
    #                         upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    #                         to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    #                         features_df = features_df.drop(columns=to_drop)
    #                         print(f"Dropped {len(to_drop)} columns: {to_drop}")

    #                     # --- B. Apply Scaler ---
    #                     scaler_type = node.data.get('scaler', 'none')
    #                     if scaler_type != 'none':
    #                         print(f"Applying {scaler_type}")
    #                         if scaler_type == 'StandardScaler':
    #                             scaler = StandardScaler()
    #                         elif scaler_type == 'MinMaxScaler':
    #                             scaler = MinMaxScaler()
                            
    #                         scaled_features = scaler.fit_transform(features_df)
    #                         features_df = pd.DataFrame(scaled_features, index=features_df.index, columns=features_df.columns)

    #                     # --- C. Apply PCA ---
    #                     if node.data.get('usePCA'):
    #                         n_components = node.data.get('pcaComponents', 5)
    #                         # Ensure n_components is not more than available features
    #                         n_components = min(n_components, len(features_df.columns), len(features_df))
    #                         print(f"Applying PCA with {n_components} components")
    #                         pca = PCA(n_components=n_components)
    #                         principal_components = pca.fit_transform(features_df)
    #                         pca_cols = [f'pca_{i+1}' for i in range(n_components)]
    #                         features_df = pd.DataFrame(data=principal_components, columns=pca_cols, index=features_df.index)

    #                     # --- D. Reconstruct the DataFrame ---
    #                     # Combine the non-feature columns with the newly processed feature columns
    #                     non_feature_df = df_processed[[col for col in df_processed.columns if col not in feature_cols]]
    #                     df_output = pd.concat([non_feature_df, features_df], axis=1)
                        
    #             elif node.type == 'dataValidation':
    #                 print("Executing Data Validation Node")
    #                 df_output = df_input.copy() # This node is informational
                    
    #                 # 1. First, calculate the split info, which does NOT require a label.
    #                 #    We use the full input dataframe for sample counts.
    #                 total_samples = len(df_output)
    #                 split_info = {}
    #                 method = node.data.get('validationMethod', 'train_test_split')

    #                 if method == 'train_test_split':
    #                     train_pct = node.data.get('trainSplit', 70) / 100.0
    #                     split_idx = int(total_samples * train_pct)
    #                     split_info = {
    #                         "method": "Train/Test Split",
    #                         "train_samples": split_idx,
    #                         "test_samples": total_samples - split_idx,
    #                         "total_samples": total_samples
    #                     }
    #                 elif method == 'walk_forward':
    #                     train_window = node.data.get('walkForwardTrainWindow', 365)
    #                     test_window = node.data.get('walkForwardTestWindow', 30)
                        
    #                     num_folds = 0
    #                     if total_samples > train_window and test_window > 0:
    #                         num_folds = 1 + (total_samples - train_window) // test_window
                        
    #                     split_info = {
    #                         "method": "Walk-Forward",
    #                         "train_window_size": f"{train_window} days",
    #                         "test_window_size": f"{test_window} days",
    #                         "approximate_folds": num_folds,
    #                         "total_samples": total_samples
    #                     }
                    
    #                 # Initialize the metadata with the universally available split info
    #                 node_metadata[node_id] = {
    #                     "validation_info": split_info
    #                 }

    #                 # 2. THEN, if a label column exists, add the label distribution.
    #                 if 'label' in df_output.columns:
    #                     print("Validation Node: Reporting on existing label column.")
    #                     df_labeled = df_output.dropna(subset=['label'])
    #                     label_counts = df_labeled['label'].value_counts()
    #                     label_distribution = {str(k): int(v) for k, v in label_counts.to_dict().items()}
                        
    #                     # Add it to the metadata dictionary
    #                     node_metadata[node_id]["label_distribution"] = label_distribution
                
    #             elif node.type == 'mlModel':
    #                 print("Executing ML Model Node (Validation Preview)")
    #                 df_output = df_input.copy() # This node is informational, pass data through
                    
    #                 model_name = node.data.get('modelName')
    #                 hyperparameters = node.data.get('hyperparameters', {})
                    
    #                 # --- 2. Check if the model is unsupervised FIRST ---
    #                 is_unsupervised = model_name in UNSUPERVISED_MODELS
                    
    #                 # --- A. Prepare Feature Data (common for both paths) ---
    #                 exclude_cols = ['timestamp', 'label']
    #                 feature_cols = [col for col in df_output.columns if col not in exclude_cols]
                    
    #                 if not feature_cols:
    #                     raise ValueError("No feature columns found for model processing.")
                    
    #                 X_features = df_output[feature_cols]
                    
    #                 # --- B. Instantiate the Model ---
    #                 model = get_model(model_name, hyperparameters)
                    
    #                 # --- C. Branching Logic: Unsupervised vs. Supervised ---
    #                 if is_unsupervised:
    #                     print(f"Processing unsupervised model: {model_name}")
                        
    #                     # Fit the model and get the cluster labels/predictions
    #                     # .fit_predict() is common for clustering, .fit_transform() for PCA
    #                     if hasattr(model, 'fit_predict'):
    #                         clusters = model.fit_predict(X_features)
    #                         # Add the cluster assignments as a new column
    #                         df_output[f'{model_name}_cluster'] = clusters
    #                     elif hasattr(model, 'fit_transform'):
    #                         # This case is for PCA, which is already handled by the DataScaling node.
    #                         # We can just show a message or pass through.
    #                         print("PCA model is for transformation; use Data Scaling node for PCA.")
                        
    #                     # For unsupervised, we don't have traditional metrics.
    #                     # The primary output is the modified DataFrame with the new cluster column.
    #                     # We can still add some info to the metadata.
    #                     node_metadata[node_id] = {
    #                         "model_info": {
    #                             "Model Name": model_name,
    #                             "Problem Type": "Unsupervised",
    #                             "Total Samples": len(X_features),
    #                             "Total Features": len(feature_cols)
    #                         }
    #                     }
                    
    #                 else: # Supervised Path
    #                     print(f"Processing supervised model: {model_name}")
                        
    #                     if 'label' not in df_output.columns:
    #                         raise ValueError(f"Supervised model '{model_name}' requires an upstream node to have created a 'label' column.")
                        
    #                     df_labeled = df_output.dropna(subset=['label']).copy()
                        
    #                     X = df_labeled[feature_cols]
    #                     y = df_labeled['label']
                        
    #                     # The rest of your supervised logic is excellent and can remain here
    #                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                        
    #                     model.fit(X_train, y_train)
                        
    #                     # --- C. Evaluate Model & Gather Metrics (MODIFIED) ---
    #                     y_pred = model.predict(X_test)
                        
    #                     model_metrics = {}
    #                     model_analysis = {}
                        
    #                     is_classification = y.dtype != 'float' and y.nunique() < 30
                        
    #                     if is_classification:
    #                         print("Calculating classification metrics...")
    #                         model_metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
    #                         # Handle binary vs. multiclass for other metrics
    #                         avg_method = 'binary' if y.nunique() == 2 else 'weighted'
    #                         model_metrics['precision'] = round(precision_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
    #                         model_metrics['recall'] = round(recall_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
    #                         model_metrics['f1_score'] = round(f1_score(y_test, y_pred, average=avg_method, zero_division=0), 4)
                            
    #                         # ROC AUC for binary classification
    #                         if avg_method == 'binary' and hasattr(model, "predict_proba"):
    #                             y_proba = model.predict_proba(X_test)[:, 1]
    #                             model_metrics['roc_auc'] = round(roc_auc_score(y_test, y_proba), 4)
                            
    #                         report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                            
    #                         # Clean up the report for JSON (convert numpy types to native Python types)
    #                         cleaned_report = {
    #                             str(key): {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else float(value)
    #                             for key, value in report.items()
    #                         }
    #                         model_analysis['classification_report'] = cleaned_report
                            
    #                         # Get class labels for the confusion matrix
    #                         class_labels = sorted(list(y.unique()))
    #                         cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    #                         model_analysis['confusion_matrix'] = {
    #                             'labels': [str(label) for label in class_labels],
    #                             'values': cm.tolist() # Convert numpy array to nested list
    #                         }
                            
    #                         # --- Get Feature Importances ---
    #                         if hasattr(model, 'feature_importances_'):
    #                             importances = model.feature_importances_
    #                             feature_importance_data = sorted(
    #                                 zip(X.columns, importances),
    #                                 key=lambda x: x[1],
    #                                 reverse=True
    #                             )
    #                             model_analysis['feature_importance'] = [
    #                                 {'feature': name, 'importance': float(imp)}
    #                                 for name, imp in feature_importance_data
    #                             ]
                            
    #                     else: # Regression
    #                         print("Calculating regression metrics...")
    #                         model_metrics['mean_squared_error'] = round(mean_squared_error(y_test, y_pred), 4)
    #                         model_metrics['r2_score'] = round(r2_score(y_test, y_pred), 4)
                            
    #                         if hasattr(model, 'feature_importances_'):
    #                             importances = model.feature_importances_
    #                             feature_importance_data = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    #                             model_analysis['feature_importance'] = [{'feature': name, 'importance': float(imp)} for name, imp in feature_importance_data]
                        
    #                     # --- D. Store results in metadata ---
    #                     node_metadata[node_id] = {
    #                         "model_metrics": model_metrics,
    #                         "model_info": {
    #                             "Model Name": model_name,
    #                             "Problem Type": "Classification" if is_classification else "Regression",
    #                             "Train Samples": len(y_train),
    #                             "Test Samples": len(y_test),
    #                             "Total Features": len(feature_cols)
    #                         },
    #                         "model_analysis": model_analysis
    #                     }
                
    #             elif node.type == 'charting':
    #                 print("Executing Charting Node")
    #                 df_output = df_input.copy() # This node is informational, pass data through
                    
    #                 chart_config = node.data
    #                 chart_type = chart_config.get('chartType')
    #                 x_axis = chart_config.get('xAxis')
    #                 y_axis = chart_config.get('yAxis')
    #                 group_by = chart_config.get('groupBy')
                    
    #                 chart_data = []
    #                 SAMPLE_LIMIT = 5000 # Max points to send to the frontend
                    
    #                 if chart_type == 'histogram' and x_axis:
    #                     # Create bins for the histogram
    #                     counts, bins = np.histogram(df_output[x_axis].dropna(), bins=20)
    #                     chart_data = [
    #                         {"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])}
    #                         for i in range(len(counts))
    #                     ]
    #                 elif x_axis and y_axis:
    #                     cols_to_keep = [x_axis, y_axis]
    #                     if group_by:
    #                         cols_to_keep.append(group_by)
                        
    #                     plot_df = df_output[list(set(cols_to_keep))].dropna()
                        
    #                     # --- NEW: APPLY SAMPLING ---
    #                     if len(plot_df) > SAMPLE_LIMIT:
    #                         print(f"Dataset too large ({len(plot_df)} rows). Sampling down to {SAMPLE_LIMIT}.")
    #                         plot_df = plot_df.sample(n=SAMPLE_LIMIT, random_state=42)
                        
    #                     chart_data = plot_df.to_dict(orient='records')
                        
    #                 node_metadata[node_id] = {
    #                     "chart_config": chart_config,
    #                     "chart_data": chart_data,
    #                     "info_message": f"Displaying {len(chart_data)} of {len(df_output)} total points." if len(chart_data) < len(df_output) else None
    #                 }
                
    #             else:
    #                 # Default case: pass data through if node type is unknown
    #                 df_output = df_input
                
    #             node_outputs[node_id] = df_output
            
    #         except Exception as e:
    #             print(f"ERROR executing node {node_id}: {e}")
    #             traceback.print_exc()
    #             error_info = {
    #                 "error": f"Error in node '{node.data.get('label', node.type)}': {e}",
    #                 "nodeId": node_id
    #             }
    #             return {"data": [], "info": error_info}


    #     # --- 3. Format and Return the Final Output ---
    #     final_df = node_outputs.get(target_node_id)
    #     if final_df is None or final_df.empty:
    #         return {"data": [], "info": {"message": "No data produced by the target node."}}
        
    #     # Use your existing helper to generate rich metadata
    #     final_info = generate_df_info(final_df.copy(), nodes[exec_order[0]].data['symbol'], nodes[exec_order[0]].data['timeframe'])
        
    #     if target_node_id in node_metadata:
    #         print(f"Merging special metadata for node {target_node_id}")
    #         final_info.update(node_metadata[target_node_id])
        
    #     # Sanitize and format for JSON
    #     final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #     df_for_json = final_df.astype(object).where(pd.notnull(final_df), None)
        
    #     if 'timestamp' in df_for_json.columns:
    #         df_for_json['timestamp'] = (pd.to_datetime(df_for_json['timestamp']).astype('int64') // 10**6)

    #     data_as_dicts = df_for_json.to_dict(orient='records')
        
    #     return {"data": data_as_dicts, "info": final_info}


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
