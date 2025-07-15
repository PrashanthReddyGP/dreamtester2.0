import traceback

try:
    import ast
    import sys
    import time
    import uuid
    import uvicorn
    import asyncio
    from typing import List
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict
    from sqlalchemy.orm import Session

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
        get_backtest_job 
    )

    from pipeline import run_unified_test_manager, run_optimization_manager, run_asset_screening_manager, run_batch_manager
    from core.connect_to_brokerage import get_client

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
            if batch_id in self.active_connections:
                # Create a list of tasks to send messages concurrently
                tasks = [connection.send_json(message) for connection in self.active_connections[batch_id]]
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
            await manager.send_json_to_batch(batch_id, message)
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

    symbol_cache = {}
    CACHE_DURATION_SECONDS = 3600  # Cache for 1 hour



    @app.on_event("startup")
    async def startup_event():
        # Create the queue here. It will be passed to tasks as needed.
        app.state.websocket_queue = asyncio.Queue()
        # Start the websocket_sender task and give it the queue
        asyncio.create_task(websocket_sender(app.state.websocket_queue, manager))
        # --- NEW: Start the heartbeat task ---
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
    async def submit_batch_backtest_endpoint(files: List[StrategyFileModel], background_tasks: BackgroundTasks, request: Request):
        """
        Accepts a batch of strategies, creates a job record, queues a background task,
        and returns a unique batch_id for the client to connect to via WebSocket.
        """
        print(f"API: Received a batch of {len(files)} strategies.")
        
        batch_id = str(uuid.uuid4())
        
        # Immediately create the job record in the database so the frontend can query its status
        create_backtest_job(batch_id)
        
        files_data = [file.model_dump() for file in files]
        
        # Get the queue and pass it
        queue = request.app.state.websocket_queue
        main_loop = asyncio.get_running_loop() # Get the loop

        background_tasks.add_task(
            run_batch_manager, 
            batch_id=batch_id, 
            files_data=files_data, 
            manager=manager,
            queue=queue,
            loop=main_loop
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

        background_tasks.add_task(
            run_unified_test_manager,
            batch_id=batch_id,
            config=config_dict,
            manager=manager,
            queue=queue,
            loop=main_loop
        )
        return {"message": "Optimization job started.", "batch_id": batch_id}

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
        try:
            while True:
                await websocket.receive_text() 
        except WebSocketDisconnect:
            manager.disconnect(websocket, batch_id)

    if __name__ == "__main__":
        # Make sure to pass the 'app' object you created earlier
        uvicorn.run(app, host="127.0.0.1", port=8000, ws_per_message_deflate=False, ws_ping_interval=20, ws_ping_timeout=60, ws_max_size=16777216)
        
except Exception as e:
    # If any error occurs, print it and wait for user input
    print("An unhandled exception occurred!")
    print(traceback.format_exc())
    input("\nPress Enter to exit...")
