import traceback

try:
    import ast
    import sys
    import uuid
    import asyncio
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
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

    from pipeline import run_indicator_optimization_manager, run_batch_manager

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

    class OptimizationSubmitBody(BaseModel):
        strategyCode: str
        asset: str
        timeframe: str
        startDate: Optional[str] = None
        endDate: Optional[str] = None
        parametersToOptimize: List[OptimizationParam]


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

    # --- 6. CREATE A SINGLE, GLOBAL INSTANCE OF THE MANAGER ---
    manager = ConnectionManager()




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

    @app.post("/api/strategies/parse-indicators")
    async def parse_indicators_from_strategy(request: Request):
        """
        Safely parses a Python strategy file to find the `self.indicators` list
        and default settings like symbol, timeframe, and dates from the __init__ method.
        """
        code_bytes = await request.body()
        code_str = code_bytes.decode('utf-8')
        
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Python syntax error in strategy file: {e}")

        parsed_data = {
            "settings": {
                "symbol": "ADAUSDT",  # Default fallback
                "timeframe": "1d",
                "startDate": "2000-01-01",
                "endDate": "2100-12-31"
            },
            "indicators": []
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
                        # Now look for assignments within __init__
                        for sub_node in item.body:
                            if isinstance(sub_node, ast.Assign):
                                target = sub_node.targets[0]
                                if isinstance(target, ast.Attribute):
                                    attr_name = target.attr
                                    # Safely evaluate the assigned value
                                    try:
                                        value = ast.literal_eval(sub_node.value)
                                        if attr_name == 'symbol':
                                            parsed_data["settings"]["symbol"] = value
                                        elif attr_name == 'timeframe':
                                            parsed_data["settings"]["timeframe"] = value
                                        # Handle both snake_case and camelCase for dates
                                        elif attr_name in ['start_date', 'startDate']:
                                            parsed_data["settings"]["startDate"] = value
                                        elif attr_name in ['end_date', 'endDate']:
                                            parsed_data["settings"]["endDate"] = value
                                        elif attr_name == 'indicators' and isinstance(value, list):
                                            # This part is for parsing the indicators list
                                            for i, element_tuple in enumerate(value):
                                                if isinstance(element_tuple, tuple) and len(element_tuple) >= 3:
                                                    indicator_name = element_tuple[0]
                                                    default_params = element_tuple[2]
                                                    
                                                    indicator_config = {"id": f"indicator_{i}", "name": indicator_name, "originalIndex": i, "params": []}
                                                    for j, p_val in enumerate(default_params):
                                                        indicator_config["params"].append({
                                                            "id": f"param_{i}_{j}", "originalIndex": j, "value": p_val,
                                                            "start": p_val, "end": p_val, "step": 1, "enabled": False
                                                        })
                                                    parsed_data["indicators"].append(indicator_config)
                                    except (ValueError, SyntaxError):
                                        # Could not evaluate the node, skip it
                                        pass
        
        return parsed_data

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
    def submit_batch_backtest_endpoint(files: List[StrategyFileModel], background_tasks: BackgroundTasks):    
        """
        Accepts a batch of strategies, creates a job record, queues a background task,
        and returns a unique batch_id for the client to connect to via WebSocket.
        """
        print(f"API: Received a batch of {len(files)} strategies.")
        
        batch_id = str(uuid.uuid4())
        
        # Immediately create the job record in the database so the frontend can query its status
        create_backtest_job(batch_id)
        
        files_data = [file.model_dump() for file in files]
        
        # Pass the manager instance to the background task so it can send updates
        background_tasks.add_task(run_batch_manager, batch_id=batch_id, files_data=files_data, manager=manager)
        
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
    def submit_optimization(body: OptimizationSubmitBody, background_tasks: BackgroundTasks):
        batch_id = str(uuid.uuid4())
        create_backtest_job(batch_id) # Reuse the same job table for tracking
        
        background_tasks.add_task(
            run_indicator_optimization_manager,
            batch_id=batch_id,
            config=body.model_dump(), # Pass the entire validated config
            manager=manager
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
        import uvicorn
        # Make sure to pass the 'app' object you created earlier
        uvicorn.run(app, host="127.0.0.1", port=8000)
        
except Exception as e:
    # If any error occurs, print it and wait for user input
    print("An unhandled exception occurred!")
    print(traceback.format_exc())
    input("\nPress Enter to exit...")
