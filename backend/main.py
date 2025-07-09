import uuid
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from sqlalchemy.orm import Session

# Import the new database functions
from database import (
    LatestResult,
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

from pipeline import run_batch_manager

##########################################

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

@app.get("/api/backtest/latest")
def get_latest_result_endpoint():
    db = SessionLocal()
    try:
        # Always fetch the one and only result row
        result_row = db.query(LatestResult).filter(LatestResult.id == 1).first()
        if result_row:
            return result_row.results_data
        return {} # Return empty dict if no result has ever been saved
    finally:
        db.close()
        
@app.delete("/api/backtest/latest", status_code=200)
def clear_latest_result_endpoint():
    """
    Finds and deletes the single record for the latest backtest result.
    This is called by the frontend before starting a new backtest run.
    """
    # Create a new database session
    db: Session = SessionLocal()
    
    try:
        # Query for the single row in the LatestResult table.
        # We assume its ID is always 1, as per our single-slot design.
        result_to_delete = db.query(LatestResult).filter(LatestResult.id == 1).first()

        if result_to_delete:
            # If the row exists, delete it.
            db.delete(result_to_delete)
            db.commit()
            print("--- Cleared latest backtest result from the database. ---")
            return {"message": "Latest result cleared successfully."}
        else:
            # If the row doesn't exist, that's fine too. Nothing to do.
            print("--- No previous backtest result found to clear. ---")
            return {"message": "No previous result to clear."}
            
    except Exception as e:
        # If anything goes wrong, roll back the transaction and raise an error.
        db.rollback()
        print(f"Error clearing latest result: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear previous results on the server.")
    finally:
        # Always close the session to free up the connection.
        db.close()

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


####################################

@app.websocket("/ws/results/{batch_id}")
async def websocket_endpoint(websocket: WebSocket, batch_id: str):
    # Manually check the origin of the WebSocket request
    origin = websocket.headers.get('origin')
    if origin not in origins:
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
