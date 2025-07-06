from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the new database functions
from database import (
    save_api_key, 
    get_api_key,
    get_strategies_tree,
    create_strategy_item,
    update_strategy_item,
    delete_strategy_item,
    move_strategy_item,
    clear_all_strategies
)
class ApiKeyBody(BaseModel):
    """Defines the expected JSON body for the save_keys endpoint."""
    apiKey: str
    apiSecret: str

class StrategyItemCreate(BaseModel):
    id: str
    name: str
    type: str # 'file' or 'folder'
    parentId: Optional[str] = None
    content: Optional[str] = None

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
        parent_id=item.parentId,
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
