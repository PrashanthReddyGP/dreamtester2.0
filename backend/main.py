from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import save_api_key, get_api_key

class ApiKeyBody(BaseModel):
    """Defines the expected JSON body for the save_keys endpoint."""
    apiKey: str
    apiSecret: str

# Create the FastAPI application instance
app = FastAPI()

origins = [
    "http://localhost:5173", # The origin of your React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.get("/api/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Python API is running!"}

@app.get("/api/keys/{exchange_name}")
def load_keys_endpoint(exchange_name: str):
    keys = get_api_key(exchange=exchange_name)
    if not keys:
        # If no keys are found, return a standard 404 Not Found error.
        raise HTTPException(
            status_code=404, 
            detail=f"API keys not found for exchange: {exchange_name}"
        )
    return keys

@app.post("/api/keys/{exchange_name}")
def save_keys_endpoint(exchange_name: str, keys: ApiKeyBody):
    # The 'keys' parameter is automatically parsed from the request body
    # into an ApiKeyBody object by FastAPI.
    result = save_api_key(
        exchange=exchange_name, 
        key=keys.apiKey, 
        secret=keys.apiSecret
    )
    return result