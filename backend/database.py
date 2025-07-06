import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Define where the database file will be stored.
# Using os.path.expanduser('~') is a good cross-platform way to get the user's home directory.
APP_DATA_DIR = os.path.join(os.path.expanduser('~'), '.dreamtester_2.0')
os.makedirs(APP_DATA_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(APP_DATA_DIR, 'database.db')}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Define the Database Model (Table) ---
class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    exchange_name = Column(String, unique=True, index=True)
    api_key = Column(String)
    api_secret = Column(String) # NOTE: In a real app, encrypt this!

# Create the table in the database if it doesn't exist
Base.metadata.create_all(bind=engine)

# --- Define the CRUD (Create, Read, Update, Delete) functions ---
def save_api_key(exchange: str, key: str, secret: str):
    db = SessionLocal()
    try:
        # Check if a key for this exchange already exists
        db_key = db.query(ApiKey).filter(ApiKey.exchange_name == exchange).first()
        if db_key:
            # Update existing key
            db_key.api_key = key
            db_key.api_secret = secret
        else:
            # Create new key
            db_key = ApiKey(exchange_name=exchange, api_key=key, api_secret=secret)
            db.add(db_key)
        
        db.commit()
        db.refresh(db_key)
        return {"status": "success", "message": f"{exchange.capitalize()} keys saved."}
    finally:
        db.close()

def get_api_key(exchange: str):
    db = SessionLocal()
    try:
        db_key = db.query(ApiKey).filter(ApiKey.exchange_name == exchange).first()
        if db_key:
            return {"exchange": db_key.exchange_name, "api_key": db_key.api_key}
        return None
    finally:
        db.close()