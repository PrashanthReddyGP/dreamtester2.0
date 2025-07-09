import os
import time
from datetime import datetime
from sqlalchemy import REAL, create_engine, Column, Integer, String, Text, ForeignKey, asc, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# --- This part remains the same ---
APP_DATA_DIR = os.path.join(os.path.expanduser('~'), '.dreamtester_2.0')

os.makedirs(APP_DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(APP_DATA_DIR, 'database.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ApiKey Model remains the same ---
class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    exchange_name = Column(String, unique=True, index=True)
    api_key = Column(String)
    api_secret = Column(String)

# --- NEW: Define the StrategyFile Model ---
class StrategyFile(Base):
    __tablename__ = "strategy_files"

    # The frontend uses UUIDs, so we'll use a String primary key
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String) # 'file' or 'folder'
    content = Column(Text, nullable=True) # Content is only for files

    # Self-referencing foreign key for hierarchy
    parent_id = Column(String, ForeignKey("strategy_files.id"), nullable=True)
    
    # Relationship to easily access children from a parent
    children = relationship("StrategyFile", back_populates="parent", cascade="all, delete-orphan")
    parent = relationship("StrategyFile", back_populates="children", remote_side=[id])

    # Convert model instance to a dictionary, matching frontend's expected structure
    def to_dict(self):
        sorted_children = sorted(self.children, key=lambda child: child.name.lower())
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "content": self.content,
            "children": [child.to_dict() for child in sorted_children]
        }

class LatestResult(Base):
    __tablename__ = "latest_result"
    # We use a fixed primary key so we can always target this one row.
    id = Column(Integer, primary_key=True, default=1) 
    results_data = Column(JSON, nullable=True)
    updated_at = Column(REAL, default=time.time, onupdate=time.time)
    

class BacktestJob(Base):
    __tablename__ = "backtest_jobs"

    # We use the batch_id from the API endpoint as the primary key
    id = Column(String, primary_key=True, index=True) 
    
    # New columns to track the job's progress and outcome
    status = Column(String, default="pending") # "pending", "running", "completed", "failed"
    error_message = Column(Text, nullable=True) 
    logs = Column(JSON, nullable=True, default=lambda: []) # Default to an empty list
    
    # This still holds the final, successful result payload
    results_data = Column(JSON, nullable=True)
    
    created_at = Column(REAL, default=time.time)
    updated_at = Column(REAL, default=time.time, onupdate=time.time)



# --- Create all tables in the database ---
Base.metadata.create_all(bind=engine)



# --- CRUD functions for ApiKey remain the same ---
def save_api_key(exchange: str, key: str, secret: str):
    db = SessionLocal()
    try:
        db_key = db.query(ApiKey).filter(ApiKey.exchange_name == exchange).first()
        if db_key:
            db_key.api_key = key
            db_key.api_secret = secret
        else:
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
            return {"exchange": db_key.exchange_name, "api_key": db_key.api_key, "api_secret": db_key.api_secret}
        return None
    finally:
        db.close()

# --- NEW: CRUD functions for Strategy Files ---

def get_strategies_tree():
    db = SessionLocal()
    try:
        # Fetch only top-level items (those without a parent)
        top_level_items = db.query(StrategyFile).filter(StrategyFile.parent_id == None).order_by(asc(StrategyFile.name)).all()
        # The to_dict method will recursively build the rest of the tree
        return [item.to_dict() for item in top_level_items]
    finally:
        db.close()

def create_strategy_item(item_id: str, name: str, type: str, content: str = None, parent_id: str = None):
    db = SessionLocal()
    try:
        new_item = StrategyFile(id=item_id, name=name, type=type, content=content, parent_id=parent_id)
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        return new_item.to_dict()
    finally:
        db.close()

def update_strategy_item(item_id: str, name: str = None, content: str = None):
    db = SessionLocal()
    try:
        item_to_update = db.query(StrategyFile).filter(StrategyFile.id == item_id).first()
        if not item_to_update:
            return None
        if name is not None:
            item_to_update.name = name
        if content is not None:
            item_to_update.content = content
        db.commit()
        db.refresh(item_to_update)
        return item_to_update.to_dict()
    finally:
        db.close()
        
def delete_strategy_item(item_id: str):
    db = SessionLocal()
    try:
        item_to_delete = db.query(StrategyFile).filter(StrategyFile.id == item_id).first()
        if not item_to_delete:
            return None
        db.delete(item_to_delete)
        db.commit()
        return {"status": "success", "message": "Item deleted."}
    finally:
        db.close()
        
def move_strategy_item(item_id: str, new_parent_id: str | None):
    db = SessionLocal()
    try:
        item_to_move = db.query(StrategyFile).filter(StrategyFile.id == item_id).first()
        if not item_to_move:
            return None # Item not found
        
        # Prevent dropping a folder into itself or its own children (complex check, omitted for brevity but good for a real app)

        item_to_move.parent_id = new_parent_id
        db.commit()
        db.refresh(item_to_move)
        return {"status": "success", "message": "Item moved."}
    finally:
        db.close()
        
def clear_all_strategies():
    db = SessionLocal()
    try:
        # This executes a "DELETE FROM strategy_files" statement, deleting all rows.
        num_rows_deleted = db.query(StrategyFile).delete()
        db.commit()
        return {"status": "success", "message": f"Successfully deleted {num_rows_deleted} items."}
    except Exception as e:
        db.rollback()
        # In a real app, you would log this error
        print(f"Error clearing strategies: {e}")
        return None
    finally:
        db.close()

def create_multiple_strategy_items(items: list[dict]):
    db = SessionLocal()
    try:
        # Create a list of StrategyFile objects from the dictionaries
        new_items = [StrategyFile(**item) for item in items]
        
        # db.add_all() efficiently stages all the new objects
        db.add_all(new_items)
        
        # A single commit writes all of them to the database
        db.commit()
        
        return {"status": "success", "message": f"Successfully imported {len(new_items)} items."}
    except Exception as e:
        db.rollback() # If any item fails, roll back the entire transaction
        print(f"Error during bulk insert: {e}")
        return None
    finally:
        db.close()



# --- 2. ADD NEW CRUD FUNCTIONS FOR BACKTEST JOBS ---

def create_backtest_job(batch_id: str):
    """
    Creates an initial record for a new backtest job in the database.
    """
    db = SessionLocal()
    try:
        # Create a new job with a 'pending' status
        new_job = BacktestJob(id=batch_id, status="pending")
        db.add(new_job)
        db.commit()
        print(f"--- Created initial DB record for batch_id: {batch_id} ---")
    finally:
        db.close()

def update_job_status(batch_id: str, status: str, log_message: str = None):
    """
    Updates the status of a job and appends a new log message.
    """
    db = SessionLocal()
    try:
        job = db.query(BacktestJob).filter(BacktestJob.id == batch_id).first()
        if job:
            job.status = status
            if log_message:
                # SQLAlchemy's JSON type tracks mutations, so we can append directly.
                # It's safer to get the list, append, and then re-assign.
                current_logs = job.logs or []
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3] # HH:MM:SS.ms
                current_logs.append(f"[{timestamp}] {log_message}")
                job.logs = current_logs
            db.commit()
    finally:
        db.close()

def fail_job(batch_id: str, error: str):
    """
    Marks a job as 'failed' and stores the error message.
    """
    db = SessionLocal()
    try:
        job = db.query(BacktestJob).filter(BacktestJob.id == batch_id).first()
        if job:
            job.status = "failed"
            job.error_message = error
            db.commit()
    finally:
        db.close()

def save_backtest_results(batch_id: str, results: dict):
    """
    Saves the final, successful result data to a completed job.
    """
    db = SessionLocal()
    try:
        job = db.query(BacktestJob).filter(BacktestJob.id == batch_id).first()
        if job:
            job.status = "completed"
            job.results_data = results
            db.commit()
    finally:
        db.close()

# --- 3. CREATE a function to get the full job details ---
def get_backtest_job(batch_id: str):
    """
    Retrieves the full details of a specific backtest job.
    """
    db = SessionLocal()
    try:
        job = db.query(BacktestJob).filter(BacktestJob.id == batch_id).first()
        if job:
            # Convert the SQLAlchemy object to a dictionary for the API
            return {
                "id": job.id, 
                "status": job.status, 
                "results": job.results_data, 
                "logs": job.logs, 
                "error": job.error_message,
                "created_at": job.created_at
            }
        return None
    finally:
        db.close()