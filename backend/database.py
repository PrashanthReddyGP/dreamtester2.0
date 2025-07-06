import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, asc
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
            return {"exchange": db_key.exchange_name, "api_key": db_key.api_key}
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

