from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union, Any

class IndicatorConfig(BaseModel):
    id: str
    name: str
    timeframe: str
    params: Dict[str, Any]