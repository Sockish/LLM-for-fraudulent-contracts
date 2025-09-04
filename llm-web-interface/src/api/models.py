from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    max_tokens: int
    temperature: float

class HealthCheckResponse(BaseModel):
    status: str
    message: str