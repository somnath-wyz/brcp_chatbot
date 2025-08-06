from pydantic import BaseModel
from typing import Dict, List, Optional

class QueryResponse(BaseModel):
    """Response model for database queries."""
    response: str
    files: Dict[str, str] = {}
    download_urls: Dict[str, str] = {}
    success: bool = True
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    created_files: List[str] = []
    thread_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None