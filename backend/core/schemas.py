from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AskRequest(BaseModel):
    """
    Defines the structure for a user's question.
    """
    query: str
    image_data: Optional[str] = Field(None, description="Base64-encoded image data") 

class SourceDocument(BaseModel):
    """
    Represents a singe retrieved source document for citation.
    """
    source_type: str
    location: str
    url: Optional[str] = None

class AskResponse(BaseModel):
    """
    Defines the structure for the AI's answer
    """
    answer: str
    sources: List[SourceDocument]=[]