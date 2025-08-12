from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ChatHistoryItem(BaseModel):
    type: Literal['human', 'ai']
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(..., description="Unique ID for the chat session.", example="user123_abc")
    chat_history: Optional[List[ChatHistoryItem]] = Field(None, description="History of the conversation.")

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[dict]