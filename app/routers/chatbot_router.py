from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.chatbot_service import chatbot_instance
router = APIRouter(
    prefix="/api/v1/chatbot",  
    tags=["Chatbot"],        
)

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def handle_chat(request_body: ChatRequest):
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot service is not available due to an initialization error."
        )

    user_message = request_body.message
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        bot_response_dict = chatbot_instance.ask(user_message)
        return bot_response_dict
    except Exception as e:
        print(f"An error occurred during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")