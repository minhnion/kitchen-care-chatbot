from fastapi import APIRouter
from ..models import ChatRequest, ChatResponse

from ..controller.chatbot_controller import chatbot_controller

router = APIRouter(
    prefix="/api/v1/chatbot",  
    tags=["Chatbot"],   
)


@router.post(
    "/chat", 
    summary="Chat with the AI assistant",
    response_model=ChatResponse
)

def handle_chat(request_body: ChatRequest):
    return chatbot_controller.handle_chat_request(request_body)