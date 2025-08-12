from fastapi import HTTPException
from ..services import chatbot_instance
from ..models import ChatRequest
class ChatbotController:
    def __init__(self):
        self.chatbot_service = chatbot_instance

    def handle_chat_request(self, request_body: ChatRequest):
        if self.chatbot_service is None:
            raise HTTPException(
                status_code=503, 
                detail="Chatbot service is not available due to an initialization error."
            )

        user_message = request_body.message
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        try:
            history_as_dict = [item.dict() for item in request_body.chat_history] if request_body.chat_history else None

            response = self.chatbot_service.ask_with_memory(
                query=request_body.message,
                session_id=request_body.session_id,
                chat_history_from_request=history_as_dict
            )
            return response
            
        except Exception as e:
            print(f"An error occurred during chat processing: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

chatbot_controller = ChatbotController()