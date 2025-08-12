from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

_chat_memory_store = {}

class ChatHistoryItem(object):
    def __init__(self, type, content):
        self.type = type
        self.content = content

def get_chat_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _chat_memory_store:
        _chat_memory_store[session_id] = ChatMessageHistory()
        
    return _chat_memory_store[session_id]

def load_history_from_request(session_id: str, chat_history_from_request: list):
    chat_history = get_chat_memory(session_id)
    chat_history.clear()  
    
    for item in chat_history_from_request:
        if item.type == 'human':
            chat_history.add_user_message(item.content)
        elif item.type == 'ai':
            chat_history.add_ai_message(item.content)
            