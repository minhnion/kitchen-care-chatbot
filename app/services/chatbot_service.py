import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from .rag_chain import create_rag_chain
from .conversational_chain import create_conversational_rag_chain
from .memory_manager import load_history_from_request, ChatHistoryItem

class ChatbotService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        if not os.path.exists(settings.FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{settings.FAISS_INDEX_PATH}'. "
                "Please run 'create_vector_store.py' first."
            )
        self.vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})

        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY, 
            temperature=0.1,
        )
        
        self.rag_chain = create_rag_chain(self.retriever, self.llm)
        self.conversational_rag_chain = create_conversational_rag_chain(self.llm, self.retriever)
        
    def ask(self, query: str) -> dict:
        if not query:
            return {
                "answer": "Vui lòng đặt một câu hỏi",
                "source_documents": []
            }
            
        response_dict = self.rag_chain.invoke(query)
        source_docs_metadata = [doc.metadata for doc in response_dict.get("source_documents",[])]
        response_dict["source_documents"] = source_docs_metadata
        
        return response_dict
    
    def ask_with_memory(self, query: str, session_id: str, chat_history_from_request: list = None) -> dict:
        if not query:
            return {"answer": "Vui lòng đặt một câu hỏi", "source_documents": []}
        if chat_history_from_request:
            history_items = [ChatHistoryItem(type=item['type'], content=item['content']) for item in chat_history_from_request]
            load_history_from_request(session_id, history_items)
            
        response_dict = self.conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        source_docs_metadata = [doc.metadata for doc in response_dict.get("context", [])]
        
        return {
            "answer": response_dict.get("answer"),
            "source_documents": source_docs_metadata
        }

try:
    chatbot_instance = ChatbotService()
except Exception as e:
    print(f"FATAL: Failed to initialize ChatbotService: {e}")
    chatbot_instance = None