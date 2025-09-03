import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from .conversational_chain import create_conversational_rag_chain, create_general_response_chain
from .memory_manager import get_chat_memory, load_history_from_request, ChatHistoryItem
from .retriever_chain import create_query_transformation_chain, create_advanced_retriever
from .intent_router import create_intent_router_chain

class ChatbotService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.0
        )

        if not os.path.exists(settings.FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at '{settings.FAISS_INDEX_PATH}'.")
            
        self.vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True
        )
        
        self.advanced_retriever = create_advanced_retriever(self.vector_store)
        self.query_transform_chain = create_query_transformation_chain(self.llm)
        self.generation_chain = create_conversational_rag_chain(self.llm)
        self.intent_router_chain = create_intent_router_chain(self.llm)
        self.general_response_chain = create_general_response_chain(self.llm)
        
        print("Multi-route ChatbotService initialized successfully.")

    def _invoke_product_rag_pipeline(self, query: str, chat_history_str: str) -> dict:
        """Pipeline chuyên dụng để xử lý các câu hỏi về sản phẩm."""
                
        # Bước 1: Biến đổi câu hỏi
        transformed_question = self.query_transform_chain.invoke({
            "question": query,
            "chat_history": chat_history_str
        })
        print(f"Original question: '{query}'")
        print(f"Transformed question: '{transformed_question}'")

        # Bước 2: Truy xuất tài liệu
        retrieved_docs = self.advanced_retriever.invoke(transformed_question)
        
        if not retrieved_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn trong cơ sở dữ liệu.",
                "source_documents": []
            }

        # Bước 3: Sinh câu trả lời
        answer = self.generation_chain.invoke({
            "question": query, 
            "context": retrieved_docs
        })

        unique_contexts = {doc.metadata.get("product_id"): doc.metadata for doc in retrieved_docs if doc.metadata.get("product_id")}
        source_documents = list(unique_contexts.values())

        return {
            "answer": answer,
            "source_documents": source_documents
        }

    def _invoke_general_pipeline(self, query: str, memory):
        """Pipeline chuyên dụng để xử lý chào hỏi và lạc đề."""
        answer = self.general_response_chain.invoke({
            "input": query,
            "chat_history": memory.messages
        })
        return {"answer": answer, "source_documents": []}

    def ask(self, query: str, session_id: str, chat_history_from_request: list = None) -> dict:
        if not query:
            return {"answer": "Vui lòng đặt một câu hỏi", "source_documents": []}
        
        memory = get_chat_memory(session_id)
        if chat_history_from_request:
            history_items = [ChatHistoryItem(**item) for item in chat_history_from_request]
            load_history_from_request(session_id, history_items)
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in memory.messages])

        intent_result = self.intent_router_chain.invoke({"query": query})
        intent = intent_result.intent
        print(f"Detected user intent: {intent}")

        if intent == "product_inquiry":
            response = self._invoke_product_rag_pipeline(query, chat_history_str)
        else: 
            response = self._invoke_general_pipeline(query, memory)

        memory.add_user_message(query)
        memory.add_ai_message(response.get("answer", ""))
        
        return response

try:
    chatbot_instance = ChatbotService()
except Exception as e:
    print(f"FATAL: Failed to initialize ChatbotService: {e}")
    import traceback
    traceback.print_exc()
    chatbot_instance = None