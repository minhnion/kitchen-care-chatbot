import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
FAISS_INDEX_PATH = "faiss_index_products"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

class ChatbotService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Please run 'create_vector_store.py' first."
            )
        self.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
        
        print(f"Loading LLM: {LLM_MODEL_NAME}")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=google_api_key, 
            temperature=0.1,
        )
        
        rag_template = """
        Bạn là một trợ lý ảo tư vấn sản phẩm cho cửa hàng 'Kitchen Care'.
        Hãy trả lời câu hỏi của người dùng chỉ dựa trên các thông tin sản phẩm được cung cấp dưới đây.
        Nếu thông tin không đủ để trả lời, hãy nói rằng "Để biết thêm thông tin chi tiết vui lòng liên lạc với đội ngũ hỗ trợ trên trang web."
        Không được tự bịa ra thông tin. Trả lời một cách thân thiện và chuyên nghiệp.

        Thông tin sản phẩm (Context):
        {context}

        Câu hỏi của người dùng (Question):
        {question}

        Câu trả lời của bạn:
        """
        self.rag_prompt = PromptTemplate.from_template(rag_template)

        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
    def ask(self, query: str) -> str:
        if not query:
            return "Vui lòng đặt một câu hỏi."
        bot_response = self.rag_chain.invoke(query)
        return bot_response

try:
    chatbot_instance = ChatbotService()
except Exception as e:
    print(f"FATAL: Failed to initialize ChatbotService: {e}")
    chatbot_instance = None