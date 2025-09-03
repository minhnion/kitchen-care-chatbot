from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from typing import List

def create_query_transformation_chain(llm: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Bạn là một trợ lý AI chuyên viết lại câu hỏi của người dùng để tối ưu hóa cho việc tìm kiếm ngữ nghĩa trong một cơ sở dữ liệu sản phẩm.\n"
         "Dựa vào lịch sử chat và câu hỏi mới, hãy tạo ra một câu hỏi độc lập, tập trung vào các từ khóa chính về sản phẩm, tính năng hoặc danh mục.\n"
         "Mục tiêu là tạo ra một câu hỏi tốt nhất để tìm kiếm các sản phẩm liên quan, không phải để LLM trả lời trực tiếp.\n"
         "Ví dụ 1:\n"
         "Lịch sử chat: (rỗng)\n"
         "Câu hỏi mới: Mình cần tư vấn mua sắm về sản phẩm bếp từ\n"
         "Câu hỏi đã tối ưu hóa: thông tin về các sản phẩm bếp từ\n"
         "Ví dụ 2:\n"
         "Lịch sử chat: Human: Bạn có bếp từ của Bosch không?\n"
         "Câu hỏi mới: Thế còn loại 4 vùng nấu thì sao?\n"
         "Câu hỏi đã tối ưu hóa: thông tin về bếp từ Bosch 4 vùng nấu"
        ),
        ("human", 
         "Lịch sử chat:\n{chat_history}\n\nCâu hỏi mới: {question}\n\nCâu hỏi đã tối ưu hóa:")
    ])
    
    chain = prompt | llm | (lambda msg: msg.content)
    return chain

def create_advanced_retriever(vector_store: FAISS) -> Runnable:
    """
    Tạo ra một retriever tiên tiến kết hợp tìm kiếm vector và re-ranking.
    """
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 20, 'fetch_k': 50} 
    )

    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=4) 

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever
    
        