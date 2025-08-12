from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from .memory_manager import get_chat_memory

def create_conversational_rag_chain(llm: BaseChatModel, retriever: VectorStoreRetriever):
    
    # 1. Chain để tạo câu hỏi độc lập (History-Aware Retriever)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt= ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 2. Chain để trả lời câu hỏi dựa trên context
    qa_system_prompt = """
    Bạn là một trợ lý ảo tư vấn sản phẩm cho cửa hàng 'Kitchen Care'.
    Hãy trả lời câu hỏi của người dùng chỉ dựa trên các thông tin sản phẩm được cung cấp dưới đây (Context).
    Nếu thông tin không đủ để trả lời, hãy nói rằng "Xin lỗi, tôi không tìm thấy thông tin liên quan."
    Không được tự bịa ra thông tin. Trả lời một cách thân thiện và chuyên nghiệp.

    Context:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Kết hợp 2 chain trên thành một retrieval chain hoàn chỉnh
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_memory,  # Sử dụng hàm quản lý bộ nhớ đã tách riêng
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain