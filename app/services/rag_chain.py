from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from app.utils import format_docs

def create_rag_chain(retriever: VectorStoreRetriever, llm: BaseChatModel) -> RunnableMap:
    rag_template = """
    Bạn là một trợ lý ảo tư vấn sản phẩm cho cửa hàng 'Kitchen Care'.
    Hãy trả lời câu hỏi của người dùng chỉ dựa trên các thông tin sản phẩm được cung cấp dưới đây.
    Nếu thông tin không đủ để trả lời, hãy nói rằng "Xin lỗi, tôi không tìm thấy thông tin liên quan."
    Không được tự bịa ra thông tin. Trả lời một cách thân thiện và chuyên nghiệp.

    Thông tin sản phẩm (Context):
    {context}

    Câu hỏi của người dùng (Question):
    {question}

    Câu trả lời của bạn:
    """
    rag_prompt = PromptTemplate.from_template(rag_template)
    
    context_chain = retriever | format_docs
    
    rag_chain = RunnableMap(
        {
            "source_documents": retriever,
            "answer": (
                {"context": context_chain, "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            ),
        }
    )
    return rag_chain