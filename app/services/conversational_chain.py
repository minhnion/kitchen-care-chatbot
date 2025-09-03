from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder

def create_conversational_rag_chain(llm: BaseChatModel):
    qa_system_prompt = """
    Bạn là một trợ lý ảo tư vấn sản phẩm chuyên nghiệp cho cửa hàng 'Kitchen Care'.
    Nhiệm vụ của bạn là trả lời **câu hỏi gốc của người dùng** dựa trên thông tin được cung cấp trong "Ngữ cảnh sản phẩm".
    Hãy tuân thủ các quy tắc sau:
    1.  Đọc kỹ **câu hỏi gốc** và ngữ cảnh để hiểu ý định của người dùng.
    2.  Nếu ngữ cảnh chứa đủ thông tin, hãy tổng hợp một câu trả lời mạch lạc và hữu ích.
    3.  Luôn trích dẫn nguồn cho các sản phẩm bạn đề cập bằng cách dùng `product_id`.
    4.  Nếu ngữ cảnh không chứa thông tin liên quan đến câu hỏi gốc, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin phù hợp với yêu cầu của bạn."
    5.  KHÔNG được bịa đặt thông tin.

    Ngữ cảnh sản phẩm:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "Câu hỏi gốc của người dùng: {question}\n\nCâu trả lời của bạn:"),
    ])
    
    return qa_prompt | llm | StrOutputParser()

def create_general_response_chain(llm: BaseChatModel):
    """
    Tạo một chain để xử lý các câu chào hỏi hoặc lạc đề.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là một trợ lý ảo thân thiện và chuyên nghiệp của cửa hàng thương mại điện tử 'Kitchen Care'. "
                "Hãy trả lời trực tiếp và ngắn gọn tin nhắn của người dùng. "
                "Nếu người dùng hỏi một câu hỏi lạc đề, hãy lịch sự nói rằng bạn chỉ có thể trả lời các câu hỏi về sản phẩm nhà bếp. "
                "Luôn giữ giọng điệu hữu ích và lịch sự. Trả lời bằng tiếng Việt."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    return prompt | llm | StrOutputParser()