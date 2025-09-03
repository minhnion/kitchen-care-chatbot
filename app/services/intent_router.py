from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from typing import Literal

Intent = Literal["greeting", "product_inquiry", "category_inquiry", "off_topic"]

class UserIntent(BaseModel):
    intent: Intent = Field(description="Phân loại ý định của người dùng.")

def create_intent_router_chain(llm: BaseChatModel):
    parser = PydanticOutputParser(pydantic_object=UserIntent)
    prompt = PromptTemplate(
        template=(
            "Bạn là một chuyên gia phân loại ý định của người dùng cho chatbot của cửa hàng 'Kitchen Care'.\n"
            "Hãy phân tích tin nhắn của người dùng và phân loại nó vào một trong các danh mục sau:\n"
            "1. 'greeting': Đối với lời chào (xin chào, chào bạn), lời tạm biệt, hoặc lời cảm ơn đơn giản.\n"
            "2. 'category_inquiry': Nếu người dùng hỏi về các loại sản phẩm chung chung, các danh mục mà cửa hàng có (ví dụ: 'cửa hàng bán gì?', 'có những loại nào?', 'ngoài X ra còn gì nữa không?').\n"
            "3. 'product_inquiry': Nếu người dùng hỏi về sản phẩm cụ thể, thương hiệu, tính năng, giá cả của một sản phẩm hoặc một loại sản phẩm xác định (ví dụ: 'bếp từ Bosch', 'máy rửa bát giá rẻ').\n"
            "4. 'off_topic': Đối với bất kỳ câu hỏi nào khác không phải là lời chào và không liên quan đến sản phẩm nhà bếp.\n\n"
            "{format_instructions}\n\n"
            "Tin nhắn của người dùng:\n{query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser