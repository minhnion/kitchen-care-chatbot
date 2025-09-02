from langchain.docstore.document import Document
from app.config import settings
import locale
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'vi_VN')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def _create_product_summary(product: dict) -> str:
    """Create a large summary full context of the product."""
    price = product.get("price", 0)
    discount = product.get("discountPercent", 0)
    star_average = product.get('starAverage', 0)
    num_reviews = product.get('numberOfReviews', 0)
    final_price = price * (1 - discount / 100) if discount > 0 else price
    
    price_text = f"Giá niêm yết là {locale.format_string('%d', price, grouping=True)} đồng."
    
    if discount > 0:
        price_text += f" Đang giảm giá {discount}%, giá chỉ còn {locale.format_string('%d', final_price, grouping=True)} đồng."
    
    rating_text = f"Sản phẩm được đánh giá trung bình {star_average}/5 sao từ {num_reviews} lượt đánh giá." if num_reviews > 0 else "Sản phẩm này chưa có đánh giá."

    specs_text = ", ".join([f"{spec.get('key', '')}: {spec.get('value', '')}" for spec in product.get('specifications', [])])
    
    summary = (
        f"Sản phẩm {product.get('name', '')} của thương hiệu {product.get('brandName')} "
        f"thuộc danh mục {product.get('categoryName')} > {product.get('subCategoryName')}. "
        f"{price_text} {rating_text} "
        f"Mô tả: {product.get('description', '')}. "
        f"Thông số kỹ thuật: {specs_text}."
    )
    return " ".join(summary.split())

def create_documents_for_indexing(products: list) -> list[Document]:
    """Create documents for indexing from product data."""
    all_documents  = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=[". ", ".\n", "\n\n", "\n", " ", ""]
    )
    
    for product in products:
        summary_context  = _create_product_summary(product)
        small_chunks  = text_splitter.split_text(summary_context)
        
        for chunk in small_chunks:
            metadata = {
                "product_id": str(product.get('_id')),
                "name": product.get('name'),
                "slug": product.get('slug'),
                "main_image": product.get('mainImage'),
                "price": product.get('price'),
                "original_context": summary_context, 
            }

            doc = Document(page_content=chunk, metadata=metadata)
            all_documents.append(doc)
    
    return all_documents