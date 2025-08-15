from langchain.docstore.document import Document
from app.config import settings
import locale

try:
    locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'vi_VN')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def create_langchain_documents(products: list) -> list:
    documents = []
    for product in products:
        price = product.get('price', 0)
        discount = product.get('discountPercent', 0)
        
        star_average = product.get('starAverage', 0)
        num_reviews = product.get('numberOfReviews', 0)
        
        quantity_sold = product.get('quantitySold', 0)

        final_price = price * (1 - discount / 100) if discount > 0 else price
        
        price_text = f"Giá niêm yết của sản phẩm là {locale.format_string('%d', price, grouping=True)} đồng. "
        if discount > 0:
            price_text += f"Hiện đang có chương trình giảm giá {discount}%, giá chỉ còn {locale.format_string('%d', final_price, grouping=True)} đồng. "

        rating_text = ""
        if star_average > 0 and num_reviews > 0:
            rating_text = f"Sản phẩm này được người dùng đánh giá trung bình {star_average}/5 sao dựa trên {num_reviews} lượt đánh giá. "

        specs_text = ", ".join([f"{spec.get('key', '')}: {spec.get('value', '')}" for spec in product.get('specifications',[])])
        
        page_content = (
            f"Thông tin về sản phẩm '{product.get('name', '')}':\n"
            f"- Thương hiệu: {product.get('brandName', 'Chưa xác định')}.\n"
            f"- Danh mục: {product.get('categoryName', 'N/A')} > {product.get('subCategoryName', 'N/A')}.\n"
            f"- Giá cả: {price_text.strip()}\n"
            f"- Đánh giá: {rating_text.strip()}\n"
            f"- Mô tả chung: {product.get('description', 'Không có mô tả chi tiết.')}\n"
            f"- Thông số kỹ thuật: {specs_text}."
        )
        
        metadata = {
            "product_id": str(product.get('_id')),
            "name": product.get('name'),
            "price": price,
            "final_price": final_price,
            "discount": discount,
            "star_average": star_average,
            "number_of_reviews": num_reviews,
            "slug": product.get('slug'),
            "main_image": product.get('mainImage'),
            "source": settings.PRODUCT_COLLECTION
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    print(f"created {len(documents)} LangChain documents.")
    return documents