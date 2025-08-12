from langchain.docstore.document import Document
from app.config import settings

def create_langchain_documents(products: list) -> list:
    documents = []
    for product in products:
        #process specifications in product
        specs_text = ", ".join([f"{spec.get('key', '')}: {spec.get('value', '')}" for spec in product.get('specifications',[])])
        page_content = (
            f"Tên: {product.get('name', '')}. "
            f"Thương hiệu: {product.get('brandName', 'N/A')}. "
            f"Danh mục: {product.get('categoryName', 'N/A')} > {product.get('subCategoryName', 'N/A')}. "
            f"Mô tả: {product.get('description', '')}. "
            f"Thông số kỹ thuật: {specs_text}."
        )
        
        metadata = {
            "product_id": str(product.get('_id')),
            "name": product.get('name'),
            "price": product.get('price'),
            "slug": product.get('slug'),
            "main_image": product.get('mainImage'),
            "source": settings.PRODUCT_COLLECTION
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    print(f"created {len(documents)} LangChain documents.")
    return documents