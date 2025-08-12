import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "kitchen-care"
PRODUCT_COLLECTION = "products"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index_products"

def get_db_connection():
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        return None

def fetch_populated_products(client: MongoClient):
    db = client[DB_NAME]
    products_collection = db[PRODUCT_COLLECTION]
    
    pipeline = [
        #Phase 1: Filter products with is not Delete
        {
            '$match': { 'isDelete': False }
        },
        
        #Phase 2: Join with parent collection
        
        {
            '$lookup': {
                'from': 'categories',
                'localField': 'category',
                'foreignField': '_id',
                'as': 'categoryDetails'
            }
        },
        
        {
            '$lookup': {
                'from': 'subcategories',
                'localField': 'subCategory',
                'foreignField': '_id',
                'as': 'subCategoryDetails'
            }
        },
        
        {
            '$lookup': {
                'from': 'brands',
                'localField': 'brand',
                'foreignField': '_id',
                'as': 'brandDetails'
            }
        },
        
        #Phase 3: Re format output
        {
            '$project': {
                '_id': 1, 'name': 1, 'description': 1, 'price': 1, 
                'specifications': 1, 'slug': 1, 'mainImage': 1,
                
                'categoryName': { '$arrayElemAt': ['$categoryDetails.name', 0]},
                'subCategoryName': { '$arrayElemAt': ['$subCategoryDetails.name', 0] },
                'brandName': { '$arrayElemAt': ['$brandDetails.name', 0] },
            }
        }
    ]
    
    populated_products = list(products_collection.aggregate(pipeline))
    print(f"Found and populated {len(populated_products)} products")
    
    return populated_products

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
            "product_id": str(product.get('id')),
            "name": product.get('name'),
            "price": product.get('price'),
            "slug": product.get('slug'),
            "main_image": product.get('mainImage'),
            "source": PRODUCT_COLLECTION
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    print(f"created {len(documents)} LangChain documents.")
    return documents

def build_and_save_vector_store(documents: list):
    if not documents:
        return
    
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print("FAISS index saved successfully.")
    
def main():
    client = get_db_connection()
    if not client:
        return
    
    try:
        products = fetch_populated_products(client)
        if(products):
            documents = create_langchain_documents(products)
            build_and_save_vector_store(documents)
    finally:
        print("Closing MongoDB connection.")
        client.close()
        
if __name__ == "__main__":
    main()