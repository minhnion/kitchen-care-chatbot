from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .data_loader import get_db_connection, fetch_populated_products
from .document_processor import create_langchain_documents
from app.config import settings

def build_and_save_vector_store(documents: list):
    if not documents:
        return
    
    embeddings = HuggingFaceEmbeddings(model_name = settings.EMBEDDING_MODEL_NAME)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(settings.FAISS_INDEX_PATH)
    print("FAISS index saved successfully.")
    
def main():
    client = get_db_connection()
    if not client:
        return
    
    try:
        products = fetch_populated_products(client)
        if products:
            documents = create_langchain_documents(products)
            build_and_save_vector_store(documents)
    finally:
        print("Closing MongoDB connection.")
        client.close()
        
if __name__ == "__main__":
    main()