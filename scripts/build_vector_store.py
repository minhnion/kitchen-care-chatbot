from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .data_loader import get_db_connection, fetch_populated_products
from .document_processor import create_documents_for_indexing
from app.config import settings

def build_and_save_vector_store(documents: list, index_path: str):
    """Build and save the FAISS vector store."""
    if not documents:
        print("No documents provided for building the vector store.")
        return
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(index_path)
    
    
def main():
    client = get_db_connection()
    if not client:
        return
    
    try:
        products = fetch_populated_products(client)
        if products:
            documents_to_index  = create_documents_for_indexing(products)
            build_and_save_vector_store(documents_to_index, settings.FAISS_INDEX_PATH)
    finally:
        print("Closing MongoDB connection.")
        client.close()
        
if __name__ == "__main__":
    main()