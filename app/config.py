from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8',extra='ignore')
    
    #MongoDB 
    MONGODB_URI: str
    DB_NAME: str = "kitchen-care"
    PRODUCT_COLLECTION: str = "products"
    
    #Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "gemini-1.5-flash-latest"
    GOOGLE_API_KEY: str
    
    #FAISS
    FAISS_INDEX_PATH: str = "faiss_index_products"
    
settings = Settings()
