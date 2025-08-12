from fastapi import FastAPI
from dotenv import load_dotenv
import os

from .routers import chatbot_router

load_dotenv()

app = FastAPI(
    title="Kitchen Care Chatbot Service",
    description="API service for the e-commerce AI chatbot using RAG.",
    version="0.1.0",
)

print(f"MongoDB URI from env: {os.getenv('MONGODB_URI')}")

app.include_router(chatbot_router.router)

@app.get("/health", tags=["Health Check"])
def heath_check():
    return {"status": "ok", "message": "Chatbot service is up and running!"}