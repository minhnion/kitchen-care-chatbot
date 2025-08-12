from fastapi import FastAPI
from .routers import chatbot_router
from .config import settings

app = FastAPI(
    title="Kitchen Care Chatbot Service",
    description="API service for the e-commerce AI chatbot using RAG.",
    version="0.1.0",
)

print(f"MongoDB URI from config: {settings.MONGODB_URI}")

app.include_router(chatbot_router.router)

@app.get("/health", tags=["Health Check"])
def heath_check():
    return {"status": "ok", "message": "Chatbot service is up and running!"}