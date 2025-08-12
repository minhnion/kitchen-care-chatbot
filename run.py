import uvicorn
import os

if __name__ == "__main__":
    try:
        uvicorn.run(
            "app.main:app", 
            host=os.getenv("HOST", "127.0.0.1"), 
            port=int(os.getenv("PORT", 8000)),
            reload=True
        )
    except Exception as e:
        print(f"An error occurred while trying to run Uvicorn: {e}")