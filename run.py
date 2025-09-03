import subprocess
import time
import os
import sys

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
APP_MODULE = "app.main:app"
UI_MODULE = "ui/app.py"

def run_server():
    command = [
        sys.executable, "-m", "uvicorn",
        APP_MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--reload"
    ]
    return subprocess.Popen(command)

def run_streamlit():
    time.sleep(10) 
    
    command = [
        sys.executable, "-m", "streamlit", "run",
        UI_MODULE,
        "--server.headless", "true" 
    ]
    return subprocess.Popen(command)

if __name__ == "__main__":
    server_process = None
    streamlit_process = None
    
    try:
        server_process = run_server()
        streamlit_process = run_streamlit()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n--- Received Ctrl+C. Shutting down processes. ---")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if server_process:
            print("Terminating FastAPI server...")
            server_process.terminate()
            server_process.wait() 
        
        if streamlit_process:
            print("Terminating Streamlit UI...")
            streamlit_process.terminate()
            streamlit_process.wait()

        print("--- All processes have been shut down. ---")