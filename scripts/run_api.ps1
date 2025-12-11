# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run FastAPI server
uvicorn src.app:app --reload --port 8000
