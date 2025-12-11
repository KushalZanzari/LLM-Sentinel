from fastapi import FastAPI
from pydantic import BaseModel
from src.main import evaluate

app = FastAPI()

class EvalRequest(BaseModel):
    chat_path: str
    ctx_path: str

@app.post("/evaluate")
def eval_endpoint(req: EvalRequest):
    result = evaluate(req.chat_path, req.ctx_path)
    return result
