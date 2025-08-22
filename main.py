from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: PromptRequest):
    response = client.text_generation(
        req.prompt,
        max_new_tokens=200,
        temperature=0.7,
        stop_sequences=["\nUser:", "\nAI:"]
    )
    return {"response": response}
