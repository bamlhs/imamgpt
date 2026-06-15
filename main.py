from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from model import generate_answer
from pdf_processor import search_relevant_text

class ChatRequest(BaseModel):
    input_text: str

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ImamGPT API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    context = search_relevant_text(request.input_text)
    response = generate_answer(request.input_text, context)
    return {"response": response}
subdomain = "imamgpt"

# Allow FastAPI to run inside Google Colab
nest_asyncio.apply()
ngrok_tunnel = ngrok.connect(8000)

print(f"Public API URL: {ngrok_tunnel.public_url}")

# Start FastAPI Server
uvicorn.run(app, host="0.0.0.0", port=8000)
