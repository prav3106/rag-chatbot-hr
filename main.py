import os
import pickle
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# config
STORE_PATH       = os.path.join(os.path.dirname(__file__), "vector_store.pkl")
RETRIEVAL_K      = 5
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

# app setup
app = FastAPI(title="SWS AI RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# loading vector store
print("[Startup] Loading vector store...")
try:
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    chunks = store["chunks"]
    bm25   = store["bm25"]
    print(f"[Startup] Loaded {len(chunks)} chunks from vector store ✓")
except FileNotFoundError:
    print("[Startup] ERROR: vector_store.pkl not found. Run `python ingest.py` first.")
    chunks, bm25 = None, None

groq_client = Groq(api_key=GROQ_API_KEY)
print("[Startup] Groq client ready ✓")

# request-response models
class ChatRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    source: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    source_chunks: List[SourceChunk]
    
