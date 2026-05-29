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
    
# Core RAG
def retrieve_chunks(question: str, k: int = RETRIEVAL_K):
    if bm25 is None:
        raise RuntimeError("Vector store not loaded. Run `python ingest.py` first.")

    tokens = question.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        {
            "text":   chunks[i]["text"],
            "source": chunks[i]["source"],
            "page":   chunks[i]["page"],
            "score":  float(scores[i]),
        }
        for i in top_indices
    ]


def build_prompt(question: str, retrieved: list) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are the SWS AI internal HR assistant. Your job is to help employees find accurate answers about company policies, benefits, and procedures.

INSTRUCTIONS:
- Answer ONLY using the context provided below.
- Be helpful, clear, and concise.
- Cite which document your answer comes from (e.g. "According to the Leave Policy...").
- If the answer is not found in the context, say exactly: "I don't have that information in the company documents."
- Do NOT use outside knowledge or make up information.
- Format lists and structured data clearly when helpful.

CONTEXT FROM COMPANY DOCUMENTS:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:"""


def generate_answer(question: str, retrieved: list) -> str:
    prompt = build_prompt(question, retrieved)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content
  
# endpoints
@app.get("/")
async def serve_ui():
    index_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SWS AI RAG Chatbot API is running."}


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if chunks else "degraded",
        "retrieval": "BM25 (rank-bm25)",
        "chunks_loaded": len(chunks) if chunks else 0,
        "retrieval_k": RETRIEVAL_K,
        "llm": "llama-3.3-70b-versatile (Groq)",
        "vector_db": "BM25 index (pickle)",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")

    try:
        retrieved = retrieve_chunks(question, k=RETRIEVAL_K)
        answer = generate_answer(question, retrieved)

        # Deduplicate sources
        seen, unique_sources = set(), []
        for c in retrieved:
            if c["source"] not in seen:
                seen.add(c["source"])
                unique_sources.append(c["source"])

        source_chunks = [
            SourceChunk(
                source=c["source"],
                page=c["page"],
                snippet=c["text"][:150] + ("..." if len(c["text"]) > 150 else "")
            )
            for c in retrieved[:3]
        ]

        return ChatResponse(answer=answer, sources=unique_sources, source_chunks=source_chunks)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if "auth" in str(e).lower() or "api key" in str(e).lower():
            raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY.")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")