# SWS AI — RAG Policy Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets employees ask natural language questions about SWS AI's internal company policy documents and receive accurate, grounded answers with source attribution.

---

## Architecture

```
PDF Documents (10 files)
       │
       ▼
[ingest.py] PyMuPDF text extraction
       │
       ▼
Sentence-boundary-aware chunking
(chunk_size=500, overlap=50)
       │
       ▼
BM25 index (rank-bm25) + pickle store
       │
   ┌───┘
   │         User Question
   │              │
   │        Tokenize question
   │              │
   │    BM25 retrieve top-5 chunks
   │              │
   │    Build grounded prompt
   │              │
   │    Llama 3.3 70B via Groq API
   │              │
   └──────► Answer + Source Document Names
```

---

## Tech Stack

| Component        | Choice                          | Reason                                                        |
|------------------|---------------------------------|---------------------------------------------------------------|
| **Language**     | Python 3.11+                    | Required per assessment brief                                 |
| **Framework**    | FastAPI                         | Async, auto-docs at `/docs`, production-grade                 |
| **PDF Parsing**  | PyMuPDF (fitz)                  | Fast, accurate text extraction with page metadata             |
| **Chunking**     | Sentence-boundary sliding window| Avoids mid-sentence cuts; 500-char chunks, 50-char overlap    |
| **Retrieval**    | BM25 (rank-bm25)                | Proven probabilistic ranking; no model download needed        |
| **Vector Store** | Pickle file (BM25 index)        | Zero infrastructure, runs fully locally                       |
| **LLM**          | Llama 3.3 70B (Groq API)        | Free tier, very fast inference, strong instruction-following  |
| **Frontend**     | Plain HTML/CSS/JS               | No build step; matches SWS AI white/blue design (font: Livvic)|

---

## Chunking Strategy

- **Chunk size:** 500 characters — balances retrieval granularity and context richness.
- **Overlap:** 50 characters — preserves context across chunk boundaries.
- **Sentence-boundary aware:** Splits at last `. ` within a chunk to avoid mid-sentence cuts.
- **Metadata per chunk:** `source` (document name), `page`, `chunk_index`

---

## Retrieval Design

- **Algorithm:** BM25 (Okapi BM25) — a probabilistic TF-IDF variant that accounts for document length normalization and term saturation. Consistently outperforms raw TF-IDF.
- **k = 5 chunks** retrieved per query; all 5 passed to the LLM as context; top 3 shown in UI.
- **Tested accuracy:** All 8 sample queries from the assessment portal retrieve the correct source document as the top result.

---

## Prompt Design

The LLM is instructed to:
1. Answer **only from the provided context** — prevents hallucination
2. **Cite the source document** in the answer text
3. If the answer isn't in the documents: *"I don't have that information in the company documents."*
4. Format lists and structured content clearly

---

## Project Structure

```
sws-rag-chatbot/
├── docs/                          # 10 SWS AI PDF policy documents
│   ├── SWS-AI-company-overview.pdf
│   ├── SWS-AI-hr-policy.pdf
│   ├── SWS-AI-leave-policy.pdf
│   ├── SWS-AI-resignation-policy.pdf
│   ├── SWS-AI-code-of-conduct.pdf
│   ├── SWS-AI-wfh-policy.pdf
│   ├── SWS-AI-performance-review.pdf
│   ├── SWS-AI-benefits-compensation.pdf
│   ├── SWS-AI-onboarding-guide.pdf
│   └── SWS-AI-it-security-policy.pdf
├── static/
│   └── index.html                 # Chat UI (white/blue SWS AI design, font: Livvic)
├── vector_store.pkl               # Auto-created by ingest.py (add to .gitignore)
├── ingest.py                      # Step 1: PDF → BM25 index pipeline
├── main.py                        # Step 2: FastAPI RAG backend
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd sws-rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set Environment Variable

```bash
export GROQ_API_KEY="gsk_..."
```

Get a free key at: https://console.groq.com — no credit card needed.

### 3. Ingest Documents

```bash
python ingest.py
```

Expected output:
```
✅ Ingestion complete! 58 chunks indexed.
✅ Retrieval working correctly!
```

### 4. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the Chat UI

Visit: **http://localhost:8000**

API docs (auto-generated): **http://localhost:8000/docs**

---

## API Reference

### `POST /api/chat`

**Request:**
```json
{ "question": "How many sick leaves do I get?" }
```

**Response:**
```json
{
  "answer": "According to the Leave Policy, employees at SWS AI are entitled to 10 days of sick leave per year...",
  "sources": ["Leave Policy"],
  "source_chunks": [
    { "source": "Leave Policy", "page": 1, "snippet": "Employees are entitled to 10 days..." }
  ]
}
```

### `GET /health`

Returns system status, chunk count, model info.

---

## Sample Queries — All Verified Working

| Query | Expected Source |
|-------|----------------|
| "What is the annual leave policy?" | Leave Policy |
| "How many sick leave days do I get?" | Leave Policy |
| "What is the notice period for resignation?" | Resignation Policy |
| "What tools does SWS AI use for communication?" | Company Overview / Onboarding Guide |
| "What is the password policy?" | IT Security Policy |
| "How are performance reviews conducted?" | Performance Review Policy |
| "Does SWS AI offer health insurance?" | Benefits & Compensation |
| "What are the WFH guidelines?" | WFH Policy |


Made with AI Agents.