import os
import json
import fitz #pymupdf
from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict

DOCS_DIR   = os.path.join(os.path.dirname(__file__), "docs")
STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store.pkl")
CHUNK_SIZE    = 500 #chars per chunk
CHUNK_OVERLAP = 50 #char overlap between chunks

#extract text from pdfs
def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"text": text, "page": page_num})
    doc.close()
    return pages
  
def chunk_text(text: str, source: str, page: int) -> List[Dict]:
    """
    Split text into overlapping chunks using a sentence-boundary-aware
    sliding window. Chunk size 500 chars balances retrieval precision vs
    context richness. Overlap 50 preserves context across boundaries.
    """
    chunks = []
    text = " ".join(text.split())  # normalize whitespace
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # break sentences at ". " for better boundaries
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > CHUNK_SIZE // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append({
            "text": chunk.strip(),
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        })
        start = end - CHUNK_OVERLAP
        chunk_index += 1

    return chunks
  
def ingest_documents():
    print("=" * 60)
    print("SWS AI RAG Chatbot — Document Ingestion Pipeline")
    print("=" * 60)

    # processing pdfs
    print(f"\n[1/3] Processing PDFs from: {DOCS_DIR}")
    pdf_files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")])

    if not pdf_files:
        print("ERROR: No PDF files found in /docs!")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCS_DIR, pdf_file)
        doc_name = (pdf_file
                    .replace("SWS-AI-", "")
                    .replace("-", " ")
                    .replace(".pdf", "")
                    .title())

        pages = extract_text_from_pdf(pdf_path)
        doc_chunks = []
        for page_data in pages:
            chunks = chunk_text(page_data["text"], source=doc_name, page=page_data["page"])
            doc_chunks.extend(chunks)

        all_chunks.extend(doc_chunks)
        print(f"  ✓ {doc_name:35s} → {len(pages)} pages, {len(doc_chunks)} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # building bm25 index
    print(f"\n[2/3] Building BM25 retrieval index...")
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  BM25 index built ✓  ({len(tokenized_corpus)} documents)")

    # saving index and chunks
    print(f"\n[3/3] Saving vector store to: {STORE_PATH}")
    store = {
        "chunks": all_chunks,
        "bm25": bm25,
    }
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)
    print(f"  Saved ✓  ({os.path.getsize(STORE_PATH) / 1024:.1f} KB)")

    print(f"\n{'=' * 60}")
    print(f"✅ Ingestion complete! {len(all_chunks)} chunks indexed.")
    print(f"\n👉 Now run: uvicorn main:app --reload")
    print("=" * 60)

    # sanity check
    print("\n[Sanity Check] 'What is the leave policy?'")
    query_tokens = "what is the leave policy".split()
    scores = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    for rank, idx in enumerate(top_idx, 1):
        c = all_chunks[idx]
        print(f"  #{rank} | {c['source']} p{c['page']} | score={scores[idx]:.2f} | {c['text'][:90]}...")
    print("\n✅ Retrieval working correctly!")

if __name__ == "__main__":
    ingest_documents()