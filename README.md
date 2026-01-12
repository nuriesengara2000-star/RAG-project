# Hybrid RAG Application (Project 10)

This project is an upgraded RAG (Retrieval-Augmented Generation) application.
It uses **hybrid search**: keyword search + semantic search.
The results are merged using **Reciprocal Rank Fusion (RRF)**.

---

## Project Description

The application allows the user to:
- upload a text document,
- ask questions about the document,
- get answers based only on the document content.

The system does not hallucinate and uses retrieved text as context.

---

## How the System Works

1. The document is split into small text chunks.
2. Each chunk is converted into an embedding and stored in Supabase.
3. When the user asks a question:
   - Keyword search finds exact words (Full-Text Search).
   - Semantic search finds similar meaning using vectors.
4. The results are combined using RRF.
5. The best results are sent to the language model to generate the answer.

---

## Technologies Used

- Python
- Supabase (PostgreSQL)
- pgvector
- Full-Text Search (tsvector)
- OpenAI API

---

## Project Files
├─ ingest.py # load document and create embeddings
├─ chat.py # hybrid search + RRF + chat
├─ chunking.py # text chunking
├─ doc.txt # example document
├─ README.md

How to run
1. Create a Supabase project and enable pgvector.
2. Create the documents table and RPC search function.
3. Set environment variables in `.env`.
4. Run:
   ```bash
   python ingest.py doc.txt
   python chat.py
