RAG Application with Supabase and pgvector

This project implements a simple Retrieval-Augmented Generation (RAG) application.

Description
The application allows users to upload a text document, store its chunks as vector embeddings in Supabase (PostgreSQL with pgvector),
and ask questions that are answered based on the document content.

Tech Stack
- Python
- Supabase (PostgreSQL)
- pgvector
- OpenAI API

Project Structure
- ingest.py — loads and processes documents (chunking + embeddings)
- chat.py — chat interface using retrieval and generation
- chunking.py — text chunking logic

How it works
1. A document is loaded and split into chunks.
2. Each chunk is converted into an embedding and stored in Supabase.
3. A semantic search retrieves relevant chunks using cosine similarity.
4. The LLM generates an answer based only on the retrieved context.

How to run
1. Create a Supabase project and enable pgvector.
2. Create the documents table and RPC search function.
3. Set environment variables in `.env`.
4. Run:
   ```bash
   python ingest.py doc.txt
   python chat.py
