# RAG Application (Projects 9, 10, 11)

This project is a RAG (Retrieval-Augmented Generation) application.
It answers questions using the content of uploaded documents.

The project was done step by step:
- **Project 9:** basic RAG with Supabase + pgvector (semantic search)
- **Project 10:** hybrid search (keyword + semantic) + RRF fusion
- **Project 11:** re-ranking + cache + timing logs

---

## Project 9 (Basic RAG)

### What it does
- Load a text file
- Split text into chunks
- Create embeddings for each chunk
- Store chunks + embeddings in Supabase
- Search by meaning (semantic search) using pgvector
- Generate an answer only from retrieved context (no hallucinations)

---

## Project 10 (Hybrid Search + RRF)

### Added features
- **Keyword search** using PostgreSQL Full-Text Search (tsvector / tsquery)
- **Semantic search** using pgvector embeddings
- **RRF (Reciprocal Rank Fusion)** to merge keyword + semantic results

RRF helps the system work well for:
- technical queries (exact words, error codes)
- normal questions (meaning-based)

---

## Project 11 (Re-rank + Cache + Logs)

### Added features
- **Re-ranking:** after retrieval (hybrid + RRF), the system reorders top-K chunks and keeps top-N best chunks.
- **Cache (LRU + TTL):** repeated questions return the saved answer fast.
- **Logs:** shows CACHE HIT/MISS and timing for retrieval / rerank / generation.

Example behavior:
- First same query → `CACHE MISS` + rerank runs
- Second same query → `CACHE HIT` (fast answer)

---

## How the system works (simple)

1. The document is split into small chunks.
2. Each chunk is converted into an embedding and stored in Supabase.
3. User asks a question:
   - keyword search finds exact words (FTS)
   - semantic search finds similar meaning (vectors)
4. Results are merged with RRF.
5. (Project 11) Re-ranking selects the best chunks.
6. The final chunks are sent to the LLM to generate the answer.

---

## Technologies used
- Python
- Supabase (PostgreSQL)
- pgvector
- Full-Text Search (tsvector)
- OpenAI API

---

## Project files
Task9/
├─ ingest.py # load document, chunking, embeddings, insert to Supabase
├─ chat.py # hybrid retrieval + RRF + rerank + cache + chat
├─ chunking.py # text chunking logic
├─ doc.txt # example document
├─ README.md
└─ .gitignore


---

## Environment variables (.env)

Required:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`

Optional (Projects 10–11):
- `RETRIEVAL_K=10`
- `RERANK_N=5`
- `RERANK_ENABLED=true`
- `RERANK_MODEL=gpt-4o-mini`
- `CACHE_ENABLED=true`
- `CACHE_TTL_SECONDS=600`
- `CACHE_MAX_ITEMS=200`
- `DEBUG_LOGS=true`

---

## How to run

1) Create a Supabase project and enable `pgvector`.

2) Run SQL in Supabase SQL Editor:
- create `documents` table
- create `match_documents` function (semantic search)
- create FTS column + index + `keyword_search_documents` function (keyword search)

3) Install packages and run:

bash
python ingest.py doc.txt
python chat.py
Tests (examples)
Query: Ошибка connection_timeout

Keyword search finds exact match

RRF + rerank puts best chunk on top

Query: Почему у меня не грузится страница?

Semantic search works better

System answers only if context exists

Security
API keys are stored in .env and are not committed to GitHub.
