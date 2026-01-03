from __future__ import annotations
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

from chunking import simple_chunk

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/file.txt")
        sys.exit(1)

    load_dotenv()

    file_path = sys.argv[1]
    doc_name = os.path.basename(file_path)

    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_ROLE_KEY")
    openai_key = get_env("OPENAI_API_KEY")
    emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    sb = create_client(supabase_url, supabase_key)
    client = OpenAI(api_key=openai_key)

    text = read_txt(file_path)
    chunks = simple_chunk(text, max_chars=1200, overlap=150)
    print(f"Loaded: {doc_name} | chars={len(text)} | chunks={len(chunks)}")

    # Пакетная векторизация (быстрее)
    resp = client.embeddings.create(model=emb_model, input=chunks)
    vectors = [d.embedding for d in resp.data]

    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        rows.append({
            "doc_name": doc_name,
            "chunk_index": i,
            "content": chunk,
            "embedding": vec
        })

    # Insert пачкой
    result = sb.table("documents").insert(rows).execute()
    if getattr(result, "error", None):
        raise RuntimeError(result.error)

    print(f"Inserted rows: {len(rows)} ✅")

if __name__ == "__main__":
    main()
