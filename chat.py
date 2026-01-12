from __future__ import annotations
import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

def rrf_fusion(keyword_results, semantic_results, k: int = 60):

    fused = {}

    def add_list(results, source_name: str):
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            if doc_id not in fused:
                fused[doc_id] = {"doc": doc, "score": 0.0, "sources": set()}
            fused[doc_id]["score"] += 1.0 / (rank + k)
            fused[doc_id]["sources"].add(source_name)

    add_list(keyword_results, "keyword")
    add_list(semantic_results, "semantic")

    merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return merged


def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def build_prompt(context: str, question: str) -> str:
    return f"""Ты помощник. Отвечай ТОЛЬКО на основе контекста.
Если в контексте нет ответа — скажи: "В документе нет информации об этом".

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""

def main():
    load_dotenv()

    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_ROLE_KEY")
    openai_key = get_env("OPENAI_API_KEY")
    emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # модель чата можешь поменять
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    sb = create_client(supabase_url, supabase_key)
    client = OpenAI(api_key=openai_key)

    print("RAG Chat. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # 1) embedding вопроса
        q_emb = client.embeddings.create(model=emb_model, input=q).data[0].embedding


        # 2) semantic search (top-5)
        sem_rpc = sb.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_threshold": 0.15,  # 0.15–0.25 обычно нормально
            "match_count": 5
        }).execute()
        semantic_results = sem_rpc.data or []

        # 3) keyword search (top-5)
        kw_rpc = sb.rpc("keyword_search_documents", {
            "query_text": q,
            "match_count": 5
        }).execute()
        keyword_results = kw_rpc.data or []

        # Если вообще ничего не нашли
        if not semantic_results and not keyword_results:
            print("Assistant: В документе нет информации об этом.\n")
            continue

        # 4) RRF fusion
        fused = rrf_fusion(keyword_results, semantic_results, k=60)

        # 5) финальный топ для контекста
        top_fused = fused[:6]

                # --- DEBUG / REPORT OUTPUT 
        print("\n--- KEYWORD TOP-5 ---")
        for i, d in enumerate(keyword_results[:5], 1):
            print(i, d["doc_name"], d["chunk_index"], f"kw_rank={d.get('kw_rank', 0):.4f}")

        print("\n--- SEMANTIC TOP-5 ---")
        for i, d in enumerate(semantic_results[:5], 1):
            print(i, d["doc_name"], d["chunk_index"], f"sim={d.get('similarity', 0):.4f}")

        print("\n--- RRF TOP-3 ---")
        for i, x in enumerate(fused[:3], 1):
            d = x["doc"]
            print(i, d["doc_name"], d["chunk_index"], f"rrf={x['score']:.4f}", "sources=", ",".join(sorted(x["sources"])))
        print()
        # --- END DEBUG ---


        context = "\n\n---\n\n".join(
            [
                f"[{x['doc']['doc_name']} | chunk {x['doc']['chunk_index']} | sources={','.join(sorted(x['sources']))} | rrf={x['score']:.4f}]\n"
                f"{x['doc']['content']}"
                for x in top_fused
            ]
        )



        # 3) генерация ответа
        prompt = build_prompt(context, q)
        completion = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "Ты аккуратный помощник по документам."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        answer = completion.choices[0].message.content
        print(f"Assistant: {answer}\n")

if __name__ == "__main__":
    main()
