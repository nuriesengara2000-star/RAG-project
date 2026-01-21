from __future__ import annotations

import os
import time
import hashlib
from collections import OrderedDict

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI


# -----------------------------
# Cache (LRU + TTL)
# -----------------------------
class LRUCacheTTL:
    def __init__(self, max_items: int = 200, ttl_seconds: int = 600):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.store = OrderedDict()  # key -> (value, expires_at)

    def get(self, key: str):
        now = time.time()
        if key not in self.store:
            return None
        value, expires_at = self.store[key]
        if expires_at < now:
            del self.store[key]
            return None
        self.store.move_to_end(key)
        return value

    def set(self, key: str, value):
        now = time.time()
        expires_at = now + self.ttl_seconds
        if key in self.store:
            del self.store[key]
        self.store[key] = (value, expires_at)
        self.store.move_to_end(key)

        while len(self.store) > self.max_items:
            self.store.popitem(last=False)


def make_cache_key(question: str, retrieval_k: int, rerank_n: int, rerank_enabled: bool) -> str:
    base = (question.strip().lower() + f"|k={retrieval_k}|n={rerank_n}|rerank={int(rerank_enabled)}").encode("utf-8")
    return hashlib.sha256(base).hexdigest()


# -----------------------------
# RRF fusion (Project 10)
# -----------------------------
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


# -----------------------------
# Helpers
# -----------------------------
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


def rerank_with_llm(
    client: OpenAI,
    model: str,
    question: str,
    candidates: list[dict],
    top_n: int,
) -> list[dict]:
    """
    candidates: список элементов fused вида {"doc": {...}, "score": float, "sources": set()}
    Возвращает список candidates, отсортированный по релевантности (top_n).
    """

    # Сформируем краткий список кандидатов: индекс + текст (обрежем, чтобы не было слишком длинно)
    items = []
    for i, x in enumerate(candidates):
        doc = x["doc"]
        text = (doc.get("content") or "").strip()
        if len(text) > 600:
            text = text[:600] + "..."
        items.append(f"{i}. {text}")

    prompt = f"""You are a reranking system.
Given a QUESTION and candidate TEXTS, choose the most relevant texts.

Return ONLY a comma-separated list of indices (example: 2,0,3).
Do not add any other words.

QUESTION:
{question}

CANDIDATES:
{chr(10).join(items)}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You rerank candidates by relevance. Output only indices."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Парсим индексы вида "2,0,3"
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        idxs = []
        for p in parts:
            if p.isdigit():
                idxs.append(int(p))

        # Фильтрация: только валидные, без дублей, в порядке вывода
        seen = set()
        idxs_clean = []
        for i in idxs:
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                idxs_clean.append(i)

        if not idxs_clean:
            # если модель вернула мусор — fallback
            return candidates[:top_n]

        reranked = [candidates[i] for i in idxs_clean][:top_n]

        # если модель вернула меньше top_n, дополним исходным порядком
        if len(reranked) < top_n:
            for x in candidates:
                if x not in reranked:
                    reranked.append(x)
                if len(reranked) == top_n:
                    break

        return reranked

    except Exception:
        # безопасный fallback
        return candidates[:top_n]


def main():
    load_dotenv()

    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_ROLE_KEY")
    openai_key = get_env("OPENAI_API_KEY")
    emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    rerank_model = os.getenv("RERANK_MODEL", chat_model)

    retrieval_k = int(os.getenv("RETRIEVAL_K", "10"))
    rerank_n = int(os.getenv("RERANK_N", "5"))  
    rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"

    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "600"))
    cache_max = int(os.getenv("CACHE_MAX_ITEMS", "200"))

    debug_logs = os.getenv("DEBUG_LOGS", "true").lower() == "true"

    cache = LRUCacheTTL(max_items=cache_max, ttl_seconds=cache_ttl) if cache_enabled else None
   

    sb = create_client(supabase_url, supabase_key)
    client = OpenAI(api_key=openai_key)

    print("RAG Chat. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # --- CACHE CHECK ---
        key = None
        if cache_enabled:
            key = make_cache_key(q, retrieval_k, rerank_n, rerank_enabled)
            cached = cache.get(key)
            if cached is not None:
                if debug_logs:
                    print("[CACHE HIT] Returning cached answer.\n")
                print(f"Assistant: {cached}\n")
                continue
            else:
                if debug_logs:
                    print("[CACHE MISS] No cached answer.\n")

        t0 = time.time()

        # 1) embedding вопроса
        q_emb = client.embeddings.create(model=emb_model, input=q).data[0].embedding

        # 2) semantic search (top-K)
        sem_rpc = sb.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_threshold": 0.15,
            "match_count": retrieval_k
        }).execute()
        semantic_results = sem_rpc.data or []

        # 3) keyword search (top-K)
        kw_rpc = sb.rpc("keyword_search_documents", {
            "query_text": q,
            "match_count": retrieval_k
        }).execute()
        keyword_results = kw_rpc.data or []

        if not semantic_results and not keyword_results:
            print("Assistant: В документе нет информации об этом.\n")
            continue

        # 4) RRF fusion
        fused = rrf_fusion(keyword_results, semantic_results, k=60)

        # 5) кандидаты для rerank: берём top-K из fused
        candidates = fused[:retrieval_k]

        t_before_rerank = time.time()

        if rerank_enabled:
            top_fused = rerank_with_llm(
                client=client,
                model=rerank_model,
                question=q,
                candidates=candidates,
                top_n=rerank_n,
            )
        else:
            top_fused = candidates[:min(6, len(candidates))]  # как раньше

        t_after_rerank = time.time()

        if debug_logs and rerank_enabled:
            print(f"[RERANK] candidates={len(candidates)} -> selected={len(top_fused)} | time={(t_after_rerank - t_before_rerank):.3f}s\n")


        # --- DEBUG / REPORT OUTPUT ---
        if debug_logs:
            print("\n--- KEYWORD TOP-5 ---")
            for i, d in enumerate(keyword_results[:5], 1):
                print(i, d["doc_name"], d["chunk_index"], f"kw_rank={d.get('kw_rank', 0):.4f}")

            print("\n--- SEMANTIC TOP-5 ---")
            for i, d in enumerate(semantic_results[:5], 1):
                print(i, d["doc_name"], d["chunk_index"], f"sim={d.get('similarity', 0):.4f}")

            print("\n--- RRF TOP-3 ---")
            for i, x in enumerate(fused[:3], 1):
                d = x["doc"]
                print(i, d["doc_name"], d["chunk_index"], f"rrf={x['score']:.4f}",
                      "sources=", ",".join(sorted(x["sources"])))
            print()

        context = "\n\n---\n\n".join(
            [
                f"[{x['doc']['doc_name']} | chunk {x['doc']['chunk_index']} | sources={','.join(sorted(x['sources']))} | rrf={x['score']:.4f}]\n"
                f"{x['doc']['content']}"
                for x in top_fused
            ]
        )

        t_retrieval = time.time()

        # 6) generation
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

        t_generation = time.time()

        if debug_logs:
            print(f"[TIMING] retrieval={(t_retrieval - t0):.3f}s, generation={(t_generation - t_retrieval):.3f}s\n")

        print(f"Assistant: {answer}\n")

        # --- CACHE SET ---
        if cache_enabled and key is not None:
            cache.set(key, answer)


if __name__ == "__main__":
    main()
