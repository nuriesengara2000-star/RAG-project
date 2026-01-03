from __future__ import annotations
import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

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

        # 2) RPC semantic search
        rpc = sb.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_threshold": 0.2,
            "match_count": 5


        }).execute()

        matches = rpc.data or []
        if not matches:
            print("Assistant: В документе нет информации об этом.\n")
            continue

        context = "\n\n---\n\n".join(
            [f"[{m['doc_name']} | chunk {m['chunk_index']} | sim={m['similarity']:.3f}]\n{m['content']}" for m in matches]
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
