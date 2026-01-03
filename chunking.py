from __future__ import annotations
import re
from typing import List

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Простой чанкинг по символам с оверлапом.
    Для старта достаточно. Потом можно улучшать (по предложениям/токенам).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks
