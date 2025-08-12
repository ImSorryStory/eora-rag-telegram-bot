from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import re

try:
    import tiktoken
except Exception:
    tiktoken = None

HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    text = HTML_TAG_RE.sub(" ", html)
    text = WS_RE.sub(" ", text)
    return text.strip()


def approx_token_len(s: str) -> int:
    if not s:
        return 0
    if tiktoken is None:
        # rough heuristic ~ 4 chars per token
        return max(1, round(len(s) / 4))
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))


def chunk_text(text: str, target_tokens: int = 400, overlap_tokens: int = 40) -> List[str]:
    if not text:
        return []
    # fall back to char-based splitting if no tiktoken
    if tiktoken is None:
        chunk_size = target_tokens * 4
        overlap = overlap_tokens * 4
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        part = tokens[i : i + target_tokens]
        chunks.append(enc.decode(part))
        i += target_tokens - overlap_tokens
    return chunks


@dataclass
class SourceMeta:
    source_id: str
    title: str | None
    url: str | None
    file_path: str | None
    extra: dict