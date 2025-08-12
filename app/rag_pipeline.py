from __future__ import annotations
from typing import Dict, List, Tuple
import json
import os

from .config import settings
from .store import VectorStore
from .llm import embed, complete
from .prompts import SYS_PROMPT, USER_PROMPT_TEMPLATE


class RAG:
    def __init__(self, index_path: str | None = None, meta_path: str | None = None):
        self.index_path = index_path or settings.INDEX_PATH
        self.meta_path = meta_path or settings.CHUNKS_PATH
        # lazy-load; we don't know dim before first embed
        self._vs: VectorStore | None = None

    def _load_vs(self):
        if self._vs is None:
            # We'll read meta to infer embedding dim by embedding one dummy string if needed
            # Simpler: try to read index; if missing, raise
            if not (self.index_path and self.meta_path and \
                    os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
                raise FileNotFoundError(
                    "Индекс не найден. Сначала запустите ingest.py для построения индекса.")

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict]:
        # Load store on demand
        import os
        if self._vs is None:
            # quick dim discovery: read faiss index
            import faiss
            if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
                raise FileNotFoundError("Нет индекса/метаданных. Запустите ingest.py")
            index = faiss.read_index(self.index_path)
            dim = index.d
            self._vs = VectorStore(dim, self.index_path, self.meta_path)
        vs = self._vs
        q_emb = embed([query])[0]
        hits = vs.search(q_emb, top_k or settings.TOP_K)
        metas = vs.metas
        results = []
        for i, score in hits:
            m = metas[i]
            results.append({"score": score, **m})
        return results

    @staticmethod
    def _make_sources_block(chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        # собираем уникальные документы в порядке появления + подбираем первый фрагмент для каждого
        ordered = []
        seen = set()
        for ch in chunks:
            key = (ch.get("url"), ch.get("file_path"))
            if key in seen:
                continue
            seen.add(key)
            snippet = (ch.get("text") or "").strip()
            if len(snippet) > 800:
                snippet = snippet[:800] + "…"
            ordered.append({
                "title": ch.get("title") or (ch.get("url") or ch.get("file_path") or "Источник"),
                "url": ch.get("url"),
                "file_path": ch.get("file_path"),
                "snippet": snippet,
            })

        lines = []
        for i, d in enumerate(ordered, start=1):
            loc = d["url"] or d["file_path"] or ""
            lines.append(f"[{i}] {d['title']} — {loc}\n{d['snippet']}")
        return "\n\n".join(lines), ordered

    def answer(self, question: str) -> Dict:
        retrieved = self.retrieve(question, top_k=settings.TOP_K)
        src_block, doc_list = self._make_sources_block(retrieved)

        user_prompt = USER_PROMPT_TEMPLATE.format(question=question, sources_block=src_block)
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = complete(messages)

        # Collect attachments (top N unique with local files)
        attachments = []
        n = 0
        seen = set()
        for ch in retrieved:
            fp = ch.get("file_path")
            if fp and os.path.exists(fp) and fp not in seen:
                attachments.append(fp)
                seen.add(fp)
                n += 1
                if n >= settings.ATTACH_TOP_N:
                    break

        return {
            "answer": text,
            "sources": doc_list,
            "attachments": attachments,
        }