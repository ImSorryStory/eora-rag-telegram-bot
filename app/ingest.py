from __future__ import annotations
import argparse
import os
import json
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx2txt

from .utils import chunk_text
from .llm import embed
from .store import VectorStore
from .config import settings
from .web_scraper import fetch_url, in_allowed

SUPPORTED_EXT = {".txt", ".md", ".pdf", ".html", ".htm", ".docx"}


def read_file(path: str) -> tuple[str, str]:
    title = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return title, f.read()
    if ext == ".pdf":
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return title, text
    if ext in {".html", ".htm"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            title = (soup.title.string or title) if soup.title else title
            return str(title), soup.get_text(" ")
    if ext == ".docx":
        text = docx2txt.process(path) or ""
        return title, text
    raise ValueError(f"Unsupported file: {path}")


def read_urls(urls: List[str]) -> List[Dict]:
    docs = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        if not in_allowed(url):
            print(f"Skip not-allowed domain: {url}")
            continue
        try:
            title, text = fetch_url(url)
            docs.append({
                "title": title or url,
                "text": text,
                "url": url,
                "file_path": None,
            })
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return docs


def read_local(dir_path: str) -> List[Dict]:
    docs = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in SUPPORTED_EXT:
                continue
            path = os.path.join(root, fn)
            try:
                title, text = read_file(path)
                docs.append({
                    "title": title,
                    "text": text,
                    "url": None,
                    "file_path": path,
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", default="links.txt")
    ap.add_argument("--local-dir", default="data/sources")
    ap.add_argument("--index", default=settings.INDEX_PATH)
    ap.add_argument("--meta", default=settings.CHUNKS_PATH)
    ap.add_argument("--chunk-tokens", type=int, default=400)
    args = ap.parse_args()

    # collect documents
    docs = []
    if os.path.exists(args.urls_file):
        with open(args.urls_file, "r", encoding="utf-8") as f:
            url_list = [line.strip() for line in f]
        docs += read_urls(url_list)
    if os.path.isdir(args.local_dir):
        docs += read_local(args.local_dir)

    # chunking & metas
    texts: List[str] = []
    metas: List[Dict] = []
    for i, d in enumerate(docs):
        chunks = chunk_text(d["text"], target_tokens=args.chunk_tokens)
        for j, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({
                "doc_id": i,
                "chunk_id": j,
                "title": d["title"],
                "url": d["url"],
                "file_path": d["file_path"],
                "text": ch,
            })

    if not texts:
        print("No texts to index.")
        return

    # embeddings
    embs = embed(texts)

    # store
    dim = len(embs[0])
    vs = VectorStore(dim, args.index, args.meta)
    vs.add(embs, metas)
    vs.save()
    print(f"Indexed {len(texts)} chunks from {len(docs)} documents.")

if __name__ == "__main__":
    main()