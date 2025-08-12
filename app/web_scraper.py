from __future__ import annotations
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from .utils import strip_html
from .config import settings

HEADERS = {"User-Agent": "EORA-RAG-Bot/1.0 (+github.com/ImSorryStory/eora-rag-telegram-bot)"}


def fetch_url(url: str) -> tuple[str, str]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    text = strip_html(html)
    return title, text


def in_allowed(url: str) -> bool:
    host = urlparse(url).netloc
    allowed = {d.strip().lower() for d in settings.ALLOWED_DOMAINS.split(",")}
    return any(host.endswith(d) for d in allowed)


def normalize(base: str, href: str) -> str:
    return urljoin(base, href)