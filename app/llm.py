from __future__ import annotations
from typing import List, Dict, Any
from openai import OpenAI
from .config import settings

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed(texts: List[str]) -> List[List[float]]:
    resp = client().embeddings.create(model=settings.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def complete(messages: List[Dict[str, Any]], max_tokens: int | None = None, temperature: float | None = None) -> str:
    resp = client().chat.completions.create(
        model=settings.GEN_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content or ""

# Заготовка для GigaChat: можно быстро переключить транспорт
# from gigachat import GigaChat
# gc = GigaChat(credentials=os.environ["GIGACHAT_TOKEN"], verify_ssl_certs=False)
# gc.chat("...")