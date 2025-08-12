from __future__ import annotations
import yaml
import re
from app.rag_pipeline import RAG


def score(answer: str, must_include: list[str]) -> float:
    a = answer.lower()
    hits = sum(1 for k in must_include if k.lower() in a)
    return hits / max(1, len(must_include))


def main():
    rag = RAG()
    with open("eval/qa_pairs.yaml", "r", encoding="utf-8") as f:
        qa = yaml.safe_load(f)
    total = 0
    s = 0.0
    for item in qa:
        q = item["q"]
        res = rag.answer(q)
        ans = res["answer"]
        sc = score(ans, item.get("must_include", []))
        print("Q:", q)
        print("A:", ans[:500], "...\n")
        print("score:", sc)
        total += 1
        s += sc
    print(f"Avg score: {s/total:.2f}")

if __name__ == "__main__":
    main()