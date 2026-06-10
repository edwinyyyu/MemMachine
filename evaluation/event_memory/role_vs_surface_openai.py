"""Role-vs-surface probe on OpenAI text-embedding-3-small, to compare biases vs
embeddinggemma-300m (see role_vs_surface_probe.py).

text-embedding-3-small is SYMMETRIC (no query/document prompts), so every text is
embedded the same way. The doc-doc contrasts (role / surface / value) are directly
comparable to embeddinggemma; the query->doc ones are symmetric similarities.

The OpenAI key is read ONLY from the authorized locomo .env and never printed.

    PYTHONPATH=<repo> uv run python evaluation/event_memory/role_vs_surface_openai.py
"""

import os
from pathlib import Path

import numpy as np

_ENV = Path(
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/locomo/.env"
)

ITEMS = [
    ("Who paid Alice?", "q", "Q:who-paid-Alice"),
    ("Who paid Bob?", "q", "Q:who-paid-Bob"),
    ("Alice paid Bob.", "d", "D:Alice->Bob"),
    ("Bob paid Alice.", "d", "D:Bob->Alice"),
    ("I saw a squirrel on my way to work near the coffee shop today.", "d", "D:squirrel"),
    ("I saw a raccoon on my way to work near the coffee shop today.", "d", "D:raccoon"),
    ("I went by the coffee shop on my way to work today.", "d", "D:coffee-only"),
    ("What did I see on my way to work today?", "q", "Q:saw-work"),
    ("What did I see on my way to work near the coffee shop today?", "q", "Q:saw-work-coffee"),
    ("I am 25.", "d", "D:25"),
    ("I am 26.", "d", "D:26"),
    ("I am 85.", "d", "D:85"),
    ("How old am I?", "q", "Q:how-old"),
]


def _load_openai_key() -> None:
    for line in _ENV.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = (
                stripped.split("=", 1)[1].strip().strip('"').strip("'")
            )
            return
    raise SystemExit("OPENAI_API_KEY not found in the authorized .env")


def main() -> None:
    _load_openai_key()
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=[it[0] for it in ITEMS]
    )
    emb = np.array([d.embedding for d in resp.data], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T
    labels = [it[2] for it in ITEMS]

    def c(a: str, b: str) -> float:
        return float(sim[labels.index(a), labels.index(b)])

    print("text-embedding-3-small (symmetric — every text embedded identically)\n")
    head = "".join(f"{j:>6}" for j in range(len(ITEMS)))
    print(f"{'':<20}{head}")
    for i, lab in enumerate(labels):
        row = "".join(f"{sim[i, j]:6.2f}" for j in range(len(ITEMS)))
        print(f"{i:>2} {lab:<17}{row}")

    print("\n================ ROLE BINDING ================")
    print(f"  D:Alice->Bob  vs  D:Bob->Alice         = {c('D:Alice->Bob','D:Bob->Alice'):.3f}")
    print("Query 'Who paid Alice?' (answer = 'Bob paid Alice'):")
    print(f"  -> D:Bob->Alice  (role-CORRECT)        = {c('Q:who-paid-Alice','D:Bob->Alice'):.3f}")
    print(f"  -> D:Alice->Bob  (role-WRONG)          = {c('Q:who-paid-Alice','D:Alice->Bob'):.3f}")
    print(f"  margin (correct - wrong)               = {c('Q:who-paid-Alice','D:Bob->Alice') - c('Q:who-paid-Alice','D:Alice->Bob'):+.3f}")
    print("Query 'Who paid Bob?' (answer = 'Alice paid Bob'):")
    print(f"  -> D:Alice->Bob  (role-CORRECT)        = {c('Q:who-paid-Bob','D:Alice->Bob'):.3f}")
    print(f"  -> D:Bob->Alice  (role-WRONG)          = {c('Q:who-paid-Bob','D:Bob->Alice'):.3f}")
    print(f"  margin (correct - wrong)               = {c('Q:who-paid-Bob','D:Alice->Bob') - c('Q:who-paid-Bob','D:Bob->Alice'):+.3f}")

    print("\n================ SURFACE (entity swap) ================")
    print(f"  D:squirrel  vs  D:raccoon              = {c('D:squirrel','D:raccoon'):.3f}")
    print(f"  D:squirrel  vs  D:coffee-only          = {c('D:squirrel','D:coffee-only'):.3f}")
    print("Query 'What did I see on my way to work today?':")
    print(f"  -> D:squirrel                          = {c('Q:saw-work','D:squirrel'):.3f}")
    print(f"  -> D:raccoon                           = {c('Q:saw-work','D:raccoon'):.3f}")
    print(f"  -> D:coffee-only (no animal)           = {c('Q:saw-work','D:coffee-only'):.3f}")

    print("\n================ VALUE sensitivity ================")
    print(f"  D:25  vs  D:26  (adjacent)             = {c('D:25','D:26'):.3f}")
    print(f"  D:25  vs  D:85  (far)                  = {c('D:25','D:85'):.3f}")
    print(f"  D:26  vs  D:85  (far)                  = {c('D:26','D:85'):.3f}")
    print("Query 'How old am I?':")
    for d in ("D:25", "D:26", "D:85"):
        print(f"  -> {d:<8}                           = {c('Q:how-old', d):.3f}")


if __name__ == "__main__":
    main()
