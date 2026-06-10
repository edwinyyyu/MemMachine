"""Probe: does embeddinggemma bind ROLES, or only SURFACE similarity?

Embeds a designed set with the correct asymmetric prompts (questions as queries,
statements as documents) and prints the pairwise cosine matrix plus three targeted
contrasts:

  ROLE   — "Alice paid Bob." vs "Bob paid Alice." (surface-identical, roles swapped)
           and whether "Who paid Alice?" prefers the role-correct answer.
  SURFACE— "squirrel" vs "raccoon" sentences (one content word differs).
  VALUE  — "I am 25/26/85." (one token differs; is 25~26 closer than 25~85?).

The point: find the cosine band where two genuinely-different memories become
indistinguishable to retrieval (strong competition) — i.e. where the embedding
can't separate them and only context/expansion can.

    PYTHONPATH=<repo> uv run python evaluation/event_memory/role_vs_surface_probe.py
"""

import asyncio

import numpy as np

# (text, mode, short label) — mode picks the asymmetric prompt.
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


def _norm(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


async def main() -> None:
    from claude_memory.engine import build_embedder

    embedder = build_embedder("embeddinggemma")

    q_idx = [i for i, it in enumerate(ITEMS) if it[1] == "q"]
    d_idx = [i for i, it in enumerate(ITEMS) if it[1] == "d"]
    q_vecs = await embedder.search_embed([ITEMS[i][0] for i in q_idx])
    d_vecs = await embedder.ingest_embed([ITEMS[i][0] for i in d_idx])

    dim = len(q_vecs[0])
    emb = np.zeros((len(ITEMS), dim), dtype=float)
    for slot, i in enumerate(q_idx):
        emb[i] = q_vecs[slot]
    for slot, i in enumerate(d_idx):
        emb[i] = d_vecs[slot]
    emb = _norm(emb)
    sim = emb @ emb.T
    labels = [it[2] for it in ITEMS]

    def c(a: str, b: str) -> float:
        return float(sim[labels.index(a), labels.index(b)])

    # ---- full matrix ----
    print("pairwise cosine (q=query-prompt, d=document-prompt):\n")
    head = "".join(f"{j:>6}" for j in range(len(ITEMS)))
    print(f"{'':<20}{head}")
    for i, lab in enumerate(labels):
        row = "".join(f"{sim[i, j]:6.2f}" for j in range(len(ITEMS)))
        print(f"{i:>2} {lab:<17}{row}")

    print("\n================ ROLE BINDING ================")
    print("Doc-doc, surface-identical, roles swapped:")
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
    asyncio.run(main())
