"""Does embeddinggemma's TASK PROMPT change the role/surface/value biases?

Sweeps every prompt in the model's `prompts` dict (Retrieval, Clustering, STS,
Classification, ...) over the same probe set, plus the asymmetric retrieval config
(query prompt for questions, document prompt for statements) we actually deploy.
For each, reports the key contrasts so we can see whether e.g. the STS
("sentence similarity") prompt binds argument roles where retrieval does not.

    PYTHONPATH=<repo> uv run python evaluation/event_memory/role_vs_surface_prompts.py
"""

import numpy as np

ITEMS = [
    ("Who paid Alice?", "q", "Q:who-paid-Alice"),
    ("Who paid Bob?", "q", "Q:who-paid-Bob"),
    ("Alice paid Bob.", "d", "D:Alice->Bob"),
    ("Bob paid Alice.", "d", "D:Bob->Alice"),
    (
        "I saw a squirrel on my way to work near the coffee shop today.",
        "d",
        "D:squirrel",
    ),
    ("I saw a raccoon on my way to work near the coffee shop today.", "d", "D:raccoon"),
    ("I went by the coffee shop on my way to work today.", "d", "D:coffee-only"),
    ("What did I see on my way to work today?", "q", "Q:saw-work"),
    (
        "What did I see on my way to work near the coffee shop today?",
        "q",
        "Q:saw-work-coffee",
    ),
    ("I am 25.", "d", "D:25"),
    ("I am 26.", "d", "D:26"),
    ("I am 85.", "d", "D:85"),
    ("How old am I?", "q", "Q:how-old"),
]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("google/embeddinggemma-300m")
    texts = [it[0] for it in ITEMS]
    modes = [it[1] for it in ITEMS]
    labels = [it[2] for it in ITEMS]

    print("available prompts:")
    for name, template in model.prompts.items():
        print(f"  {name:<22} {template!r}")
    print(f"default_prompt_name = {model.default_prompt_name}\n")

    names = list(model.prompts)
    # locate the asymmetric retrieval pair (query / document) by common aliases
    q_name = next((n for n in ("query", "Retrieval-query") if n in names), None)
    d_name = next((n for n in ("document", "Retrieval-document") if n in names), None)

    def encode(texts_subset: list[str], prompt_name: str) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts_subset, prompt_name=prompt_name, normalize_embeddings=True
            ),
            dtype=float,
        )

    def matrix_symmetric(prompt_name: str) -> np.ndarray:
        emb = encode(texts, prompt_name)
        return emb @ emb.T

    def matrix_retrieval() -> np.ndarray:
        emb = np.zeros((len(texts), model.get_sentence_embedding_dimension()))
        qi = [i for i, m in enumerate(modes) if m == "q"]
        di = [i for i, m in enumerate(modes) if m == "d"]
        emb[qi] = encode([texts[i] for i in qi], q_name)
        emb[di] = encode([texts[i] for i in di], d_name)
        return emb @ emb.T

    def contrasts(sim: np.ndarray) -> dict[str, float]:
        def c(a: str, b: str) -> float:
            return float(sim[labels.index(a), labels.index(b)])

        return {
            "role-swap AB~BA": c("D:Alice->Bob", "D:Bob->Alice"),
            "role-margin (Alice)": c("Q:who-paid-Alice", "D:Bob->Alice")
            - c("Q:who-paid-Alice", "D:Alice->Bob"),
            "role-margin (Bob)": c("Q:who-paid-Bob", "D:Alice->Bob")
            - c("Q:who-paid-Bob", "D:Bob->Alice"),
            "squirrel~raccoon": c("D:squirrel", "D:raccoon"),
            "25~26 (adjacent)": c("D:25", "D:26"),
            "25~85 (far)": c("D:25", "D:85"),
        }

    strategies: dict[str, np.ndarray] = {}
    if q_name and d_name:
        strategies[f"retrieval-asym({q_name}/{d_name})"] = matrix_retrieval()
    for name in names:
        strategies[f"sym:{name}"] = matrix_symmetric(name)

    rows = list(contrasts(next(iter(strategies.values()))).keys())
    print(f"{'contrast':<22}" + "".join(f"{s[:20]:>22}" for s in strategies))
    data = {s: contrasts(m) for s, m in strategies.items()}
    for r in rows:
        print(f"{r:<22}" + "".join(f"{data[s][r]:>+22.3f}" for s in strategies))
    print(
        "\nrole-margin > 0 means the query prefers the role-CORRECT answer "
        "(binds roles); <= 0 means role-blind."
    )


if __name__ == "__main__":
    main()
