"""
07 - Semantic search with real sentence embeddings.

Everything so far used random or TF-IDF vectors. This is the payoff: real
embeddings from a transformer, where cosine similarity tracks *meaning* rather
than shared words. "How do I fix a noisy engine?" should match "the motor is
making a grinding sound" even though they share almost no vocabulary.

We use sentence-transformers' all-MiniLM-L6-v2: small (~80 MB), fast, and a
solid baseline. It maps each sentence to a 384-dimensional vector. These
models are trained so that semantic closeness shows up as high cosine
similarity, which is exactly why cosine is the default metric in every vector
DB that stores embeddings.

The corpus below deliberately mixes three worlds -- AI/ML, engine maintenance,
and random filler -- so that a good query lights up its own cluster and leaves
the distractors cold. That contrast is the thing to watch.

First run downloads the model (one time, a few hundred MB of cache). After
that it is fully offline.
"""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run the project setup (make.ps1) or: pip install -r requirements.txt"
    ) from exc


CORPUS = [
    # --- AI / ML ---
    "Cosine similarity is the default metric for comparing text embeddings.",
    "Transformer models map sentences into high-dimensional vector spaces.",
    "Vector databases like Pinecone and pgvector index embeddings for search.",
    "Fine-tuning adapts a pretrained language model to a narrow task.",
    "Retrieval-augmented generation grounds an LLM in your own documents.",
    "Backpropagation computes gradients to train a neural network.",
    "Tokenization splits raw text into the units a model actually reads.",
    "Approximate nearest neighbor search trades a little recall for speed.",
    # --- Engine / vehicle maintenance ---
    "The engine is making a loud grinding noise when it idles.",
    "Replace the oil and the oil filter every five thousand miles.",
    "A worn timing belt can cause catastrophic engine failure.",
    "Check the coolant level before a long drive in hot weather.",
    "The motor stalls at low RPM and the check-engine light is on.",
    "Spark plugs that are fouled will make the engine misfire.",
    "Low tire pressure hurts fuel economy and tire wear.",
    "The transmission slips when shifting from second to third gear.",
    # --- Random distractors ---
    "The bakery on the corner sells fresh sourdough every morning.",
    "She planted tomatoes and basil in the garden last weekend.",
    "The hiking trail offers a wide view of the valley at sunrise.",
    "A good espresso depends on grind size, dose, and water temperature.",
    "The orchestra tuned their instruments before the evening performance.",
]

QUERIES = [
    "Why is cosine similarity used in vector search?",
    "My car's motor sounds rough and won't run smoothly.",
    "What should I cook for dinner tonight?",
]


def top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int = 5):
    """Return (indices, scores) of the k highest-cosine corpus rows.

    The embeddings are normalized at encode time, so cosine is a plain dot
    product here -- the normalize-once trick from file 03 in the wild.
    """
    scores = corpus_vecs @ query_vec
    k = min(k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]], None


def _demo() -> None:
    print("Loading all-MiniLM-L6-v2 (first run downloads the model)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # normalize_embeddings=True returns unit vectors, so dot product == cosine.
    corpus_vecs = model.encode(CORPUS, normalize_embeddings=True)
    print(f"Encoded {len(CORPUS)} sentences into {corpus_vecs.shape[1]}-dim vectors.")
    print()

    for query in QUERIES:
        q_vec = model.encode(query, normalize_embeddings=True)
        scores = corpus_vecs @ q_vec
        order = np.argsort(scores)[::-1][:5]

        print(f'QUERY: "{query}"')
        print("  rank  cosine   sentence")
        for rank, i in enumerate(order, 1):
            print(f"  {rank:>4}  {scores[i]:+.4f}  {CORPUS[i]}")
        print()

    print("Notice how each query pulls back its own topic cluster and pushes the")
    print("two unrelated clusters to the bottom -- with essentially no shared")
    print("keywords. Cosine over learned embeddings is matching meaning, not")
    print("spelling. That is the entire reason semantic search works.")


if __name__ == "__main__":
    _demo()
