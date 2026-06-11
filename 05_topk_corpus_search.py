"""
05 - Top-K corpus search: the operation a vector DB actually performs.

"Find the K most similar items to this query" is the core primitive behind
semantic search, recommendations, retrieval-augmented generation, and dedup.
This file builds it from the pieces you already have.

The matrix-multiplication framing
----------------------------------
Stack your corpus of N vectors as the rows of a matrix C (shape N x d). Take a
single query vector q (shape d). If everything is unit-normalized, then the
cosine similarity of q against ALL N corpus vectors at once is a single
matrix-vector product:

        scores = C @ q          (shape N,)

scores[i] is the cosine similarity between the query and corpus row i. One
BLAS call computes the entire similarity column. To find the top K, you then
pick the K largest scores -- and you use np.argpartition, not a full sort,
because you do not care about the order of the other N-K results.

This is the whole game. A production vector DB adds an approximate index on
top (see 09) so it does not have to touch every row, but the exact version is
just "matmul, then partial sort."
"""

from __future__ import annotations

import time

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row of a 2-D array (or a single 1-D vector) to unit length."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        n = np.linalg.norm(matrix)
        return matrix if n == 0.0 else matrix / n
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pairwise cosine, exposed so this file can be compared against 01-04."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a / na) @ (b / nb))


def top_k(query: np.ndarray, corpus: np.ndarray, k: int = 5):
    """
    Return (indices, scores) of the K corpus rows most similar to `query`.

    Normalizes internally so you can pass raw vectors. Uses argpartition to
    find the top K in O(N) instead of an O(N log N) full sort, then sorts only
    those K.
    """
    q = normalize(query)
    c = normalize(corpus)
    scores = c @ q                      # the entire similarity column, one matmul
    k = min(k, len(scores))
    # argpartition puts the k largest somewhere in the last k slots (unordered).
    top_unsorted = np.argpartition(scores, -k)[-k:]
    # Now sort just those k by score, descending.
    order = top_unsorted[np.argsort(scores[top_unsorted])[::-1]]
    return order, scores[order]


def _demo() -> None:
    rng = np.random.default_rng(0)
    n, dim, k = 100_000, 256, 5

    print(f"Corpus: {n:,} vectors x {dim} dims. Query for top-{k}.")
    corpus = rng.standard_normal((n, dim))
    query = rng.standard_normal(dim)

    # Pre-normalize the corpus once (index build); time the query separately.
    corpus_unit = normalize(corpus)
    query_unit = normalize(query)

    t0 = time.perf_counter()
    scores = corpus_unit @ query_unit               # C @ q, the whole column
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    elapsed = time.perf_counter() - t0

    print()
    print(f"Exact top-{k} found in {elapsed * 1e3:.2f} ms (one matmul + partial sort)")
    print("-" * 48)
    for rank, i in enumerate(idx, 1):
        print(f"  #{rank}  corpus[{i:>6}]  cosine = {scores[i]:+.4f}")

    # Cross-check the convenience wrapper agrees.
    idx2, scores2 = top_k(query, corpus, k)
    assert np.array_equal(idx, idx2)
    print()
    print("Brute-force exact search scales linearly with N: every query touches")
    print("every row. That is fine up to a few million vectors. Past that, you")
    print("reach for an approximate index (file 09) to stop scanning everything.")


if __name__ == "__main__":
    _demo()
