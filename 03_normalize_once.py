"""
03 - Normalize once, then it is just a dot product.

The key realization that makes vector databases fast.

The naive cosine formula recomputes both magnitudes on every comparison:

        cos(theta) = (A . B) / (||A|| * ||B||)

But in a search workload you compare ONE query against MILLIONS of corpus
vectors. The corpus vectors never change between queries. So why recompute
||C|| a million times per query?

The trick: pre-normalize every vector to unit length once, up front. A unit
vector has magnitude 1. If ||A|| = ||B|| = 1, the denominator becomes
1 * 1 = 1 and the whole formula collapses to:

        cos(theta) = A . B        (when ||A|| = ||B|| = 1)

Cosine similarity on normalized vectors is *literally* the dot product. The
expensive part (the square roots and the division) is paid once at index-build
time, and every subsequent query is a bare matrix multiply. This is exactly
why FAISS, hnswlib, pgvector, Pinecone, and friends ask you to normalize (or do
it for you) and then expose "inner product" as the metric.

This file benchmarks naive cosine vs normalize-once on a 1M-vector corpus
search to show the payoff.
"""

from __future__ import annotations

import time

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Scale each row to unit L2 length.

    Returns a new array where every row has magnitude 1 (rows that were the
    zero vector stay zero, guarded against divide-by-zero).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # avoid 0/0; a zero row stays zero
    return matrix / norms


def cosine_naive(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of `query` against every row of `corpus`, recomputing
    all norms every call. This is the "do it the obvious way" baseline.
    """
    q_norm = np.linalg.norm(query)
    c_norms = np.linalg.norm(corpus, axis=1)
    denom = c_norms * q_norm
    denom[denom == 0.0] = 1.0
    return (corpus @ query) / denom


def cosine_prenormalized(query_unit: np.ndarray, corpus_unit: np.ndarray) -> np.ndarray:
    """
    Cosine similarity when both query and corpus are ALREADY unit vectors.
    No norms, no division: just one matrix-vector product. This is the form
    that runs inside a production vector index.
    """
    return corpus_unit @ query_unit


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pairwise cosine via the normalize-once trick: scale both vectors to unit
    length, then the cosine is simply their dot product. Same answer as the
    naive formula, exposed for one-to-one comparison with the other files.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a / na) @ (b / nb))


def _benchmark() -> None:
    rng = np.random.default_rng(7)
    n_vectors = 1_000_000
    dim = 128

    print(f"Building a {n_vectors:,} x {dim} corpus ({n_vectors * dim * 8 / 1e6:.0f} MB)...")
    corpus = rng.standard_normal((n_vectors, dim))
    query = rng.standard_normal(dim)

    # --- Naive path: norms recomputed on every query ---
    t0 = time.perf_counter()
    scores_naive = cosine_naive(query, corpus)
    t_naive = time.perf_counter() - t0

    # --- Normalize-once path ---
    # The normalization is a one-time index-build cost, timed separately.
    t0 = time.perf_counter()
    corpus_unit = normalize(corpus)
    t_build = time.perf_counter() - t0

    query_unit = query / np.linalg.norm(query)

    # Per-query cost is just the dot product. Time only that.
    t0 = time.perf_counter()
    scores_fast = cosine_prenormalized(query_unit, corpus_unit)
    t_query = time.perf_counter() - t0

    # Both paths must rank identically.
    assert np.allclose(scores_naive, scores_fast, atol=1e-9)

    print()
    print("1M-vector search: naive cosine vs normalize-once")
    print("=" * 56)
    print(f"  naive cosine (per query)        : {t_naive * 1e3:8.2f} ms")
    print(f"  normalize-once (one-time build) : {t_build * 1e3:8.2f} ms")
    print(f"  dot-product query (per query)   : {t_query * 1e3:8.2f} ms")
    print("-" * 56)
    print(f"  per-query speedup               : {t_naive / t_query:8.1f}x")
    print()
    print("The build cost is paid ONCE when you index the corpus. Every query")
    print("after that is a pure dot product. Across millions of queries the")
    print("amortized win is enormous, which is why every vector DB stores")
    print("normalized embeddings and calls cosine 'inner product'.")


if __name__ == "__main__":
    _benchmark()
