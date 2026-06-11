"""
09 - Approximate Nearest Neighbors with HNSW: where vector DBs live.

Exact search (file 05) touches every row of the corpus on every query. That is
O(N) per query and it stops being free somewhere past a few million vectors.
Production vector databases do NOT do this. They build an index that lets a
query reach its neighbors while visiting only a tiny fraction of the corpus.

HNSW (Hierarchical Navigable Small World) is the index most of them use --
Pinecone, Weaviate, Qdrant, Milvus, pgvector's hnsw, and Lucene all ship a
variant. The idea: build a multi-layer graph where each node links to its
nearest neighbors, with sparse "express lane" layers on top for long jumps.
A search greedily hops toward the query, descending layers, and converges in
roughly O(log N) steps instead of O(N).

The catch is in the name: APPROXIMATE. HNSW can miss a true neighbor, so you
measure RECALL (what fraction of the true top-K it returned) against the
latency you bought. This file builds an HNSW index over a 100k-vector corpus
with cosine space, benchmarks it against exact brute force, and reports both
the speedup and the recall so the tradeoff is concrete.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import hnswlib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "hnswlib is not installed. Run make.ps1 or pip install -r requirements.txt"
    ) from exc


def exact_top_k(query: np.ndarray, corpus_unit: np.ndarray, k: int):
    """Ground-truth top-k by exhaustive cosine (corpus already unit length)."""
    scores = corpus_unit @ query
    idx = np.argpartition(scores, -k)[-k:]
    return set(idx[np.argsort(scores[idx])[::-1]].tolist())


def _make_clustered_corpus(rng, n, dim, n_clusters):
    """
    Build a corpus with real neighborhood structure, the way embeddings
    actually look: points gathered around cluster centers, not scattered
    uniformly. Pure random Gaussian noise has NO structure, so every point is
    almost equidistant from every other and ANN recall looks pathologically
    bad -- which would misrepresent how HNSW performs on real data. Clusters
    fix that and make the benchmark faithful to production embeddings.
    """
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    corpus = centers[assign] + 0.35 * rng.standard_normal((n, dim)).astype(np.float32)
    return corpus, centers


def _demo() -> None:
    rng = np.random.default_rng(123)
    n, dim, k = 100_000, 256, 10
    n_queries = 200
    n_clusters = 500

    print(f"Corpus: {n:,} x {dim}. Index: HNSW (cosine space). k = {k}.")
    print(f"Data has {n_clusters} clusters (real embeddings have structure too).")
    corpus, centers = _make_clustered_corpus(rng, n, dim, n_clusters)
    corpus_unit = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    # Queries: sit near a random cluster, like a real lookup near known content.
    q_centers = centers[rng.integers(0, n_clusters, size=n_queries)]
    queries = (q_centers + 0.35 * rng.standard_normal((n_queries, dim))).astype(np.float32)
    queries_unit = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # --- Build the HNSW index ---
    # space='cosine' tells hnswlib to use cosine distance (1 - cosine sim).
    # M and ef_construction trade index quality/size against build time.
    index = hnswlib.Index(space="cosine", dim=dim)
    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(corpus, np.arange(n))
    t_build = time.perf_counter() - t0
    print(f"Index built in {t_build:.2f} s (M=16, ef_construction=200).")
    print()

    # --- Exact brute-force baseline (the ground truth + its cost) ---
    t0 = time.perf_counter()
    truth = [exact_top_k(q, corpus_unit, k) for q in queries_unit]
    t_exact = time.perf_counter() - t0

    # --- HNSW at a few 'ef' settings. Higher ef = search wider = better
    #     recall, slower query. This single knob IS the recall/latency dial. ---
    print(f"{'ef':>5} | {'HNSW (ms/q)':>12} | {'exact (ms/q)':>13} | {'speedup':>8} | {'recall@10':>10}")
    print("-" * 62)
    for ef in (10, 50, 100, 200):
        index.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = index.knn_query(queries, k=k)
        t_hnsw = time.perf_counter() - t0

        recalls = [len(set(labels[i].tolist()) & truth[i]) / k for i in range(n_queries)]
        recall = float(np.mean(recalls))

        ms_hnsw = t_hnsw / n_queries * 1e3
        ms_exact = t_exact / n_queries * 1e3
        print(f"{ef:>5} | {ms_hnsw:>12.3f} | {ms_exact:>13.3f} | {ms_exact / ms_hnsw:>7.1f}x | {recall:>10.3f}")

    print("-" * 62)
    print()
    print("Read the table as a dial: crank ef up and recall climbs toward 1.0")
    print("while latency rises; turn it down and queries get cheaper but you")
    print("start missing true neighbors. There is no 'correct' setting -- you")
    print("pick the recall your product can tolerate and pay the latency for it.")
    print("THIS knob, on THIS kind of index, is what 'tuning a vector DB' means.")


if __name__ == "__main__":
    _demo()
