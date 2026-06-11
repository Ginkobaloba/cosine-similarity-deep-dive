"""
08 - Cosine vs Euclidean vs raw dot product, side by side.

Three metrics, three different questions:

  cosine(A, B)    = (A . B) / (||A|| ||B||)   -> pure direction, length removed
  dot(A, B)       =  A . B                     -> direction AND magnitude mixed
  euclidean(A, B) =  ||A - B||                 -> straight-line distance

On UNIT-normalized vectors they are tightly related: maximizing cosine,
maximizing dot product, and minimizing Euclidean distance all produce the SAME
ranking (because ||A - B||^2 = 2 - 2 cos(theta) when both are unit length). So
for normalized embeddings the choice is mostly cosmetic.

They DISAGREE the moment magnitudes vary:
  - Raw dot product rewards long vectors. A vector that points in a so-so
    direction but is very long can outscore a perfectly-aligned short one.
  - Euclidean penalizes magnitude gaps even when the direction is identical.
  - Cosine ignores magnitude entirely and asks only "same direction?".

This file embeds the file-07 corpus, ranks one query by all three metrics so
you can see where they agree and where they split, then projects everything to
2D with PCA and plots the query plus each metric's top-K in its own color.
The figure is saved to outputs/metric_comparison.png.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sentence-transformers is not installed. Run make.ps1 or "
        "pip install -r requirements.txt"
    ) from exc

import matplotlib

matplotlib.use("Agg")  # headless-safe; we save a PNG instead of opening a window
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reuse the corpus from the semantic-search file.
from importlib import import_module

_sem = import_module("07_semantic_search")
CORPUS = _sem.CORPUS


def rank_by_cosine(query_unit, corpus_unit):
    return corpus_unit @ query_unit  # higher = better


def rank_by_dot(query_raw, corpus_raw):
    return corpus_raw @ query_raw    # higher = better, magnitude counts


def rank_by_euclidean(query_raw, corpus_raw):
    return -np.linalg.norm(corpus_raw - query_raw, axis=1)  # negate so higher = better


def _top(scores, k):
    order = np.argsort(scores)[::-1][:k]
    return list(order)


def _demo() -> None:
    k = 5
    query = "Why is cosine similarity used in vector search?"
    print(f'QUERY: "{query}"')
    print(f"Comparing the three metrics' top-{k} on the same corpus.")
    print()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Raw (un-normalized) embeddings so dot and euclidean see real magnitudes.
    corpus_raw = model.encode(CORPUS, normalize_embeddings=False)
    query_raw = model.encode(query, normalize_embeddings=False)

    # Unit versions for cosine.
    corpus_unit = corpus_raw / np.linalg.norm(corpus_raw, axis=1, keepdims=True)
    query_unit = query_raw / np.linalg.norm(query_raw)

    cos_scores = rank_by_cosine(query_unit, corpus_unit)
    dot_scores = rank_by_dot(query_raw, corpus_raw)
    euc_scores = rank_by_euclidean(query_raw, corpus_raw)

    top_cos = _top(cos_scores, k)
    top_dot = _top(dot_scores, k)
    top_euc = _top(euc_scores, k)

    print(f"{'rank':>4} | {'cosine':<28} | {'dot product':<28} | {'euclidean':<28}")
    print("-" * 100)
    for r in range(k):
        c = CORPUS[top_cos[r]][:26]
        d = CORPUS[top_dot[r]][:26]
        e = CORPUS[top_euc[r]][:26]
        print(f"{r + 1:>4} | {c:<28} | {d:<28} | {e:<28}")
    print()

    agree_cd = set(top_cos) == set(top_dot)
    print(f"cosine vs dot   top-{k} identical set? {agree_cd}")
    print(f"cosine vs euclid top-{k} identical set? {set(top_cos) == set(top_euc)}")
    print("  Where embeddings have near-uniform magnitude the three agree. The")
    print("  splits you see are exactly the magnitude-sensitive cases dot and")
    print("  euclidean care about and cosine throws away.")

    # --- 2D PCA visualization ---
    os.makedirs("outputs", exist_ok=True)
    all_raw = np.vstack([corpus_raw, query_raw])
    coords = PCA(n_components=2, random_state=0).fit_transform(all_raw)
    corpus_xy, query_xy = coords[:-1], coords[-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    panels = [
        ("Cosine top-5", top_cos, "#2563eb"),
        ("Dot product top-5", top_dot, "#dc2626"),
        ("Euclidean top-5", top_euc, "#059669"),
    ]
    for ax, (title, top_idx, color) in zip(axes, panels):
        ax.scatter(corpus_xy[:, 0], corpus_xy[:, 1], c="#cbd5e1", s=40, label="corpus")
        ax.scatter(
            corpus_xy[top_idx, 0], corpus_xy[top_idx, 1],
            c=color, s=90, edgecolors="black", linewidths=0.6, label="top-5", zorder=3,
        )
        ax.scatter(
            query_xy[0], query_xy[1], marker="*", c="#f59e0b", s=320,
            edgecolors="black", linewidths=0.8, label="query", zorder=4,
        )
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    fig.suptitle("Same query, three metrics, top-5 highlighted (PCA to 2D)", fontsize=13)
    fig.tight_layout()
    out = os.path.join("outputs", "metric_comparison.png")
    fig.savefig(out, dpi=130)
    print()
    print(f"Saved figure -> {out}")
    print("(2D PCA flattens 384 dims to a picture, so trust the printed rankings")
    print(" over the plot for the fine distinctions -- the plot is for intuition.)")


if __name__ == "__main__":
    _demo()
