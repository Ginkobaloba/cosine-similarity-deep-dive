"""
06 - A k-Nearest-Neighbors classifier built on cosine similarity.

kNN is the most direct application of a similarity metric to a real ML task:
to label a new point, find its k closest neighbors in the training set and let
them vote. The only design choice that matters here is "closest by what?" --
and swapping cosine for Euclidean changes the answer.

This file implements a small cosine-kNN from scratch, trains it on sklearn's
8x8 handwritten digits, evaluates accuracy, and compares it head to head with
a Euclidean kNN on the identical split.

When does cosine win for kNN?
-----------------------------
Cosine wins when the *direction* of the feature vector carries the signal and
the *magnitude* is noise. For the digits data, a brighter scan of the same
digit has larger pixel values everywhere -- bigger magnitude, same shape.
Cosine ignores that brightness and compares the pattern of strokes.

Watch the actual numbers, though: on this dataset the two metrics land
essentially tied (Euclidean is even a hair ahead at k=1). That is the honest
lesson. Cosine is not automatically better. Digits pixel vectors have fairly
uniform magnitude, so removing magnitude buys little here. Cosine pulls
clearly ahead only when magnitude really is noise (long vs short documents,
differing scan brightness, varying record counts). In genuinely
magnitude-meaningful data the advantage flips the other way (see file 10).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise unit normalization (also handles a single 1-D vector)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        n = np.linalg.norm(matrix)
        return matrix if n == 0.0 else matrix / n
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pairwise cosine, exposed for cross-file equality testing."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a / na) @ (b / nb))


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Predict labels for every row of x_query by majority vote of k nearest
    training neighbors. metric is "cosine" or "euclidean".

    For cosine we rank by *highest* similarity; for euclidean by *smallest*
    distance. Both reduce to one matrix operation over the whole query batch.
    """
    if metric == "cosine":
        train_u = normalize(x_train)
        query_u = normalize(x_query)
        sims = query_u @ train_u.T              # (n_query, n_train), higher = closer
        neighbors = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    elif metric == "euclidean":
        # ||q - t||^2 = ||q||^2 - 2 q.t + ||t||^2; rank by this, lower = closer.
        q2 = np.sum(x_query ** 2, axis=1, keepdims=True)
        t2 = np.sum(x_train ** 2, axis=1)
        d2 = q2 - 2.0 * (x_query @ x_train.T) + t2
        neighbors = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    preds = np.empty(len(x_query), dtype=y_train.dtype)
    for i, nbr_idx in enumerate(neighbors):
        votes = y_train[nbr_idx]
        preds[i] = np.bincount(votes).argmax()
    return preds


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _demo() -> None:
    digits = load_digits()
    x, y = digits.data, digits.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Digits dataset: {x.shape[0]} samples x {x.shape[1]} features (8x8 pixels)")
    print(f"  train: {len(x_train)}   test: {len(x_test)}")
    print()
    print(f"{'k':>3} | {'cosine acc':>11} | {'euclidean acc':>14}")
    print("-" * 36)
    for k in (1, 3, 5, 7):
        cos_pred = knn_predict(x_train, y_train, x_test, k=k, metric="cosine")
        euc_pred = knn_predict(x_train, y_train, x_test, k=k, metric="euclidean")
        print(f"{k:>3} | {_accuracy(y_test, cos_pred):>11.4f} | {_accuracy(y_test, euc_pred):>14.4f}")
    print("-" * 36)
    print()
    print("On digits the two metrics land neck and neck (Euclidean is even a")
    print("hair ahead at k=1). The honest read: removing magnitude barely helps")
    print("here because the pixel vectors already have similar magnitude. The")
    print("metric is the only thing that changed between those two columns --")
    print("same data, same vote, different notion of 'near' -- and on THIS data")
    print("it barely moves the needle. Cosine's edge shows up elsewhere (file 10).")


if __name__ == "__main__":
    _demo()
