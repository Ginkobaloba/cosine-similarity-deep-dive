"""
04 - Sparse vectors: TF-IDF text, where cosine truly shines.

Text turned into vectors is mostly zeros. A vocabulary has tens of thousands
of terms; any single document uses a few hundred. Storing that as a dense
array wastes almost all of its memory on zeros. scipy.sparse stores only the
non-zero entries (term -> weight), so a 50,000-term vocabulary costs you only
as much as the words a document actually contains.

Cosine similarity works identically on sparse vectors. The dot product only
needs the terms two documents share (everything else multiplies by zero), and
the norms only need the non-zero weights. The math does not care whether the
zeros are stored or implied.

Why cosine beats Euclidean for text
-----------------------------------
Document length is a nuisance variable. A 2,000-word article and a 200-word
summary of the same topic should be "close," but in raw count space the long
document sits much farther from the origin, so Euclidean distance reports them
as far apart purely because one is longer. Cosine divides length out and
compares only the *mix* of terms -- the direction in vocabulary space -- which
is what "about the same thing" actually means.

This file builds a small TF-IDF matrix, computes cosine similarity directly on
the sparse representation, and shows side by side how Euclidean distance gets
fooled by length while cosine does not.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.metrics.pairwise import euclidean_distances


def sparse_cosine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> float:
    """
    Cosine similarity between two sparse row vectors, computed by hand so the
    mechanics are visible. Works purely on stored non-zero entries.
    """
    # a @ b.T is a 1x1 sparse matrix; pull the scalar out.
    dot = (a @ b.T).toarray()[0, 0]
    norm_a = np.sqrt((a.multiply(a)).sum())
    norm_b = np.sqrt((b.multiply(b)).sum())
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _demo() -> None:
    docs = [
        "the engine needs an oil change and a new oil filter",
        # Same topic, padded out to ~3x the length. Cosine should still see it
        # as the closest match to doc 0; Euclidean will be misled by length.
        "the engine needs an oil change and a new oil filter because the old "
        "oil is dirty and the engine oil pressure is low so change the oil and "
        "replace the oil filter and check the oil level after the oil change",
        "machine learning models embed text into dense vectors for search",
        "the cat sat quietly on the warm windowsill in the afternoon sun",
    ]
    labels = ["oil-change (short)", "oil-change (long)", "ML embeddings", "cat nap"]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)  # sparse CSR matrix

    print("Sparse TF-IDF matrix")
    print("--------------------")
    print(f"  shape          : {tfidf.shape} (docs x vocabulary)")
    print(f"  stored entries : {tfidf.nnz} of {tfidf.shape[0] * tfidf.shape[1]} cells")
    density = 100 * tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1])
    print(f"  density        : {density:.1f}% (the rest are implied zeros)")
    print(f"  type           : {type(tfidf).__name__}")
    print()

    query = tfidf[0]  # the short oil-change doc

    # Hand-rolled sparse cosine vs sklearn, to prove they match.
    print("Cosine similarity of doc 0 ('oil-change short') vs every doc")
    print("-----------------------------------------------------------")
    sk_scores = sk_cosine(query, tfidf).ravel()
    for i, label in enumerate(labels):
        mine = sparse_cosine(query, tfidf[i])
        flag = "  <- self" if i == 0 else ""
        print(f"  {label:<20} hand={mine:+.4f}  sklearn={sk_scores[i]:+.4f}{flag}")
        assert abs(mine - sk_scores[i]) < 1e-9

    print()
    print("Cosine vs Euclidean: which one is fooled by length?")
    print("---------------------------------------------------")
    euc = euclidean_distances(query, tfidf).ravel()
    print(f"  {'document':<20} {'cosine sim':>11} {'euclidean dist':>15}")
    for i, label in enumerate(labels):
        print(f"  {label:<20} {sk_scores[i]:>11.4f} {euc[i]:>15.4f}")
    print()
    print("  By cosine, the long oil-change doc is the nearest non-self match:")
    print("  same term mix, length divided out. Watch how the Euclidean number")
    print("  inflates for the long doc even though it is about the exact same")
    print("  topic -- length, not meaning, is driving that distance.")


if __name__ == "__main__":
    _demo()
