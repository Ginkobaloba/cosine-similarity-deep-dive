"""
The one test that earns its keep: every cosine implementation in files 01-06
must return the SAME number on the SAME input, within floating-point tolerance.

This is the safety net for the optimized versions. The scalar implementation in
01 is dead simple and easy to trust by eye. Files 02-06 add vectorization, the
normalize-once trick, sparse storage, and batch matrix math -- each an
opportunity to introduce a subtle bug (a wrong axis, a missing norm, a
transpose). If any of them disagrees with the scalar baseline, that is a bug in
the optimization, and this test catches it.
"""

from importlib import import_module

import numpy as np
import pytest
from scipy import sparse

# Import each stage's pairwise cosine. Digit-prefixed filenames can't use the
# normal `import` statement, so we go through importlib.
scalar = import_module("01_from_scratch").cosine_similarity
numpy_cos = import_module("02_numpy_vectorized").cosine_similarity
normalize_cos = import_module("03_normalize_once").cosine_similarity
_sparse_mod = import_module("04_sparse_vectors")
topk_cos = import_module("05_topk_corpus_search").cosine_similarity
knn_cos = import_module("06_knn_classifier").cosine_similarity


def sparse_cos(a, b):
    """Adapter: file 04 works on sparse rows, so wrap dense input as 1xN CSR."""
    sa = sparse.csr_matrix(np.asarray(a, dtype=np.float64).reshape(1, -1))
    sb = sparse.csr_matrix(np.asarray(b, dtype=np.float64).reshape(1, -1))
    return _sparse_mod.sparse_cosine(sa, sb)


# (name, callable) for every implementation under test.
IMPLEMENTATIONS = [
    ("01_scalar", lambda a, b: scalar(list(a), list(b))),
    ("02_numpy", lambda a, b: numpy_cos(np.asarray(a), np.asarray(b))),
    ("03_normalize_once", lambda a, b: normalize_cos(np.asarray(a), np.asarray(b))),
    ("04_sparse", sparse_cos),
    ("05_topk", lambda a, b: topk_cos(np.asarray(a), np.asarray(b))),
    ("06_knn", lambda a, b: knn_cos(np.asarray(a), np.asarray(b))),
]

TOL = 1e-9


def _vector_pairs():
    """A spread of pairs: hand-checkable, random, and edge cases."""
    rng = np.random.default_rng(2024)
    pairs = [
        ([1.0, 0.0], [1.0, 1.0]),               # 45 degrees -> ~0.7071
        ([2.0, 1.0], [4.0, 2.0]),               # collinear -> 1.0
        ([1.0, 0.0], [0.0, 1.0]),               # orthogonal -> 0.0
        ([1.0, 1.0], [-1.0, -1.0]),             # opposite -> -1.0
        ([3.0, -2.0, 5.0], [-1.0, 4.0, 2.0]),   # arbitrary 3D
    ]
    for _ in range(25):  # random higher-dim pairs
        a = rng.standard_normal(16)
        b = rng.standard_normal(16)
        pairs.append((a, b))
    return pairs


PAIRS = _vector_pairs()


@pytest.mark.parametrize("a, b", PAIRS)
def test_all_implementations_match_scalar(a, b):
    """Every implementation agrees with the pure-Python baseline."""
    baseline = scalar(list(a), list(b))
    for name, impl in IMPLEMENTATIONS:
        got = impl(a, b)
        assert got == pytest.approx(baseline, abs=TOL), (
            f"{name} disagrees with scalar baseline: {got} != {baseline}"
        )


@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
def test_known_45_degree_value(name, impl):
    """cos(45 degrees) = 1/sqrt(2). A value we can check by hand."""
    assert impl([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / np.sqrt(2), abs=TOL)


@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
def test_collinear_is_one(name, impl):
    assert impl([2.0, 1.0], [4.0, 2.0]) == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
def test_orthogonal_is_zero(name, impl):
    assert impl([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
def test_opposite_is_minus_one(name, impl):
    assert impl([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0, abs=TOL)


@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
def test_zero_vector_guard_returns_zero(name, impl):
    """No NaN, no divide-by-zero: the zero vector is guarded to 0.0 everywhere."""
    result = impl([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert result == pytest.approx(0.0, abs=TOL)
    assert not np.isnan(result)


def test_symmetry():
    """cosine(a, b) == cosine(b, a) for every implementation."""
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    for name, impl in IMPLEMENTATIONS:
        assert impl(a, b) == pytest.approx(impl(b, a), abs=TOL), name
