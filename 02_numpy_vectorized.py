"""
02 - The same math, vectorized with numpy.

The scalar version in 01 is correct and readable, but it walks every element
of the vector in a Python `for` loop. Python loops pay interpreter overhead on
every single iteration. numpy pushes the loop down into compiled C (and SIMD
instructions on your CPU), so the per-element cost collapses.

This file proves the point with a microbenchmark: compute cosine similarity
between pairs of random vectors at N = 10, 1,000, and 100,000 dimensions, and
print the speedup of the numpy version over the pure-Python version.

Takeaway: the *formula* is identical. What changes is who runs the loop.
"""

from __future__ import annotations

import time

import numpy as np

# Reuse the pure-Python implementation as the baseline so the comparison is
# apples-to-apples (same definition, different execution engine).
from importlib import import_module

_scalar = import_module("01_from_scratch")
scalar_cosine = _scalar.cosine_similarity


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vectorized cosine similarity for two 1-D numpy arrays.

    np.dot is a single compiled call; np.linalg.norm computes the L2 norm in
    C. Same three steps as the scalar version (dot, two norms, divide), just
    without a Python-level loop.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _time_it(fn, repeats: int) -> float:
    """Return the best (min) wall-clock seconds over `repeats` runs."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _benchmark() -> None:
    rng = np.random.default_rng(42)
    sizes = [10, 1_000, 100_000]

    print("Microbenchmark: scalar Python vs numpy (best of several runs)")
    print("=" * 64)
    print(f"{'dimension N':>12} | {'scalar (ms)':>12} | {'numpy (ms)':>11} | {'speedup':>8}")
    print("-" * 64)

    for n in sizes:
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        a_list = a.tolist()
        b_list = b.tolist()

        # Bigger N is slower, so do fewer repeats to keep the run snappy.
        repeats = 50 if n <= 1_000 else 5

        t_scalar = _time_it(lambda: scalar_cosine(a_list, b_list), repeats)
        t_numpy = _time_it(lambda: cosine_similarity(a, b), repeats)

        # Confirm both engines agree before reporting the speedup.
        assert abs(scalar_cosine(a_list, b_list) - cosine_similarity(a, b)) < 1e-9

        speedup = t_scalar / t_numpy if t_numpy > 0 else float("inf")
        print(f"{n:>12,} | {t_scalar * 1e3:>12.3f} | {t_numpy * 1e3:>11.3f} | {speedup:>7.1f}x")

    print("-" * 64)
    print()
    print("Why vectorization matters")
    print("-------------------------")
    print("  At N = 10 the numpy call can actually LOSE: the fixed overhead of")
    print("  building arrays and dispatching into C dwarfs the tiny loop. As N")
    print("  grows, that overhead amortizes and numpy pulls far ahead. This is")
    print("  the whole reason vector databases store embeddings as dense float")
    print("  arrays and never touch a Python loop on the hot path.")


if __name__ == "__main__":
    _benchmark()
