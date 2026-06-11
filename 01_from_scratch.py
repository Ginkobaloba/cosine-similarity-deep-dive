"""
01 - Cosine similarity from scratch (pure Python, no dependencies).

The whole idea in one sentence: cosine similarity measures the angle between
two vectors, ignoring how long they are. Two vectors that point the same
direction score 1.0 no matter their magnitude. Two that are perpendicular
score 0.0. Two that point in opposite directions score -1.0.

The math, step by step
-----------------------
Given two vectors A and B of the same dimension:

1. Dot product:           A . B = sum(a_i * b_i)
   This is large when the vectors line up component-by-component and small
   (or negative) when they don't. It is the raw, un-normalized measure of
   agreement.

2. Magnitude (L2 norm):   ||A|| = sqrt(sum(a_i^2))
   The length of the vector, i.e. its Euclidean distance from the origin.

3. Divide out the lengths: cos(theta) = (A . B) / (||A|| * ||B||)
   Dividing by both magnitudes strips length out of the picture entirely.
   What survives is pure direction.

Why "cosine"? Because the dot product has a second, geometric definition:

        A . B = ||A|| * ||B|| * cos(theta)

where theta is the angle between the two vectors. Rearrange it and the
length terms cancel:

        cos(theta) = (A . B) / (||A|| * ||B||)

So the formula literally returns the cosine of the angle between the
vectors. cos(0 degrees) = 1 (same direction), cos(90 degrees) = 0
(orthogonal, "unrelated"), cos(180 degrees) = -1 (opposite).

The zero-vector trap
--------------------
A zero vector has no direction and zero magnitude, so the formula divides
by zero. There is no meaningful angle to a vector that points nowhere, so we
guard for it and return 0.0 by convention.
"""

from __future__ import annotations

import math
from typing import Sequence


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Sum of element-wise products. Raw, un-normalized agreement."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def magnitude(a: Sequence[float]) -> float:
    """Euclidean (L2) length of the vector: sqrt(sum of squares)."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine of the angle between a and b.

    Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector is the
    zero vector (no direction -> no defined angle), which also avoids a
    division-by-zero blow-up.
    """
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)


def _worked_example() -> None:
    """A 2D example you can verify by hand on a whiteboard."""
    # Two vectors 45 degrees apart: one points straight right (along x),
    # the other points up-and-right at 45 degrees.
    a = [1.0, 0.0]   # 0 degrees
    b = [1.0, 1.0]   # 45 degrees

    dp = dot_product(a, b)
    ma = magnitude(a)
    mb = magnitude(b)
    cos = cosine_similarity(a, b)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    print("2D worked example")
    print("-----------------")
    print(f"A = {a}   (points along the x-axis, 0 degrees)")
    print(f"B = {b}   (points up-and-right, 45 degrees)")
    print()
    print(f"  dot(A, B)   = (1*1) + (0*1)           = {dp}")
    print(f"  ||A||       = sqrt(1^2 + 0^2)         = {ma}")
    print(f"  ||B||       = sqrt(1^2 + 1^2)         = {mb:.6f}")
    print(f"  cos(theta)  = {dp} / ({ma} * {mb:.6f}) = {cos:.6f}")
    print(f"  theta       = acos({cos:.6f})         = {angle_deg:.2f} degrees")
    print()
    print("  Sanity check: cos(45 degrees) = 0.7071..., and the two vectors")
    print("  really are 45 degrees apart. The math agrees with the geometry.")
    print()

    # A few more cases to build intuition.
    cases = [
        ("identical direction", [2.0, 1.0], [4.0, 2.0]),   # B = 2A -> cos 1.0
        ("orthogonal",          [1.0, 0.0], [0.0, 5.0]),   # 90 deg -> cos 0.0
        ("opposite direction",  [1.0, 1.0], [-1.0, -1.0]), # 180 deg -> cos -1.0
        ("zero vector guard",   [0.0, 0.0], [1.0, 2.0]),   # guarded -> 0.0
    ]
    print("Intuition table")
    print("---------------")
    for label, x, y in cases:
        print(f"  {label:<22} cos({x}, {y}) = {cosine_similarity(x, y):+.4f}")


if __name__ == "__main__":
    _worked_example()
