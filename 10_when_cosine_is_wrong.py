"""
10 - When cosine is the WRONG choice. The honest counterweight.

Cosine is the default for embedding search, and the previous nine files earned
that default. But "default" is not "always." Reaching for cosine reflexively on
the wrong data gives you confident, wrong answers. Three failure modes, each
with runnable proof.

1. MAGNITUDE-MEANINGFUL VECTORS
   When length carries real information, throwing it away is throwing away the
   signal. Physical coordinates, raw counts where the total matters, sensor
   readings, prices -- cosine says two of these are "identical" whenever they
   point the same way, even if one is ten times the other.

2. HIGH-DIM SPARSE: COSINE CONCENTRATES
   In very high dimensions, random sparse vectors become nearly orthogonal and
   their cosine similarities pile up in a narrow band. The metric loses
   contrast -- everything looks about equally (dis)similar -- so a threshold
   that separates "related" from "unrelated" gets brittle.

3. EMBEDDINGS NOT TRAINED FOR COSINE
   An embedding space has a geometry baked in by its training objective. If a
   model was trained with Euclidean/L2 distance (or dot product over
   un-normalized vectors), scoring it with cosine silently mismatches the
   space the model actually learned. Always score with the metric the model
   was trained on.

The lesson is not "cosine bad." It is: know what your vectors mean, and pick
the metric that respects it.
"""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def case_1_magnitude_meaningful() -> None:
    print("Case 1: magnitude carries the meaning")
    print("-------------------------------------")
    # Monthly spend profiles. Direction = WHERE money goes; magnitude = HOW MUCH.
    budget_shopper = np.array([50.0, 30.0, 20.0])    # rent, food, fun ($100 total)
    big_spender = np.array([500.0, 300.0, 200.0])    # same SPLIT, 10x the money
    different_split = np.array([20.0, 20.0, 60.0])   # same $100, different priorities

    print(f"  budget_shopper  = {budget_shopper}  (total ${budget_shopper.sum():.0f})")
    print(f"  big_spender     = {big_spender}  (total ${big_spender.sum():.0f})")
    print(f"  different_split = {different_split}  (total ${different_split.sum():.0f})")
    print()
    print(f"  cosine(budget, big_spender)     = {cosine(budget_shopper, big_spender):.4f}  <- 'identical'")
    print(f"  cosine(budget, different_split) = {cosine(budget_shopper, different_split):.4f}")
    print(f"  euclid(budget, big_spender)     = {euclidean(budget_shopper, big_spender):.2f}  <- far apart")
    print(f"  euclid(budget, different_split) = {euclidean(budget_shopper, different_split):.2f}")
    print()
    print("  Cosine calls the $100 and $1000 budgets identical because they")
    print("  spend in the same proportions. If your question is 'how much do")
    print("  they spend?', cosine erased the answer. Use Euclidean (or compare")
    print("  the raw totals) when magnitude IS the signal.")
    print()


def case_2_high_dim_concentration() -> None:
    print("Case 2: high-dim sparse -> cosine concentrates, contrast collapses")
    print("-----------------------------------------------------------------")
    rng = np.random.default_rng(0)
    # Model documents of FIXED length (nnz terms each) drawn from a vocabulary
    # of growing size (dim). This is the TF-IDF reality: short docs, big
    # vocabulary. Holding the term count fixed isolates the pure effect of
    # dimensionality, so the trend is clean instead of confounded by vectors
    # that happen to come out empty at small dim.
    nnz = 8
    n = 4_000
    for dim in (16, 64, 256, 1_024, 4_096):
        sims = np.empty(n)
        for j in range(n):
            a = np.zeros(dim)
            b = np.zeros(dim)
            a[rng.choice(dim, nnz, replace=False)] = rng.random(nnz)
            b[rng.choice(dim, nnz, replace=False)] = rng.random(nnz)
            sims[j] = cosine(a, b)
        print(f"  dim = {dim:>5} (nnz={nnz}):  mean cos = {sims.mean():.4f}   "
              f"std = {sims.std():.4f}   P(cos>0) = {np.mean(sims > 0):.2f}")
    print()
    print("  As the vocabulary grows, two fixed-length documents share a term")
    print("  less and less often. Cosine values collapse toward 0 and their")
    print("  spread shrinks: most pairs score exactly 0 and the rest barely")
    print("  register. A fixed 'similar if > threshold' cutoff loses its grip")
    print("  because there is almost no contrast left to threshold on. This is")
    print("  the curse of dimensionality biting sparse cosine, and it is why")
    print("  raw high-dim sparse vectors get dimensionality-reduced (LSA, or a")
    print("  learned dense embedding) before similarity search.")
    print()


def case_3_wrong_training_metric() -> None:
    print("Case 3: scoring with a metric the embedding was not trained for")
    print("---------------------------------------------------------------")
    rng = np.random.default_rng(1)
    # Pretend a model learned a space where SIMILAR items sit at the same
    # location plus a magnitude that encodes confidence/frequency. Distance in
    # this space is meant to be read as L2, not as an angle.
    anchor = np.array([3.0, 3.0])
    # Same direction, very different magnitude (model meant: low confidence).
    same_dir_far = np.array([0.2, 0.2])
    # Slightly different direction, similar magnitude (model meant: near match).
    near_in_space = np.array([3.4, 2.6])

    print(f"  anchor        = {anchor}")
    print(f"  same_dir_far  = {same_dir_far}   (cosine LOVES this)")
    print(f"  near_in_space = {near_in_space}   (L2 says THIS is the neighbor)")
    print()
    print(f"  cosine(anchor, same_dir_far)  = {cosine(anchor, same_dir_far):.4f}")
    print(f"  cosine(anchor, near_in_space) = {cosine(anchor, near_in_space):.4f}")
    print(f"  euclid(anchor, same_dir_far)  = {euclidean(anchor, same_dir_far):.4f}")
    print(f"  euclid(anchor, near_in_space) = {euclidean(anchor, near_in_space):.4f}")
    print()
    print("  Cosine and Euclidean pick DIFFERENT nearest neighbors here. If the")
    print("  model was trained with an L2 objective, cosine's answer is simply")
    print("  wrong for this space. Rule: score embeddings with the metric they")
    print("  were trained under -- check the model card before you assume cosine.")
    print()


def _demo() -> None:
    case_1_magnitude_meaningful()
    case_2_high_dim_concentration()
    case_3_wrong_training_metric()
    print("Bottom line: cosine answers 'same direction?'. When that is the right")
    print("question (most embedding search), it wins. When magnitude, density, or")
    print("the model's own geometry is the real question, it misleads. Match the")
    print("metric to the meaning of the vector, every time.")


if __name__ == "__main__":
    _demo()
