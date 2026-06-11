# Cosine Similarity Deep Dive

A runnable, build-up-from-zero course on the single most important number in
modern vector search. By the end you can put cosine similarity on a whiteboard,
implement it five different ways without notes, and know exactly when it is the
wrong tool to reach for.

> **One-line definition:** cosine similarity is the cosine of the angle between
> two vectors. It measures whether they point the same direction and ignores how
> long they are.

## The geometric picture, in words

Picture two arrows from the origin. Cosine similarity does not care how long
either arrow is. It cares only about the angle between them.

- Same direction (angle 0) gives **1.0**. Maximum similarity.
- Right angle (90 degrees) gives **0.0**. Unrelated.
- Opposite directions (180 degrees) gives **-1.0**. Maximum dissimilarity.

That "ignore the length, keep the direction" property is the whole reason it
dominates text and embedding search. A long document and a short summary of the
same topic point the same direction in word-space even though one arrow is much
longer. Cosine sees them as nearly identical. Euclidean distance, which measures
the straight-line gap between the arrow tips, gets fooled by the length
difference and reports them as far apart.

## The formula

For two vectors A and B:

```
                A . B              sum(a_i * b_i)
cos(theta) = ----------- = ----------------------------------
             ||A|| ||B||    sqrt(sum a_i^2) * sqrt(sum b_i^2)
```

Three steps:

1. **Dot product** `A . B` measures raw agreement, component by component.
2. **Magnitudes** `||A||` and `||B||` are the lengths (L2 norms) of each vector.
3. **Divide** the dot product by both lengths to strip magnitude out and leave
   pure direction.

It is called *cosine* because the dot product also equals
`||A|| ||B|| cos(theta)`. Rearrange and the lengths cancel, leaving the cosine
of the angle. The formula does not approximate the angle, it returns it exactly.

The one gotcha: a zero vector has no direction and zero length, so the formula
divides by zero. Every implementation in this repo guards for it and returns
`0.0` by convention.

## Why it shows up everywhere

Once embeddings became the default representation for text, images, and audio,
cosine similarity became the default way to compare them, for a few reasons that
compound:

- **Magnitude is usually noise in embedding space.** Models like
  sentence-transformers are trained so that *direction* encodes meaning. Two
  paraphrases land in the same direction; their magnitudes are an artifact of
  token count and other incidentals you want to ignore.
- **Normalize once and it becomes a dot product.** If every vector is scaled to
  unit length, the denominator is `1 * 1 = 1` and cosine collapses to a bare dot
  product. That turns a corpus search into a single matrix multiply, which is
  the fastest thing a modern CPU or GPU does. This is why Pinecone, pgvector,
  Chroma, Weaviate, Qdrant, and FAISS all store normalized vectors and expose
  "inner product" as the metric. You have been using this optimization without
  knowing it had a name.
- **It is bounded and interpretable.** Always in `[-1, 1]`, so thresholds and
  rankings behave predictably across datasets.

## The files, in order

Each file runs on its own and prints something worth reading. They build on each
other, so go top to bottom the first time through.

| # | File | What it teaches |
|---|------|-----------------|
| 01 | `01_from_scratch.py` | Pure Python, scalar. The math step by step with a 2D worked example and the zero-vector guard. |
| 02 | `02_numpy_vectorized.py` | The same formula in numpy. Microbenchmark at N = 10 / 1k / 100k showing why vectorization wins (and when it does not). |
| 03 | `03_normalize_once.py` | The "normalize once, then it is just a dot product" trick. Benchmarked against naive cosine on a 1M-vector search. |
| 04 | `04_sparse_vectors.py` | TF-IDF sparse vectors with scipy.sparse. Why cosine beats Euclidean for text, with the length-bias shown concretely. |
| 05 | `05_topk_corpus_search.py` | Top-K search as one matrix multiply `C @ q` plus a partial sort. The core primitive of every vector DB. |
| 06 | `06_knn_classifier.py` | A cosine-kNN classifier on sklearn digits, head to head with Euclidean kNN. When cosine wins and why. |
| 07 | `07_semantic_search.py` | Real embeddings (all-MiniLM-L6-v2). Semantic search over a mixed corpus where queries light up their own cluster. |
| 08 | `08_cosine_vs_euclidean_vs_dot.py` | Three metrics side by side on the same embeddings, where they agree and split, with a 2D PCA plot. |
| 09 | `09_ann_intro.py` | HNSW approximate search on 100k vectors. The recall-vs-latency dial that "tuning a vector DB" actually means. |
| 10 | `10_when_cosine_is_wrong.py` | The honest counterweight. Three failure modes where cosine misleads, each with runnable proof. |

The progression is deliberate: scalar truth (01) -> make it fast (02-03) ->
make it real with text (04) -> make it a search engine (05) -> make it a
classifier (06) -> make it semantic (07) -> compare the alternatives (08) ->
scale it the way production does (09) -> learn when not to use it (10).

## Quick start (Windows)

```powershell
# From the repo root:
.\make.ps1 setup      # create .venv and install everything (pulls torch, give it a minute)
.\make.ps1 run 01     # run lesson 01
.\make.ps1 all        # run every lesson 01-10 in order
.\make.ps1 test       # run the pytest suite
```

Not on Windows, or prefer manual control:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 01_from_scratch.py
pytest -q
```

Lessons 07 and 08 download the `all-MiniLM-L6-v2` model on first run (a few
hundred MB, cached after that). Everything else is offline and fast.

## The test suite is the point

`tests/test_implementations_agree.py` verifies that **every cosine
implementation in files 01-06 returns the same number on the same input**,
within floating-point tolerance. The scalar version in 01 is easy to trust by
eye. Files 02-06 add vectorization, normalization, sparse storage, and batch
matrix math, and each of those is a place a subtle bug can hide (a wrong axis, a
dropped norm, an accidental transpose). The equality test is the net that
catches it. If you modify an optimized version and it drifts from the scalar
baseline, the suite goes red immediately.

```powershell
.\make.ps1 test
```

## Further reading

The actual literature, not blog summaries:

- **TF-IDF and the vector space model.** Salton, Wong, and Yang (1975),
  *A Vector Space Model for Automatic Indexing*, Communications of the ACM 18(11).
  Salton and Buckley (1988), *Term-Weighting Approaches in Automatic Text
  Retrieval*, Information Processing & Management 24(5). This is where comparing
  documents by the angle between their term vectors started.
- **Sentence embeddings.** Reimers and Gurevych (2019),
  *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP.
  The paper behind the sentence-transformers library used in lessons 07 and 08.
  https://arxiv.org/abs/1908.10084
- **Approximate nearest neighbor (HNSW).** Malkov and Yashunin (2018),
  *Efficient and robust approximate nearest neighbor search using Hierarchical
  Navigable Small World graphs*, IEEE TPAMI. The index behind lesson 09 and most
  production vector databases. https://arxiv.org/abs/1603.09320
- **The curse of dimensionality.** Aggarwal, Hinneburg, and Keim (2001),
  *On the Surprising Behavior of Distance Metrics in High Dimensional Space*,
  ICDT. Useful background for the concentration effect in lesson 10.

---

Built by [Paradigm Coding Solutions](https://github.com/Ginkobaloba). MIT
licensed, so fork it, teach from it, and rip pieces out for your own work.
