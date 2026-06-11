# Learning Path

Read this top to bottom once, then start running files. The repo is built as a
ladder: every file assumes you ran the one before it. You do not need to read
the source of all ten before you start, you need to run them in order and watch
what each one prints, because the printed output is the lesson. Run setup first
(`.\make.ps1 setup`), then `.\make.ps1 run 01` through `run 10`, or
`.\make.ps1 all` to fire them in sequence. The goal at the end is concrete: you
can derive the formula on a whiteboard, you can type out a working
implementation under interview pressure without notes, and you can say out loud
when cosine is the wrong choice and what to use instead.

**If you have 30 minutes**, do the spine and skip the scaling detours. Run `01`
and actually read its output, the 2D worked example is the whole geometric
intuition in one screen, and make sure you can re-derive `cos(theta) = A.B /
(||A|| ||B||)` from "the dot product equals the product of lengths times the
cosine of the angle." Then run `03` to internalize the one idea that makes
vector databases fast: normalize every vector to unit length once, and cosine
collapses into a bare dot product, which is why Pinecone and pgvector ask for
normalized vectors and call the metric "inner product." Then run `07` to see it
do real work, semantic search where queries match meaning instead of keywords.
Close with `10` so you leave knowing the failure modes, not just the wins. Three
files, one core truth each: what it is, why it is fast, and when it lies.

**If you have 2 hours**, go straight through `01` to `10` in order and run the
test suite (`.\make.ps1 test`) somewhere in the middle, because that suite is the
part most people skip and it is the part that proves you actually understand the
math. It checks that all six implementations in files `01` through `06` return
the identical number on the identical input, which is the real test of whether
the optimized versions (vectorized, normalized, sparse, batched) are faithful to
the dead-simple scalar one. Spend the most time on `02` and `03` (why
vectorization wins and the normalize-once trick), `05` (top-K search as a single
matrix multiply `C @ q`, which is literally what a vector DB does on every
query), and `09` (HNSW and the recall-versus-latency dial, which is where
production vector search actually lives). Files `04`, `06`, and `08` are the
"prove it on real-ish data" reinforcement: sparse text, a kNN classifier, and a
three-metric showdown with a plot. By the end you should be able to whiteboard
the formula, explain why normalization turns it into a dot product, sketch how
top-K search scales and where ANN takes over, and name at least two situations
where you would reach for Euclidean or raw dot product instead.

**The one thing to over-index on:** the relationship between cosine, the dot
product, and normalization. Almost everything practical about vector search
falls out of a single fact, that on unit vectors cosine similarity, the dot
product, and (inverted) Euclidean distance all produce the same ranking. Once
that clicks, the speed tricks in `03` and `05`, the metric choices in `08`, and
the failure modes in `10` stop being separate facts and become one idea seen
from different angles. If you can explain that fact cold, you can explain
everything else in this repo.
