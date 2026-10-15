# Plan: relevance-feedback re-ranking of batches

## Context
The UI shows the pool in batches of `N_PER_BATCH` (10). When the curator presses **Next** at the frontier, `nextBatch()` in [templates/index.html](templates/index.html) sends `/next` the full pool plus a `SelectionHistory`:
- `accepted`: the items in the batch the curator kept
- `rejected`: the items in the batch the curator did not keep, plus items removed from the selection
- `seen`: everything shown so far

`RandomStrategy` ignores this feedback. The goal is a `SelectionStrategy` that re-ranks the unseen part of the pool after every batch, using the embeddings in the representation banks ([modules/representation/memory_bank.py](modules/representation/memory_bank.py)), so the feedback on batch *k* shapes batch *k+1*. Feedback accumulates over the whole query.

The plan follows the same rules as [extension_plan.md](extension_plan.md):
- Nearly all the code goes in one new file.
- Shared files only get small, additive hooks.
- `SELECTION=random` keeps today's behaviour.

The `SelectionStrategy.select(pool, n, history)` interface stays the same. The representation service is passed in through the constructor, which is the route the representation plan already left open.

## Feedback model
- **Positives P** = `history.accepted` ∪ the ids in the selection bank. With `RERANK_CROSS_QUERY=false`, only selection-bank items that are also in the current pool count, so a deliberately different query isn't pulled back towards earlier themes.
  - The selection bank is the source of truth for what is kept, because the curator can go back and keep an item from an earlier batch.
- **Negatives N** = `history.rejected` − P.
  - "Not kept" is a weak signal. It can mean "fine, but not for this exhibition", so negatives carry a lower weight than positives.
- **Candidates C** = pool items whose id is not in `history.seen`.

## Scoring
The view is set by `RERANK_VIEW`, which defaults to `fused`. All vectors are L2-normalised, so a dot product is a cosine.

1. **Rocchio (default).** Build `q = β·centroid(P) − γ·centroid(N)`, then score each candidate as `s(c) = c · q`, using `β=1.0` and `γ=0.25`.
   - This is cheap: one matrix–vector product over the search bank.
2. **kNN (option, `RERANK_SCORING=knn`).** Score each candidate as `s(c) = mean top-k sim(c, P) − γ · mean top-k sim(c, N)`.
   - This works better when the curator builds an exhibition around several themes at once. A single centroid averages those themes into a point that matches none of them.
   - It is built from `rep.similarity(view, ids, bank)`.

Write the score to `item["score"]`, as the interface intends. The browser sends it back, and the logger records it.

## Composing a batch of n
- **Exploitation:** `n − n_explore` items, picked greedily with **MMR**: `argmax λ·s(c) − (1−λ)·max sim(c, already picked)` with `λ=0.7`. This stops the top 10 from being near-duplicates, which is the curatorial concern the diversity evaluator also has.
- **Exploration:** `n_explore = round(ε·n)` items drawn at random from the candidates (`ε=0.2` → 2 of 10). This keeps feedback flowing on parts of the pool the model hasn't seen, and avoids a filter bubble.
- **Cold start:**
  - If P and N are both empty (the first batch from `/search`, or nothing kept or rejected yet), fall back to random.
  - If only N is non-empty, use the negative term alone (move away from it) plus exploration.

## Unencoded items
`/search` returns while the pool is still being embedded in the background, at roughly 0.8 s per object with images. With a pool of several hundred objects, much of it will have no vectors when the second batch is requested. The strategy:
- Waits up to `RERANK_WAIT` (about 1 s) for the **feedback items** (P ∪ N) to have vectors, checking `rep.vector(id, view)`, so `/next` stays responsive.
  - It does not use `rep.wait("search")`: that waits for the whole pool, which would nearly always time out and add a full second to every `/next`.
- Ranks only the candidates that have a vector.
- Fills any remaining slots from the unencoded candidates at random, counted as exploration.
- Logs how many candidates were encoded (`coverage`), so we can tell when ranking was effectively random.
- Ignores P or N items that have no vector yet.

**Displayed batch is encoded first.** `RepresentationService.prioritize(ids)` exists, and `/search` and `/next` in [app.py](app.py) already call it with the displayed ids. Each batch on screen is therefore queued for encoding while the curator looks at it, and by the time they press Next its vectors are usually ready. The strategy needs no change for this.
  - One limit: a pool batch that is already being encoded is not interrupted, so with `REPRESENTATION_BATCH_SIZE=8` the feedback items can still be up to one batch late. That is what `RERANK_WAIT` covers.

## Files
- **New: `modules/selection/relevance_feedback.py`**, containing `RelevanceFeedbackStrategy(SelectionStrategy)`.
  - `__init__(self, representation, view, scoring, beta, gamma, k, mmr_lambda, explore, wait)`
  - `select(pool, n, history)` builds P, N and C, waits briefly, scores, then composes the batch with MMR and exploration.
  - Helpers `_score_rocchio` and `_score_knn` are pure numpy over `bank.matrix(view)`, so they can be unit-tested without the model.
  - A `last_debug` dict holds the coverage, |P|, |N|, the explored ids and the score range, for the logger.
  - If `representation` is `None`, log a warning and delegate to `RandomStrategy`, so a misconfiguration never breaks the app.
- **[config.py](config.py)**:
  - Add `REGISTRY["selection"]["relevance_feedback"]`.
  - Change `_build_selection(key, representation=None)` so it reads the `RERANK_*` env vars when the key is `relevance_feedback`.
  - In `build_modules()`, build the representation first and pass it in.
- **[app.py](app.py)**: no change is required. Optionally, `/next` passes `strategy.last_debug` to `log_next`.
- **[modules/run_logger.py](modules/run_logger.py)**: add an optional `rerank` field to `log_next`, holding the scores, coverage and explored ids. This is what the evaluation runs on.
- **[templates/index.html](templates/index.html)**: no visible UI change; there is no explicit "not relevant" button. One logic fix only. `keepItem` should remove the id from `serverHistory.rejected`. Today, an item kept after navigating back stays in both `accepted` and `rejected`. P ∪ N handles this on the server, but the history should be consistent.
- **[.env.example](.env.example)**: add a section with these variables:
  ```
  SELECTION=random              # random | relevance_feedback (needs REPRESENTATION)
  # RERANK_VIEW=fused           # fused | image | text (must be in REPRESENTATION_VIEWS)
  # RERANK_SCORING=rocchio      # rocchio | knn
  # RERANK_BETA=1.0
  # RERANK_GAMMA=0.25
  # RERANK_K=3
  # RERANK_MMR_LAMBDA=0.7
  # RERANK_EXPLORE=0.2
  # RERANK_WAIT=1.0             # seconds /next waits for the feedback items' vectors
  # RERANK_CROSS_QUERY=true     # items kept under earlier queries count as positives
  ```
- **[DESIGN.md](DESIGN.md) / [README.md](README.md)**: add a module-table row and a short "Relevance feedback" section covering the feedback model, scoring, exploration and the env vars.

## Steps
1. Implement the scoring and composition helpers in pure numpy, with unit tests on synthetic vectors:
   - A positive cluster should be ranked first.
   - Negatives should push items down.
   - MMR should spread picks across clusters.
   - Exploration should take `round(ε·n)` slots.
2. Write `RelevanceFeedbackStrategy` on top of the banks, then add the fallback paths: no representation, no feedback, and low coverage.
3. Wire it in through `config.py` and `.env.example`, then log the extra fields.
4. Apply the frontend history fix.
5. Docs.

## Evaluation
- **Offline simulation.** This is the main check, because it doesn't need a curator. Take an exported pool and define a simulated curator who keeps items that match a hidden target concept (a `type`/`material`/`place` URI from the Artifact). Run 5–10 batches with `random` and with `relevance_feedback`, and compare:
  - the per-batch acceptance rate
  - cumulative recall
  - diversity, using [diversity_evaluator.py](modules/retrieval_evaluator/diversity_evaluator.py)

  This would be a CLI, `python -m modules.selection.relevance_feedback export.json --target <uri>`, like the other module CLIs.
- **Live logs.** `runs.log` already records the history for every `next` event. The acceptance rate per batch index, compared across the `random` and `relevance_feedback` conditions, is the in-use metric. It should rise over batches without diversity collapsing.
- **Sanity checks:**
  - With `SELECTION=random`, the app behaves as it does now.
  - With `REPRESENTATION` blank and `SELECTION=relevance_feedback`, a warning is logged and selection is random.
  - Keep three objects of one kind and press Next → more objects of that kind should appear, plus about 2 exploratory ones.

## Decisions
1. **Scope of positives:** controlled by the flag `RERANK_CROSS_QUERY`, default `true`. Items kept under earlier queries count as positives. Setting it to `false` limits positives to kept items that are in the current pool.
2. **"Not kept" as feedback:** it stays an implicit, low-weight negative (`RERANK_GAMMA=0.25`). There is no explicit "not relevant" button, and the UI stays as it is.

## Deferred
- **Query text as the Rocchio anchor `q0`.** This would make the first batch non-random. It needs a text-only query embedding, and the encoder is built for objects, so it waits until it can be done together with the embeddings owner. Until then, the first batch is random.
