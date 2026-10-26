# Plan: relevance-feedback re-ranking of batches

> Design notes written before the strategy was built. Since then it is `SELECTION=rerank` (`RerankStrategy` in [rerank_strategy.py](rerank_strategy.py)), the Representation module became the Embedding module ([modules/embedding/](../embedding/), `EMBEDDING=metadata_text | multimodal`), and the server-side search/selection banks were replaced by the browser sending the pool, the kept items and the negatives with every `/next`. The implemented behaviour is documented in [DESIGN.md](../../DESIGN.md) §4.

## Context
The UI shows the pool in batches of `N_PER_BATCH` (10). When the curator presses **Next** at the frontier, `nextBatch()` in [templates/index.html](../../templates/index.html) sends `/next` the full pool plus a `SelectionHistory`:
- `accepted`: the items in the batch the curator kept
- `rejected`: the items in the batch the curator did not keep, plus items removed from the selection
- `seen`: everything shown so far

`RandomStrategy` ignores this feedback. The goal is a `SelectionStrategy` that re-ranks the unseen part of the pool after every batch, using the embeddings in the representation banks ([modules/representation/memory_bank.py](../embedding/service.py)), so the feedback on batch *k* shapes batch *k+1*. Feedback accumulates over the whole query.

The plan follows three rules:
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

**Displayed batch is encoded first.** `RepresentationService.prioritize(ids)` exists, and `/search` and `/next` in [app.py](../../app.py) already call it with the displayed ids. Each batch on screen is therefore queued for encoding while the curator looks at it, and by the time they press Next its vectors are usually ready. The strategy needs no change for this.
  - One limit: a pool batch that is already being encoded is not interrupted, so with `REPRESENTATION_BATCH_SIZE=8` the feedback items can still be up to one batch late. That is what `RERANK_WAIT` covers.

## Files
- **New: `modules/selection/relevance_feedback.py`**, containing `RelevanceFeedbackStrategy(SelectionStrategy)`.
  - `__init__(self, representation, view, scoring, beta, gamma, k, mmr_lambda, explore, wait)`
  - `select(pool, n, history)` builds P, N and C, waits briefly, scores, then composes the batch with MMR and exploration.
  - Helpers `_score_rocchio` and `_score_knn` are pure numpy over `bank.matrix(view)`, so they can be unit-tested without the model.
  - A `last_debug` dict holds the coverage, |P|, |N|, the explored ids and the score range, for the logger.
  - If `representation` is `None`, log a warning and delegate to `RandomStrategy`, so a misconfiguration never breaks the app.
- **[config.py](../../config.py)**:
  - Add `REGISTRY["selection"]["relevance_feedback"]`.
  - Change `_build_selection(key, representation=None)` so it reads the `RERANK_*` env vars when the key is `relevance_feedback`.
  - In `build_modules()`, build the representation first and pass it in.
- **[app.py](../../app.py)**: no change is required. Optionally, `/next` passes `strategy.last_debug` to `log_next`.
- **[modules/run_logger.py](../../modules/run_logger.py)**: add an optional `rerank` field to `log_next`, holding the scores, coverage and explored ids. This is what the evaluation runs on.
- **[templates/index.html](../../templates/index.html)**: no visible UI change; there is no explicit "not relevant" button. One logic fix only. `keepItem` should remove the id from `serverHistory.rejected`. Today, an item kept after navigating back stays in both `accepted` and `rejected`. P ∪ N handles this on the server, but the history should be consistent.
- **[.env.example](../../.env.example)**: add a section with these variables:
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
- **[DESIGN.md](../../DESIGN.md) / [README.md](../../README.md)**: add a module-table row and a short "Relevance feedback" section covering the feedback model, scoring, exploration and the env vars.

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
  - diversity, using [diversity_evaluator.py](../../modules/retrieval_evaluator/diversity_evaluator.py)

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

## Addendum: explicit negatives

### Context
Today the browser builds `history.rejected` implicitly. Two things put an item there:
- it was in a batch at the frontier and wasn't kept when the curator pressed Next ([templates/index.html:1123](templates/index.html#L1123)), or
- it was kept and then removed from the selection (`removeItem`, [templates/index.html:975](templates/index.html#L975)).

Both are weak, ambiguous signals. "Not kept" can mean "fine, but not here", and un-keeping can just be tidying up. This addendum replaces them with **explicit negatives**: the curator marks an item "not relevant" and can see and edit that list. This supersedes Decision 2 above ("not kept" as an implicit negative, no "not relevant" button).

### Behaviour
- **Marking a negative.** A pool card shows a small × in its top-right corner on mouse-over. It reuses the `.sel-remove` look from the selection cards.
  - Clicking it marks the item negative: a red outline, and the image dimmed. The × then stays visible.
  - Clicking × again un-marks the item.
  - Cards that are kept show no ×.
  - Pressing **Keep** on a negative card moves it from the negatives to the selection. Keep and negative never overlap.
- **Nothing implicit.** Pressing Next no longer rejects the items that weren't kept, and removing an item from the selection doesn't make it a negative.
- **Persistence.** Negatives persist across searches, like the selection. They are a page-level list of items and are not reset in `doSearch`.
  - On the server they follow `RERANK_CROSS_QUERY`, as positives do. With `true` all of them count. With `false` only the negatives in the current pool count.
- **Tabs in the selection panel.** The panel gets two tabs, **Selection (n)** and **Negatives (n)**.
  - The Selection tab is today's grid.
  - The Negatives tab shows the same small cards, and the × on each removes that item from the negatives. If the item's card is in the batch on screen, the card is updated too.
  - Export and Build Exhibition stay below both tabs and act on the selection only.
- **Timing.** A change to the negatives takes effect on the next **Next**. Marking an item doesn't re-rank anything immediately.
- **Not shown again.** The relevance-feedback strategy leaves negatives out of the candidates. A negative from an earlier search therefore won't come back in a later one. `RandomStrategy` is unchanged.

### Feedback model (revised)
- **P** is unchanged: `history.accepted` ∪ the selection bank.
- **N** = `history.rejected` − P. `history.rejected` now carries the explicit negatives only.
  - With `RERANK_CROSS_QUERY=false`, N is limited to ids in the pool.
- **C** = pool − `history.seen` − N.
- **Weight.** The default `RERANK_GAMMA` goes from 0.25 to **0.5**. An explicit negative is a stronger signal than "not kept", but it still weighs less than a positive (β=1.0), so one negative doesn't outweigh the positives.
- **Earlier searches.** A negative from an earlier search usually has no entry in the current search bank. `rep.vector()` already falls back to the encoder cache ([memory_bank.py:221](modules/representation/memory_bank.py#L221)), so such negatives still count while they are cached. A negative that has been evicted from the cache is ignored, as items without vectors already are.

`SelectionHistory` and the `select(pool, n, history)` interface stay the same. Only the meaning of `rejected` changes.

### Files
- **[templates/index.html](../../templates/index.html)**, where nearly all of the change is:
  - **State.**
    - Add `let negatives = []`, a list of item objects like `selected`, and an `isNegative(id)` helper.
    - `serverHistory.rejected` is no longer kept up to date by hand. Each `/next` request instead sends `history: {...serverHistory, rejected: negatives.map(i => i.id)}`. There are two calls: the multi-query first batch in `doSearch` and `nextBatch`.
  - **`accepted` simplified.**
    - `keepItem` adds the id to `serverHistory.accepted` (no duplicates) and `removeItem` removes it.
    - `nextBatch` no longer pushes anything into `accepted` or `rejected`.
    - Delete the `rejIdx` workaround in `keepItem`; it is no longer needed.
  - **Pool card.**
    - `renderPool` adds `<button class="sel-remove neg-btn" title="Not relevant">×</button>` to cards that aren't kept, and the class `negative` when the item is a negative.
    - New `toggleNegative(item, card)` adds or removes the negative, updates the card, and calls `renderNegatives()`.
    - `keepItem` drops the item from the negatives first, then removes the card's `.negative` class and its `.neg-btn`.
    - `removeItem` adds the `.neg-btn` back to the card when it restores it.
  - **CSS.**
    - `.artifact-card { position: relative; }`
    - `.neg-btn { display: none }`, with `.artifact-card:hover .neg-btn` and `.artifact-card.negative .neg-btn` set to `display: block`.
    - `.artifact-card.negative`: a red box-shadow (the `rgba(160,30,30,…)` already used by `.sel-remove:hover`) and the image at reduced opacity.
  - **Tabs.**
    - Replace `.section-meta` in `#selected-section` with a tab bar of two buttons.
    - Add `#negative-grid` next to `#selected-grid`, with one of the two hidden.
    - Turn the card-list half of `renderSelected` into `renderCardList(grid, items, onRemove, emptyMsg)` and use it for both grids.
    - Add `renderNegatives()`. The tab labels carry the counts that `#sel-stat` shows today.
- **[modules/selection/relevance_feedback.py](rerank_strategy.py)**:
  - In `_feedback`, with `cross_query` off, filter N to the pool as well (today only the kept ids are filtered).
  - In `select`, leave N out of the candidates.
  - Update the module docstring: N is now explicit negatives, and negatives are excluded from candidates.
  - **CLI simulation.** `_simulate` currently rejects every relevant item that wasn't shown. Add `--neg-rate` (default 0.3): the fraction of the non-relevant items shown in a batch that the simulated curator marks negative. It approximates an explicit curator, who marks only the clear misses.
- **[modules/base.py](../base.py)** and **[modules/selection/_stub.py](_stub.py)**: change the comment and docstring on `rejected` to "item IDs the user explicitly marked not relevant".
- **[config.py](../../config.py)**: change the `RERANK_GAMMA` default to `"0.5"`.
- **[.env.example](../../.env.example)**: `RERANK_GAMMA=0.5`. Rewrite the comment so it describes explicit negatives, and say that `RERANK_CROSS_QUERY` applies to negatives too.
- **[tests/test_relevance_feedback.py](../../tests/test_rerank_strategy.py)**: add tests that
  - negatives are excluded from the candidates;
  - with `cross_query=False`, negatives outside the pool are dropped;
  - an id that is in both `rejected` and P counts as a positive.
- **[DESIGN.md](../../DESIGN.md) / [README.md](../../README.md)**:
  - Revise the Negatives bullet (DESIGN.md:206) and the `SelectionHistory` row.
  - Describe the × and the Negatives tab.
  - Note that `rejected` in `runs.log` means explicit negatives from now on, so older logs are not comparable on that field.

### Steps
1. Server: `_feedback` and candidate exclusion, then the tests, then `--neg-rate` in the CLI.
2. Config and `.env.example`: the gamma default and comments.
3. Frontend: the state and history changes, then the × on pool cards, then the tabs.
4. Docs, base.py and _stub.py comments.

### Verification
- `pytest tests/test_relevance_feedback.py`
- `python -m modules.selection.relevance_feedback <export.json> --target <uri> --neg-rate 0.3` should still beat random on recall.
- In the app (`SELECTION=relevance_feedback`):
  - Hovering a pool card shows the ×. Clicking it outlines the card in red and adds the item to the Negatives tab. Clicking it again undoes that.
  - Keep on a negative card moves it to the Selection tab.
  - Pressing Next with nothing marked: `runs.log` shows `rejected: []` and `n_neg: 0`.
  - Mark 3 items negative, then Next: `n_neg: 3`, and those items never come back.
  - Removing an item from the selection doesn't add it to the negatives.
  - Removing a negative in the Negatives tab restores the card on screen.
  - After a new search, the Negatives tab still lists the earlier negatives.
- With `SELECTION=random`: the × and the tabs work, and selection is random as before.
