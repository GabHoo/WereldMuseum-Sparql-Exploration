"""
RerankStrategy — relevance feedback: re-ranks the unseen pool after every
batch, using the curator's feedback and the Embedding module's object vectors
(EMBEDDING=metadata_text | multimodal, in the EMBEDDING_VIEW space). It only
needs L2-normalised vectors, so it works with any embedding.

Feedback (the browser sends it with every /next):
    positives P  history.accepted: every kept object, also from earlier queries.
    negatives N  history.rejected − P: the objects the curator explicitly
                 marked not relevant. N weighs less than P (gamma < beta).
    cross_query=False keeps only the P and N objects in the current pool.
    Negatives are never shown again, also under a later query.

Scoring of each unseen candidate c (vectors are L2-normalised, dot = cosine):
    rocchio  s(c) = c · (beta·centroid(P) − gamma·centroid(N))
    knn      s(c) = mean top-k sim(c, P) − gamma · mean top-k sim(c, N)
             (better when the curator collects several themes at once)

A batch of n: round(explore·n) random candidates (exploration), the rest picked
greedily by MMR — argmax lambda·s(c) − (1−lambda)·max sim(c, picked) — so the
batch is not ten near-duplicates. The batch is shuffled, so ranked and explored
items are not told apart by position.

Vectors are filled in the background, so not every candidate has one yet:
only encoded candidates are ranked, and free slots are filled at random. The
strategy waits at most `wait` seconds for the feedback items' own vectors
(app.py prioritises the batch on screen, so they are usually ready).

Without feedback (e.g. the first batch of a query), or without the Embedding
module, it selects at random.

Usage (CLI, offline simulation against the random baseline):
    python -m modules.selection.rerank_strategy <pool.json | export.json | runs.log> \\
        [--target <concept URI>] [--batches 8] [--n 10] [--runs 5] [--neg-rate 0.3]
"""
import logging
import random
import sys

import numpy as np

from modules.base import SelectionHistory, SelectionStrategy

log = logging.getLogger(__name__)



# ── Scoring and batch composition (pure numpy) ───────────────────────────────

def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _score_rocchio(cands: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                   beta: float, gamma: float) -> np.ndarray:
    """cands m × d, pos p × d, neg q × d (p or q may be 0) → scores (m,)."""
    query = np.zeros(cands.shape[1], dtype=np.float32)
    if len(pos):
        query += beta * _unit(pos.mean(axis=0))
    if len(neg):
        query -= gamma * _unit(neg.mean(axis=0))
    return cands @ query


def _mean_top_k(sims: np.ndarray, k: int) -> np.ndarray:
    """Row-wise mean of the k largest values of sims (m × p)."""
    k = min(k, sims.shape[1])
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def _score_knn(cands: np.ndarray, pos: np.ndarray, neg: np.ndarray,
               beta: float, gamma: float, k: int) -> np.ndarray:
    scores = np.zeros(len(cands), dtype=np.float32)
    if len(pos):
        scores += beta * _mean_top_k(cands @ pos.T, k)
    if len(neg):
        scores -= gamma * _mean_top_k(cands @ neg.T, k)
    return scores


def _mmr(cands: np.ndarray, scores: np.ndarray, n: int, lam: float) -> list:
    """Indices of n candidates picked greedily by maximal marginal relevance."""
    picked: list = []
    max_sim = np.full(len(cands), -np.inf, dtype=np.float32)
    available = np.ones(len(cands), dtype=bool)
    for _ in range(min(n, len(cands))):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        gain = np.where(available, lam * scores - (1 - lam) * redundancy, -np.inf)
        best = int(np.argmax(gain))
        picked.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, cands @ cands[best])
    return picked


# ── Strategy ──────────────────────────────────────────────────────────────────

class RerankStrategy(SelectionStrategy):
    def __init__(self, embedding=None, view: str = "fused", scoring: str = "rocchio",
                 beta: float = 1.0, gamma: float = 0.5, k: int = 3, mmr_lambda: float = 0.7,
                 explore: float = 0.2, wait: float = 1.0, cross_query: bool = True,
                 rng: random.Random | None = None):
        if scoring not in ("rocchio", "knn"):
            raise ValueError(f"Unknown RERANK_SCORING: {scoring!r} (rocchio | knn)")
        if embedding is None:
            log.warning("SELECTION=rerank needs EMBEDDING; selecting at random")
        elif view not in embedding.encoder.views:
            raise ValueError(f"EMBEDDING_VIEW={view!r} is not computed by this embedding "
                             f"{list(embedding.encoder.views)}")
        self.emb = embedding
        self.view = view
        self.scoring = scoring
        self.beta, self.gamma, self.k = beta, gamma, k
        self.mmr_lambda = mmr_lambda
        self.explore = explore
        self.wait = wait
        self.cross_query = cross_query
        self._rng = rng or random.Random()
        self.last_debug: dict = {}   # what the last select() did, for the run log

    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        seen = set(history.seen) | set(history.rejected)
        cands = [item for item in pool if item["id"] not in seen]
        if self.emb is None:
            self.last_debug = {"mode": "random", "reason": "no embedding"}
            return self._rng.sample(cands, min(n, len(cands)))

        pos_ids, neg_ids = self._feedback(pool, history)
        if not pos_ids and not neg_ids:
            self.last_debug = {"mode": "random", "reason": "no feedback"}
            return self._rng.sample(cands, min(n, len(cands)))

        self.emb.wait(pos_ids + neg_ids, self.wait)
        pos = self._stack(pos_ids)
        neg = self._stack(neg_ids)
        if not len(pos) and not len(neg):
            self.last_debug = {"mode": "random", "reason": "feedback not encoded",
                               "n_pos": len(pos_ids), "n_neg": len(neg_ids)}
            return self._rng.sample(cands, min(n, len(cands)))

        # Candidates with a vector are ranked; the rest can only be explored.
        by_id = {item["id"]: item for item in cands}
        enc_ids, enc_mat = self.emb.matrix(list(by_id), self.view)

        if self.scoring == "knn":
            scores = _score_knn(enc_mat, pos, neg, self.beta, self.gamma, self.k) if enc_ids else np.zeros(0)
        else:
            scores = _score_rocchio(enc_mat, pos, neg, self.beta, self.gamma) if enc_ids else np.zeros(0)
        for item_id, score in zip(enc_ids, scores):
            by_id[item_id]["score"] = round(float(score), 4)

        n = min(n, len(cands))
        n_explore = round(self.explore * n)
        ranked = [enc_ids[i] for i in _mmr(enc_mat, scores, n - n_explore, self.mmr_lambda)] if enc_ids else []
        rest = [i for i in by_id if i not in set(ranked)]
        explored = self._rng.sample(rest, n - len(ranked))

        self.last_debug = {
            "mode":        self.scoring,
            "view":        self.view,
            "n_pos":       len(pos), "n_neg": len(neg),
            "coverage":    round(len(enc_ids) / len(cands), 3) if cands else 0.0,
            "ranked_ids":  ranked,
            "explored_ids": explored,
            "scores":      {i: by_id[i]["score"] for i in ranked},
            "score_range": [round(float(scores.min()), 4), round(float(scores.max()), 4)] if len(scores) else None,
        }
        batch = [by_id[i] for i in ranked + explored]
        self._rng.shuffle(batch)
        return batch

    # ── Private ───────────────────────────────────────────────────────────────

    def _feedback(self, pool: list, history: SelectionHistory) -> tuple:
        """(positive ids, negative ids), each without duplicates."""
        kept, rejected = history.accepted, history.rejected
        if not self.cross_query:
            in_pool = {item["id"] for item in pool}
            kept = [i for i in kept if i in in_pool]
            rejected = [i for i in rejected if i in in_pool]
        pos = list(dict.fromkeys(kept))
        pos_set = set(pos)
        neg = [i for i in dict.fromkeys(rejected) if i not in pos_set]
        return pos, neg

    def _stack(self, ids: list) -> np.ndarray:
        vecs = [v for v in (self.emb.vector(i, self.view) for i in ids) if v is not None]
        return np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)


# ── CLI: offline simulation ───────────────────────────────────────────────────

_CONCEPT_FIELDS = ("type_uris", "material_uris", "place_uris")


def _load_pool(path: str) -> list:
    """A pool list, an export {meta, items}, or a run log (last search event's pool)."""
    import json
    with open(path, encoding="utf-8") as f:
        if path.endswith((".log", ".jsonl")):
            pools = [r["pool"] for r in map(json.loads, filter(str.strip, f)) if r.get("event") == "search"]
            if not pools:
                raise SystemExit(f"No search event in {path}")
            return pools[-1]
        data = json.load(f)
    return data if isinstance(data, list) else data.get("items", [])


def _concepts(item: dict) -> set:
    props = item.get("properties", {})
    return {uri for field in _CONCEPT_FIELDS for uri in props.get(field) or []}


def _simulate(strategy, rep, pool: list, relevant: set, batches: int, n: int,
              neg_rate: float, rng: random.Random) -> list:
    """Per batch: (acceptance rate, cumulative recall, diversity of the batch).

    The simulated curator keeps every relevant item shown and marks a share
    neg_rate of the other items shown as negatives (only the clear misses).
    """
    history, kept, rows = SelectionHistory(), [], []
    for _ in range(batches):
        shown = strategy.select(pool, n, history)
        if not shown:
            break
        hits = [item for item in shown if item["id"] in relevant]
        kept += hits
        history.seen += [item["id"] for item in shown]
        history.accepted += [item["id"] for item in hits]
        misses = [item["id"] for item in shown if item["id"] not in relevant]
        history.rejected += rng.sample(misses, round(neg_rate * len(misses)))
        vecs = [v for v in (rep.vector(item["id"], strategy.view) for item in shown) if v is not None]
        if len(vecs) >= 2:
            mat = np.stack(vecs)
            sims = mat @ mat.T
            diversity = float(1 - sims[np.triu_indices(len(vecs), 1)].mean())
        else:
            diversity = float("nan")
        rows.append((len(hits) / len(shown), len(kept) / max(len(relevant), 1), diversity))
    # Pool exhausted early: pad, so runs line up per batch.
    return rows + [(float("nan"),) * 3] * (batches - len(rows))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.selection.rerank_strategy <pool.json | export.json | runs.log>"
              " [--target <concept URI>] [--batches 8] [--n 10] [--runs 5] [--neg-rate 0.3]")
        sys.exit(1)

    from collections import Counter

    from dotenv import load_dotenv
    load_dotenv()
    import os

    import config

    def _arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    pool = _load_pool(sys.argv[1])
    batches, n, runs = int(_arg("--batches", 8)), int(_arg("--n", 10)), int(_arg("--runs", 5))
    neg_rate = float(_arg("--neg-rate", 0.3))
    counts = Counter(uri for item in pool for uri in _concepts(item))
    target = _arg("--target", None)
    if target is None:
        print(f"Pool: {len(pool)} items. Pick a --target concept; most common in this pool:")
        for uri, count in counts.most_common(15):
            print(f"  {count:4d}  {uri}")
        sys.exit(0)
    relevant = {item["id"] for item in pool if target in _concepts(item)}
    print(f"Pool: {len(pool)} items, {len(relevant)} relevant to {target}")
    if not relevant:
        sys.exit(1)

    key = os.getenv("EMBEDDING", "").strip() or "metadata_text"
    print(f"Loading model ({key}) …")
    rep = config.build_embedding(key)
    print("Encoding the pool …")
    rep.ensure(pool)
    rep.wait([item["id"] for item in pool], timeout=None)

    kwargs = {"view": config.embedding_view(key), **config.rerank_kwargs()}
    strategies = {
        "random": lambda seed: RerankStrategy(None, rng=random.Random(seed), **kwargs),
        "rerank": lambda seed: RerankStrategy(rep, rng=random.Random(seed), **kwargs),
    }
    logging.getLogger(__name__).setLevel(logging.ERROR)   # the random baseline's warning
    for name, make in strategies.items():
        results = np.array([_simulate(make(seed), rep, pool, relevant, batches, n,
                                      neg_rate, random.Random(seed)) for seed in range(runs)])
        mean = np.nanmean(results, axis=0)
        print(f"\n── {name} (mean of {runs} runs) ──")
        print("batch  accept  recall  diversity")
        for b, (acc, rec, div) in enumerate(mean, 1):
            print(f"{b:5d}  {acc:6.2f}  {rec:6.2f}  {div:9.3f}")
