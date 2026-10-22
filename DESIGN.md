# Exhibition Curation Tool — Design Document

## Vision

A modular, graph-agnostic tool that helps museum curators build exhibitions by querying a collection in natural language, exploring results interactively, and exporting a curated set of artifacts. Every processing step is a **pluggable module** behind a stable interface — implementations can be swapped without touching anything else.

---

## User Flow

```
User types NL query (or clicks a category)
        │
        ▼
  [QueryGeneration] ──converts──▶ FIND query (SELECT DISTINCT ?artifact … LIMIT 100)
        │   concept_query only:              │
        │   [ConceptMatcher] text ──▶ concepts {uri, label, feature}
        │                                    ▼
        │                        [KnowledgeBase] ──executes──▶ artifact URIs
        │                                    │
        │                                    ▼
        │                        [ArtifactFetcher] ──FETCH query──▶ Artifact objects ──▶ pool
        │                                    │      (same fields for every object, lists for multi-values)
        └─ query features ─▶ [RetrievalEvaluator] ◀── pool   (optional: diversity / surprise)
                                             ▼
                       [SelectionStrategy] ──picks N (~10)──▶ items shown to user
                                             │
             User reviews cards (Yes / No)   │      [RunLogger] ← records query,
                     │                               features, pool, history, exports
                     ├─ "Next batch" ──▶ SelectionStrategy picks next N (seen items never return)
                     │
                     └─ enough Yes ──▶ [Exporter] ──▶ JSON download
```

---

## Module Specifications

| # | Module | Env var | Interface |
|---|--------|---------|-----------|
| 1 | QueryGeneration | `QUERY_GENERATION` | `convert(nl_query, context) -> str` |
| 2 | ConceptMatcher | `CONCEPT_MATCHER` | `match(nl_query) -> list[dict]` (used inside `concept_query`) |
| 3 | KnowledgeBase | `KNOWLEDGE_BASE` | `execute(sparql) -> list[dict]` |
| — | ArtifactFetcher | `ARTIFACT_FIELDS_FILE` | `fetch(ids) -> list[Artifact]` (infrastructure, configured per graph) |
| 4 | SelectionStrategy | `SELECTION` | `select(pool, n, history) -> list[dict]` |
| 5 | RetrievalEvaluator | `RETRIEVAL_EVALUATOR` | `score(pool, query_features) -> dict` (optional) |
| 6 | Exporter | `EXPORTER` | `export(items, meta) -> bytes` |
| 7 | RunLogger | `LOG_FILE`, `LOG_CONSOLE` | infrastructure, not pluggable |
| 8 | ChatModel | `CHAT_MODEL` | `chat(messages) -> str` (AI curation assistant, optional) |

All ABCs live in `modules/base.py` (RetrievalEvaluator sub-interfaces in `modules/retrieval_evaluator/base.py`).

### 1. QueryGeneration

Converts user input into a SPARQL **find** query for the target graph. This is the module `app.py` talks to on `/search`. It only decides *which* artifacts match; their metadata is fetched afterwards by the ArtifactFetcher, so query generators never deal with display fields.

| | |
|--|--|
| **Input** | `nl_query: str` — the user's input (a category selection or free text) |
| | `context: dict` — optional per-call overrides, e.g. `context["sparql_template"]` |
| **Output** | `str` — a SPARQL `SELECT DISTINCT ?artifact` query; the `?artifact` column must hold artifact URIs, and `LIMIT` therefore counts artifacts |

```python
class QueryGeneration(ABC):
    def convert(self, nl_query: str, context: dict) -> str: ...          # abstract
    def get_categories(self) -> list[str]: ...                           # button labels; default []
    def match_concepts(self, nl_query: str) -> list[dict]: ...           # concepts for the UI; default []
    def get_query_features(self, nl_query: str) -> dict[str, str]: ...   # pinned features; default {}
```

**Implementations (both shipped):**

| Name | Key | How it works |
|------|-----|-------------|
| `CategorySelect` | `category_select` | UI presents a fixed list of category buttons (`CATEGORIES_FILE` overrides the defaults). User picks one → known URI → substituted into a find-query template (that type or any sub-type, with an image). No text input, no concept matching, no query features. |
| `ConceptQuery` | `concept_query` | UI shows a free-text box. Delegates text → concepts to the configured **ConceptMatcher** (§2), then only builds SPARQL: URIs are grouped by `feature` and each group becomes one required `VALUES` + `skos:narrower*` block (AND across features, OR within one). The outer template keeps only artifacts with an image and returns their URIs. The matcher runs once per query string; `match_concepts`, `get_query_features` and `convert` reuse the cached result (the cache is read into a local first, so concurrent requests can't swap concepts). `keyword_matching` is a legacy alias key. |

**Why the split.** Before, one `KeywordMatching` module both matched text to thesaurus concepts and filled the SPARQL template. Now *which concepts the user meant* (ConceptMatcher) is separate from *what to fetch for them* (ConceptQuery), so a new matcher (embeddings, an LLM) reuses the same templates, and template changes don't touch matching. ConceptQuery receives its matcher by constructor injection from `config.py`; `app.py` is unaware of the matcher.

#### Shared feature schema

Feature names are **Artifact field names**. A ConceptMatcher tags each concept with one of them, `get_query_features()` returns `{feature: [labels]}` with those keys, and every pool item's `properties` (an Artifact dict) has a field of the same name:

| Feature | Find-query anchor (in `concept_query`) | Artifact fields (labels / concept URIs) |
|---------|----------------------------------------|----------------|
| `type` | `?artifact crm:P2_has_type` | `type`, `type_uris` |
| `material` | `?artifact crm:P45_consists_of` | `material`, `material_uris` |
| `place` | `?artifact crm:P108i_was_produced_by/crm:P7_took_place_at` | `place`, `place_uris` |

`get_query_features()` returns the searched **concept URIs** per feature, e.g. `{"type": ["https://hdl.handle.net/…/termmaster10069604"]}`, which compare directly with `properties["type_uris"]` of every item — this is how the `RetrievalEvaluator` measures surprise, without any key translation or label matching.

`get_query_features()` returns `{}` on implementations that don't produce typed concept matches (e.g. `CategorySelect`), in which case the evaluator treats the query as unpinned and reports only diversity.

---

### 2. ConceptMatcher

Maps free text to concepts in the collection's vocabulary. Knows nothing about SPARQL templates. Used by `ConceptQuery`; selected with `CONCEPT_MATCHER`.

| | |
|--|--|
| **Input** | `nl_query: str` — free text typed by the user |
| **Output** | `list[dict]` — one entry per concept: `{"uri": str, "label": str \| None, "feature": str}` |

- `uri` — concept URI to anchor the query on.
- `label` — human-readable label, or `None` when the URI is outside the vocabulary (e.g. an ancestor reached by generalisation). Unlabelled concepts are still queried but not shown in the UI or used as query features.
- `feature` — which item feature the concept constrains, from the shared feature schema above.

```python
class ConceptMatcher(ABC):
    def match(self, nl_query: str) -> list[dict]: ...
```

**Implementations:**

| Name | Key | How it works |
|------|-----|-------------|
| `ThesaurusMatcher` | `thesaurus` | Words (minus filler words) are substring-matched against thesaurus labels (CSV: concept, prefLabel, broad, altLabels; files via `KEYWORD_MATCHING_THESAURUS_FILES`). Each word's matches are collapsed to coherent superconcepts by walking the `broad` column up while two or more matches share a parent. Each concept's top-level `skos:broader*` ancestor, looked up on `SPARQL_ENDPOINT_URL` and cached, names its feature: `Objecttrefwoord` → `type`, `Culturele Herkomst` / `Geografische Herkomst` → `place`, `Materiaal en techniek` → `material`. Without an endpoint every concept falls back to `type`. |

Known limitations of `thesaurus` (to fix inside the matcher): substring matching is noisy (`houten` hits unrelated tool concepts), matches whose parent is not shared are dropped during collapsing, and collapsing can climb close to a tree root.

Future swap-ins: embedding-based matching over thesaurus labels; LLM-based matching that maps text to URIs + features.

---

### 3. KnowledgeBase

Executes a SPARQL query against a knowledge graph and returns raw results. The caller never needs to know whether the graph is remote or local.

| | |
|--|--|
| **Input** | `sparql: str` — a valid SPARQL SELECT query |
| **Output** | `list[dict]` — one dict per result row; keys are variable names, values are strings |

```python
class KnowledgeBase(ABC):
    def execute(self, sparql: str) -> list[dict]: ...
```

**Skeleton implementations (both shipped):**

| Name | Key | How it works |
|------|-----|-------------|
| `SPARQLEndpoint` | `sparql_endpoint` | Fires query at a remote HTTP endpoint via SPARQLWrapper. Configured with `endpoint_url`. |
| `RDFFile` | `rdf_file` | Loads a local `.ttl` / `.n3` / `.rdf` file with rdflib, executes SPARQL in-process. Configured with `file_path`. |

---

#### ArtifactFetcher (infrastructure on top of the KnowledgeBase)

Turns the artifact URIs from the find query into `Artifact` objects (see *Artifact Schema* below). Not a pluggable module type: the mechanism is generic, and what is graph-specific — how each field is reached in the graph — is configuration in a JSON file (`ARTIFACT_FIELDS_FILE`, default `collections/wereld_museum/artifact_fields.json`):

```json
{
  "prefixes": { "crm": "http://www.cidoc-crm.org/cidoc-crm/", "...": "..." },
  "fields": {
    "title":    "?artifact dct:title ?value .",
    "material": "?artifact crm:P45_consists_of/skos:prefLabel ?value .",
    "place":    "?artifact crm:P108i_was_produced_by/crm:P7_took_place_at/skos:prefLabel ?value ."
  }
}
```

Each pattern binds `?value` for `?artifact`. The fetcher builds one query per batch of 100 URIs with one `UNION` branch per field, so every row is a single `(artifact, field, value)` triple — multi-valued fields never multiply each other and nothing is concatenated:

```sparql
SELECT ?artifact ?field ?value WHERE {
  VALUES ?artifact { <uri1> <uri2> … }
  { ?artifact dct:title ?value . BIND("title" AS ?field) }
  UNION { ?artifact crm:P45_consists_of/skos:prefLabel ?value . BIND("material" AS ?field) }
  UNION …
}
```

Rows are folded into `{field: [values]}` and then into `Artifact.from_values()`. Field names in the file must be Artifact fields (checked at startup); Artifact fields missing from the file stay empty. Labels use `skos:prefLabel` (one Dutch label per concept), not `skos:altLabel` (mixed-language synonyms).

---

### 4. SelectionStrategy

Picks which N items to show the user from the remaining pool, using feedback history.

| | |
|--|--|
| **Input** | `pool: list[dict]` — all pool items (`id`, `title`, `score`, `properties`, optional `selection`) |
| | `n: int` — number of items to return |
| | `history: SelectionHistory` — `accepted` (ids kept so far in this search), `rejected` (shown, not kept), `seen` (every id shown) |
| **Output** | `list[dict]` — up to n items from pool, none of them in `history.seen` |
| **Side effect** | May write `item["score"]` / `item["selection"]`; the changes travel with the pool into the next call |

```python
class SelectionStrategy(ABC):
    def prepare(self, pool: list[dict]) -> None: ...                                  # optional, once per /search
    def select(self, pool: list[dict], n: int, history: SelectionHistory) -> list[dict]: ...
```

**State lives on the items.** The server keeps nothing between requests: the pool goes to the browser and comes back with every `/next`. A strategy that needs per-search data (clusters, scores) computes it in `prepare()` — called by `/search` on the fresh pool, before the first `select()` — and stores small, JSON-serialisable results in `item["selection"]`. Never store vectors there (100 × 768 floats would add >1 MB per request); keep them server-side, e.g. in the shared `TextEmbedder` cache. The browser fills `history.accepted` with the ids of kept cards when the user asks for the next batch.

**Reusable building blocks** for new strategies:

| Piece | File | What it does |
|-------|------|--------------|
| `unseen(pool, history)` | `modules/selection/utils.py` | items not shown yet |
| `empty_streak(history, n)` | `modules/selection/utils.py` | batches in a row (latest first) without a keep |
| `metadata_linearization(fields)` | `modules/selection/utils.py` | returns item → one text line from the given Artifact fields (validated at startup) |
| `largest_remainder(weights, total)` | `modules/selection/utils.py` | splits a number of slots over keys proportionally, in whole numbers |
| `Clusterer` / `KMeansClusterer` | `modules/selection/clustering.py` | vectors → cluster labels (swap in another algorithm by implementing `fit`) |
| `TextEmbedder` | `modules/embedding.py` | cached, normalised text embeddings; the model is loaded once per process and shared with the evaluator |

**Implementations:**

| Name | Key | How it works |
|------|-----|--------------|
| `RandomStrategy` | `random` | `random.sample` over unseen items; ignores feedback. |
| `TitleClusterStrategy` | `title_cluster` | Clusters the pool, shows one item per cluster first, then steers towards the clusters the user keeps from (below). |

#### TitleClusterStrategy (`title_cluster`)

*prepare (once per search):* each item's metadata is linearized into one text by `metadata_linearization(TITLE_CLUSTER_FIELDS)` (default `title`), embedded with the shared `TextEmbedder`, and clustered with k-means into `TITLE_CLUSTER_N_CLUSTERS` clusters (default = 2 × batch size, 18, so clusters remain unshown after the first batch). Each item gets `item["selection"] = {"cluster": "<search-id>:<k>", "cluster_label": <most common text in the cluster>}`. The search-id prefix (a hash of the pool's ids) keeps clusters of different searches apart when the browser merges multi-category pools.

*select (every batch):*

```
streak  = batches in a row (most recent first) in which nothing was kept   (from history.seen order)
r       = min(1, s · streak)          (s = TITLE_CLUSTER_SHUFFLE_STEP, default 0.25)
steered = round(w · (1 − r) · n)      (w = TITLE_CLUSTER_STEERING_WEIGHT, default 0.6)
          only once something is kept: shared over the clusters of history.accepted,
          in proportion to how many items were kept from each (largest remainder)
random  = round(r · remaining) slots: uniform over all unseen items, ignoring clusters
explore = the rest: one item per cluster, least-shown clusters first (never-shown first),
          cycling until the batch is full
within a cluster: random among its unseen items
order:  steered, then exploration, then random
```

* **First batch** (nothing shown yet): one item from each of 9 of the 18 clusters.
* **After a keep**: the streak is 0, so ~60 % of the next batch comes from the kept item's cluster(s), the rest explores — never-shown clusters first.
* **After a batch with no keep**: the next batch prefers the never-shown clusters, and randomness grows by `s` per empty batch in a row — steering towards earlier keeps fades (`w · (1 − r)`) and a growing share of slots is drawn purely at random. With `s = 0.25`, four empty batches in a row make the batch fully random. Any keep resets the streak.
* `w = 0` never steers; `w = 1` fills the batch from kept clusters. `s = 0` disables the shuffling (empty batches change nothing).
* Paging through the whole pool shows every item exactly once.

Simulated on the *masker* pool (95 objects, 18 clusters), a curator who never keeps: batch 2 shows 8 never-shown clusters, then `r` = 0.25 → 0.50 → 0.75 → 1.0. Keeping in batch 1, nothing in 2–4, keeping again in 5: steering fades to 0 and returns right after the second keep.

*Trade-off: number of clusters.* More clusters leave unshown clusters to rotate to, but make clusters small: with 18 clusters the *masker* pool's sizes are 17, 15, 13, 9, 8, 5, 4, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1. Keeping an item from a cluster of 3 can steer at most 2 more cards; the remaining steered slots fall back to exploration. `TITLE_CLUSTER_N_CLUSTERS=9` gives bigger clusters (stronger steering) but nothing left unshown after batch 1.

*Choosing the text (`TITLE_CLUSTER_FIELDS`).* A comma-separated list of Artifact fields; their values are joined with ` · ` (list fields contribute all values joined with `, `, empty fields are skipped). Field names are checked against `Artifact` at startup.

| `TITLE_CLUSTER_FIELDS` | Text for an object |
|------|--------------------|
| `title` | `HOUTEN VROUWENFIGUUR VERBONDEN MET HET YASSI-GENOOTSCHAP` |
| `title,type,material,place` | `HOUTEN VROUWENFIGUUR … · beelden · glas (materiaal), hout, tand (dierlijk), katoen · Liberia` |
| `title,type,material,place` (untitled object) | `kokosnoot · Liberia` |

*Choosing the fields.* Titles are often missing or generic in this collection (51 of 95 *masker* objects have none), which puts half the pool into one "(untitled)" cluster. Clustering on `TITLE_CLUSTER_FIELDS=title,type,material,place` gives far more balanced and readable groups:

| Search | Fields | Cluster sizes |
|--------|--------|---------------|
| masker | `title` | 51, 24, 12, 3, 1, 1, 1, 1, 1 |
| masker | `title,type,material,place` | 23, 17, 16, 11, 11, 6, 5, 4, 2 |
| Liberia | `title` | 26, 21, 12, 6, 5, 4, 3, 1, 1 |
| Liberia | `title,type,material,place` | 18, 15, 14, 8, 7, 5, 5, 5, 2 (e.g. "gras · Liberia", "hout · Liberia", "hoeden · plantenvezel · Liberia") |

---

### 5. RetrievalEvaluator (optional)

Scores the whole retrieved pool on two separate axes. Enabled by setting `RETRIEVAL_EVALUATOR`; when unset, `/search` returns `metrics: null` and the embedding model is never loaded.

| | |
|--|--|
| **Input** | `pool: list[dict]` — normalised pool items |
| | `query: dict` — `{feature: [concept URIs]}` from `QueryGeneration.get_query_features()`; `{}` means all features free |
| **Output** | `{"diversity": float, "surprise": float \| null}` |

```python
class RetrievalEvaluator(ABC):
    def score(self, pool: list[dict], query: dict) -> dict: ...
```

`DiversityEvaluator` (key `diversity`) is composed of three sub-modules (interfaces in `modules/retrieval_evaluator/base.py`):

- `FeatureEncoder` — `MultilingualFeatureEncoder` turns feature values into vectors.
- `MetricsCalculator` — `PairwiseDiversityCalculator` turns those vectors into **diversity**.
- `SurpriseCalculator` — `ConceptSurpriseCalculator` computes **surprise** from concept URIs (no embeddings).

#### How the metrics are calculated

**Pinned vs free.** *Pinned* features are the ones the query searched on (keys of `get_query_features()`); *free* features are all others present on any item. Diversity looks only at free features; surprise only at pinned ones.

**Diversity** — "how varied is the pool where the user left it open?"

1. *Features per item*: the item's Artifact fields minus non-semantic keys (`id`, `images`, `inventory_number`, `dimensions`, `description`, and the `*_uris` fields). Typically `title`, `type`, `material`, `place`, `date`, `maker`, `intended_use`, `acquisition`, `acquired_from`; empty fields are dropped.
2. *Encode*: every text value is embedded with a multilingual sentence-transformer (`EMBEDDING_MODEL`, default `paraphrase-multilingual-mpnet-base-v2`) into an L2-normalised vector, so `cosine_distance(a, b) = 1 − a·b`. A list field (e.g. `material: ["hout", "kleurstof"]`) becomes the re-normalised **mean** of its values' vectors. A single value that parses as a number becomes a float. Embeddings are cached per text for the server's lifetime.
3. *Score*:

```
for each free feature f:
    V_f = vectors of the items that have f            (skip f if fewer than 2)
    diversity_f = mean over all pairs (i, j) in V_f of (1 − v_i · v_j)
diversity = unweighted mean of diversity_f over free features
```

Numeric features use mean pairwise `|a − b|` instead. If a feature mixes numbers and text across items (e.g. a title that is just a number), the minority kind is ignored. With sentence embeddings, values cluster in a narrow middle band (~0.2–0.5), so compare pools against each other rather than reading the number absolutely.

**Surprise** — "how much of the pool falls outside what was searched?"

For each pinned feature, the searched concepts are first expanded with everything below them in the thesaurus (`skos:narrower*`, one SPARQL query per feature, cached) — the same expansion the find query uses, so a *dansmasker* counts as within *maskers (kleding)*. Then:

```
k = number of pinned features (0 → surprise = 0)
for each item, for each pinned feature f:
    deviates on f  ⇔  item has ≥ 1 concept for f   (properties["<f>_uris"])
                      and none of them is a searched concept or below one
d(item)  = number of pinned features the item deviates on
surprise = mean over the pool of d(item) / k        (0 … 1)
```

- **Surprise > 0 exactly when some item deviates.** With two searched concepts on one feature (e.g. place = Ivoorkust, Liberia), an item deviates only if its places include *neither* of them (nor anything below them) — something else entirely.
- **Not recorded is not surprising.** An item with no value for a pinned feature does not count as deviating on it.
- **Concepts are compared by URI**, so differences in label wording never create surprise; extra values do not either (an object typed both *maskers* and *diapositieven* still contains the searched type).
- **It is 0 for today's searches by construction.** The find query requires every pinned feature, so every retrieved item is within the searched concepts. Surprise becomes informative once retrieval also returns near-misses (relaxed retrieval: drop one pinned constraint at a time, see the diversity design doc).
- If the expansion query fails, surprise is reported as `null` rather than a wrong number.

Worked example (checked against the live collection): scoring the 95-object *masker* pool against `place = Ivoorkust` → 93 items deviate on 1 of 1 feature → surprise `93/95 = 0.979`; against `place = Ivoorkust, Liberia` → 92 deviate → `0.968`; against `type = maskers` **and** `place = Ivoorkust` → 93 items deviate on 1 of 2 → `93 × 0.5 / 95 = 0.489`.

**Both scores describe the whole pool** (up to 100 objects), not the batch on screen.

The method and its rationale are in `modules/retrieval_evaluator/diversity_recommender_design.md`.

---

### 6. Exporter

Serialises the final selection into a downloadable file.

| | |
|--|--|
| **Input** | `items: list[dict]` — the Yes-selected items (pool format, score stripped) |
| | `meta: dict` — `{ query, sparql, timestamp }` |
| **Output** | `bytes` — file content ready for HTTP download |

```python
class Exporter(ABC):
    def export(self, items: list[dict], meta: dict) -> bytes: ...
```

---

### 7. RunLogger

Records every curation event to a **JSON Lines** file (one JSON object per line) and/or stdout. This is not a pluggable processing step — it is infrastructure that observes the pipeline without influencing it.

| | |
|--|--|
| **Events logged** | `search` — query, query_features, generated SPARQL, full pool, first displayed batch |
| | `next` — current history (accepted / rejected / seen), next displayed batch |
| | `export` — exported items and metadata |
| **Format** | JSON Lines (`.jsonl` / `.log`): one `json.loads()`-able object per line, append-only |
| **Config** | `LOG_FILE` — path to log file (default `runs.log`, gitignored via `*.log`) |
| | `LOG_CONSOLE` — `"true"` to also print each record to stdout |

```python
class RunLogger:
    def log_search(self, query: str, sparql: str, pool: list, displayed: list, query_features: dict = None) -> None: ...
    def log_next(self, history: dict, displayed: list) -> None: ...
    def log_export(self, items: list, meta: dict) -> None: ...
```

Each record automatically gets a `timestamp` field (UTC ISO-8601).

---

### 8. ChatModel (AI curation assistant)

The chat box under the selection sends the curator's message plus the kept items to `POST /api/chat`. The route adds the exported selection JSON to the message and `modules/selection/textual_interaction.py` asks the configured ChatModel for a reply (system prompt: summarise commonalities and differences of the selection, concisely).

```python
class ChatModel(ABC):
    def chat(self, messages: list[dict]) -> str: ...   # [{"role", "content"}, ...] → reply
```

Implementations raise `ChatModelError` with a readable message (missing login or key, model unavailable); `/api/chat` returns it as the error shown in the chat instead of crashing.

| Name | Key | How it works |
|------|-----|--------------|
| `LocalHFChatModel` | `local_hf` | Hugging Face `transformers` pipeline on this machine (`CHAT_LOCAL_MODEL`, default `meta-llama/Llama-3.2-1B-Instruct`; `CHAT_DEVICE` mps → cuda → cpu). Loaded on the first chat, not at startup. At startup it looks for a token (`HF_TOKEN`, then a saved Hugging Face login); with none and a terminal attached it starts the Hugging Face login prompt. Gated models give a message that says whether you are logged in and lack access (request it on the model page, or set `CHAT_LOCAL_MODEL` to an open model such as `Qwen/Qwen2.5-1.5B-Instruct`) or are not logged in. |
| `NebulaChatModel` | `nebula` | The VU Nebula service (`https://nebula.cs.vu.nl/api/`, OpenAI-compatible) via the official `openai` client, as in `networkinstitutevu/nebula-api-examples`. `NEBULA_API_KEY`, `NEBULA_MODEL` (default `deepseek-r1:1.5b`). `<think>…</think>` reasoning is stripped from replies. `python -m modules.chat_model.nebula` lists the models your key can use. |

Leave `CHAT_MODEL` blank to disable the assistant (`/api/chat` answers 503). New models: copy `modules/chat_model/_stub.py`.

---

## Default Configuration & Implementation Selection

Configuration lives in two files:

**`.env`** (not committed, copied from `.env.example`):
```ini
QUERY_GENERATION=category_select   # category_select | concept_query
CONCEPT_MATCHER=thesaurus          # used by concept_query
KNOWLEDGE_BASE=sparql_endpoint
SELECTION=random
EXPORTER=json
RETRIEVAL_EVALUATOR=               # diversity | blank = off
TITLE_CLUSTER_FIELDS=title         # used by SELECTION=title_cluster
TITLE_CLUSTER_STEERING_WEIGHT=0.6
TITLE_CLUSTER_SHUFFLE_STEP=0.25
ARTIFACT_FIELDS_FILE=collections/wereld_museum/artifact_fields.json

# Module settings (only the relevant ones are used)
SPARQL_ENDPOINT_URL=https://api.colonialcollections.nl/datasets/nmvw/collection-archives/sparql
RDF_FILE_PATH=
KEYWORD_MATCHING_THESAURUS_FILES=collections/wereld_museum/thesaurus1.csv,...
```

**`config.py`** reads `.env` and resolves each key to a concrete class:

```python
REGISTRY = {
    "query_generation": {
        "category_select":  "modules.query_generation.category_select.CategorySelect",
        "concept_query":    "modules.query_generation.concept_query.ConceptQuery",
        "keyword_matching": "modules.query_generation.concept_query.ConceptQuery",  # legacy alias
    },
    "concept_matcher": {
        "thesaurus": "modules.concept_matching.thesaurus_matcher.ThesaurusMatcher",
    },
    "knowledge_base": {
        "sparql_endpoint": "modules.knowledge_base.sparql_endpoint.SPARQLEndpoint",
        "rdf_file":        "modules.knowledge_base.rdf_file.RDFFile",
    },
    "selection": {
        "random":        "modules.selection.random_strategy.RandomStrategy",
        "title_cluster": "modules.selection.title_cluster.TitleClusterStrategy",
    },
    "exporter": {
        "json": "modules.exporter.json_exporter.JSONExporter",
    },
    "retrieval_evaluator": {
        "diversity": "modules.retrieval_evaluator.diversity_evaluator.DiversityEvaluator",
    },
}
```

Each module type has a `_build_<type>()` function in `config.py` that passes constructor arguments from `.env`. `build_modules()` also builds the `ArtifactFetcher` around the chosen KnowledgeBase. Composition happens there too: `_build_query_generation("concept_query")` calls `_build_concept_matcher(CONCEPT_MATCHER)` and injects the result, and `_build_retrieval_evaluator` assembles encoder + calculator.

Adding a new implementation = add one registry entry + set the env var (plus a `_build_*` branch if it needs constructor args). No other files change.

---

## Artifact Schema (internal)

`modules/artifact.py` defines one dataclass that every search result is turned into, whatever the graph. Each field's type decides whether it is single- or multi-valued; missing data is `None` or `[]`, so **every artifact always has every key**.

| Field | Type | Wereld Museum source (`artifact_fields.json`) |
|-------|------|-----------------------------------------------|
| `id` | `str` | artifact URI |
| `title` | `str \| None` | `dct:title` |
| `inventory_number` | `str \| None` | identifier labelled "Identification number" |
| `intended_use` | `str \| None` | `P103_was_intended_for` |
| `dimensions` | `str \| None` | `P43_has_dimension` label |
| `type` | `list[str]` | `P2_has_type` prefLabel |
| `material` | `list[str]` | `P45_consists_of` prefLabel |
| `place` | `list[str]` | production `P7_took_place_at` prefLabel |
| `date` | `list[str]` | production label (free text, e.g. "ca. 1900"; the role label "Maker" is filtered out) |
| `maker` | `list[str]` | production `P14_carried_out_by` label |
| `description` | `list[str]` | text labelled "Description" |
| `acquisition` | `list[str]` | acquisition event label (e.g. "Schenking") |
| `acquired_from` | `list[str]` | acquisition `P23_transferred_title_from` label |
| `images` | `list[str]` | `P65 / digitally_shown_by / access_point` |
| `type_uris` | `list[str]` | `P2_has_type` concept URIs |
| `material_uris` | `list[str]` | `P45_consists_of` concept URIs |
| `place_uris` | `list[str]` | production `P7_took_place_at` concept URIs |

Pool items wrap an artifact for the UI and the selection loop:

```json
{
  "id": "https://hdl.handle.net/20.500.11840/102545",
  "title": "HOUTEN MASKER VOOR DE GOLI DANS",
  "score": 1.0,
  "properties": {
    "id": "https://hdl.handle.net/20.500.11840/102545",
    "title": "HOUTEN MASKER VOOR DE GOLI DANS",
    "inventory_number": "TM-3305-18",
    "intended_use": "Masker",
    "dimensions": "58,5 × 44 × 5,5cm",
    "type": ["maskers (kleding)"],
    "material": ["kleurstof", "hout"],
    "place": ["Ivoorkust"],
    "date": ["voor 1963"],
    "maker": [],
    "description": ["Het masker danst tijdens de Goli, …"],
    "acquisition": ["Verwerving: aankoop", "Aankoop"],
    "acquired_from": ["A.P. (André) Schaller"],
    "images": ["https://…/TM-3305-18.jpg", "https://…"],
    "type_uris": ["https://hdl.handle.net/20.500.11840/termmaster10069604"],
    "material_uris": ["…/termmaster10065614", "…/termmaster10065560"],
    "place_uris": ["…/termmaster10054917"]
  }
}
```

`properties` is `Artifact.to_dict()`. `title` is repeated at the top level (`"Untitled"` when missing) for display. Images already known to be placeholders are removed; artifacts left without images are dropped. `score` is internal and stripped on export.

---

## Export JSON

```json
{
  "meta": {
    "query": "funerary objects from West Africa",
    "sparql": "SELECT ...",
    "timestamp": "2026-07-02T12:00:00Z"
  },
  "items": [
    {
      "id": "...",
      "title": "...",
      "properties": { "id": "...", "title": "...", "type": ["..."], "material": ["..."], "images": ["..."], ... }
    }
  ]
}
```

---

## Curation UI (`templates/index.html`)

A single page; all state (pool, history, selection) lives in the browser.

```
┌ Search ─ keyword box (concept_query) or category buttons (category_select) ┐
├ Concepts: chips for each matched concept (link to its URI)                 ┤
├ Pool section ──────────────────────────────────────────────────────────────┤
│  EXPLORING  100 objects            ← total pool size, fixed per search     │
│  DIVERSITY 0.320  SURPRISE 0.000   ← only when RETRIEVAL_EVALUATOR is set  │
│ ┌ In this pool ┐ ┌ cards (N per batch: thumbnail + title + Keep) ────────┐ │
│ │ TYPE         │ │                                                       │ │
│ │  [searched]  │ │                                                       │ │
│ │ MATERIAL  ▾  │ │                                                       │ │
│ │ PLACE     ▾  │ ├───────────────────────────────────────────────────────┤ │
│ └──────────────┘ │ BATCH 3 / 10   ← Previous   Next batch →              │ │
│                  └───────────────────────────────────────────────────────┘ │
├ Selection: kept items → Export / Build Exhibition                          ┤
└────────────────────────────────────────────────────────────────────────────┘
```

**Feature sidebar ("In this pool").** One block per main feature — `type`, `material`, `place` (the shared feature schema; `MAIN_FEATURES` in the page script):

- A feature the query **searched on** (some concept in `/search`'s `concepts` has that `feature`) shows the searched labels as fixed chips marked *searched*.
- Every **other** feature is a collapsible list: the summary shows the most common value and the number of distinct values; expanded, it lists every value with its count and a bar, most common first, plus *not recorded* for objects without that field.
- Counts are computed in the browser from the pool's Artifact dicts: each object counts **once per distinct value**, so an object with two materials counts for both and totals can exceed the pool size. Counts cover the **whole pool**, not the visible batch, so they don't change when paging.
- In category mode no concepts are returned, so all three features are lists. Multi-category searches are merged in the browser first, so the counts cover the merged pool.

**Batching.** `/search` returns the first `N` cards (`N_PER_BATCH`, default 9); *Next batch* asks `/next` for `N` unseen ones (pure in-memory selection, ~1 ms — no SPARQL). The bottom-left indicator shows `Batch i / ⌈pool / N⌉`. *Previous* / *Forward* re-show earlier batches from browser history without calling the server.

**Card grid.** `layoutGrid()` picks the column count so a batch fills balanced rows: all `N` cards in one row when the grid is wide enough (cards ≥ 96 px), otherwise the fewest rows with near-equal columns (9 → 5 + 4, or 3 × 3). It re-runs on resize.

**Images: thumbnails, placeholders, preloading.** The museum's image proxy only serves full-size originals (often 0.5–3 MB, no resize parameters), which made paging slow. Cards therefore load `/thumb?url=<original>` instead (see *ImageCache* below), ~20–40 KB each. After the first batch renders, the page preloads the thumbnails of the whole pool in the background (4 at a time; a new search cancels the previous preload), so every later batch shows instantly. If an image is the collection's "image not available" placeholder, `/thumb` answers 404 and the card falls through to the object's next image; only when every image is a placeholder does it show a neutral *no image* box.

---

### ImageCache (`modules/image_cache.py`)

Infrastructure behind `/thumb`. On the first request for an image URL it downloads the original once, shrinks it to a JPEG thumbnail (`THUMBNAIL_SIZE`, default 480 px longest side) and stores it under `THUMBNAIL_CACHE_DIR` (default `.cache/thumbnails/`, gitignored) as `<sha1(url)>.jpg`; later requests are served from disk in ~2 ms, across server restarts. The same download is compared (perceptual average hash, Hamming distance ≤ 5) with `PLACEHOLDER_IMAGE_FILE`; a match is recorded as `<sha1(url)>.placeholder` and the image is never shown. `_build_pool` drops images already known to be placeholders, and artifacts left with no image at all. A per-URL lock makes concurrent requests download an image only once. Only `http(s)` URLs on `IMAGE_HOSTS` (default `collectie.wereldculturen.nl`) are fetched, so `/thumb` is not an open proxy.

---

## Flask API (stateless server)

**State lives entirely in the browser** (pool, history, current display). The server is a pure function: same inputs → same outputs.

| Method | Path | Request body | Response |
|--------|------|-------------|----------|
| `GET` | `/` | — | Serves `index.html` |
| `POST` | `/search` | `{ query: str, n: int }` | `{ sparql: str, pool: [...], displayed: [...], concepts: [{label, uri, feature}], metrics: {diversity, surprise} \| null }` |
| `POST` | `/next` | `{ pool: [...], n: int, history: {...} }` | `{ displayed: [...], pool: [...] }` |
| `POST` | `/export` | `{ items: [...], meta: {...} }` | JSON file download |
| `GET` | `/thumb?url=…` | — | Cached JPEG thumbnail; `404` if the image is a placeholder, `403` if the host is not allowed, `502` if it can't be fetched |

---

## File Structure

```
museum_curation/
├── modules/
│   ├── base.py                          # ABCs + SelectionHistory dataclass
│   ├── artifact.py                      # Artifact dataclass — the fixed object schema
│   ├── artifact_fetcher.py              # artifact URIs → Artifact objects (fetch query)
│   ├── image_cache.py                   # /thumb: cached thumbnails + placeholder detection
│   ├── embedding.py                     # shared sentence-transformer + TextEmbedder cache
│   ├── run_logger.py                    # RunLogger — append-only JSON Lines event log
│   ├── query_generation/
│   │   ├── _stub.py                     # copy-paste template for new impls
│   │   ├── category_select.py           # baseline: fixed category buttons
│   │   └── concept_query.py             # matched concepts → SPARQL (VALUES + narrower*)
│   ├── concept_matching/
│   │   ├── _stub.py
│   │   └── thesaurus_matcher.py         # text → concept URIs via thesaurus CSVs
│   ├── knowledge_base/
│   │   ├── _stub.py
│   │   ├── sparql_endpoint.py           # remote HTTP endpoint via SPARQLWrapper
│   │   └── rdf_file.py                  # local TTL/RDF file via rdflib
│   ├── selection/
│   │   ├── _stub.py
│   │   ├── utils.py                     # unseen(), largest_remainder(), metadata_linearization()
│   │   ├── clustering.py                # Clusterer ABC + KMeansClusterer
│   │   ├── title_cluster.py          # cluster → one per cluster → steer to kept clusters
│   │   └── random_strategy.py           # random.sample(unseen, n)
│   ├── retrieval_evaluator/
│   │   ├── _stub.py
│   │   ├── base.py                      # FeatureEncoder, MetricsCalculator, SurpriseCalculator ABCs
│   │   ├── diversity_evaluator.py       # DiversityEvaluator (+ CLI)
│   │   ├── feature_encoder.py           # multilingual sentence-transformer encoder
│   │   ├── metrics_calculator.py        # diversity (pairwise embedding distance)
│   │   ├── surprise_calculator.py       # surprise (items outside the searched concepts)
│   │   └── diversity_recommender_design.md
│   ├── exporter/
│   │   ├── _stub.py
│   │   └── json_exporter.py
│   ├── chat_model/
│   │   ├── _stub.py
│   │   ├── local_hf.py                  # local transformers pipeline + HF login
│   │   └── nebula.py                    # VU Nebula (OpenAI-compatible) via openai client
│   └── exhibition_builder/              # downstream consumer of the export (skeleton)
│       ├── _stub.py
│       └── digital_wall.py
├── collections/
│   └── wereld_museum/                   # graph-specific utility space
│       ├── wereldmuseum_categories.json # label → concept URI mapping (optional override)
│       ├── thesaurus1-3.csv             # vocabulary for the thesaurus ConceptMatcher
│       ├── artifact_fields.json         # Artifact field → SPARQL pattern for this graph
│       └── notavailable.jpg             # reference placeholder image for perceptual hash filter
├── config.py                            # module registry, reads from .env
├── .env                                 # not committed
├── .env.example                         # committed, documents all options
├── app.py                               # Flask: thin orchestration only
├── templates/
│   └── index.html
├── requirements.txt
├── .cache/thumbnails/                 # gitignored; written by ImageCache
└── runs.log                             # gitignored; written by RunLogger
```

---

## How to Implement a New Module

1. Copy `modules/<module>/_stub.py` to a new file in the same folder.
2. Implement the single abstract method.
3. Add an entry to the `REGISTRY` in `config.py`.
4. Set the env var in `.env`.

### Example stub (SelectionStrategy)

```python
# modules/selection/_stub.py
from modules.base import SelectionStrategy, SelectionHistory

class MySelectionStrategy(SelectionStrategy):
    def select(self, pool: list[dict], n: int, history: SelectionHistory) -> list[dict]:
        unseen = [item for item in pool if item["id"] not in history.seen]
        # your logic here — may mutate item["score"]
        return unseen[:n]
```

---

## Graph-Specific Utility Space (`collections/`)

Each graph (knowledge base) that the tool connects to gets its own subfolder under `collections/`. These folders are the only place where graph-specific, non-code assets live. The rest of the codebase stays graph-agnostic.

```
collections/
└── wereld_museum/
    ├── wereldmuseum_categories.json   # label → concept URI mapping
    ├── thesaurus1-3.csv               # concept vocabulary (concept, prefLabel, broad, altLabels)
    ├── artifact_fields.json           # how each Artifact field is read from this graph
    └── notavailable.jpg               # reference "image not available" placeholder
```

**What belongs here:**
- Category / concept URI mappings (`CATEGORIES_FILE` env var)
- Thesaurus vocabularies for concept matching (`KEYWORD_MATCHING_THESAURUS_FILES`)
- The Artifact field mapping (`ARTIFACT_FIELDS_FILE`)
- Placeholder / dummy images (`PLACEHOLDER_IMAGE_FILE` — perceptual hash comparison via imagehash + Pillow, done by ImageCache when a thumbnail is first made; results persist in `THUMBNAIL_CACHE_DIR`)
- Future: schema snippets for LLM prompts, local RDF dumps, domain vocabulary files

**How it's used:**
- Env vars in `.env` point to these files
- `config.py` reads the env vars and passes file paths as constructor args to the relevant modules
- The modules themselves load and use the files — `config.py` never inspects their content

Switching to a different graph = create a new `collections/<graph>/` folder, point the env vars there, done.

---

## Out of Scope (for now)

- LLM- or embedding-based ConceptMatcher (interface ready, impl deferred)
- Relaxed retrieval and MMR-style diversity-aware selection (see the diversity design doc)
- Further feedback-driven SelectionStrategies (e.g. nearest neighbours of kept items, rejected-cluster down-weighting)
- Object validation
- User accounts or persistent sessions
- Docker / container deployment
