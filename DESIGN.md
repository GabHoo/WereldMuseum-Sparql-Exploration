# Exhibition Curation Tool — Design Document

## Vision

A modular, graph-agnostic tool that helps museum curators build exhibitions by querying a collection in natural language, exploring results interactively, and exporting a curated set of artifacts. Every processing step is a **pluggable module** behind a stable interface — implementations can be swapped without touching anything else.

---

## User Flow

```
User types NL query
        │
        ▼
  [Text2SPARQL]  ──converts──▶  SPARQL query string
        │
        ▼
  [KnowledgeBase]  ──executes──▶  result pool
        │
        ▼
  [SelectionStrategy]  ──picks N (~10)──▶  items shown to user
        │                                          │
  User reviews cards (Yes / No)            [RunLogger] ← records query,
        │                                    pool, history, exports
        ├─ "Next batch" ──▶ SelectionStrategy picks next N (seen items never return)
        │
        └─ enough Yes ──▶ [Exporter]  ──▶  JSON download
```

---

## Module Specifications

### 1. Text2SPARQL

Converts user input into a valid SPARQL string for the target graph.

| | |
|--|--|
| **Input** | `nl_query: str` — the user's input (a category selection, keywords, or free text) |
| | `context: dict` — graph-specific hints passed from config: known entity URIs, ontology prefixes, SPARQL templates, domain vocabulary, RDF schema snippets |
| **Output** | `str` — a valid SPARQL SELECT query |

```python
class Text2SPARQL(ABC):
    def convert(self, nl_query: str, context: dict) -> str: ...
```

**Skeleton implementations (both shipped):**

| Name | Key | How it works |
|------|-----|-------------|
| `CategorySelect` | `category_select` | UI presents a fixed list of choices from `context["categories"]`. User picks one → known URI → substituted into a SPARQL template. No text input. |
| `TemplateKeyword` | `template_keyword` | UI shows a text box. User types keywords. These are substituted into a parameterised SPARQL template stored in `context["keyword_sparql_template"]`. |

Future swap-in: LLM-based (`llm`) that uses `context["schema_snippet"]` to prompt a model.

---

### 2. KnowledgeBase

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

### 3. SelectionStrategy

Picks which N items to show the user from the remaining pool, using feedback history.

| | |
|--|--|
| **Input** | `pool: list[dict]` — all items from the query, each with a `score: float` field |
| | `n: int` — number of items to return |
| | `history: SelectionHistory` — `accepted: list[str]`, `rejected: list[str]`, `seen: set[str]` (item IDs) |
| **Output** | `list[dict]` — n items from pool, all with IDs not in `history.seen` |
| **Side effect** | May mutate `item["score"]` in pool to propagate rankings to the next round |

```python
class SelectionStrategy(ABC):
    def select(self, pool: list[dict], n: int, history: SelectionHistory) -> list[dict]: ...
```

The random baseline filters `seen` items, then `random.sample(remaining, n)`. A future active-learning implementation uses `accepted` item properties to re-score the pool before sampling.

---

### 4. Exporter

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

### 5. RunLogger

Records every curation event to a **JSON Lines** file (one JSON object per line) and/or stdout. This is not a pluggable processing step — it is infrastructure that observes the pipeline without influencing it.

| | |
|--|--|
| **Events logged** | `search` — query, generated SPARQL, full pool, first displayed batch |
| | `next` — current history (accepted / rejected / seen), next displayed batch |
| | `export` — exported items and metadata |
| **Format** | JSON Lines (`.jsonl` / `.log`): one `json.loads()`-able object per line, append-only |
| **Config** | `LOG_FILE` — path to log file (default `runs.log`, gitignored via `*.log`) |
| | `LOG_CONSOLE` — `"true"` to also print each record to stdout |

```python
class RunLogger:
    def log_search(self, query: str, sparql: str, pool: list, displayed: list) -> None: ...
    def log_next(self, history: dict, displayed: list) -> None: ...
    def log_export(self, items: list, meta: dict) -> None: ...
```

Each record automatically gets a `timestamp` field (UTC ISO-8601).

---

## Default Configuration & Implementation Selection

Configuration lives in two files:

**`.env`** (not committed, copied from `.env.example`):
```ini
TEXT2SPARQL=category_select
KNOWLEDGE_BASE=sparql_endpoint
SELECTION=random
EXPORTER=json

# KnowledgeBase settings (only the relevant ones are used)
SPARQL_ENDPOINT_URL=https://api.colonialcollections.nl/datasets/nmvw/collection-archives/sparql
RDF_FILE_PATH=
```

**`config.py`** reads `.env` and resolves each key to a concrete class:

```python
REGISTRY = {
    "text2sparql": {
        "category_select": "modules.text2sparql.category_select.CategorySelect",
        "template_keyword": "modules.text2sparql.template_keyword.TemplateKeyword",
    },
    "knowledge_base": {
        "sparql_endpoint": "modules.knowledge_base.sparql_endpoint.SPARQLEndpoint",
        "rdf_file":        "modules.knowledge_base.rdf_file.RDFFile",
    },
    "selection": {
        "random": "modules.selection.random_strategy.RandomStrategy",
    },
    "exporter": {
        "json": "modules.exporter.json_exporter.JSONExporter",
    },
}
```

Adding a new implementation = add one registry entry + set the env var. No other files change.

---

## Pool Item Schema (internal)

```json
{
  "id": "https://hdl.handle.net/.../artifact123",
  "title": "Funeral mask",
  "score": 1.0,
  "properties": {
    "url_photo": "...",
    "specificType": "...",
    "any_other_sparql_field": "..."
  }
}
```

`score` is internal. All SPARQL-returned fields land in `properties` verbatim. `score` is stripped on export.

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
      "properties": { "url_photo": "...", "specificType": "...", ... }
    }
  ]
}
```

---

## Flask API (stateless server)

**State lives entirely in the browser** (pool, history, current display). The server is a pure function: same inputs → same outputs.

| Method | Path | Request body | Response |
|--------|------|-------------|----------|
| `GET` | `/` | — | Serves `index.html` |
| `POST` | `/search` | `{ query: str, n: int }` | `{ sparql: str, pool: [...], displayed: [...] }` |
| `POST` | `/next` | `{ pool: [...], n: int, history: {...} }` | `{ displayed: [...], pool: [...] }` |
| `POST` | `/export` | `{ items: [...], meta: {...} }` | JSON file download |

---

## File Structure

```
museum_curation/
├── modules/
│   ├── base.py                          # ABCs + SelectionHistory dataclass
│   ├── run_logger.py                    # RunLogger — append-only JSON Lines event log
│   ├── text2sparql/
│   │   ├── _stub.py                     # copy-paste template for new impls
│   │   ├── category_select.py           # baseline: fixed category buttons
│   │   └── template_keyword.py          # keyword fills SPARQL template
│   ├── knowledge_base/
│   │   ├── _stub.py
│   │   ├── sparql_endpoint.py           # remote HTTP endpoint via SPARQLWrapper
│   │   └── rdf_file.py                  # local TTL/RDF file via rdflib
│   ├── selection/
│   │   ├── _stub.py
│   │   └── random_strategy.py           # random.sample(unseen, n)
│   └── exporter/
│       ├── _stub.py
│       └── json_exporter.py
├── collections/
│   └── wereld_museum/                   # graph-specific utility space
│       ├── wereldmuseum_categories.json # label → concept URI mapping (optional override)
│       └── notavailable.jpg             # reference placeholder image for perceptual hash filter
├── config.py                            # module registry, reads from .env
├── .env                                 # not committed
├── .env.example                         # committed, documents all options
├── app.py                               # Flask: thin orchestration only
├── templates/
│   └── index.html
├── requirements.txt
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
    └── notavailable.jpg               # reference "image not available" placeholder
```

**What belongs here:**
- Category / concept URI mappings (`CATEGORIES_FILE` env var)
- Placeholder / dummy images (`PLACEHOLDER_IMAGE_FILE` — perceptual hash comparison via imagehash + Pillow; each URL fetched once and cached for the server lifetime)
- Future: schema snippets for LLM prompts, local RDF dumps, domain vocabulary files

**How it's used:**
- Env vars in `.env` point to these files
- `config.py` reads the env vars and passes file paths as constructor args to the relevant modules
- The modules themselves load and use the files — `config.py` never inspects their content

Switching to a different graph = create a new `collections/<graph>/` folder, point the env vars there, done.

---

## Out of Scope (for now)

- LLM-based Text2SPARQL (interface + context dict ready, impl deferred)
- Scored / active-learning SelectionStrategy (interface + score field ready, impl deferred)
- Object validation
- User accounts or persistent sessions
- Docker / container deployment
