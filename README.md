# Exhibition Curation Tool — Wereld Museum

A modular, graph-agnostic tool for curating an exhibition from a linked collection.
Curators query the collection, browse artifacts, mark selections, and export a curated JSON set.

Please see [DESIGN.md](DESIGN.md) for the full architecture.

---

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # configure modules (defaults work out of the box)
python app.py
```

Open `http://127.0.0.1:5000`. If port 5000 is taken (another checkout of the app, or macOS AirPlay Receiver), `app.py` uses the next free port — 5001, 5002, … — and prints the address it runs on.

---

## Configure modules

All module selection is done via `.env`. Copy `.env.example` to `.env` and set:

```ini
QUERY_GENERATION=concept_query    # concept_query
CONCEPT_MATCHER=thesaurus         # thesaurus (used by concept_query)
KNOWLEDGE_BASE=sparql_endpoint    # sparql_endpoint | rdf_file
SELECTION=random                  # random | clustering | rerank (clustering and rerank need EMBEDDING)
EXPORTER=json                     # json
EXHIBITION_BUILDER=digital_wall   # digital_wall
RETRIEVAL_EVALUATOR=              # diversity | blank = off
CHAT_MODEL=local_hf               # local_hf | nebula | blank = off (curation assistant)
EMBEDDING=                        # metadata_text | multimodal | blank = off (vectors for clustering / rerank)
```

### Search results

Every search runs two SPARQL queries. The query generator's *find* query returns matching artifact URIs (at most 100 artifacts). The artifact fetcher then reads the same fields for each one into an `Artifact` (see `modules/artifact.py`): title, type, material, place, date, maker, description, acquisition, images and more. Multi-valued fields are lists.

How each field is read from the graph is configured in a JSON file:
```ini
ARTIFACT_FIELDS_FILE=collections/wereld_museum/artifact_fields.json   # default
```

### Knowledge base

| Key | What it needs |
|-----|---------------|
| `sparql_endpoint` | `SPARQL_ENDPOINT_URL=https://...` |
| `rdf_file` | `RDF_FILE_PATH=path/to/collection.ttl` |

### Query generation

| Key | How it works |
|-----|-------------|
| `concept_query` | Free-text box; a concept matcher maps the text to concept URIs, which are pinned as type, material and origin in the SPARQL query. `keyword_matching` still works as an alias. |

### Concept matcher (used by `concept_query`)

| Key | How it works |
|-----|-------------|
| `thesaurus` | Substring-matches words against thesaurus labels and collapses matches to shared parent concepts |

`thesaurus` needs one or more thesaurus CSV files:
```ini
KEYWORD_MATCHING_THESAURUS_FILES=collections/wereld_museum/thesaurus1.csv,collections/wereld_museum/thesaurus2.csv,collections/wereld_museum/thesaurus3.csv
```
Expected columns: `concept, prefLabel, broad, altLabels` (altLabels pipe-separated).

### Exhibition theme and curation assistant

The **Theme** box next to the search describes the exhibition's perspective, e.g. *"Central Asian items significant to the nomadic lifestyle"*. It is stored as `exhibition_theme` in the export meta and the run log, and given to the curation assistant. It does not change the search results.

The curation assistant (chat box under the selection) answers questions about the selected items; `summary` gives the recurring themes and connections in the selection rather than item-by-item descriptions.

| Key | How it works |
|-----|-------------|
| `local_hf` | Hugging Face model on this machine (`CHAT_LOCAL_MODEL`); gated models need a Hugging Face login |
| `nebula` | VU Nebula service (`NEBULA_API_KEY`, `NEBULA_MODEL`) |

`CHAT_CONTEXT_FIELDS` sets which Artifact fields of each selected item the model sees. Leave `CHAT_MODEL` blank to hide the assistant.

### Exhibition builder

*Build Exhibition* opens the selection in the page of the configured builder. *← Back to selection* returns to the curation page with the search, batch, selection, theme and chat as you left them, so you can add objects and build again; the wall keeps its composition too. Both are kept in the browser tab's `sessionStorage`, so they last until the tab is closed.

| Key | How it works |
|-----|-------------|
| `digital_wall` | Wall editor: drag the selected objects onto a gallery wall, draw, add text and props. *Done → Generate* turns the scene into photorealistic exhibition renders via the OpenAI image-edit API (needs `OPENAI_API_KEY`; `DIGITAL_WALL_IMAGE_MODEL`, default `gpt-image-2`). |

### Selection and embedding

Two independent choices. `SELECTION` decides which objects each batch shows — it is where the curator's feedback turns into recommendations. `EMBEDDING` decides how objects become vectors for the strategies that work on vectors. Every embedding-based strategy works with every embedding:

| `SELECTION` | How it works |
|-------------|-------------|
| `random` | Unseen objects at random; ignores feedback |
| `clustering` | Clusters the pool on its vectors, shows one object per cluster first, then steers towards the clusters you keep from. Settings: `CLUSTERING_*` |
| `rerank` | Relevance feedback: re-ranks each new batch — kept objects pull similar ones up, objects marked not relevant (the × that appears on a pool card under the mouse; listed under the selection panel's Negatives tab) push them down and are not shown again, and 2 of 10 objects are still picked at random. Settings: `RERANK_*`. Compare it with random selection offline: `python -m modules.selection.rerank_strategy <pool or runs.log> --target <concept URI>` |

| `EMBEDDING` | Vectors from | Speed |
|-------------|-------------|-------|
| `metadata_text` | the Artifact fields in `EMBEDDING_TEXT_FIELDS` as one line of text (multilingual sentence-transformer, `EMBEDDING_MODEL`) | ~1 s per 100 objects |
| `multimodal` | image + metadata with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (1.5 GB, needs `transformers` 5.19+); `EMBEDDING_VIEW` = `fused` (default), `image` or `text` | ~30 s per 100 objects |

Vectors are computed in the background and cached per object. The first batch of a search is random for `rerank` (no feedback yet), and for `clustering` when the vectors are not ready within `CLUSTERING_WAIT` seconds (multimodal); clustering then starts as soon as they are. The server keeps nothing per curator — the browser sends the pool, kept items and negatives with every request — so several tabs work side by side. Progress: `GET /embedding/status`. See DESIGN.md §4.

---

## Add a new module implementation

1. Copy `modules/<type>/_stub.py` to a new file in the same folder.
2. Implement the single abstract method (instructions are in the stub).
3. Add an entry to `REGISTRY[<type>]` in `config.py`.
4. Set the env var in `.env`.

---

## Files

```
app.py              Flask app — thin orchestration layer
config.py           Module registry and wiring; reads .env
modules/
  base.py           ABCs and SelectionHistory dataclass
  artifact.py       Artifact dataclass — fixed fields for every search result
  artifact_fetcher.py  Fetches Artifact fields for found artifact URIs
  image_cache.py    Cached card thumbnails (/thumb) + placeholder detection
  query_generation/ Query generation implementations
  concept_matching/ Text → concept URI matchers used by concept_query
  knowledge_base/   SPARQL execution backends
  selection/        Selection strategies (random, clustering, rerank) + shared helpers
  embedding/        Object embeddings for selection (metadata_text, multimodal) + EmbeddingService
  retrieval_evaluator/ Optional pool diversity / surprise metrics
  exporter/         Export formats
  chat_model/       Chat models for the curation assistant (local_hf, nebula)
  curation_assistant.py  Chat prompt + selection context sent to the chat model
  exhibition_builder/ Exhibition pages built from the selection (digital_wall)
templates/
  index.html        Single-page curation UI
  digital_wall.html Wall editor page (DigitalWall)
static/
  digital_wall/     Wall editor script, styles and prop images
tests/              Unit tests, no model needed (python -m tests.test_rerank_strategy, …)
.env.example        Documents all environment variables
DESIGN.md           Full architecture and module specifications
```
