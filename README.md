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

Open `http://127.0.0.1:5000`.

---

## Configure modules

All module selection is done via `.env`. Copy `.env.example` to `.env` and set:

```ini
TEXT2SPARQL=category_select       # category_select | template_keyword
KNOWLEDGE_BASE=sparql_endpoint    # sparql_endpoint | rdf_file
SELECTION=random                  # random
EXPORTER=json                     # json
```

### Knowledge base

| Key | What it needs |
|-----|---------------|
| `sparql_endpoint` | `SPARQL_ENDPOINT_URL=https://...` |
| `rdf_file` | `RDF_FILE_PATH=path/to/collection.ttl` |

### Text-to-SPARQL

| Key | How it works |
|-----|-------------|
| `category_select` | Buttons with predefined categories; no text input |
| `template_keyword` | Free-text box; tries to match a known category, falls back to title search |

`template_keyword` accepts an optional extra entity-URI mapping:
```ini
TEMPLATE_KEYWORD_ENTITY_URIS_FILE=path/to/entity_uris.json
```
Format: `{"Label": "https://hdl.handle.net/..."}`.

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
  text2sparql/      Text-to-SPARQL implementations
  knowledge_base/   SPARQL execution backends
  selection/        Item selection strategies
  exporter/         Export formats
templates/
  index.html        Single-page curation UI
.env.example        Documents all environment variables
DESIGN.md           Full architecture and module specifications
```
