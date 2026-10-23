import json
import os
from datetime import datetime, timezone

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

import config
from modules.artifact import Artifact
from modules.base import ChatModelError, SelectionHistory
from modules.selection.textual_interaction import generate_chat_response

app = Flask(__name__)
MODULES = config.build_modules()
IMAGES  = MODULES["image_cache"]   # thumbnails + placeholder detection


# ── Pool normalisation ────────────────────────────────────────────────────────

def _build_pool(artifacts: list[Artifact]) -> list:
    """
    Convert Artifact objects into the internal pool item schema:
        { id, title, score, properties }   — properties is Artifact.to_dict()

    Images already known to be placeholders (from earlier thumbnail requests)
    are removed from each artifact; artifacts left without any image are
    dropped. Unknown images are kept: /thumb detects placeholders when the
    browser requests them, and the card falls through to the next image.
    """
    pool = []
    for a in artifacts:
        a.images = [url for url in a.images if not IMAGES.is_placeholder(url)]
        if not a.images:
            continue
        pool.append({
            "id":         a.id,
            "title":      a.title or "Untitled",
            "score":      1.0,
            "properties": a.to_dict(),
        })
    return pool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    t2s = MODULES["query_generation"]
    return render_template(
        "index.html",
        query_generation_mode=MODULES["query_generation_key"],
        categories=t2s.get_categories(),
        n_per_batch=config.N_PER_BATCH,
    )


@app.route("/thumb")
def thumb():
    """
    Small cached thumbnail of a collection image: ?url=<original image URL>.
    404 when the image is the collection's "not available" placeholder (the
    card then tries the artifact's next image), 502 when it can't be fetched.
    """
    url = request.args.get("url", "")
    if not IMAGES.is_allowed(url):
        abort(403)
    path, is_placeholder = IMAGES.thumbnail(url)
    if path is None:
        abort(404 if is_placeholder else 502)
    return send_file(path, mimetype="image/jpeg", max_age=7 * 24 * 3600)


@app.route("/search", methods=["POST"])
def search():
    # Read the JSON body sent by the frontend, defaulting to an empty dict when
    # the request has no payload or cannot be decoded.
    data = request.get_json(silent=True) or {}
    # Extract the free-text query and the requested batch size from the request.
    nl_query = data.get("query", "").strip()
    n = int(data.get("n", config.N_PER_BATCH))
    # Optional curator framing of the exhibition (e.g. "items significant to
    # the nomadic lifestyle"). Not used for retrieval yet; only logged.
    exhibition_theme = data.get("exhibition_theme", "").strip()

    # Backend validation: reject empty searches early so the rest of the pipeline
    # never runs on an invalid request.
    if not nl_query:
        return jsonify({"error": "query is required"}), 400

    # Grab the query-generation module and derive both semantic concepts and the
    # feature vector used later for retrieval evaluation and ranking diagnostics.
    qg = MODULES["query_generation"]
    concepts       = qg.match_concepts(nl_query)
    query_features = qg.get_query_features(nl_query)
    try:
        # Step 1: translate the natural-language request into a SPARQL query.
        # This returns the URIs of candidate artifacts in the knowledge base.
        sparql = qg.convert(nl_query, {})
        rows = MODULES["knowledge_base"].execute(sparql)
        # Keep only valid artifact references; ignore any malformed rows.
        ids = [r["artifact"] for r in rows if r.get("artifact")]

        # Step 2: fetch the full metadata for each matched artifact so the backend
        # can build a display pool and later rank/return the chosen items.
        artifacts = MODULES["artifact_fetcher"].fetch(ids)
    except Exception as exc:
        # Any issue in query generation, SPARQL execution, or artifact fetching is
        # surfaced to the client while still returning the matched concepts.
        return jsonify({"error": str(exc), "concepts": concepts}), 500

    # Build the pool of candidates that the selection mechanism can rank. This
    # step also strips placeholder images and removes artifacts without usable images.
    pool = _build_pool(artifacts)
    if not pool:
        # If no valid artifacts remain after filtering, return an empty pool while
        # still including the underlying SPARQL and concepts for debugging.
        return jsonify({"sparql": sparql, "pool": [], "displayed": [], "concepts": concepts, "metrics": None})

    # Reset selection history for this request so the first batch is chosen from a
    # fresh state, then ask the selector to choose the next N items.
    history = SelectionHistory()
    MODULES["selection"].prepare(pool)   # per-search work, e.g. clustering
    displayed = MODULES["selection"].select(pool, n, history)

    # Optional retrieval evaluation: score the current pool against the query
    # features when an evaluator is configured.
    evaluator = MODULES.get("retrieval_evaluator")
    metrics = evaluator.score(pool, query_features) if evaluator else None

    # Persist the search trail for auditing/logging before returning the result.
    MODULES["logger"].log_search(nl_query, sparql, pool, displayed, query_features, exhibition_theme)
    return jsonify({"sparql": sparql, "pool": pool, "displayed": displayed, "concepts": concepts, "metrics": metrics})


@app.route("/next", methods=["POST"])
def next_batch():
    data = request.get_json(silent=True) or {}
    pool = data.get("pool", [])
    n = int(data.get("n", config.N_PER_BATCH))
    history = SelectionHistory.from_dict(data.get("history", {}))

    displayed = MODULES["selection"].select(pool, n, history)
    MODULES["logger"].log_next(data.get("history", {}), displayed)
    return jsonify({"displayed": displayed, "pool": pool})


@app.route("/export", methods=["POST"])
def export():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    meta = data.get("meta", {})
    meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    if not items:
        return jsonify({"error": "No items to export"}), 400

    MODULES["logger"].log_export(items, meta)
    payload = MODULES["exporter"].export(items, meta)
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=exhibition.json"},
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """Route die de tentoonstelling exporteert naar bestand en de data meegeeft aan Llama."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    items = data.get("items", [])
    meta = data.get("meta", {})

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    context_prefix = ""

    json_string_context = None

    # Als er items in de huidige sessie zijn, trigger dan de JSON-export op de achtergrond
    if items:
        try:
            # 1. Log de export (net als in de /export route)
            MODULES["logger"].log_export(items, meta)

            # 2. Genereer de JSON payload via de bestaande exporter
            payload = MODULES["exporter"].export(items, meta)

            # 3. Sla het bestand fysiek op als 'exhibition.json' in je werkmap
            export_filename = "exhibition.json"
            with open(export_filename, "wb") as f:
                f.write(payload if isinstance(payload, bytes) else payload.encode('utf-8'))

            # 4. Lees het bestand direct weer in als tekst-context voor het LLM
            with open(export_filename, "r", encoding="utf-8") as f:
                exhibition_data = json.load(f)

            # Formatteer dit compact voor het model
            json_string = json.dumps(exhibition_data, indent=2, ensure_ascii=False)
            context_prefix = f"[Systeem info: exhibition.json is succesvol opgeslagen]\nHier is de huidige tentoonstellingsdata uit exhibition.json:\n```json\n{json_string}\n```\n\nGebruikersvraag over deze data: "

        except Exception as e:
            print(f"Achtergrond export mislukt: {str(e)}")
            # We gaan door, maar zonder de bestandscontext
            context_prefix = "[Systeem info: Automatische export naar exhibition.json mislukt]\n"

    model = MODULES.get("chat_model")
    if model is None:
        return jsonify({"error": "No chat model configured (set CHAT_MODEL in .env)."}), 503

    try:
        # Combineer de bestandsinhoud met de vraag van de gebruiker
        full_message = f"{context_prefix}{user_message}"
        reply = generate_chat_response(model, full_message)
        return jsonify({"reply": reply})
    except ChatModelError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Chat model error: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
