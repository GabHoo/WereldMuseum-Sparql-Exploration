import json
import os
import socket
from datetime import datetime, timezone

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

import config
from modules.artifact import Artifact
from modules.base import ChatModelError, SelectionHistory
from modules.embedding.service import DISPLAYED, FEEDBACK, POOL

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
    return render_template(
        "index.html",
        n_per_batch=config.N_PER_BATCH,
        chat_model_key=MODULES.get("chat_model_key"),
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
        # Preloaded demo queries are served from disk (see modules/preload.py).
        pool_cache = MODULES.get("pool_cache")
        artifacts = pool_cache.get(sparql) if pool_cache else None
        if artifacts is None:
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

    # Optional embedding: encode the results in the background for
    # embedding-based selection strategies (clustering, rerank).
    embedding = MODULES.get("embedding")
    if embedding:
        embedding.ensure(pool, POOL)

    if not pool:
        # If no valid artifacts remain after filtering, return an empty pool while
        # still including the underlying SPARQL and concepts for debugging.
        return jsonify({"sparql": sparql, "pool": [], "displayed": [], "concepts": concepts, "metrics": None})

    # Reset selection history for this request so the first batch is chosen from a
    # fresh state, then ask the selector to choose the next N items.
    history = SelectionHistory()
    MODULES["selection"].prepare(pool)   # per-search work, e.g. clustering
    displayed = MODULES["selection"].select(pool, n, history)
    if embedding:
        embedding.ensure(displayed, DISPLAYED)   # feedback on these comes next

    # Optional retrieval evaluation: score the current pool against the query
    # features when an evaluator is configured.
    evaluator = MODULES.get("retrieval_evaluator")
    metrics = evaluator.score(pool, query_features) if evaluator else None

    # Persist the search trail for auditing/logging before returning the result.
    MODULES["logger"].log_search(nl_query, sparql, pool, displayed, query_features, exhibition_theme,
                                 selection_debug=getattr(MODULES["selection"], "last_debug", None))
    return jsonify({"sparql": sparql, "pool": pool, "displayed": displayed, "concepts": concepts, "metrics": metrics})


@app.route("/next", methods=["POST"])
def next_batch():
    data = request.get_json(silent=True) or {}
    pool = data.get("pool", [])
    n = int(data.get("n", config.N_PER_BATCH))
    history = SelectionHistory.from_dict(data.get("history", {}))

    # Everything the strategy needs comes with the request: the pool, and the
    # kept items and negatives (full items, so they can be encoded even when
    # they came from an earlier search). Nothing is remembered per curator.
    embedding = MODULES.get("embedding")
    if embedding:
        embedding.ensure(data.get("feedback_items", []), FEEDBACK)
        embedding.ensure(pool, POOL)    # this tab's pool to the front again
    displayed = MODULES["selection"].select(pool, n, history)
    if embedding:
        embedding.ensure(displayed, DISPLAYED)
    MODULES["logger"].log_next(data.get("history", {}), displayed,
                               selection_debug=getattr(MODULES["selection"], "last_debug", None))
    return jsonify({"displayed": displayed, "pool": pool})


@app.route("/embedding/status")
def embedding_status():
    embedding = MODULES.get("embedding")
    status = embedding.status() if embedding else {"enabled": False}
    if PRELOADER:
        status["preload"] = PRELOADER.status()
    return jsonify(status)


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
    """The curator's chat message about the current selection → assistant reply."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    items = data.get("items", [])
    exhibition_theme = data.get("exhibition_theme", "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    assistant = MODULES.get("curation_assistant")
    if assistant is None:
        return jsonify({"error": "No chat model configured (set CHAT_MODEL in .env)."}), 503

    try:
        reply = assistant.reply(message, items, exhibition_theme)
    except ChatModelError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Chat model error: {e}"}), 500

    MODULES["logger"].log_chat(message, reply, items, exhibition_theme)
    return jsonify({"reply": reply})


@app.route("/exhibition", methods=["POST"])
def exhibition():
    """
    Build Exhibition: the curation page posts the selection as a form field
    `export` = JSON { items, meta }. Returns the page produced by the
    configured ExhibitionBuilder.
    """
    try:
        data = json.loads(request.form.get("export", "{}"))
    except ValueError:
        return jsonify({"error": "export is not valid JSON"}), 400
    items = data.get("items", [])
    meta = data.get("meta", {})
    meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    if not items:
        return jsonify({"error": "No items to build an exhibition from"}), 400

    export_data = {
        "meta":  meta,
        "items": [
            {"id": item["id"], "title": item["title"], "properties": item.get("properties", {})}
            for item in items
        ],
    }
    return MODULES["exhibition_builder"].build(export_data)


@app.route("/exhibition/render", methods=["POST"])
def exhibition_render():
    """
    Multipart form: scene (PNG composed on the exhibition page), description,
    variation (0 = faithful). Returns the rendered exhibition as a PNG.
    """
    scene = request.files.get("scene")
    if scene is None:
        return jsonify({"error": "scene image is required"}), 400
    try:
        png = MODULES["exhibition_builder"].render(
            scene.read(),
            request.form.get("description", "").strip(),
            int(request.form.get("variation", 0)),
        )
    except NotImplementedError:
        return jsonify({"error": "This exhibition builder does not render scenes"}), 501
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return Response(png, mimetype="image/png")


# Demo warm-up (PRELOAD_QUERIES). Under the debug reloader this module is
# loaded twice; only the serving child process (WERKZEUG_RUN_MAIN) preloads.
PRELOADER = config.build_preloader(MODULES, _build_pool)
if PRELOADER and (__name__ != "__main__" or os.environ.get("WERKZEUG_RUN_MAIN") == "true"):
    PRELOADER.start()


def _free_port(start: int) -> int:
    """First port from `start` up that nothing on localhost is listening on."""
    port = start
    while True:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


if __name__ == "__main__":
    # Several checkouts (git worktrees) can run side by side: 5000, 5001, …
    # The debug reloader re-runs this file in a child process, which must
    # reuse the parent's choice, so it is passed on through the environment.
    port = int(os.environ.get("APP_PORT") or _free_port(5000))
    os.environ["APP_PORT"] = str(port)
    app.run(debug=True, port=port)
