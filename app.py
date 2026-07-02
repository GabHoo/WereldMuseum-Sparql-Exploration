import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request

import config
from modules.base import SelectionHistory

app = Flask(__name__)
MODULES = config.build_modules()


# ── Placeholder photo detection via perceptual image hashing ─────────────────
# Set PLACEHOLDER_IMAGE_FILE in .env to the path of a reference "not available"
# image (e.g. collections/wereld_museum/notavailable.jpg).
#
# At startup its average hash is computed once.  For each photo URL returned by
# a query the image is fetched and its hash compared.  A persistent in-process
# cache keyed by URL means every URL is fetched at most once per server
# lifetime — first search pays the cost, every repeat is instant.
#
# Requires: Pillow + imagehash  (both in requirements.txt)

_ref_hash = None
_placeholder_cache: dict[str, bool] = {}   # url → is_placeholder
_HASH_THRESHOLD = 5                         # max hamming distance to match

try:
    from PIL import Image
    import imagehash as _imagehash_lib

    _placeholder_image_file = os.getenv("PLACEHOLDER_IMAGE_FILE", "").strip()
    if _placeholder_image_file and os.path.exists(_placeholder_image_file):
        _ref_hash = _imagehash_lib.average_hash(Image.open(_placeholder_image_file))
except ImportError:
    pass  # Pillow/imagehash not installed; placeholder detection disabled


def _check_url(url: str) -> bool:
    """Fetch url, compute hash, compare to picture to reference not_available picture. Caches result."""
    if url in _placeholder_cache:
        return _placeholder_cache[url]
    result = False
    if _ref_hash is not None:
        try:
            with urllib.request.urlopen(url, timeout=4) as resp:
                data = resp.read()
            h = _imagehash_lib.average_hash(Image.open(io.BytesIO(data)))
            result = (h - _ref_hash) <= _HASH_THRESHOLD
        except Exception:
            result = False
    _placeholder_cache[url] = result
    return result


def _prefetch_placeholders(urls: set[str]) -> None:
    """Fetch all cache-miss URLs in parallel (max 10 workers)."""
    uncached = [u for u in urls if u not in _placeholder_cache]
    if not uncached or _ref_hash is None:
        return
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_check_url, u): u for u in uncached}
        for f in as_completed(futures):
            f.result()  # populate cache; errors already caught inside _check_url


# ── Pool normalisation ────────────────────────────────────────────────────────

def _normalize_pool(raw_rows: list) -> list:
    """
    Convert raw SPARQL result rows into the internal pool item schema:
        { id, title, score, properties }

    Dedup key is (artifact_id, photo_url):
      - same artifact + same photo → one card (collapses duplicate type rows)
      - same artifact + different photo → separate cards (distinct visual entries)
      - same artifact + no photo (or cleared placeholder) → one card
    """
    fm = config.FIELD_MAP

    # Parallel-fetch all unique photo URLs not yet in the cache, then compare
    # each against the reference placeholder hash.  Cache-hits are free on
    # every subsequent search that returns the same URLs.
    unique_photos = {
        row.get("url_photo", "").strip()
        for row in raw_rows
        if row.get("url_photo", "").strip()
    }
    _prefetch_placeholders(unique_photos)

    seen: set = set()
    pool = []
    for row in raw_rows:
        item_id = row.get(fm["id"], "").strip()
        if not item_id:
            continue
        photo = row.get("url_photo", "").strip()
        if not photo or _check_url(photo):
            continue
        dedup_key = (item_id, photo)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        pool.append({
            "id":         item_id,
            "title":      row.get(fm["title"], "").strip() or "Untitled",
            "score":      1.0,
            "properties": row,
        })
    return pool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    t2s = MODULES["text2sparql"]
    return render_template(
        "index.html",
        text2sparql_mode=MODULES["text2sparql_key"],
        categories=t2s.get_categories(),
        n_per_batch=config.N_PER_BATCH,
    )


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    nl_query = data.get("query", "").strip()
    n = int(data.get("n", config.N_PER_BATCH))

    if not nl_query:
        return jsonify({"error": "query is required"}), 400

    try:
        sparql = MODULES["text2sparql"].convert(nl_query, {})
        raw = MODULES["knowledge_base"].execute(sparql)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # Normalize the raw SPARQL results into the internal pool item schema
    pool = _normalize_pool(raw)
    if not pool:
        return jsonify({"sparql": sparql, "pool": [], "displayed": []})

    history = SelectionHistory()
    displayed = MODULES["selection"].select(pool, n, history)

    MODULES["logger"].log_search(nl_query, sparql, pool, displayed)
    return jsonify({"sparql": sparql, "pool": pool, "displayed": displayed})


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
