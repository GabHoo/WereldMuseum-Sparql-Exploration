"""
RunLogger — records every curation event to a JSON Lines file and/or stdout.

Each call to log_*() appends one JSON object on its own line so the log is
machine-readable (one json.loads() per line) and never needs to be fully
loaded into memory.

Controlled by two env vars (set in .env, documented in .env.example):
    LOG_FILE     path to the log file  (default: runs.log)
    LOG_CONSOLE  "true" to also print each record to stdout  (default: false)
"""
import json
from datetime import datetime, timezone


class RunLogger:
    def __init__(self, log_file: str = "runs.log", console: bool = False):
        self._path    = log_file
        self._console = console

    # ── public API ────────────────────────────────────────────────────────────

    def log_search(self, query: str, sparql: str, pool: list, displayed: list, query_features: dict = None) -> None:
        self._emit({
            "event":          "search",
            "query":          query,
            "query_features": query_features or {},
            "sparql":         sparql,
            "pool_size":      len(pool),
            "displayed_ids":  [i["id"] for i in displayed],
            "pool":           pool,
            "displayed":      displayed,
        })

    def log_next(self, history: dict, displayed: list) -> None:
        self._emit({
            "event":         "next",
            "history":       history,
            "displayed_ids": [i["id"] for i in displayed],
            "displayed":     displayed,
        })

    def log_export(self, items: list, meta: dict) -> None:
        self._emit({
            "event":       "export",
            "meta":        meta,
            "items_count": len(items),
            "items":       items,
        })

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, record: dict) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self._path:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._console:
            print(line, flush=True)
