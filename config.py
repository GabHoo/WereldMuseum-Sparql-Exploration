"""
Central wiring: reads .env, resolves module keys to concrete classes,
and instantiates them with their module-specific constructor arguments.

To swap a module:
  1. Add an entry to REGISTRY[<type>] pointing to your class.
  2. If your class needs extra constructor args, add a branch in the
     corresponding _build_*() function and document the env var in .env.example.
  3. Set the env var in .env.
"""
import importlib
import os

from dotenv import load_dotenv

load_dotenv()

# ── Registry: key → dotted class path ────────────────────────────────────────

REGISTRY = {
    "query_generation": {
        "category_select":  "modules.query_generation.category_select.CategorySelect",
        "concept_query":    "modules.query_generation.concept_query.ConceptQuery",
        # Legacy alias: before the matcher/query split this key named the
        # combined module. Kept so existing .env files keep working.
        "keyword_matching": "modules.query_generation.concept_query.ConceptQuery",
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
    "exhibition_builder": {
        "digital_wall": "modules.exhibition_builder.digital_wall.DigitalWall",
    },
}


def _load_class(module_type: str, key: str):
    dotted = REGISTRY[module_type][key]
    mod_path, class_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), class_name)


# ── General settings ──────────────────────────────────────────────────────────

N_PER_BATCH = int(os.getenv("N_PER_BATCH", "9"))


# ── Module builders ───────────────────────────────────────────────────────────

def _build_query_generation(key: str):
    cls = _load_class("query_generation", key)
    if key == "category_select":
        return cls(categories_file=os.getenv("CATEGORIES_FILE", ""))
    if key in ("concept_query", "keyword_matching"):
        return cls(matcher=_build_concept_matcher(os.getenv("CONCEPT_MATCHER", "thesaurus")))
    return cls()


def _build_concept_matcher(key: str):
    cls = _load_class("concept_matcher", key)
    if key == "thesaurus":
        files_raw = os.getenv("KEYWORD_MATCHING_THESAURUS_FILES", "")
        files = [f.strip() for f in files_raw.split(",") if f.strip()]
        return cls(
            thesaurus_files=files,
            sparql_endpoint_url=os.getenv("SPARQL_ENDPOINT_URL", ""),
        )
    return cls()


def _build_knowledge_base(key: str):
    cls = _load_class("knowledge_base", key)
    if key == "sparql_endpoint":
        return cls(endpoint_url=os.getenv("SPARQL_ENDPOINT_URL", ""))
    if key == "rdf_file":
        return cls(file_path=os.getenv("RDF_FILE_PATH", ""))
    return cls()


def _build_selection(key: str):
    cls = _load_class("selection", key)
    if key == "title_cluster":
        from modules.embedding import DEFAULT_MODEL, TextEmbedder
        from modules.selection.clustering import KMeansClusterer
        from modules.selection.utils import metadata_linearization
        return cls(
            embedder=TextEmbedder(os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)),
            clusterer=KMeansClusterer(),
            linearize=metadata_linearization(os.getenv("TITLE_CLUSTER_FIELDS", "title")),
            n_clusters=int(os.getenv("TITLE_CLUSTER_N_CLUSTERS", str(2 * N_PER_BATCH))),
            steering_weight=float(os.getenv("TITLE_CLUSTER_STEERING_WEIGHT", "0.6")),
            shuffle_step=float(os.getenv("TITLE_CLUSTER_SHUFFLE_STEP", "0.25")),
        )
    return cls()


def _build_exporter(key: str):
    return _load_class("exporter", key)()


def _build_exhibition_builder(key: str):
    cls = _load_class("exhibition_builder", key)
    if key == "digital_wall":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            image_model=os.getenv("DIGITAL_WALL_IMAGE_MODEL", "gpt-image-2"),
            image_size=os.getenv("DIGITAL_WALL_IMAGE_SIZE", "1024x1024"),
        )
    return cls()


def _build_logger():
    from modules.run_logger import RunLogger
    return RunLogger(
        log_file=os.getenv("LOG_FILE", "runs.log"),
        console=os.getenv("LOG_CONSOLE", "false").lower() == "true",
    )


def _build_artifact_fetcher(knowledge_base):
    from modules.artifact_fetcher import ArtifactFetcher
    return ArtifactFetcher(
        knowledge_base=knowledge_base,
        fields_file=os.getenv(
            "ARTIFACT_FIELDS_FILE", "collections/wereld_museum/artifact_fields.json"
        ),
    )


def _build_image_cache():
    from modules.image_cache import ImageCache
    hosts = os.getenv("IMAGE_HOSTS", "collectie.wereldculturen.nl")
    return ImageCache(
        cache_dir=os.getenv("THUMBNAIL_CACHE_DIR", ".cache/thumbnails"),
        allowed_hosts=[h.strip() for h in hosts.split(",")],
        size=int(os.getenv("THUMBNAIL_SIZE", "480")),
        placeholder_file=os.getenv("PLACEHOLDER_IMAGE_FILE", "").strip(),
    )


def _build_retrieval_evaluator(key: str, knowledge_base):
    # Imports are inside the function so sentence-transformers is not loaded
    # unless RETRIEVAL_EVALUATOR is explicitly set in .env.
    if key == "diversity":
        from modules.retrieval_evaluator.diversity_evaluator import build_default
        return build_default(knowledge_base)
    raise ValueError(f"Unknown retrieval_evaluator key: {key!r}")


def build_modules() -> dict:
    qg_key  = os.getenv("QUERY_GENERATION", "category_select")
    kb_key  = os.getenv("KNOWLEDGE_BASE", "sparql_endpoint")
    sel_key = os.getenv("SELECTION", "random")
    exp_key = os.getenv("EXPORTER", "json")
    eb_key  = os.getenv("EXHIBITION_BUILDER", "digital_wall")
    re_key  = os.getenv("RETRIEVAL_EVALUATOR", "").strip()

    knowledge_base = _build_knowledge_base(kb_key)
    modules = {
        "query_generation":     _build_query_generation(qg_key),
        "knowledge_base":       knowledge_base,
        "artifact_fetcher":     _build_artifact_fetcher(knowledge_base),
        "image_cache":          _build_image_cache(),
        "selection":            _build_selection(sel_key),
        "exporter":             _build_exporter(exp_key),
        "exhibition_builder":   _build_exhibition_builder(eb_key),
        "logger":               _build_logger(),
        "query_generation_key": qg_key,
    }
    if re_key:
        modules["retrieval_evaluator"] = _build_retrieval_evaluator(re_key, knowledge_base)
    return modules
