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
        "random": "modules.selection.random_strategy.RandomStrategy",
    },
    "exporter": {
        "json": "modules.exporter.json_exporter.JSONExporter",
    },
    "retrieval_evaluator": {
        "diversity": "modules.retrieval_evaluator.diversity_evaluator.DiversityEvaluator",
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
    return _load_class("selection", key)()


def _build_exporter(key: str):
    return _load_class("exporter", key)()


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


def _build_retrieval_evaluator(key: str):
    # Imports are inside the function so sentence-transformers is not loaded
    # unless RETRIEVAL_EVALUATOR is explicitly set in .env.
    if key == "diversity":
        from modules.retrieval_evaluator.feature_encoder import MultilingualFeatureEncoder
        from modules.retrieval_evaluator.metrics_calculator import EntropyDiversityCalculator
        from modules.retrieval_evaluator.diversity_evaluator import DiversityEvaluator

        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        encoder    = MultilingualFeatureEncoder(model_name=model_name)
        calculator = EntropyDiversityCalculator()
        return DiversityEvaluator(encoder=encoder, calculator=calculator)
    raise ValueError(f"Unknown retrieval_evaluator key: {key!r}")


def build_modules() -> dict:
    qg_key  = os.getenv("QUERY_GENERATION", "category_select")
    kb_key  = os.getenv("KNOWLEDGE_BASE", "sparql_endpoint")
    sel_key = os.getenv("SELECTION", "random")
    exp_key = os.getenv("EXPORTER", "json")
    re_key  = os.getenv("RETRIEVAL_EVALUATOR", "").strip()

    knowledge_base = _build_knowledge_base(kb_key)
    modules = {
        "query_generation":     _build_query_generation(qg_key),
        "knowledge_base":       knowledge_base,
        "artifact_fetcher":     _build_artifact_fetcher(knowledge_base),
        "image_cache":          _build_image_cache(),
        "selection":            _build_selection(sel_key),
        "exporter":             _build_exporter(exp_key),
        "logger":               _build_logger(),
        "query_generation_key": qg_key,
    }
    if re_key:
        modules["retrieval_evaluator"] = _build_retrieval_evaluator(re_key)
    return modules
