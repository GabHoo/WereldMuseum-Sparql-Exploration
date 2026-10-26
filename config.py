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
        "random":     "modules.selection.random_strategy.RandomStrategy",
        "clustering": "modules.selection.clustering_strategy.ClusteringStrategy",
        "rerank":     "modules.selection.rerank_strategy.RerankStrategy",
    },
    "exporter": {
        "json": "modules.exporter.json_exporter.JSONExporter",
    },
    "retrieval_evaluator": {
        "diversity": "modules.retrieval_evaluator.diversity_evaluator.DiversityEvaluator",
    },
    "chat_model": {
        "local_hf": "modules.chat_model.local_hf.LocalHFChatModel",
        "nebula":   "modules.chat_model.nebula.NebulaChatModel",
    },
    "exhibition_builder": {
        "digital_wall": "modules.exhibition_builder.digital_wall.DigitalWall",
    },
    "embedding": {
        "metadata_text": "modules.embedding.metadata_text.MetadataTextEncoder",
        "multimodal":    "modules.embedding.multimodal.MultimodalEncoder",
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


def rerank_kwargs() -> dict:
    """RerankStrategy settings from the RERANK_* env vars."""
    return {
        "scoring":     os.getenv("RERANK_SCORING", "rocchio").strip() or "rocchio",
        "beta":        float(os.getenv("RERANK_BETA", "1.0")),
        "gamma":       float(os.getenv("RERANK_GAMMA", "0.5")),
        "k":           int(os.getenv("RERANK_K", "3")),
        "mmr_lambda":  float(os.getenv("RERANK_MMR_LAMBDA", "0.7")),
        "explore":     float(os.getenv("RERANK_EXPLORE", "0.2")),
        "wait":        float(os.getenv("RERANK_WAIT", "1.0")),
        "cross_query": os.getenv("RERANK_CROSS_QUERY", "true").strip().lower() != "false",
    }


def _build_selection(key: str, embedding=None, embedding_key: str = ""):
    # Embedding-based strategies get the EmbeddingService (None when EMBEDDING
    # is unset: they then select at random) and the view to work in.
    cls = _load_class("selection", key)
    if key == "clustering":
        from modules.selection.clustering import KMeansClusterer
        return cls(
            embedding=embedding,
            clusterer=KMeansClusterer(),
            view=embedding_view(embedding_key),
            n_clusters=int(os.getenv("CLUSTERING_N_CLUSTERS", str(2 * N_PER_BATCH))),
            steering_weight=float(os.getenv("CLUSTERING_STEERING_WEIGHT", "0.6")),
            shuffle_step=float(os.getenv("CLUSTERING_SHUFFLE_STEP", "0.25")),
            wait=float(os.getenv("CLUSTERING_WAIT", "3")),
        )
    if key == "rerank":
        return cls(embedding=embedding, view=embedding_view(embedding_key), **rerank_kwargs())
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


def _build_chat_model(key: str):
    cls = _load_class("chat_model", key)
    if key == "local_hf":
        return cls(
            model_id=os.getenv("CHAT_LOCAL_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
            device=os.getenv("CHAT_DEVICE", "").strip() or None,   # default: mps → cuda → cpu
        )
    if key == "nebula":
        from modules.chat_model.nebula import NEBULA_BASE_URL
        return cls(
            api_key=os.getenv("NEBULA_API_KEY", ""),
            model=os.getenv("NEBULA_MODEL", "deepseek-r1:1.5b"),
            base_url=os.getenv("NEBULA_BASE_URL", NEBULA_BASE_URL),
        )
    return cls()


def _build_curation_assistant(chat_key: str):
    from modules.curation_assistant import DEFAULT_CONTEXT_FIELDS, CurationAssistant
    return CurationAssistant(
        model=_build_chat_model(chat_key),
        context_fields=os.getenv("CHAT_CONTEXT_FIELDS", DEFAULT_CONTEXT_FIELDS),
    )


def embedding_view(key: str) -> str:
    """EMBEDDING_VIEW, or the embedding's own default (multimodal: fused, else text)."""
    return os.getenv("EMBEDDING_VIEW", "").strip() or ("fused" if key == "multimodal" else "text")


def _build_encoder(key: str):
    cls = _load_class("embedding", key)
    if key == "metadata_text":
        from modules.embedding.sentence_model import DEFAULT_MODEL
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
            fields=os.getenv("EMBEDDING_TEXT_FIELDS", "title"),
        )
    if key == "multimodal":
        from modules.embedding.multimodal import DEFAULT_MODEL
        dim = os.getenv("EMBEDDING_DIM", "").strip()
        image_tokens = os.getenv("EMBEDDING_IMAGE_TOKENS", "").strip()
        return cls(
            model_name=os.getenv("EMBEDDING_MULTIMODAL_MODEL", DEFAULT_MODEL),
            views=(embedding_view(key),),
            dim=int(dim) if dim else None,
            device=os.getenv("EMBEDDING_DEVICE", "").strip() or None,
            dtype=os.getenv("EMBEDDING_DTYPE", "auto").strip() or "auto",
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            image_tokens=int(image_tokens) if image_tokens else None,
        )
    return cls()


def build_embedding(key: str, image_cache=None):
    """EmbeddingService around the encoder for key. Imports happen here, so no
    embedding model is loaded unless EMBEDDING is set."""
    from modules.embedding.service import EmbeddingService
    return EmbeddingService(
        _build_encoder(key),
        image_cache or _build_image_cache(),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "5000")),
    )


def build_preloader(modules: dict, build_pool):
    """Preloader for the demo queries in PRELOAD_QUERIES, or None when unset."""
    path = os.getenv("PRELOAD_QUERIES", "").strip()
    if not path:
        return None
    import json
    import pathlib
    from modules.preload import Preloader
    queries = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return Preloader(queries, modules, modules["pool_cache"], build_pool)


def build_modules() -> dict:
    qg_key  = os.getenv("QUERY_GENERATION", "concept_query")
    kb_key  = os.getenv("KNOWLEDGE_BASE", "sparql_endpoint")
    sel_key = os.getenv("SELECTION", "random")
    exp_key = os.getenv("EXPORTER", "json")
    eb_key  = os.getenv("EXHIBITION_BUILDER", "digital_wall")
    re_key  = os.getenv("RETRIEVAL_EVALUATOR", "").strip()
    chat_key = os.getenv("CHAT_MODEL", "local_hf").strip()
    emb_key  = os.getenv("EMBEDDING", "").strip()

    knowledge_base = _build_knowledge_base(kb_key)
    modules = {
        "query_generation":     _build_query_generation(qg_key),
        "knowledge_base":       knowledge_base,
        "artifact_fetcher":     _build_artifact_fetcher(knowledge_base),
        "image_cache":          _build_image_cache(),
        "exporter":             _build_exporter(exp_key),
        "exhibition_builder":   _build_exhibition_builder(eb_key),
        "logger":               _build_logger(),
    }
    if os.getenv("PRELOAD_QUERIES", "").strip():
        from modules.preload import PoolCache
        modules["pool_cache"] = PoolCache(os.getenv("POOL_CACHE_DIR", ".cache/pools"))
    if re_key:
        modules["retrieval_evaluator"] = _build_retrieval_evaluator(re_key, knowledge_base)
    if emb_key:
        modules["embedding"] = build_embedding(emb_key, modules["image_cache"])
    # After the embedding: embedding-based strategies get it injected.
    modules["selection"] = _build_selection(sel_key, modules.get("embedding"), emb_key)
    if chat_key:
        modules["curation_assistant"] = _build_curation_assistant(chat_key)
        modules["chat_model_key"]     = chat_key
    return modules
