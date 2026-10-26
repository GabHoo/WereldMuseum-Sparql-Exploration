"""
Module interfaces. Every pipeline step is an ABC here; config.py picks the
concrete class per step from .env.

Pipeline (/search):
    QueryGeneration ──▶ KnowledgeBase ──▶ ArtifactFetcher ──▶ SelectionStrategy
      │  (finds ?artifact URIs)            (URIs → Artifact objects, modules/artifact.py)
      │  (concept_query delegates text → concepts to a ConceptMatcher)
      └─ query features ──▶ RetrievalEvaluator (optional)
Then /next re-runs SelectionStrategy, /export runs Exporter, /exhibition
runs ExhibitionBuilder and /api/chat runs a ChatModel (through
modules/curation_assistant.py).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SelectionHistory:
    accepted: list = field(default_factory=list)  # item IDs user said yes to
    rejected: list = field(default_factory=list)  # item IDs user said no to
    seen: list = field(default_factory=list)       # all shown item IDs (accepted + rejected)

    @classmethod
    def from_dict(cls, d: dict) -> "SelectionHistory":
        return cls(
            accepted=d.get("accepted", []),
            rejected=d.get("rejected", []),
            seen=d.get("seen", []),
        )


class ConceptMatcher(ABC):
    """
    Text → concepts. Decides *which concepts the user meant*; knows nothing
    about SPARQL. Used by concept-based QueryGeneration implementations
    (ConceptQuery), which decide *what to fetch* for those concepts.
    """

    @abstractmethod
    def match(self, nl_query: str) -> list:
        """
        Map user text to concepts in the collection's vocabulary.

        Returns a list of {"uri": str, "label": str | None, "feature": str}:
          uri     : concept URI to anchor the query on
          label   : human-readable label, or None if the URI is not in the
                    vocabulary (e.g. an ancestor reached by generalisation)
          feature : which item feature the concept constrains, using the
                    shared feature vocabulary, which matches Artifact
                    field names (type | material | place)
        Returns [] when nothing matches.
        """


class QueryGeneration(ABC):
    """
    User input → SPARQL. The entry point app.py talks to for /search.
    Implementations may delegate concept matching to a ConceptMatcher
    (ConceptQuery) or map input to URIs directly.
    """

    @abstractmethod
    def convert(self, nl_query: str, context: dict) -> str:
        """
        Convert user input to a SPARQL *find* query: a SELECT whose ?artifact
        column lists the URIs of matching artifacts (use DISTINCT; LIMIT then
        counts artifacts). Metadata is not selected here — ArtifactFetcher
        fetches it for the returned URIs.
        """

    def match_concepts(self, nl_query: str) -> list:
        """
        Return the concepts derived from nl_query, for display in the UI.
        Each entry is {"label": str, "uri": str, "feature": str} — the
        ConceptMatcher output, minus concepts that have no label.
        Default is empty — override in implementations that do explicit concept matching.
        """
        return []

    def get_query_features(self, nl_query: str) -> dict:
        """
        Extract pinned feature constraints from nl_query as a flat dict.
        Returns {feature_name: [concept URIs]} for each feature the query
        constrains. Feature names match Artifact field names (type | material |
        place); the URIs are compared with the artifact's <feature>_uris field.
        Default returns {} (all features are free; only diversity is computed).
        Override in subclasses that perform structured feature extraction.
        """
        return {}


class KnowledgeBase(ABC):
    @abstractmethod
    def execute(self, sparql: str) -> list:
        """Execute a SPARQL query and return result rows as a list of dicts."""


class SelectionStrategy(ABC):
    """
    Decides which pool items the user sees, batch by batch.

    The server keeps no state between requests: the pool travels to the
    browser and back with every /next call. Anything a strategy needs to
    remember about an item must therefore be stored *on the item*, as
    JSON-serialisable data:
      item["score"]      float, free to use for rankings
      item["selection"]  dict, free-form annotations written by prepare()
    Keep annotations small (ids, labels, numbers) — never vectors.
    """

    def prepare(self, pool: list) -> None:
        """
        Called once per /search on the freshly built pool, before the first
        select(). Use it for per-search work such as embedding or clustering,
        and store the results in item["selection"]. Default: nothing.
        """

    @abstractmethod
    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        """
        Pick n items to display from pool, excluding history.seen.
        history.accepted / rejected hold the ids the user kept / passed over.
        May mutate item["score"] / item["selection"]; the changes travel with
        the pool into the next call.
        """


class Exporter(ABC):
    @abstractmethod
    def export(self, items: list, meta: dict) -> bytes:
        """Serialise selected items + meta into bytes for download."""


class RetrievalEvaluator(ABC):
    @abstractmethod
    def score(self, pool: list[dict], query: dict) -> dict:
        """
        Compute pool-level metrics from retrieved items and pinned query features.
        pool : pool items whose "properties" is an Artifact dict
        query: {feature_name: [concept URIs]} for each pinned feature
               (QueryGeneration.get_query_features); {} means all features free.
        Returns {"diversity": float, "surprise": float | None}
        """


class ExhibitionBuilder(ABC):
    """
    Turns a curated selection into an exhibition. app.py calls build() when the
    user clicks *Build Exhibition* and serves the returned page (/exhibition).
    """

    @abstractmethod
    def build(self, export_data: dict) -> str:
        """
        Produce a digital exhibition from a curated selection.
        export_data: same shape as the export file — { meta: {...}, items: [...] },
                     items are { id, title, properties: Artifact dict }
        Returns the exhibition as an HTML page.
        """

    def render(self, scene_png: bytes, description: str, variation: int) -> bytes:
        """
        Turn a scene the user composed on the exhibition page into a
        photorealistic exhibition image (PNG bytes), served by /exhibition/render.
        variation 0 is a faithful render; higher numbers are re-imaginings.
        Default: not supported — only meaningful for builders whose page lets
        the user compose a scene (DigitalWall).
        """
        raise NotImplementedError


class ChatModelError(Exception):
    """A chat model could not answer (missing login/key, model unavailable, ...).
    The message is meant to be shown to the user."""


class ChatModel(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system" | "user" | "assistant", "content": str}, ...]
        Returns the assistant's reply. Raise ChatModelError with a readable
        message when the model cannot answer.
        """
