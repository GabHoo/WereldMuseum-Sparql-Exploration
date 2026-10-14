"""
Module interfaces. Every pipeline step is an ABC here; config.py picks the
concrete class per step from .env.

Pipeline (/search):
    QueryGeneration ──▶ KnowledgeBase ──▶ ArtifactFetcher ──▶ SelectionStrategy
      │  (finds ?artifact URIs)            (URIs → Artifact objects, modules/artifact.py)
      │  (concept_query delegates text → concepts to a ConceptMatcher)
      └─ query features ──▶ RetrievalEvaluator (optional)
Then /next re-runs SelectionStrategy and /export runs Exporter.
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
    (ConceptQuery) or map input to URIs directly (CategorySelect).
    """

    @abstractmethod
    def convert(self, nl_query: str, context: dict) -> str:
        """
        Convert user input to a SPARQL *find* query: a SELECT whose ?artifact
        column lists the URIs of matching artifacts (use DISTINCT; LIMIT then
        counts artifacts). Metadata is not selected here — ArtifactFetcher
        fetches it for the returned URIs.
        """

    def get_categories(self) -> list:
        """
        Return display labels for category-based UI (e.g. button labels).
        Default is empty — only meaningful for CategorySelect-style implementations.
        """
        return []

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
    @abstractmethod
    def select(self, pool: list, n: int, history: SelectionHistory) -> list:
        """
        Pick n items to display from pool, excluding history.seen.
        May mutate item["score"] to persist rankings across rounds.
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
    @abstractmethod
    def build(self, export_data: dict) -> None:
        """
        Produce a digital exhibition from a curated selection.
        export_data: the dict produced by Exporter — { meta: {...}, items: [...] }
        """
