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


class QueryGeneration(ABC):
    @abstractmethod
    def convert(self, nl_query: str, context: dict) -> str:
        """Convert natural-language input to a SPARQL SELECT query string."""

    def get_categories(self) -> list:
        """
        Return display labels for category-based UI (e.g. button labels).
        Default is empty — only meaningful for CategorySelect-style implementations.
        """
        return []

    def match_concepts(self, nl_query: str) -> list:
        """
        Return the intermediate concepts derived from nl_query before SPARQL generation.
        Each entry is {"label": str, "uri": str}.
        Default is empty — override in implementations that do explicit concept matching.
        """
        return []


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


class ExhibitionBuilder(ABC):
    @abstractmethod
    def build(self, export_data: dict) -> None:
        """
        Produce a digital exhibition from a curated selection.
        export_data: the dict produced by Exporter — { meta: {...}, items: [...] }
        """
