"""
ArtifactFetcher — turns a list of artifact URIs into Artifact objects.

Second half of every search:
  1. QueryGeneration's SPARQL *finds* artifacts (returns ?artifact URIs only).
  2. ArtifactFetcher *fetches* their metadata with one query per batch:

       SELECT ?artifact ?field ?value WHERE {
         VALUES ?artifact { <uri1> <uri2> ... }
         { <title pattern>    BIND("title"    AS ?field) }
         UNION { <material pattern> BIND("material" AS ?field) }
         ...
       }

     Each row is one (artifact, field, value) triple, so multi-valued fields
     never multiply each other (no cross product, no GROUP_CONCAT). Rows are
     folded into {field: [values]} and then into Artifact objects.

The field → SPARQL pattern mapping is graph-specific and lives in a JSON file
(ARTIFACT_FIELDS_FILE, default collections/wereld_museum/artifact_fields.json).
"""
import json
import pathlib

from modules.artifact import Artifact
from modules.base import KnowledgeBase

_BATCH_SIZE = 100  # artifacts per fetch query


class ArtifactFetcher:
    """Load a graph-specific field mapping and fetch artifact metadata rows.

    Each knowledge graph exposes different RDF predicates for the same logical
    Artifact fields (for example title/material/creator). The JSON file passed to
    the constructor contains the SPARQL patterns that map each Artifact field to
    the right predicate(s) for that graph. The fetcher validates that every
    configured field is known to the Artifact model before running any queries.
    """

    def __init__(self, knowledge_base: KnowledgeBase, fields_file: str):
        """Read the field specification, configure prefixes, and validate fields.

        The file is expected to contain a JSON object with optional "prefixes" and
        required "fields" entries. The field names in "fields" must match the
        names defined on Artifact; otherwise building fetch queries would produce
        invalid data mappings.
        """
        spec = json.loads(pathlib.Path(fields_file).read_text(encoding="utf-8"))
        self._kb = knowledge_base
        self._prefixes = "\n".join(
            f"PREFIX {p}: <{uri}>" for p, uri in spec.get("prefixes", {}).items()
        )
        self._patterns: dict[str, str] = spec["fields"]
        unknown = set(self._patterns) - set(Artifact.field_names())
        if unknown:
            raise ValueError(
                f"{fields_file}: unknown Artifact field(s) {sorted(unknown)}. "
                f"Known: {Artifact.field_names()}"
            )

    def build_query(self, artifact_ids: list[str]) -> str:
        values = " ".join(f"<{i}>" for i in artifact_ids)
        branches = "\n  UNION\n  ".join(
            f'{{ {pattern} BIND("{name}" AS ?field) }}'
            for name, pattern in self._patterns.items()
        )
        return (
            f"{self._prefixes}\n"
            "SELECT ?artifact ?field ?value WHERE {\n"
            f"  VALUES ?artifact {{ {values} }}\n"
            f"  {branches}\n"
            "}"
        )

    def fetch(self, artifact_ids: list[str]) -> list[Artifact]:
        """Return one Artifact per id, in the order given. Duplicate ids are fetched once."""
        ids = list(dict.fromkeys(artifact_ids))
        collected: dict[str, dict[str, list[str]]] = {i: {} for i in ids}

        for start in range(0, len(ids), _BATCH_SIZE):
            batch = ids[start:start + _BATCH_SIZE]
            for row in self._kb.execute(self.build_query(batch)):
                art, name, value = row.get("artifact"), row.get("field"), row.get("value", "").strip()
                if art not in collected or not value:
                    continue
                vals = collected[art].setdefault(name, [])
                if value not in vals:
                    vals.append(value)

        return [Artifact.from_values(i, collected[i]) for i in ids]
