"""
Artifact — the internal, graph-agnostic description of one collection object.

Every artifact has the same fields, whatever the source graph. A field is
either single-valued (str | None) or multi-valued (list[str]); the type
annotation below decides which. Multiple values are kept as separate list
entries, never concatenated into one string.

How each field is read from a particular graph is configured outside the code,
in a fields file (e.g. collections/wereld_museum/artifact_fields.json) used by
ArtifactFetcher. A field the graph does not provide simply stays None / [].
"""
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Artifact:
    id: str                                   # artifact URI
    title: str | None = None
    inventory_number: str | None = None
    intended_use: str | None = None
    dimensions: str | None = None
    type: list[str] = field(default_factory=list)          # object type labels
    material: list[str] = field(default_factory=list)      # material / technique labels
    place: list[str] = field(default_factory=list)         # place of production labels
    date: list[str] = field(default_factory=list)          # production date, as free text
    maker: list[str] = field(default_factory=list)         # person or group that made it
    description: list[str] = field(default_factory=list)
    acquisition: list[str] = field(default_factory=list)   # how the museum acquired it
    acquired_from: list[str] = field(default_factory=list) # previous owner(s)
    images: list[str] = field(default_factory=list)        # image URLs
    # Concept URIs behind type / material / place (same concepts as the labels
    # above). Used to compare an artifact with the searched concepts by
    # identity rather than by label text.
    type_uris: list[str] = field(default_factory=list)
    material_uris: list[str] = field(default_factory=list)
    place_uris: list[str] = field(default_factory=list)

    @classmethod
    def field_names(cls) -> list[str]:
        """All fields except id, in declaration order."""
        return [f.name for f in fields(cls) if f.name != "id"]

    @classmethod
    def is_multi(cls, name: str) -> bool:
        """True if the field holds a list of values."""
        return cls.__dataclass_fields__[name].default_factory is list

    @classmethod
    def from_values(cls, artifact_id: str, values: dict[str, list[str]]) -> "Artifact":
        """
        Build an Artifact from {field_name: [values...]}, as collected from the graph.
        Multi-valued fields keep every value; single-valued fields keep the first.
        Unknown field names are ignored.
        """
        kwargs = {}
        for name in cls.field_names():
            vals = values.get(name) or []
            if cls.is_multi(name):
                kwargs[name] = list(vals)
            elif vals:
                kwargs[name] = vals[0]
        return cls(id=artifact_id, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Artifact":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
