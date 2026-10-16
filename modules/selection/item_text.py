"""
ItemText — turns a pool item into the text that gets embedded.

One spec string covers everything from a single field to a full sentence:

    "title"                                  → the title
    "title,type,material,place"              → fields joined with " · "
    "{title}[ — {type}][, made of {material}][, from {place}]"
                                             → a template

Template rules:
  {field}      any Artifact field (modules/artifact.py); list fields are
               joined with ", ", a missing value renders as "".
  [ ... ]      optional part: dropped entirely if any {field} inside it is
               empty, so "[, made of {material}]" disappears for items
               without a material instead of leaving ", made of ".

Field names are checked against the Artifact class when the spec is parsed,
so a typo fails at startup rather than silently embedding empty text.
"""
import re
import string

from modules.artifact import Artifact

_OPTIONAL = re.compile(r"\[([^\[\]]*)\]")


def _value(props: dict, field: str) -> str:
    v = props.get(field)
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return ", ".join(x.strip() for x in v if isinstance(x, str) and x.strip())
    return ""


def _fields_in(template: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


class ItemText:
    def __init__(self, spec: str, sep: str = " · "):
        spec = spec.strip()
        if "{" in spec:
            self._template = spec
        else:
            fields = [f.strip() for f in spec.split(",") if f.strip()]
            self._template = "".join(
                ("{%s}" % f) if i == 0 else ("[%s{%s}]" % (sep, f)) for i, f in enumerate(fields)
            )
        unknown = set(_fields_in(self._template)) - set(Artifact.field_names())
        if unknown:
            raise ValueError(f"Unknown Artifact field(s) {sorted(unknown)} in text spec {spec!r}. "
                             f"Known: {Artifact.field_names()}")
        self.spec = spec

    def __call__(self, item: dict) -> str:
        props = item.get("properties", {})

        def render(part: str) -> str:
            return part.format(**{f: _value(props, f) for f in _fields_in(part)})

        def optional(m: re.Match) -> str:
            part = m.group(1)
            return render(part) if all(_value(props, f) for f in _fields_in(part)) else ""

        text = render(_OPTIONAL.sub(optional, self._template))
        return re.sub(r"\s+", " ", text).strip(" ·,—-")
