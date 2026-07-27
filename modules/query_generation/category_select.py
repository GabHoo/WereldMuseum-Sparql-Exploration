import json
import pathlib

from modules.base import QueryGeneration

# Default categories for the Wereld Museum / Colonial Collections graph.
# These are bundled as the "dummy" baseline implementation.
# Override at deployment by pointing CATEGORIES_FILE to a different JSON file.
_DEFAULT_CATEGORIES = {
    "Funerary Objects":     "https://hdl.handle.net/20.500.11840/termmaster10068883",
    "Ceremonial Objects":   "https://hdl.handle.net/20.500.11840/termmaster10068882",
    "Holders & Containers": "https://hdl.handle.net/20.500.11840/termmaster10070376",
    "Costumes":             "https://hdl.handle.net/20.500.11840/termmaster10069572",
}

# SPARQL template for the Wereld Museum / Colonial Collections graph.
# Uses CIDOC-CRM + SKOS + Linked Art ontologies.
# Walks skos:narrower* to include all sub-types of the chosen category.
# Override at runtime via context["sparql_template"] if needed.
_SPARQL_TEMPLATE = """
    PREFIX la:   <https://linked.art/ns/terms/>
    PREFIX crm:  <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct:  <http://purl.org/dc/terms/>
    SELECT
      ?specificType
      (GROUP_CONCAT(DISTINCT ?typeLabelRaw; separator=" | ") AS ?typeLabel)
      ?artifact
      ?artifacttitle
      ?url_photo
    WHERE {{
      BIND(<{category_uri}> AS ?chosenCategory)
      ?chosenCategory skos:narrower* ?specificType .
      ?artifact crm:P2_has_type ?specificType .
      OPTIONAL {{ ?specificType skos:altLabel ?typeLabelRaw . }}
      OPTIONAL {{ ?artifact dct:title ?artifacttitle . }}
      OPTIONAL {{
        ?artifact a crm:E22_Human-Made_Object .
        ?artifact crm:P65_shows_visual_item ?vi .
        ?vi la:digitally_shown_by ?o2 .
        ?o2 la:access_point ?url_photo .
      }}
      FILTER(BOUND(?url_photo))
    }}
    GROUP BY ?specificType ?artifact ?artifacttitle ?url_photo
    LIMIT 100
"""


class CategorySelect(QueryGeneration):
    """
    Baseline QueryGeneration implementation: presents the user with a fixed list
    of category buttons. Each button maps to a known concept URI which is
    substituted into the SPARQL template.

    Ships with Wereld Museum categories as built-in defaults so it works
    out of the box with no configuration. To target a different collection:
      - Point CATEGORIES_FILE in .env to a JSON file {label: uri}
      - The file entries are merged on top of (and override) the defaults

    context keys (all optional overrides at call time):
        categories      : dict {label: uri}
        sparql_template : full SPARQL template string with {category_uri}
    """

    def __init__(self, categories_file: str = ""):
        self._categories = dict(_DEFAULT_CATEGORIES)
        if categories_file:
            loaded = json.loads(
                pathlib.Path(categories_file).read_text(encoding="utf-8")
            )
            self._categories.update(loaded)

    def get_categories(self) -> list:
        return list(self._categories.keys())

    def convert(self, nl_query: str, context: dict) -> str:
        categories = {**self._categories, **context.get("categories", {})}
        template = context.get("sparql_template") or _SPARQL_TEMPLATE

        uri = categories.get(nl_query)
        if not uri:
            raise ValueError(
                f"Unknown category {nl_query!r}. "
                f"Available: {list(categories.keys())}"
            )
        return template.format(category_uri=uri)
