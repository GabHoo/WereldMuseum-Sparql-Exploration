from flask import Flask, jsonify, render_template, request
from SPARQLWrapper import SPARQLWrapper, JSON
import json

app = Flask(__name__)

# Museo collection SPARQL endpoint
SPARQL_ENDPOINT = "https://api.colonialcollections.nl/datasets/nmvw/collection-archives/sparql"

# Map entity names to their URIs
ENTITY_URIS = {
    "Funerary Objects": "https://hdl.handle.net/20.500.11840/termmaster10068883",
    "Ceremonial Objects": "https://hdl.handle.net/20.500.11840/termmaster10068882",
    "Holders & Containers": "https://hdl.handle.net/20.500.11840/termmaster10070376",
    "Costumes": "https://hdl.handle.net/20.500.11840/termmaster10069572",
}

# Single query template with placeholder for category URI
QUERY_TEMPLATE="""
    PREFIX la: <https://linked.art/ns/terms/>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT  
    ?chosenCategory
    (GROUP_CONCAT(DISTINCT ?categoryLabelRaw; separator=" | ") AS ?chosenCategoryLabel) 
    ?specificType 
    (GROUP_CONCAT(DISTINCT ?typeLabelRaw; separator=" | ") AS ?typeLabel)
    ?artifact 
    ?artifacttitle
    ?url_photo 
    WHERE {{
    
    BIND(<{category_uri}> AS ?chosenCategory)
    ?chosenCategory skos:altLabel ?categoryLabelRaw .

    # 2. Recursively find the category itself and all of its narrower concepts
    ?chosenCategory skos:narrower* ?specificType .
    
    # 3. Find the objects that have these types
    ?artifact crm:P2_has_type ?specificType .

    # 4. Get labels and title for the objects and types
    OPTIONAL {{ 
        ?specificType skos:altLabel ?typeLabelRaw . 
    }}
    OPTIONAL {{ 
        ?artifact dct:title ?artifacttitle . 
    }}
    OPTIONAL {{
        ?artifact a crm:E22_Human-Made_Object.
        ?artifact crm:P65_shows_visual_item ?vi.
        ?vi la:digitally_shown_by ?o2.
        ?o2 la:access_point ?url_photo .
    }}
    }}
    GROUP BY ?chosenCategory ?specificType ?artifact ?artifacttitle ?url_photo
    LIMIT 20
    """


@app.route("/")
def index():
    return render_template("index.html", entities=ENTITY_URIS)


@app.route("/entities", methods=["GET"])
def get_entities():
    """Return the list of entities for the frontend to render buttons."""
    return jsonify({"entities": ENTITY_URIS})


@app.route("/query", methods=["POST"])
def run_query():
    data = request.get_json(silent=True) or {}
    entity_name = data.get("entity")

    if entity_name not in ENTITY_URIS:
        return jsonify({"error": "Unknown entity"}), 400

    # Get the URI for this entity
    category_uri = ENTITY_URIS[entity_name]
    
    # Format the query with the category URI binding
    query = QUERY_TEMPLATE.format(category_uri=category_uri)
    
    try:
        # Set up the connection using SPARQLWrapper
        sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # Execute query and convert to JSON
        results = sparql.query().convert()
        
        # Parse JSON results
        bindings = results.get("results", {}).get("bindings", [])
        
        # Extract values from the SPARQL JSON structure
        items = []
        for row in bindings:
            item = {var: row[var]["value"] for var in row}
            items.append(item)
        
        return jsonify({"entity": entity_name, "items": items})
    
    except Exception as e:
        return jsonify({"error": "SPARQL query failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)


@app.route("/save_selections", methods=["POST"])
def save_selections():
    """Save final selections (JSON array) to a file and return the path."""
    data = request.get_json(silent=True) or {}
    selections = data.get("selections")
    if not selections:
        return jsonify({"error": "Missing selections in request"}), 400

    out_path = "saved_selections.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(selections, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({"error": "Failed to write file", "details": str(e)}), 500

    return jsonify({"status": "saved", "path": out_path})
