# WereldMuseum-Sparql-Exploration

Find in the notebook all information on how to navigate the linked data infrastrucutre of the Wereled Museum (https://collectie.wereldmuseum.nl/thesaurus/?query=search=purl=[TERMMASTER10068899]&showtype=record#/query/c8364281-326d-4a75-85fe-fe35418cb4f8 )

## SPARQL Web Demo

A small Flask app is included that serves an HTML page with two buttons. Clicking either button runs a different Wikidata SPARQL query and displays the results on the page.

### Run locally

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

### Files

- `app.py` - Flask backend that proxies SPARQL queries to Wikidata.
- `templates/index.html` - HTML page with buttons and JavaScript to fetch query results.
- `requirements.txt` - required Python packages.
