# Recipe Helper

Recipe Helper is a REST API for finding recipes based on natural-language requests.

Users can specify ingredients, ingredients to exclude, time constraints, and softer preferences such as "spicy", "comforting", or "festive". The API supports requests in multiple languages and returns matching recipes in English.

# Setup and Running

The application was developed and tested on Windows with Python 3.14.7.

It requires:
- Python 3.14.7
- an OpenAI API key
- internet access when interpreting queries and generating embeddings

Python package dependencies and tested versions are listed in `requirements.txt`.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Set the OPENAI_API_KEY environment variable before running the application.

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### 4. Extract the recipe dataset

20170107-061401-recipeitems.json.zip

The extracted file should be placed in the project root as:

20170107-061401-recipeitems.json

### 5. Build the recipe embeddings

The recipe embeddings are generated once from the static dataset and stored locally in a compressed NumPy file.

```powershell
python build_embeddings_npz.py
```

This creates:

recipe_embeddings.npz

The full embedding build uses the OpenAI API and may take some time and incur API usage costs.

### 6. Start the API

```powershell
python -m uvicorn main:app --reload
```

The API will run at:

http://127.0.0.1:8000

Interactive FastAPI documentation and endpoint testing are available at:

http://127.0.0.1:8000/docs

## API Usage

The API exposes one search endpoint:

```text
POST /recipes/search
```

The request body contains a natural-language recipe query:

```json
{
  "query": "Jag vill ha något festligt med kyckling utan lök som tar max en timme"
}
```

Example response:

```json
{
  "query": "Jag vill ha något festligt med kyckling utan lök som tar max en timme",
  "interpreted_query": {
    "include_ingredients": ["chicken"],
    "exclude_ingredients": ["onion"],
    "max_time_minutes": 60,
    "semantic_preference": "festive"
  },
  "number_of_matches": 3035,
  "results": [
    {
      "recipe": {
        "name": "Autumn Chicken",
        "ingredients": "...",
        "url": "...",
        "cookTime": "PT30M",
        "prepTime": "PT10M",
        "source": "allrecipes",
        "recipeYield": "4 servings",
        "description": "..."
      },
      "similarity": 0.6056
    }
  ]
}
```

The API returns the three highest-ranked matching recipes. `number_of_matches` is the number of recipes that passed the hard filters before semantic ranking. `similarity` is the cosine similarity score used for ranking. It is null when the request contains neither included ingredients nor a semantic preference.

# AI Usage and Architecture

The solution separates hard constraints from semantic preferences. AI is used to interpret natural language and to rank recipes by semantic relevance, while constraints that can be evaluated exactly are handled with deterministic Python logic.

The search flow is:

```text
Natural-language query
        ↓
LLM query interpretation
        ↓
Deterministic filtering
        ↓
Embedding-based semantic ranking
        ↓
Top 3 recipes
```

### Natural-language interpretation

`gpt-5-mini` is used to convert the user's free-text request into a structured representation containing:

- ingredients to include
- ingredients to exclude
- maximum cooking time
- semantic preferences such as "spicy", "comforting", or "festive"

For example:

```text
"Jag vill ha något festligt med kyckling utan lök som tar max en timme"
```

is interpreted as:

```json
{
  "include_ingredients": ["chicken"],
  "exclude_ingredients": ["onion"],
  "max_time_minutes": 60,
  "semantic_preference": "festive"
}
```

The LLM also translates extracted ingredients and preferences to English. This allows the API to accept queries in multiple languages while searching an English-language recipe dataset.

### Deterministic filtering

Explicit requirements are handled in Python rather than by an LLM. Recipes are filtered using required ingredients, excluded ingredients, and maximum time before semantic ranking takes place.

This makes hard constraints deterministic: for example, a recipe containing an explicitly excluded ingredient is removed rather than merely receiving a lower semantic score.

### Semantic ranking with embeddings

After hard filtering, the remaining recipes are ranked using embeddings from `text-embedding-3-small` and cosine similarity.

Recipe embeddings are generated from the recipe name and description. If a description is unavailable, the ingredient list is used as a fallback. This representation was chosen because the name and description usually provide useful information about the character of a dish, while explicit ingredient requirements are already handled by the filtering step.

The ranking query combines included ingredients with any semantic preference. For example, a request for festive chicken produces a ranking query containing both "chicken" and "festive". This helps distinguish recipes where the requested ingredient is central to the dish while also ranking according to softer preferences.

Recipe embeddings are precomputed once by `build_embeddings_npz.py` and stored locally. At request time, only the search query needs a new embedding. Cosine similarity is then calculated locally against the embeddings of recipes that passed the hard filters.