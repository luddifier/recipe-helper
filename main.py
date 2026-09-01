from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import json
import re
import numpy as np

client = OpenAI()
app = FastAPI(
    title="Recipe Helper"
)

recipes = []

QUERY_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

with open("20170107-061401-recipeitems.json", "r", encoding="utf-8") as file:
    for line in file:
        recipes.append(json.loads(line))

embedding_data = np.load("recipe_embeddings.npz")

recipe_ids = embedding_data["recipe_ids"]
recipe_embedding_matrix = embedding_data["embeddings"]

recipe_id_to_index = {
    recipe_id: index
    for index, recipe_id in enumerate(recipe_ids)
}

class SearchRequest(BaseModel):
    query: str


class QueryInterpretation(BaseModel):
    include_ingredients: list[str]
    exclude_ingredients: list[str]
    max_time_minutes: int | None
    semantic_preference: str | None


def interpret_query(query: str):
    response = client.responses.parse(
        model=QUERY_MODEL,
        input=[
            {
                "role": "system",
                "content": """
                Interpret a user's recipe request.

                Separate explicit requirements from softer semantic preferences.

                - include_ingredients: ingredients the user explicitly wants
                - exclude_ingredients: ingredients the user explicitly does not want
                - max_time_minutes: explicit maximum total time, otherwise null
                - semantic_preference: softer preferences such as spicy,
                  comforting, fresh, festive, etc. Otherwise null.

                Translate ingredient names and semantic preferences to English.
                Return ingredient names in their common singular form when possible
                (e.g. "mushroom", not "mushrooms").
                """
            },
            {
                "role": "user",
                "content": query
            }
        ],
        text_format=QueryInterpretation
    )

    return response.output_parsed


def parse_time(time_string: str | None) -> int | None:
    if not time_string:
        return None

    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?",
        time_string
    )

    if not match:
        return None

    # "PT" contains no actual time information.
    if match.group(1) is None and match.group(2) is None:
        return None

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0

    return hours * 60 + minutes


def filter_recipes(recipe_list, interpreted_query):
    filtered_recipes = []

    for recipe in recipe_list:
        ingredients = recipe.get("ingredients", "").lower()

        matches = True

        # Required ingredients
        for ingredient in interpreted_query.include_ingredients:
            if ingredient.lower() not in ingredients:
                matches = False
                break

        if not matches:
            continue

        # Excluded ingredients
        for ingredient in interpreted_query.exclude_ingredients:
            if ingredient.lower() in ingredients:
                matches = False
                break

        if not matches:
            continue

        # Maximum total time
        if interpreted_query.max_time_minutes is not None:
            if recipe.get("totalTime"):
                recipe_time = parse_time(recipe.get("totalTime"))

                if recipe_time is None:
                    matches = False
                elif recipe_time > interpreted_query.max_time_minutes:
                    matches = False

            elif recipe.get("prepTime") and recipe.get("cookTime"):
                prep_time = parse_time(recipe.get("prepTime"))
                cook_time = parse_time(recipe.get("cookTime"))

                if prep_time is None or cook_time is None:
                    matches = False
                else:
                    recipe_time = prep_time + cook_time

                    if recipe_time > interpreted_query.max_time_minutes:
                        matches = False

            else:
                matches = False

        if matches:
            filtered_recipes.append(recipe)

    return filtered_recipes


def rank_recipes(interpreted_query, filtered_recipes):
    # Build the text that will be used for semantic ranking.
    # Required ingredients are useful ranking signals even when
    # the user has no softer semantic preference.
    ranking_parts = []

    ranking_parts.extend(
        interpreted_query.include_ingredients
    )

    if interpreted_query.semantic_preference:
        ranking_parts.append(
            interpreted_query.semantic_preference
        )

    ranking_query = " ".join(ranking_parts)

    # If the query only contains hard constraints such as
    # excluded ingredients or maximum time, there is nothing
    # meaningful to rank semantically.
    if not ranking_query:
        return [
            {
                "recipe": format_recipe_result(recipe),
                "similarity": None
            }
            for recipe in filtered_recipes
        ]

    # Create one embedding for the ranking query.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=ranking_query
    )

    query_embedding = np.array(
        response.data[0].embedding,
        dtype=np.float32
    )

    # Collect embeddings for recipes that passed the hard filters.
    candidate_recipes = []
    candidate_embeddings = []

    for recipe in filtered_recipes:
        recipe_id = recipe["_id"]["$oid"]

        index = recipe_id_to_index.get(recipe_id)

        if index is None:
            continue

        candidate_recipes.append(recipe)
        candidate_embeddings.append(
            recipe_embedding_matrix[index]
        )

    if not candidate_recipes:
        return []

    candidate_embeddings = np.array(
        candidate_embeddings,
        dtype=np.float32
    )

    # Calculate cosine similarity for all candidates at once.
    dot_products = candidate_embeddings @ query_embedding

    recipe_norms = np.linalg.norm(
        candidate_embeddings,
        axis=1
    )

    query_norm = np.linalg.norm(query_embedding)

    similarities = dot_products / (
        recipe_norms * query_norm
    )

    # Highest similarity first.
    ranked_indexes = np.argsort(similarities)[::-1]

    ranked_recipes = []

    for candidate_index in ranked_indexes:
        ranked_recipes.append({
            "recipe": format_recipe_result(
                candidate_recipes[candidate_index]
            ),
            "similarity": float(similarities[candidate_index])
        })

    return ranked_recipes


def format_recipe_result(recipe):
    return {
        "name": recipe.get("name"),
        "ingredients": recipe.get("ingredients"),
        "url": recipe.get("url"),
        "image": recipe.get("image"),
        "cookTime": recipe.get("cookTime"),
        "prepTime": recipe.get("prepTime"),
        "source": recipe.get("source"),
        "recipeYield": recipe.get("recipeYield"),
        "description": recipe.get("description")
    }


@app.post("/recipes/search")
def search_recipes(request: SearchRequest):
    interpreted_query = interpret_query(request.query)

    filtered_recipes = filter_recipes(recipes, interpreted_query)

    ranked_recipes = rank_recipes(interpreted_query, filtered_recipes)

    return {
        "query": request.query,
        "interpreted_query": interpreted_query,
        "number_of_matches": len(filtered_recipes),
        "results": ranked_recipes[:3]
    }