from openai import OpenAI
import json
import numpy as np

client = OpenAI()

SOURCE_FILE = "20170107-061401-recipeitems.json"
OUTPUT_FILE = "recipe_embeddings.npz"

EMBEDDING_MODEL = "text-embedding-3-small"

BATCH_SIZE = 500

# Set to an integer such as 1000 when testing.
# Use None for the full dataset.
MAX_RECIPES = None


def load_recipes():
    loaded_recipes = []

    with open(SOURCE_FILE, "r", encoding="utf-8") as file:
        for line in file:
            loaded_recipes.append(json.loads(line))

            if MAX_RECIPES is not None and len(loaded_recipes) >= MAX_RECIPES:
                break

    return loaded_recipes


def recipe_to_text(recipe):
    name = recipe.get("name", "")
    description = recipe.get("description", "")
    ingredients = recipe.get("ingredients", "")

    if description:
        return f"{name}. {description}"

    return f"{name}. {ingredients}"


def build_embeddings(recipes):
    recipe_ids = []
    embedding_batches = []

    for batch_start in range(0, len(recipes), BATCH_SIZE):
        batch = recipes[batch_start:batch_start + BATCH_SIZE]

        texts = [
            recipe_to_text(recipe)
            for recipe in batch
        ]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        batch_embeddings = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32
        )

        batch_ids = [
            recipe["_id"]["$oid"]
            for recipe in batch
        ]

        recipe_ids.extend(batch_ids)
        embedding_batches.append(batch_embeddings)

        print(
            f"Embedded {min(batch_start + BATCH_SIZE, len(recipes))} "
            f"/ {len(recipes)} recipes"
        )

    embeddings = np.vstack(embedding_batches)

    recipe_ids = np.array(recipe_ids)

    return recipe_ids, embeddings


def save_embeddings(recipe_ids, embeddings):
    np.savez_compressed(
        OUTPUT_FILE,
        recipe_ids=recipe_ids,
        embeddings=embeddings
    )


def main():
    print("Loading recipes...")

    recipes = load_recipes()

    print(f"Loaded {len(recipes)} recipes")

    print("Building embeddings...")

    recipe_ids, embeddings = build_embeddings(recipes)

    print("Embedding matrix shape:", embeddings.shape)
    print("Embedding data type:", embeddings.dtype)

    print("Saving compressed NumPy file...")

    save_embeddings(
        recipe_ids,
        embeddings
    )

    print(
        f"Done! Saved embeddings for {len(recipe_ids)} recipes "
        f"to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()