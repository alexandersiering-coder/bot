"""Vegane Rezeptvorschläge über die Spoonacular-API (food-api.spoonacular.com).

Ohne RECIPE_API_KEY bleibt das Feature aus, genau wie Bring/Reminders ohne
ihre jeweiligen Env-Vars.
"""

import os
import random

import httpx

RECIPE_API_KEY = os.getenv("RECIPE_API_KEY")
RECIPES_ENABLED = bool(RECIPE_API_KEY)

_BASE_URL = "https://api.spoonacular.com"


async def fetch_vegan_recipe(exclude_ids: set[str]) -> dict | None:
    """Holt ein gesundes, veganes Rezept. `exclude_ids` (zuletzt verschickte
    Rezept-IDs als Strings) wird nach Möglichkeit vermieden, für Abwechslung
    bei wiederkehrenden Erinnerungen."""
    async with httpx.AsyncClient(timeout=15) as client:
        search = await client.get(
            f"{_BASE_URL}/recipes/complexSearch",
            params={
                "apiKey": RECIPE_API_KEY,
                "diet": "vegan",
                "sort": "healthiness",
                "number": 10,
                "instructionsRequired": "true",
            },
        )
        search.raise_for_status()
        results = search.json().get("results", [])
        if not results:
            return None
        candidates = [r for r in results if str(r["id"]) not in exclude_ids] or results
        recipe_id = random.choice(candidates)["id"]

        info = await client.get(
            f"{_BASE_URL}/recipes/{recipe_id}/information",
            params={"apiKey": RECIPE_API_KEY, "includeNutrition": "false"},
        )
        info.raise_for_status()
        return info.json()


def format_recipe(data: dict) -> str:
    title = data.get("title", "Rezept")
    ready = data.get("readyInMinutes")
    servings = data.get("servings")
    url = data.get("sourceUrl") or data.get("spoonacularSourceUrl") or ""
    ingredients = data.get("extendedIngredients") or []

    parts = [f"🥗 {title}"]
    meta = []
    if ready:
        meta.append(f"⏱ {ready} Min.")
    if servings:
        meta.append(f"🍽 {servings} Portionen")
    if meta:
        parts.append(" · ".join(meta))
    if ingredients:
        ing_lines = "\n".join(f"• {i.get('original') or i.get('name', '')}" for i in ingredients)
        parts.append(f"Zutaten:\n{ing_lines}")
    if url:
        parts.append(f"🔗 {url}")
    return "\n\n".join(parts)
