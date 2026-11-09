"""Recipes (030): the work kept on a dataset, replayed on another file of the same shape.

recipe = dx.Recipe.from_dataset(ds, "Monthly sales", kpis={"Revenue": {...}})
report = recipe.run(dx.open("november.csv"))
recipe.to_json(); recipe.to_python()
"""

from quantilo.recipes.model import Recipe, RecipeError, parse

__all__ = ["Recipe", "RecipeError", "parse"]
