"""Detect PixMo-style counting questions that should not use pointing."""

from __future__ import annotations

import re

__all__ = ["is_pixmo_point_and_count_question"]

non_countable_quantities = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "[a-z]*seconds",
    "(tera|giga|mega|deci|kilo||micro|centi|milli|nano|pico|deca)meters",
    "meters",
    "metres",
    "acres",
    "leagues",
    "fathoms",
    "nautical miles",
    "hectares",
    "(square |SQ )?inches",
    "(square |SQ )feet",
    "(square |SQ )?ft",
    "(square |SQ )?miles",
    "(square | SQ)?yards",
    "passing yards",
    "dollars",
    "cents",
    "pounds",
    "euros",
    "seed",
    "mph",
    "kph",
    "more",
    "fewer",
    "less",
    "likes",
    "cubic",
    "gallons",
    "quarts",
    "pints",
    "fluid ounces",
    "[a-z]*liters",
    "weight",
    "[a-z]*grams",
    "pounds",
    "tons",
    "ounces",
    "ways",
    "different ways",
    "degrees",
    "calories",
    "hertz",
    "horsepower",
    "[a-z]*bytes",
    "psi",
    "atmospheres",
    "[a-z]*watts",
]
non_countable_re_str = "|".join(non_countable_quantities)
non_countable_end_re_str = "|".join(non_countable_quantities + ["money", "the"])

counting_patterns = [
    f"how ?many (?!{non_countable_re_str})",
    r"(?<!do not )(count|tally) (all|every|each|the) ",
    "(there are|a total of) _{3,4}",
    f"(what|(what's|what (is|was|were)|states?|indicates?) the( exact| precise)?) (total count|count|total|total number|number|num|total amount|amount) of (?!{non_countable_end_re_str})",
]
count_any = re.compile(
    "^(?!approximately).*(\\b|^|\n)(?P<all>" + "|".join(counting_patterns) + ")\\b.*",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def is_pixmo_point_and_count_question(question: str, answer: str = "") -> bool:
    """Return True if the question is a counting question that should use pointing."""
    del answer
    return bool(count_any.fullmatch(question))
