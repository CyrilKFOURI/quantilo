"""A curve of x typed by the user (075, R-REG-PHY-07): `a * exp(-b * x) + c`, read by Python's
own parser into a tree that only numbers, `x`, single-letter parameters, + − × ÷ and powers, and
a few functions may enter; nothing is evaluated as code. Pyodide-compatible."""

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

MAX_LENGTH = 200
MAX_PARAMETERS = 6
FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exp": lambda v: np.exp(np.clip(v, -700, 700)),
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "atan": np.arctan,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
PARAMETER = re.compile(r"^[a-df-wyz]$|^[a-z][0-9]$")  # one letter (not e, x) or k1, b2…


class FormulaError(ValueError):
    pass


@dataclass
class Formula:
    text: str
    names: list[str]  # the parameters, in order of appearance
    function: Callable[[np.ndarray, np.ndarray], np.ndarray]


def parse(text: str) -> Formula:
    """R-REG-PHY-07: the formula, its parameters, and a function (x, p) → y."""
    source = text.strip().replace("^", "**").replace("·", "*").replace("−", "-")
    if not source:
        raise FormulaError("Type a formula of x, such as a * exp(b * x) + c.")
    if len(source) > MAX_LENGTH:
        raise FormulaError(f"The formula is longer than {MAX_LENGTH} characters.")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise FormulaError(f"The formula cannot be read: {error.msg}.") from error
    names: list[str] = []
    seen_x = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id == "x":
                seen_x = True
            elif node.id in FUNCTIONS or node.id in CONSTANTS:
                continue
            elif PARAMETER.match(node.id):
                if node.id not in names:
                    names.append(node.id)
            else:
                raise FormulaError(
                    f"`{node.id}` is not x, a parameter (one letter, such as a, b, k) or a "
                    f"function ({', '.join(sorted(FUNCTIONS))})."
                )
    if not seen_x:
        raise FormulaError("The formula must use x.")
    names.sort(key=lambda name: _first(tree, name))  # in the order they are written
    if not names:
        raise FormulaError("The formula needs at least one parameter to fit, such as a or b.")
    if len(names) > MAX_PARAMETERS:
        raise FormulaError(f"At most {MAX_PARAMETERS} parameters.")
    _check(tree.body)
    body = tree.body
    order = {name: i for i, name in enumerate(names)}

    def function(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = _evaluate(body, x, p, order)
        return np.broadcast_to(np.asarray(value, dtype=float), x.shape).astype(float)

    return Formula(text.strip(), names, function)


def _first(tree: ast.AST, name: str) -> tuple[int, int]:
    return min(
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name
    )


def _check(node: ast.AST) -> None:
    allowed_binary = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, allowed_binary):
            raise FormulaError("Only + − * / and ^ (or **) are allowed.")
        _check(node.left)
        _check(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub | ast.UAdd):
            raise FormulaError("Only a minus sign may stand before a term.")
        _check(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise FormulaError(f"Allowed functions: {', '.join(sorted(FUNCTIONS))}.")
        if len(node.args) != 1 or node.keywords:
            raise FormulaError(f"`{node.func.id}` takes one argument.")
        _check(node.args[0])
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float) or isinstance(node.value, bool):
            raise FormulaError("Only numbers may appear as constants.")
    elif not isinstance(node, ast.Name):
        raise FormulaError("The formula may hold numbers, x, parameters, + − * / ^ and functions.")


def _evaluate(node: ast.AST, x: np.ndarray, p: np.ndarray, order: dict[str, int]) -> np.ndarray:
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, x, p, order)
        right = _evaluate(node.right, x, p, order)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return np.power(left, right)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, x, p, order)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        return FUNCTIONS[node.func.id](_evaluate(node.args[0], x, p, order))
    if isinstance(node, ast.Constant):
        return np.asarray(float(cast(float, node.value)))
    assert isinstance(node, ast.Name)
    if node.id == "x":
        return x
    if node.id in CONSTANTS:
        return np.asarray(CONSTANTS[node.id])
    return np.asarray(p[order[node.id]])


def starts(count: int, seed: int = 42) -> list[np.ndarray]:
    """R-REG-PHY-07: no written start for a typed formula: 1, 0.1, −1, then 5 seeded draws."""
    rng = np.random.default_rng(seed)
    found = [np.full(count, 1.0), np.full(count, 0.1), np.full(count, -1.0)]
    found += [rng.normal(0, 1, count) for _ in range(5)]
    return found
