# tools.py
from __future__ import annotations

import math
import ast
import html
from typing import Any, Dict

import numpy as np


_ALLOWED_NAMES = {
    # math
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pow": pow,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,

    # handy builtins for expressions (your schema example already used range)
    "range": range,
    "enumerate": enumerate,
    "len": len,
    "round": round,
    "float": float,
    "int": int,

    # numpy basics
    "np": np,
}


def safe_calc(expr: str) -> Any:
    # 1) unescape HTML (so &#10; becomes real newlines)
    expr = html.unescape(expr or "").strip()

    # 2) enforce "single expression" only (no def/for/assignments)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(
            "calc expects a SINGLE Python expression (no def/for/assignments). "
            "Use list comprehensions / numpy expressions."
        ) from e

    code = compile(tree, "<calc>", "eval")
    return eval(code, {"__builtins__": {}}, dict(_ALLOWED_NAMES))


def tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "calc",
            "description": (
                "Calculator: evaluate a SINGLE Python expression (math/numpy). "
                "No statements/defs/loops. Example: 'sum(300/(1.12)**t for t in range(1,6))'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Single Python expression."}
                },
                "required": ["expr"],
            },
        },
    }


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name != "calc":
        return {"error": f"Unknown tool: {name}"}

    expr = args.get("expr", "")
    try:
        value = safe_calc(expr)

        # Make numpy scalars JSON-able
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        return {"value": value}
    except Exception as e:
        # IMPORTANT: never crash the whole run because the model sent bad tool code
        return {"error": f"{type(e).__name__}: {e}"}
