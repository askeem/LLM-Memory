# tools.py
from __future__ import annotations

import math
import ast
import html
from typing import Any, Dict
import numpy as np


_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
    "pow": pow, "abs": abs, "min": min, "max": max, "sum": sum,
    "range": range, "enumerate": enumerate, "len": len, "round": round,
    "float": float, "int": int,
    "np": np,
}


def _to_jsonable(x: Any) -> Any:
    # primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # complex numbers (common from np.roots)
    if isinstance(x, complex):
        # if it's effectively real, drop the imaginary part
        if abs(x.imag) < 1e-12:
            return float(x.real)
        # otherwise encode as an object (or return repr(x) if you prefer)
        return {"re": float(x.real), "im": float(x.imag)}

    # numpy scalars (may become complex after .item())
    if isinstance(x, np.generic):
        return _to_jsonable(x.item())

    # numpy arrays (tolist may contain complex -> recurse)
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())

    # range
    if isinstance(x, range):
        return list(x)

    # containers
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # try numeric cast
    try:
        return float(x)
    except Exception:
        return repr(x)


def safe_calc(expr: str) -> Any:
    expr = html.unescape(expr or "").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(
            "calc expects a SINGLE Python expression (no def/for/assignments)."
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
                "Must return a JSON-serializable result (number or list)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
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
        return {"value": _to_jsonable(value)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
