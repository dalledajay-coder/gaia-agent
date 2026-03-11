"""Mathematical computation tools."""

from claude_agent_sdk import tool
from typing import Any


@tool(
    "calculate",
    "Evaluate a mathematical expression or perform calculations. Supports Python math expressions including math module functions.",
    {"expression": str},
)
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    expression = args["expression"]

    try:
        import math
        import statistics

        # Safe evaluation with math functions available
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "sorted": sorted,
            "int": int, "float": float, "str": str,
            "range": range, "list": list, "tuple": tuple,
            "mean": statistics.mean, "median": statistics.median,
            "stdev": statistics.stdev,
        })

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"content": [{"type": "text", "text": f"Result: {result}"}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Calculation error: {str(e)}\nExpression: {expression}"}]}
