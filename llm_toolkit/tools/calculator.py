import ast
import operator

from ..tool_types import Tool


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](
            _eval(node.left),
            _eval(node.right),
        )

    raise ValueError("Unsupported expression")


def calculator(expression: str):
    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression. Use this for exact calculations.",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A basic arithmetic expression, for example '365 * 263'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    run=calculator,
)
