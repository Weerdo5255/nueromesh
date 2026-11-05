"""Calculator tool: safe arithmetic (AST-evaluated, no eval())."""
import ast
import json
import operator
import re
import sys

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
WORDS = [(r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"), (r"\btimes\b", "*"), (r"\bplus\b", "+"),
         (r"\bminus\b", "-"), (r"\bover\b", "/"), (r"\bsquared\b", "**2"), (r"(?<=\d)\s*x\s*(?=\d)", "*")]


def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("unsupported expression")


def extract(text):
    t = text.lower()
    for pattern, repl in WORDS:
        t = re.sub(pattern, repl, t)
    found = [c.strip() for c in re.findall(r"[\d.\s+\-*/()]+", t) if re.search(r"\d", c)]
    if not found:
        raise ValueError("no arithmetic expression found")
    return max(found, key=len)


def main():
    request = json.load(sys.stdin)
    expr = extract(request["input"])
    value = evaluate(ast.parse(expr, mode="eval"))
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 6)
    print(json.dumps({"output": value, "expression": expr, "say": f"{' '.join(expr.split())} = {value}"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
