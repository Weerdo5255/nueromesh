"""Calculator tool: arithmetic, equations, calculus, number theory, statistics and matrices
(mathsolver.py: pure Python, no eval()). When an order names a quantity it does not know ("the speed
of light times 3"), it asks a director ({"query": ...}) and is run again with the answer as context.
Input is checked first (mathsolver.tidy): braces, LaTeX, labels and ratios with ':', other ways of writing
symbols, unbalanced brackets. An order it can't read is answered with why ("bad_input"), which is not
held against the tool."""
import json
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathsolver import MathError, NeedToKnow, answer, evaluate, Context, out  # noqa: E402


def legacy(text):
    """The first calculator's way: the longest run of digits and operators (it must contain an operator)."""
    t = text.lower()
    for pattern, repl in [(r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"), (r"\btimes\b", "*"), (r"\bplus\b", "+"),
                          (r"\bminus\b", "-"), (r"\bover\b", "/"), (r"(?<=\d)\s*x\s*(?=\d)", "*")]:
        t = re.sub(pattern, repl, t)
    found = [c.strip() for c in re.findall(r"[\d.\s+\-*/()^]+", t) if re.search(r"\d\s*[-+*/^]\s*[\d(]", c)]
    if not found:
        return None
    return evaluate(max(found, key=len), Context(), text)


def main():
    request = json.load(sys.stdin)
    text = request["input"]
    try:
        result = answer(text, request.get("context") or [])
    except NeedToKnow as need:
        print(json.dumps({"query": {"ask": need.ask, "about": need.phrase,
                                    "why": f"to calculate with {need.phrase}"}}))
        return
    except MathError as exc:
        try:
            result = legacy(text)
        except MathError:
            result = None
        if result is None:  # the order, not this code: said plainly, and not held against the tool
            print(json.dumps({"error": str(exc), "not_found": True, "bad_input": True}))
            return
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a fault in this code: the trace lets the mesh find it (tooldoctor.py)
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}))
