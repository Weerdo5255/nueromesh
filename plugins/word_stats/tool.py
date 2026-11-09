"""Word statistics tool."""
import json
import re
import sys
import traceback


def target_text(text):
    quoted = re.search(r"[\"“](.+?)[\"”]", text, re.S)
    if quoted:
        return quoted.group(1)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def main():
    text = target_text(json.load(sys.stdin)["input"])
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = max(1, len(re.findall(r"[.!?]+", text))) if text.strip() else 0
    w, c = len(words), len(text)
    say = (f"That text has {w} word{'s' if w != 1 else ''}, {c} character{'s' if c != 1 else ''} and "
           f"{sentences} sentence{'s' if sentences != 1 else ''}.")
    print(json.dumps({"output": f"{w} words, {c} characters, {sentences} sentence(s)", "say": say}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a fault in this code: the trace lets the mesh find it (tooldoctor.py)
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}))
