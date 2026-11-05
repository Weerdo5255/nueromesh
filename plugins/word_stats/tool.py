"""Word statistics tool."""
import json
import re
import sys


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
    print(json.dumps({"output": f"{len(words)} words, {len(text)} characters, {sentences} sentence(s)"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
