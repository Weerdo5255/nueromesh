"""Glossary lookup tool (reads glossary.json from its own folder)."""
import json
import re
import sys
from pathlib import Path


def main():
    glossary = json.loads((Path(__file__).parent / "glossary.json").read_text())
    words = re.findall(r"[a-z]+", json.load(sys.stdin)["input"].lower())
    for word in words:
        if word in glossary:
            print(json.dumps({"output": f"{word}: {glossary[word]}"}))
            return
    raise ValueError("word not in glossary")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
