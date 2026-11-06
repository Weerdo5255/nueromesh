# Challenge suites

Each folder here is a challenge suite: questions with gold answers that NeuroMesh is tested
against and learns from (see "Challenges" in the main README). The `sample-*` suites are small,
written for NeuroMesh in the style of well-known open benchmarks. Download the real ones with
`/challenge fetch <adapter> [rows] [split]`; they go to `state/challenges/`.

    manifest.toml   name, kind (classify | numeric | choice | yesno | short), description,
                    capability + labels (+ close pairs) for classify suites, retain (short),
                    exam_percent, source, license
    items.jsonl     one item per line: {"id", "input", "answer", "choices"?, "context"?, "split"?}

answer: a label (classify), a number (numeric), an option letter with "choices": [["A", "..."], ...]
(choice), true/false (yesno), or a list of acceptable strings (short). "split": "practice" or
"exam" pins an item; otherwise `exam_percent` of the items are held out as the exam, chosen by a
stable hash of the item id.
