"""Rewrite answer and correct in results/claude and results/api with the stated-answer rule (spatialcells/answers.py;
docs/experiments.md, "Scoring correction for API models"). The stored value stays in answer_first, set only the first
time, so a rerun changes nothing. No API call: the replies are already stored.

    .venv-claude/bin/python scripts/rescore_api.py [--dry-run]

Numbers in a relabeled run's reply are labels: they are read against labels_shown and mapped back to the frame numbers.
api_test.py stored only the last 600 characters, so a reply of that length or more has no first line (tail_only).
Records without the fields the rule needs are left as they are and reported.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialcells.answers import stated_answer

API_TAIL = 600
NEEDED = ("a", "b", "near", "text", "answer", "correct")


def stated(r, api):
    """The option this record's reply states, in frame numbers, else -1."""
    a, b, text = r["a"], r["b"], r["text"]
    tail_only = api and len(text or "") >= API_TAIL
    if "labels_shown" in r:
        la, lb = r["labels_shown"]
        return {la: a, lb: b}.get(stated_answer(text, la, lb, tail_only), -1)
    return stated_answer(text, a, b, tail_only)


def rescore(path, write=True):
    """Rescore one file, in record order; returns its counts."""
    api = path.startswith("results/api/")
    n = dict(records=0, changed=0, to_right=0, to_wrong=0, now_unparsed=0, newly_parsed=0, skipped=0)
    old_lines = open(path, encoding="utf-8").read().split("\n")[:-1]
    new_lines = []
    for line in old_lines:
        r = json.loads(line)
        n["records"] += 1
        if any(k not in r for k in NEEDED) or not isinstance(r["a"], int) or not isinstance(r["b"], int):
            n["skipped"] += 1
            new_lines.append(line)
            continue
        old_answer, old_correct = r["answer"], r["correct"]
        answer = stated(r, api)
        correct = answer == r["near"]
        n["changed"] += answer != old_answer
        n["to_right"] += correct and not old_correct
        n["to_wrong"] += old_correct and not correct
        n["now_unparsed"] += answer == -1 and old_answer != -1
        n["newly_parsed"] += answer != -1 and old_answer == -1
        r.setdefault("answer_first", old_answer)
        r["answer"], r["correct"] = answer, correct
        new_lines.append(json.dumps(r))
    if write and new_lines != old_lines:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                       # never a half-written reply file
    return n


def main():
    write = "--dry-run" not in sys.argv
    total = {}
    for path in sorted(glob.glob("results/claude/*.jsonl") + glob.glob("results/api/*.jsonl")):
        n = rescore(path, write)
        print(f"{path}: " + " ".join(f"{k} {v}" for k, v in n.items()), flush=True)
        for k, v in n.items():
            total[k] = total.get(k, 0) + v
    print("total: " + " ".join(f"{k} {v}" for k, v in total.items()) + ("  (dry run, nothing written)" if not write else ""))


if __name__ == "__main__":
    main()
