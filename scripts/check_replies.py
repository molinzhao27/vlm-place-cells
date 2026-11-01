"""Fine-tuned Qwen answer files: every reply must be a bare number of at most 4 tokens (Exp 30 amendment 1).

A vast-3 fault gives "!" (token 0) and long replies, and the job still ends with rc 0, so its files are checked as it ends.
Standard library only, so it runs on vast-3's host. Exit 1 if any file is damaged, 2 if nothing matched.
    python3 scripts/check_replies.py 'results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-e4-s5-room*.jsonl'
"""
import glob
import json
import re
import sys

n_files = n_rows = 0
damaged = []
for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern)):
        rows = [json.loads(line) for line in open(path)]
        bad = sum(re.fullmatch(r"\s*\d{1,3}\s*", r["text"]) is None or r["gen_tokens"] > 4 for r in rows)
        n_files += 1
        n_rows += len(rows)
        if bad:
            damaged.append(f"{bad}/{len(rows)} {path}")
print(f"{n_files} files, {n_rows} replies, {len(damaged)} damaged")
print("\n".join(damaged))
sys.exit(1 if damaged else 0 if n_files else 2)
