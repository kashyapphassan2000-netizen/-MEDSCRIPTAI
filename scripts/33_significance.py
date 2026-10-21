"""Elite check: is model B REALLY better than model A, or is it noise? Paired bootstrap on QA accuracy.

python scripts/33_significance.py --a qa --b qa-dpo
"""
import argparse
import random
from common import OUT, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--rounds", type=int, default=10000)
args = p.parse_args()


def correct(row):
    pred = row["prediction"] or ""
    letter = pred.split("Answer:", 1)[1].strip()[:1].upper() if "Answer:" in pred else None
    return int(letter == row["answer_letter"])


a = [correct(r) for r in read_jsonl(OUT / "preds" / "qa" / f"{args.a}.jsonl")]
b = [correct(r) for r in read_jsonl(OUT / "preds" / "qa" / f"{args.b}.jsonl")]
n = min(len(a), len(b))
diffs = []
rng = random.Random(0)
for _ in range(args.rounds):
    idx = [rng.randrange(n) for _ in range(n)]          # resample the SAME questions for both models
    diffs.append(sum(b[i] - a[i] for i in idx) / n)
diffs.sort()
low, high = diffs[int(0.025 * args.rounds)], diffs[int(0.975 * args.rounds)]
observed = (sum(b[:n]) - sum(a[:n])) / n
print(f"accuracy {args.a}={sum(a[:n]) / n:.3f}  {args.b}={sum(b[:n]) / n:.3f}  diff={observed:+.3f}")
print(f"95% CI of the difference: [{low:+.3f}, {high:+.3f}]")
print("SIGNIFICANT" if low > 0 or high < 0 else "NOT significant: the CI contains 0, do not claim an improvement")
