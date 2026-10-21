"""Elite check: can you trust the LLM judge? Compare its pairwise verdicts with YOUR labels (Cohen's kappa).

1) Open outputs/reports/human_labels.csv (created on first run) and fill the 'human' column
   with the model name you prefer, or 'tie', for each row (read the two answers in the preds files).
2) Run again: python scripts/35_judge_agreement.py --judge-file outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from common import OUT

p = argparse.ArgumentParser()
p.add_argument("--judge-file", required=True)
p.add_argument("--n", type=int, default=30, help="how many items you will label by hand")
a = p.parse_args()

judge = json.loads(Path(a.judge_file).read_text())["per_item"][: a.n]
labels = OUT / "reports" / "human_labels.csv"
if not labels.exists():                                     # first run: make the sheet for you to fill
    with labels.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "human"])
        w.writerows([[i, ""] for i in range(len(judge))])
    print(f"fill the 'human' column in {labels}, then run this script again")
    raise SystemExit

human = {int(r["i"]): r["human"].strip() for r in csv.DictReader(labels.open()) if r["human"].strip()}
pairs = [(judge[i], h) for i, h in human.items() if i < len(judge)]
n = len(pairs)
observed = sum(j == h for j, h in pairs) / n                 # raw agreement
jc, hc = Counter(j for j, _ in pairs), Counter(h for _, h in pairs)
expected = sum(jc[k] * hc[k] for k in set(jc) | set(hc)) / (n * n)   # agreement by pure chance
kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
print(f"items={n}  agreement={observed:.2f}  Cohen's kappa={kappa:.2f}")
print("kappa > 0.6 = judge is usable; 0.4-0.6 = use with care; < 0.4 = do not trust this judge")
