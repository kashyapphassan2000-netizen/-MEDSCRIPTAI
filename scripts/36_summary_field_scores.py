"""Elite check: criterion-level scoring for the summary task (rubric idea from HealthBench, made verifiable).

Every field path in the REFERENCE JSON is one criterion, e.g. "patient information.age".
recall    = reference fields the model filled       (missing facts lower it)
precision = model fields that exist in the reference (invented fields lower it = hallucination signal)
"""
import argparse
import json
from common import OUT, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["base", "summary"])
a = p.parse_args()


def paths(value, prefix=""):
    """{"a": {"b": 1}, "c": [{"d": 2}]} -> {"a.b", "c.d"} (list items are merged)."""
    out = set()
    if isinstance(value, dict):
        for k, v in value.items():
            out |= paths(v, f"{prefix}{k.lower().strip()}.")
    elif isinstance(value, list):
        for v in value:
            out |= paths(v, prefix)
    elif value not in (None, "", "None"):
        out.add(prefix.rstrip("."))
    return out


def load(text):
    try:
        return json.loads((text or "").strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        return None


for model in a.models:
    rows = read_jsonl(OUT / "preds" / "summary" / f"{model}.jsonl")
    tp = fp = fn = invalid = 0
    for r in rows:
        ref, pred = paths(load(r["reference"]) or {}), load(r["prediction"])
        if pred is None:
            invalid += 1
            fn += len(ref)                     # an unreadable answer misses every criterion
            continue
        got = paths(pred)
        tp += len(ref & got)
        fp += len(got - ref)
        fn += len(ref - got)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"{model:10} field precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} invalid_json={invalid}/{len(rows)}")
