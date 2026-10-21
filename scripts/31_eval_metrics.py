"""Evaluation step B: automatic metrics for every saved prediction file.

qa      -> accuracy (letter match), ROUGE-L, BERTScore
notes   -> ROUGE-L, BERTScore
summary -> JSON-valid rate, ROUGE-L, BERTScore
"""
import argparse
import json
import os
from pathlib import Path
from rouge_score import rouge_scorer                 # Google's reference ROUGE implementation
from bert_score import score as bert_score           # BERTScore (Zhang et al., 2020)
from common import OUT, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--tasks", nargs="+", default=["qa", "notes", "summary"])
p.add_argument("--bert-model", default=os.getenv("BERTSCORE_MODEL", "roberta-large"))
a = p.parse_args()

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)   # stemmer: "infections" ~ "infection"


def letter(text):
    """'Answer: C) ...' -> 'C' (None if the model did not follow the format)."""
    if text and "Answer:" in text:
        tail = text.split("Answer:", 1)[1].strip()
        return tail[:1].upper() or None
    return None


def is_json(text):
    try:
        json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        return True
    except (json.JSONDecodeError, AttributeError):
        return False


results = {}
for task in a.tasks:
    for path in sorted(Path(OUT / "preds" / task).glob("*.jsonl")):
        rows = read_jsonl(path)
        preds = [r["prediction"] or "" for r in rows]
        refs = [r["reference"] for r in rows]
        m = {"n": len(rows)}
        m["rougeL"] = sum(rouge.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)) / len(rows)
        _, _, f1 = bert_score(preds, refs, model_type=a.bert_model, lang="en", verbose=False)
        m["bertscore_f1"] = f1.mean().item()
        if task == "qa":
            m["accuracy"] = sum(letter(p_) == r["answer_letter"] for p_, r in zip(preds, rows)) / len(rows)
            m["format_ok"] = sum(letter(p_) is not None for p_ in preds) / len(rows)
        if task == "summary":
            m["json_valid"] = sum(is_json(p_) for p_ in preds) / len(rows)
        results[f"{task}/{path.stem}"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
        print(f"{task:8} {path.stem:12} {results[f'{task}/{path.stem}']}")

report = OUT / "reports" / "metrics.json"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(results, indent=1))
print(f"saved {report}")
