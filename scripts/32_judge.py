"""Evaluation step C: LLM-as-a-judge.

--mode score     one answer at a time, 1-5 rubric score            (Part 5)
--mode pairwise  model A vs model B, run twice with order swapped    (Part 6)
                 -> removes position bias: a "win" must win in BOTH orders
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from common import OUT, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["qa", "notes", "summary"])
p.add_argument("--mode", default="score", choices=["score", "pairwise"])
p.add_argument("--models", nargs="+", required=True, help="score: any number; pairwise: exactly two (A B)")
p.add_argument("--n", type=int, default=100)
a = p.parse_args()

judge = OpenAI(base_url=os.getenv("JUDGE_URL", "http://localhost:8001/v1"), api_key="EMPTY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "teacher")
RUBRIC = {
    "qa": "correct final answer (vs the reference), sound medical reasoning, no unsafe advice",
    "notes": "faithful to the conversation (no invented facts), complete key findings, clear clinical style",
    "summary": "valid JSON, every value supported by the note, no important fact missing",
}


def ask_json(prompt):
    """Call the judge at temperature 0 and pull the first {...} object out of the reply."""
    text = judge.chat.completions.create(
        model=JUDGE_MODEL, temperature=0.0, max_tokens=300,
        messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def score_prompt(row):
    return (f"You are a strict medical expert grader. Criteria: {RUBRIC[a.task]}.\n\n"
            f"TASK INPUT:\n{row['prompt'][:4000]}\n\nREFERENCE ANSWER:\n{row['reference'][:3000]}\n\n"
            f"MODEL ANSWER:\n{row['prediction'][:3000]}\n\n"
            'Return ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}')


def pair_prompt(row, first, second):
    return (f"You are a strict medical expert judge. Criteria: {RUBRIC[a.task]}.\n\n"
            f"TASK INPUT:\n{row['prompt'][:4000]}\n\nREFERENCE ANSWER:\n{row['reference'][:3000]}\n\n"
            f"ANSWER A:\n{first[:3000]}\n\nANSWER B:\n{second[:3000]}\n\n"
            'Which answer is better? Return ONLY a JSON object: {"winner": "A" or "B" or "tie", "reason": "..."}')


preds = {m: read_jsonl(OUT / "preds" / a.task / f"{m}.jsonl")[: a.n] for m in a.models}
report = {}
with ThreadPoolExecutor(max_workers=16) as pool:
    if a.mode == "score":
        for m, rows in preds.items():
            scores = [r.get("score") for r in pool.map(lambda row: ask_json(score_prompt(row)), rows)]
            valid = [s for s in scores if isinstance(s, (int, float))]
            report[m] = {"mean_score": round(sum(valid) / max(len(valid), 1), 3), "judged": len(valid)}
    else:
        ma, mb = a.models
        rows_a, rows_b = preds[ma], preds[mb]

        def duel(i):
            ab = ask_json(pair_prompt(rows_a[i], rows_a[i]["prediction"], rows_b[i]["prediction"]))
            ba = ask_json(pair_prompt(rows_a[i], rows_b[i]["prediction"], rows_a[i]["prediction"]))
            first, second = ab.get("winner"), ba.get("winner")
            if first == "A" and second == "B":
                return ma                                  # A won in both orders
            if first == "B" and second == "A":
                return mb
            return "tie"                                   # disagreement across orders = tie

        outcomes = list(pool.map(duel, range(min(len(rows_a), len(rows_b)))))
        n = len(outcomes)
        report = {f"{ma}_wins": round(outcomes.count(ma) / n, 3),
                  f"{mb}_wins": round(outcomes.count(mb) / n, 3),
                  "ties": round(outcomes.count("tie") / n, 3), "n": n}

print(json.dumps(report, indent=1))
path = OUT / "reports" / f"judge_{a.task}_{a.mode}_{'_vs_'.join(a.models)}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=1))
