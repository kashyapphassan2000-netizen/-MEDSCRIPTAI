"""Evaluation step A: send every test prompt to vLLM for each model/adapter and save the answers.

python scripts/30_generate_preds.py --task qa --models base qa qa-dpo
"base" = Llama-3.1-8B-Instruct with no adapter; "qa" etc. = LoRA adapter names given to vLLM.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor   # many requests at once = vLLM batches them on the GPU
from openai import OpenAI                            # vLLM speaks the OpenAI API
from common import SFT, OUT, read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["qa", "notes", "summary"])
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--n", type=int, default=None, help="only the first N test rows")
p.add_argument("--url", default=os.getenv("VLLM_URL", "http://localhost:8000/v1"))
a = p.parse_args()

client = OpenAI(base_url=a.url, api_key="EMPTY")    # vLLM ignores the key, the SDK needs one
rows = read_jsonl(SFT / f"{a.task}_test.jsonl")[: a.n]
max_tokens = 400 if a.task == "qa" else 900


def ask(model, row):
    resp = client.chat.completions.create(
        model=model,
        messages=row["messages"][:2],                # system + user; never send the reference
        temperature=0.0,                             # greedy = reproducible scores
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


for model in a.models:
    with ThreadPoolExecutor(max_workers=16) as pool:
        preds = list(pool.map(lambda r: ask(model, r), rows))
    out = [{"i": i, "source": r.get("source"), "answer_letter": r.get("answer_letter"),
            "prompt": r["messages"][1]["content"], "reference": r["messages"][2]["content"],
            "prediction": pred} for i, (r, pred) in enumerate(zip(rows, preds))]
    write_jsonl(OUT / "preds" / a.task / f"{model}.jsonl", out)
