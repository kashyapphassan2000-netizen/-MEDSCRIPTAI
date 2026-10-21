"""Part 5: use distilabel + a teacher LLM to write explanations for MedQA answers (synthetic SFT data).

MedQA only gives the correct letter. A model trained on letters alone learns to guess, not to reason.
The teacher (Qwen2.5-14B-Instruct served by vLLM as "teacher") explains WHY the gold answer is right.
"""
import argparse
import os
import re
from distilabel.models import OpenAILLM                  # talks to any OpenAI-compatible server (vLLM)
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from common import DATA, SFT, read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None, help="only explain the first N MedQA rows (test run)")
a = p.parse_args()

TEACHER_URL = os.getenv("TEACHER_URL", "http://localhost:8001/v1")   # vLLM teacher endpoint
TEACHER = os.getenv("TEACHER_MODEL", "teacher")                       # --served-model-name in vLLM
TEACHER_SYSTEM = ("You are a senior physician and medical educator. Explain clearly and accurately "
                  "in at most 120 words. Do not change the given correct answer.")

rows = read_jsonl(SFT / "qa_train.jsonl")
todo = [r for r in rows if r["source"] == "medqa" and "Explanation:" not in r["messages"][2]["content"]]
todo = todo[: a.limit] if a.limit else todo
inputs = [{
    "question_key": r["question_key"],
    "gold": r["messages"][2]["content"],                          # e.g. "Answer: D) Nitrofurantoin"
    "instruction": (f"{r['messages'][1]['content']}\n\nThe correct answer is: "
                    f"{r['messages'][2]['content'].removeprefix('Answer: ')}\n"
                    "Explain step by step why it is correct and briefly why the other options are wrong."),
} for r in todo]
print(f"{len(inputs)} MedQA rows need an explanation")

with Pipeline(name="medscriptai-explanations") as pipeline:
    load = LoadDataFromDicts(data=inputs, batch_size=64)          # step 1: feed our rows in
    explain = TextGeneration(                                       # step 2: ask the teacher
        llm=OpenAILLM(model=TEACHER, base_url=TEACHER_URL, api_key="EMPTY",
                      generation_kwargs={"temperature": 0.3, "max_new_tokens": 300}),
        system_prompt=TEACHER_SYSTEM,
        input_batch_size=32,                                        # 32 requests in flight at a time
    )
    load >> explain                                                 # ">>" connects the steps

distiset = pipeline.run(use_cache=False)                           # returns a Hugging Face DatasetDict
generated = distiset["default"]["train"]


def good(text, gold):
    """Quality filter: non-empty, sane length, and it mentions the gold option."""
    if not text or not (80 <= len(text) <= 1500):
        return False
    option = gold.split(") ", 1)[-1].lower()
    letter = gold.removeprefix("Answer: ")[0]
    return option[:25] in text.lower() or f"{letter})" in text


explanations = {g["question_key"]: g["generation"].strip()
                for g in generated if good(g["generation"], g["gold"])}
write_jsonl(DATA / "synth" / "medqa_explanations.jsonl",
            [{"question_key": k, "explanation": v} for k, v in explanations.items()])

for r in rows:                                                      # merge back into the SFT data
    if r["question_key"] in explanations:
        r["messages"][2]["content"] += f"\nExplanation: {explanations[r['question_key']]}"
write_jsonl(SFT / "qa_train_synth.jsonl", rows)
print(f"kept {len(explanations)}/{len(inputs)} explanations after the quality filter")
