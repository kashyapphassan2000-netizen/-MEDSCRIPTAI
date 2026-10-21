"""Part 6: build DPO preference pairs (prompt, chosen, rejected) with distilabel.

For each held-out prompt we collect 3 candidate answers:
  2 samples from YOUR SFT adapter (served by vLLM under the adapter name, e.g. "qa")
  1 answer from the teacher model
The teacher then rates all 3 (UltraFeedback, 1-5). For qa we ALSO check the letter against the gold answer,
because a verifiable signal beats any judge. Best candidate = chosen, worst = rejected.
"""
import argparse
import os
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from common import PREF, SYSTEM, read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--task", default="qa", choices=["qa", "notes", "summary"])
p.add_argument("--limit", type=int, default=1500)
a = p.parse_args()

STUDENT_URL = os.getenv("STUDENT_URL", "http://localhost:8000/v1")  # vLLM with your SFT adapters
TEACHER_URL = os.getenv("TEACHER_URL", "http://localhost:8001/v1")  # vLLM with the teacher
TEACHER = os.getenv("TEACHER_MODEL", "teacher")

rows = read_jsonl(PREF / f"{a.task}_dpo.jsonl")[: a.limit]
inputs = [{"instruction": r["messages"][1]["content"],                 # the user message
           "gold_letter": r.get("answer_letter")} for r in rows]       # None for notes/summary


def llm(model, url, temperature):
    return OpenAILLM(model=model, base_url=url, api_key="EMPTY",
                     generation_kwargs={"temperature": temperature, "max_new_tokens": 700})


with Pipeline(name=f"medscriptai-dpo-{a.task}") as pipeline:
    load = LoadDataFromDicts(data=inputs, batch_size=64)
    student_1 = TextGeneration(name="student_1", llm=llm(a.task, STUDENT_URL, 0.9),
                               system_prompt=SYSTEM[a.task], input_batch_size=32)
    student_2 = TextGeneration(name="student_2", llm=llm(a.task, STUDENT_URL, 0.9),
                               system_prompt=SYSTEM[a.task], input_batch_size=32)
    teacher = TextGeneration(name="teacher", llm=llm(TEACHER, TEACHER_URL, 0.3),
                             system_prompt=SYSTEM[a.task], input_batch_size=32)
    group = GroupColumns(columns=["generation", "model_name"],          # 3 single answers -> 1 list
                         output_columns=["generations", "generation_models"])
    rate = UltraFeedback(llm=llm(TEACHER, TEACHER_URL, 0.0), aspect="overall-rating", input_batch_size=16)
    load >> [student_1, student_2, teacher] >> group >> rate            # fan out, then join

data = pipeline.run(use_cache=False)["default"]["train"]


def letter_of(text):
    """Read the letter after 'Answer:' ('Answer: C) ...' -> 'C')."""
    if text and "Answer:" in text:
        tail = text.split("Answer:", 1)[1].strip()
        return tail[:1].upper() if tail else None
    return None


pairs = []
for row in data:
    scored = []
    for text, rating in zip(row["generations"], row["ratings"]):
        if not text or rating is None:
            continue
        correct = 1 if row["gold_letter"] and letter_of(text) == row["gold_letter"] else 0
        scored.append((correct * 10 + rating, text))    # correctness dominates, rating breaks ties
    if len(scored) < 2:
        continue
    scored.sort(key=lambda s: s[0])
    (low, rejected), (high, chosen) = scored[0], scored[-1]
    if high - low < 2:                                  # too similar = weak training signal, skip
        continue
    system = {"role": "system", "content": SYSTEM[a.task]}
    pairs.append({"prompt": [system, {"role": "user", "content": row["instruction"]}],
                  "chosen": [{"role": "assistant", "content": chosen}],
                  "rejected": [{"role": "assistant", "content": rejected}],
                  "score_chosen": high, "score_rejected": low})

write_jsonl(PREF / f"{a.task}_pairs.jsonl", pairs)
print(f"{len(pairs)} preference pairs from {len(inputs)} prompts")
