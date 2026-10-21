"""Step 5: ask the same test questions to the base model and to base + your adapter, side by side."""
from unsloth import FastLanguageModel                    # Unsloth first, as always

import argparse
import torch
from common import SFT, OUT, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--task", default="qa", choices=["qa", "notes", "summary"])
p.add_argument("--adapter", default=None, help="adapter folder (default outputs/adapters/sft-<task>)")
p.add_argument("--n", type=int, default=3, help="how many test examples to show")
a = p.parse_args()
adapter = a.adapter or str(OUT / "adapters" / f"sft-{a.task}")

# Loading the ADAPTER folder makes Unsloth read adapter_config.json, load the 4-bit base, then attach the LoRA.
model, tokenizer = FastLanguageModel.from_pretrained(adapter, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)                    # switch on Unsloth's fast generation path


def answer(messages):
    """Greedy-decode one reply for a list of chat messages (system + user)."""
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=400, do_sample=False)   # do_sample=False = deterministic
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


for row in read_jsonl(SFT / f"{a.task}_test.jsonl")[: a.n]:
    prompt = row["messages"][:2]                           # system + user only; hide the reference answer
    with model.disable_adapter():                          # temporarily turn LoRA OFF = plain base model
        base = answer(prompt)
    tuned = answer(prompt)                                 # LoRA back ON
    print("=" * 80, "\nUSER:", prompt[1]["content"][:500])
    print("\n--- BASE ---\n", base[:700])
    print("\n--- FINE-TUNED ---\n", tuned[:700])
    print("\n--- REFERENCE ---\n", row["messages"][2]["content"][:700])
    torch.cuda.empty_cache()
