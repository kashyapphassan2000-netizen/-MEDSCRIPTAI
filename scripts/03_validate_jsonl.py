"""Step 3: prove every .jsonl file is valid chat data before you pay for GPU time."""
import json                                         # parse each line
import statistics                                   # median / max token counts
import sys                                          # exit code for pass/fail
from pathlib import Path
from common import SFT, BASE_MODEL

ROLES = ["system", "user", "assistant"]
errors = 0

try:                                                # token counts need the Llama tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
except Exception as exc:                            # still validate structure without it
    tok = None
    print(f"(tokenizer not loaded, skipping token counts: {exc.__class__.__name__})")

for path in sorted(Path(SFT).glob("*.jsonl")):
    lengths = []
    for n, line in enumerate(path.open(encoding="utf-8"), start=1):
        try:
            row = json.loads(line)                  # 1) each line must be valid JSON
            roles = [m["role"] for m in row["messages"]]
            assert roles == ROLES, f"roles {roles}"  # 2) exactly system -> user -> assistant
            assert all(m["content"].strip() for m in row["messages"]), "empty message"
            if row["task"] == "summary":            # 3) summary targets must be valid JSON
                json.loads(row["messages"][-1]["content"])
            if tok:                                 # 4) length in tokens after the chat template
                ids = tok.apply_chat_template(row["messages"], tokenize=True)
                ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
                lengths.append(len(ids))
        except Exception as exc:
            errors += 1
            print(f"ERROR {path.name}:{n}: {exc}")
    stats = f" tokens median={statistics.median(lengths):.0f} max={max(lengths)}" if lengths else ""
    print(f"{path.name:22} rows={n:>6}{stats}")

print("ALL_OK" if errors == 0 else f"FAILED with {errors} errors")
sys.exit(1 if errors else 0)
