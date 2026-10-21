"""Shared settings and helpers used by every MedScriptAI script."""
import json                      # read/write JSON text
import os                        # read environment variables
from pathlib import Path         # build file paths that work on any OS

# Project folders. ROOT is the repo folder (one level above scripts/).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"             # all datasets live here
RAW = DATA / "raw"               # untouched downloads
SFT = DATA / "sft"               # chat-format .jsonl for SFT
PREF = DATA / "pref"             # preference pairs for DPO
OUT = ROOT / "outputs"           # adapters, predictions, reports

# Base model. The unsloth mirror is ungated; you still must accept Meta's licence.
BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Llama-3.1-8B-Instruct")
BASE_MODEL_4BIT = os.getenv("BASE_MODEL_4BIT", "unsloth/Llama-3.1-8B-Instruct-bnb-4bit")

# One system prompt per task = one LoRA adapter per task.
SYSTEM = {
    "qa": (
        "You are MedScriptAI, a careful medical education assistant. "
        "Answer the multiple-choice question. First line: 'Answer: <letter>) <option>'. "
        "Then give a short explanation. Educational use only, not medical advice."
    ),
    "notes": (
        "You are MedScriptAI, a clinical documentation assistant. "
        "Write a concise clinical case note from the doctor-patient conversation. "
        "Use only facts stated in the conversation. Do not invent findings."
    ),
    "summary": (
        "You are MedScriptAI, a clinical information extractor. "
        "Return ONLY valid JSON that summarises the clinical note. "
        "Include only fields that are stated in the note."
    ),
}


def write_jsonl(path, rows):
    """Write a list of dicts to a .jsonl file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)       # create folders if missing
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>6} rows -> {path.relative_to(ROOT)}")


def read_jsonl(path):
    """Read a .jsonl file back into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat(task, user, assistant=None):
    """Build the 'messages' list in the chat format TRL and vLLM understand."""
    msgs = [{"role": "system", "content": SYSTEM[task]},
            {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs
