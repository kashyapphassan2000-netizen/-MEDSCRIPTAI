"""Step 2: clean the raw data and convert it to chat-format .jsonl (one file per task and split)."""
import json                                        # parse the JSON summaries
import random                                      # repeatable shuffling
import re                                          # simple text clean-up
from common import RAW, SFT, PREF, chat, read_jsonl, write_jsonl

random.seed(42)                                    # same split every time you run it
MAX_CHARS = 7000                                   # ~1,750 tokens: fits max_seq_length 2048
LETTERS = "ABCD"


def clean(text):
    """Collapse repeated spaces/newlines and strip the ends."""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", str(text or ""))).strip()


def key(text):
    """Normalised text used to find duplicates (lower-case, letters and digits only)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())[:300]


def mcq_prompt(question, options):
    """Question + lettered options, the exact user message format for the qa adapter."""
    lines = [f"{letter}) {text}" for letter, text in options]
    return f"{clean(question)}\n\nOptions:\n" + "\n".join(lines)


def short(text, limit=900):
    """Cut long explanations at the last full stop before `limit` characters."""
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(".", 1)[0]
    return cut + "."


# ---------- QA task: MedQA + MedMCQA + PubMedQA ----------
def from_medqa(row):
    opts = sorted(row["options"].items())          # [("A", "..."), ("B", "..."), ...]
    letter = row["answer_idx"]
    answer = f"Answer: {letter}) {clean(row['answer'])}"   # no explanation in MedQA (Part 5 adds one)
    return {"task": "qa", "source": "medqa", "answer_letter": letter,
            "question_key": key(row["question"]),
            "messages": chat("qa", mcq_prompt(row["question"], opts), answer)}


def from_medmcqa(row):
    if row.get("choice_type") != "single" or row.get("cop") not in (0, 1, 2, 3):
        return None                                # skip multi-answer or broken rows
    opts = list(zip(LETTERS, [row["opa"], row["opb"], row["opc"], row["opd"]]))
    letter = LETTERS[row["cop"]]
    answer = f"Answer: {letter}) {clean(opts[row['cop']][1])}"
    if row.get("exp") and len(clean(row["exp"])) > 40:
        answer += f"\nExplanation: {short(row['exp'])}"
    return {"task": "qa", "source": "medmcqa", "answer_letter": letter,
            "question_key": key(row["question"]),
            "messages": chat("qa", mcq_prompt(row["question"], opts), answer)}


def from_pubmedqa(row):
    context = clean(" ".join(row["context"]["contexts"]))
    opts = [("A", "yes"), ("B", "no"), ("C", "maybe")]
    letter = {"yes": "A", "no": "B", "maybe": "C"}[row["final_decision"]]
    question = f"Context: {context}\n\nQuestion: {row['question']}"
    answer = f"Answer: {letter}) {row['final_decision']}\nExplanation: {short(row['long_answer'])}"
    return {"task": "qa", "source": "pubmedqa", "answer_letter": letter,
            "question_key": key(row["question"]),
            "messages": chat("qa", mcq_prompt(question, opts), answer)}


# ---------- notes + summary tasks: augmented clinical notes ----------
def compact(value):
    """Drop empty fields ("None", "", [], {}) so the target JSON is short and factual."""
    if isinstance(value, dict):
        out = {k: compact(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        out = [compact(v) for v in value]
        return [v for v in out if v not in (None, "", [], {})]
    if isinstance(value, str) and value.strip().lower() in ("none", "n/a", ""):
        return None
    return value


def from_clinical(row):
    """One raw row gives one 'notes' example and one 'summary' example (or None if broken)."""
    conversation, note = clean(row["conversation"]), clean(row["note"])
    try:
        summary = compact(json.loads(row["summary"]))   # ~25% of summaries are invalid JSON
    except (json.JSONDecodeError, TypeError):
        return None
    if not conversation or not note or not summary:
        return None
    summary_text = json.dumps(summary, ensure_ascii=False)
    notes_ex = {"task": "notes", "source": "clinical_notes", "id": row["idx"],
                "messages": chat("notes", conversation, note)}
    summary_ex = {"task": "summary", "source": "clinical_notes", "id": row["idx"],
                  "messages": chat("summary", note, summary_text)}
    return notes_ex, summary_ex


def fits(example):
    """Keep only examples short enough for max_seq_length=2048."""
    return sum(len(m["content"]) for m in example["messages"]) <= MAX_CHARS


def build(rows, fn):
    """Apply a converter, drop None/too-long rows."""
    out = [fn(r) for r in rows]
    return [x for x in out if x is not None]


# ---- QA: build, de-duplicate, split ----
def sample(rows, n):
    """Random n rows (or all of them if there are fewer than n)."""
    return random.sample(rows, min(n, len(rows)))


qa_test = (sample(build(read_jsonl(RAW / "medqa_test.jsonl"), from_medqa), 200)
           + sample(build(read_jsonl(RAW / "medmcqa_val.jsonl"), from_medmcqa), 200))
pubmed = build(read_jsonl(RAW / "pubmedqa.jsonl"), from_pubmedqa)
random.shuffle(pubmed)
qa_test += pubmed[:100]                                   # 100 PubMedQA rows only for testing
qa_pool = (build(read_jsonl(RAW / "medqa_train.jsonl"), from_medqa)
           + build(read_jsonl(RAW / "medmcqa_train.jsonl"), from_medmcqa)
           + pubmed[100:])

test_keys = {x["question_key"] for x in qa_test}          # decontamination: never train on test questions
seen, qa_clean = set(), []
for ex in qa_pool:
    k = ex["question_key"]
    if k in seen or k in test_keys or not fits(ex):
        continue                                          # duplicate, leaked, or too long
    seen.add(k)
    qa_clean.append(ex)
random.shuffle(qa_clean)
write_jsonl(SFT / "qa_val.jsonl", qa_clean[:300])
write_jsonl(SFT / "qa_train.jsonl", qa_clean[300:])
write_jsonl(SFT / "qa_test.jsonl", qa_test)

# DPO prompts: 3,000 MedMCQA questions the SFT model never saw (and not in the test set)
dpo_qa = [x for x in build(read_jsonl(RAW / "medmcqa_dpo.jsonl"), from_medmcqa)
          if x["question_key"] not in seen and x["question_key"] not in test_keys and fits(x)]
write_jsonl(PREF / "qa_dpo.jsonl", dpo_qa)

# ---- notes + summary: one shared split so a test note is never a training note ----
pairs, seen_notes = [], set()
for p in build(read_jsonl(RAW / "clinical_notes.jsonl"), from_clinical):
    k = key(p[1]["messages"][1]["content"])             # the note text: ~13% of rows repeat a note
    if k in seen_notes or not (fits(p[0]) and fits(p[1])):
        continue                                         # duplicate note (would leak into test) or too long
    seen_notes.add(k)
    pairs.append(p)
random.shuffle(pairs)
splits = {"test": pairs[:200], "val": pairs[200:400], "dpo": pairs[400:1400], "train": pairs[1400:]}
for split, rows in splits.items():
    folder = PREF if split == "dpo" else SFT               # DPO prompts are held out from SFT training
    write_jsonl(folder / f"notes_{split}.jsonl", [p[0] for p in rows])
    write_jsonl(folder / f"summary_{split}.jsonl", [p[1] for p in rows])
