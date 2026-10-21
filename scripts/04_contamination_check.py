"""Elite check: is any TEST question a near-copy of a TRAINING question? (13-gram overlap, GPT-3 paper style)

Exact-match de-duplication (step 2) misses near-copies: the same question with one word changed.
Medical exams share stock phrases ("is brought to the emergency department because of a 2-day history of",
lab panels), so ONE shared 13-gram proves nothing. We flag a test question only when at least
50% of its 13-grams come from a SINGLE training question.
"""
import re
from collections import Counter, defaultdict
from common import SFT, read_jsonl

N = 13                       # n-gram length in words
THRESHOLD = 0.5              # share of the test question's n-grams found in one training question


def grams(text):
    text = text.split("\n\nOptions:")[0]                 # compare the question stem, not the option labels
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + N]) for i in range(len(words) - N + 1)}


for task in ["qa", "notes", "summary"]:
    index = defaultdict(set)                             # n-gram -> ids of training rows containing it
    path = SFT / f"{task}_train.jsonl"
    if path.exists():
        for row_id, row in enumerate(read_jsonl(path)):
            for g in grams(row["messages"][1]["content"]):
                index[g].add(row_id)
    test = read_jsonl(SFT / f"{task}_test.jsonl")
    flagged = []
    for row in test:
        test_grams = grams(row["messages"][1]["content"])
        per_train_row = Counter(i for g in test_grams for i in index.get(g, ()))
        best = per_train_row.most_common(1)[0][1] if per_train_row else 0
        if test_grams and best / len(test_grams) >= THRESHOLD:
            flagged.append(row["messages"][1]["content"][:110].replace("\n", " "))
    print(f"{task:8} test rows={len(test):4}  near-duplicates of training rows={len(flagged):3} "
          f"({len(flagged) / max(len(test), 1):.1%})")
    for text in flagged[:3]:
        print("   e.g.", text)
