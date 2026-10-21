"""Step 1: download the open medical datasets and save them untouched as .jsonl."""
import argparse                                   # read command-line options
from datasets import load_dataset                 # Hugging Face dataset loader
from common import RAW                            # data/raw folder

# name on the Hub, config, split, file we save, rows to keep as (start, end) after a seeded shuffle
SOURCES = [
    ("GBaker/MedQA-USMLE-4-options", None, "train", "medqa_train", None),
    ("GBaker/MedQA-USMLE-4-options", None, "test", "medqa_test", None),
    ("openlifescienceai/medmcqa", None, "train", "medmcqa_train", (0, 20000)),
    ("openlifescienceai/medmcqa", None, "train", "medmcqa_dpo", (20000, 23000)),  # unseen prompts for DPO
    ("openlifescienceai/medmcqa", None, "validation", "medmcqa_val", None),
    ("qiaojin/PubMedQA", "pqa_labeled", "train", "pubmedqa", None),
    ("AGBonnet/augmented-clinical-notes", None, "train", "clinical_notes", (0, 12000)),
]

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="tiny test run: keep N rows per file")
args = parser.parse_args()

RAW.mkdir(parents=True, exist_ok=True)            # make data/raw if it does not exist
for name, config, split, out_name, rows in SOURCES:
    ds = load_dataset(name, config, split=split)  # downloads once, then uses the local cache
    if rows:                                      # same seed = same order, so slices never overlap
        ds = ds.shuffle(seed=42).select(range(rows[0], min(rows[1], len(ds))))
    if args.limit:                                # --limit: tiny test run
        ds = ds.select(range(min(args.limit, len(ds))))
    path = RAW / f"{out_name}.jsonl"
    ds.to_json(str(path), lines=True, force_ascii=False)  # one JSON object per line
    print(f"{out_name:16} {len(ds):>6} rows  <- {name} [{split}]")
