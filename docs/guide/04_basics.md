# PART 4 - BASICS: your first medical fine-tune (data -> .jsonl -> QLoRA SFT -> test -> publish)

## BEFORE YOU START: what you will build in this part
You will create 6 files by hand, run them in order, and end with your first trained adapter (sft-qa) that answers medical exam questions in a fixed format, published on the Hugging Face Hub and copied to (Fake)S3.
```
01_download_data.py  -> data/raw/*.jsonl        (7 files, untouched copies of the datasets)
02_build_sft_jsonl.py -> data/sft/*.jsonl       (9 files: qa / notes / summary x train / val / test)
                       data/pref/*.jsonl        (3 files of prompts kept aside for DPO in Part 6A)
03_validate_jsonl.py  -> "ALL_OK"               (proof the files are correct before using the GPU)
10_train_sft.py       -> outputs/adapters/sft-qa (your first LoRA adapter)
11_chat_test.py       -> base vs fine-tuned answers side by side
12_push_to_hub.py     -> your adapter on huggingface.co
```
The four datasets (all open, all free):
| Dataset | What it contains | Licence | Used for |
| MedQA (USMLE 4-options) | US medical licensing exam questions, correct letter only | CC-BY-4.0 | qa |
| MedMCQA | Indian medical entrance exam questions with expert explanations | Apache-2.0 | qa (+ DPO prompts) |
| PubMedQA (labelled) | research abstract + yes/no/maybe question + long answer | MIT | qa |
| augmented-clinical-notes | case reports with a doctor-patient conversation and a JSON summary | MIT | notes and summary |
What one training example looks like after Step 4.5 (this is a REAL line from the data):
```
{"task": "qa", "source": "medmcqa", "answer_letter": "C",
 "messages": [
  {"role": "system", "content": "You are MedScriptAI, a careful medical education assistant. ..."},
  {"role": "user", "content": "A patient presents with symptoms of Hypoglycemia. ... C-peptide ... normal levels ...\n\nOptions:\nA) Insulinoma\nB) Accidental sulfonylurea ingestion\nC) Accidental exogenous Insulin administration\nD) Accidental Metformin ingestion"},
  {"role": "assistant", "content": "Answer: C) Accidental exogenous Insulin administration\nExplanation: ... C peptide ... is suppressed ..."}]}
```
This "messages" format (system, user, assistant) is the conversational format that TRL, Unsloth and vLLM all understand.
Time and cost: about 1.5 GPU-hours (about US$2). The training run itself is roughly 10-15 minutes on an A100 (estimate; your progress bar shows the real speed).

## SECTION 4.1 - Create scripts/common.py (settings shared by every script)
WHERE: VS Code Explorer (left panel), folder scripts
GOAL: one file that holds folder paths, model names and the three system prompts.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type common.py and press Enter.
Step 4. Paste the grey box below (Ctrl+V).
Step 5. Press Ctrl+S.
@@include scripts/common.py
| Line | What it does (plain English) |
| `import json`, `import os`, `from pathlib import Path` | standard Python tools: JSON text, environment variables, file paths that work on every OS |
| `ROOT = Path(__file__).resolve().parent.parent` | the project folder = two levels above this file (scripts/common.py -> medscriptai) |
| `DATA`, `RAW`, `SFT`, `PREF`, `OUT` | one variable per folder, so no script hard-codes a path |
| `BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Llama-3.1-8B-Instruct")` | the 16-bit base model; os.getenv lets you override it with an environment variable without editing code |
| `BASE_MODEL_4BIT = ...-bnb-4bit` | the same model already compressed to 4-bit NF4 by Unsloth (5.7 GB download instead of 16 GB) |
| `SYSTEM = {...}` | one system prompt per task; one task = one adapter. The API in Part 6A imports this SAME dict, so training and serving prompts can never differ |
| `def write_jsonl(path, rows)` | writes a list of dicts, one JSON object per line; `mkdir(parents=True, exist_ok=True)` creates missing folders |
| `json.dumps(row, ensure_ascii=False)` | dict -> JSON text; ensure_ascii=False keeps symbols such as °C readable |
| `print(f"wrote {len(rows):>6} rows -> ...")` | progress message; `:>6` right-aligns the number in 6 characters |
| `def read_jsonl(path)` | reads a .jsonl file back into a list of dicts; `if line.strip()` skips empty lines |
| `def chat(task, user, assistant=None)` | builds the messages list: system, user and (for training rows) the assistant answer |
Step 6. Check the file has no typing errors:
```
cd ~/medscriptai && .venv-train/bin/python -c "import sys; sys.path.insert(0,'scripts'); import common; print(common.BASE_MODEL, list(common.SYSTEM))"
```
You should see: unsloth/Llama-3.1-8B-Instruct ['qa', 'notes', 'summary']
CHECKPOINT: the line above prints.
If not: SyntaxError = the paste was cut or changed; delete everything in the file (Ctrl+A, Delete), paste again, save.

## SECTION 4.2 - Create scripts/01_download_data.py
WHERE: VS Code Explorer, folder scripts
GOAL: the script that downloads the four datasets.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 01_download_data.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/01_download_data.py
| Line | What it does (plain English) |
| `import argparse` | lets you pass options such as --limit on the command line |
| `from datasets import load_dataset` | Hugging Face's loader: downloads once, caches in HF_HOME, returns a table |
| `from common import RAW` | the data/raw folder from common.py |
| `SOURCES = [...]` | the shopping list: Hub name, sub-set name, split, output file name, rows to keep |
| `("GBaker/MedQA-USMLE-4-options", None, "train", "medqa_train", None)` | MedQA training questions; None = keep all rows |
| `("openlifescienceai/medmcqa", ..., (0, 20000))` | MedMCQA: keep rows 0-19,999 after shuffling (the full set has 182k; 20k is plenty) |
| `(..., "medmcqa_dpo", (20000, 23000))` | 3,000 DIFFERENT MedMCQA questions kept aside for DPO, so they are new to the SFT model |
| `("qiaojin/PubMedQA", "pqa_labeled", ...)` | the 1,000 expert-labelled PubMedQA questions |
| `("AGBonnet/augmented-clinical-notes", ..., (0, 12000))` | 12,000 clinical notes with conversations and JSON summaries |
| `parser.add_argument("--limit", ...)` | optional tiny test run: keep only N rows per file |
| `RAW.mkdir(parents=True, exist_ok=True)` | create data/raw (no error if it exists) |
| `ds = load_dataset(name, config, split=split)` | download the first time, load from the cache afterwards |
| `ds.shuffle(seed=42).select(range(rows[0], ...))` | shuffle with a fixed seed (same order every run) and take a slice; same seed = the SFT slice and the DPO slice never overlap |
| `ds.to_json(str(path), lines=True, force_ascii=False)` | save as .jsonl (lines=True = one object per line) |
| `print(f"{out_name:16} ...")` | one progress line per file |
CHECKPOINT: ls scripts shows 01_download_data.py and common.py

## SECTION 4.3 - Run the download
WHERE: VS Code Terminal, inside a tmux session
GOAL: seven raw .jsonl files in data/raw.
Step 1. Start a tmux session:
```
tmux new -s data
```
Step 2. Go to the scripts folder and activate the training environment:
```
cd ~/medscriptai/scripts && source ../.venv-train/bin/activate
```
Step 3. Run:
```
python 01_download_data.py
```
You should see (2-5 minutes; these are the real numbers):
```
medqa_train       10178 rows  <- GBaker/MedQA-USMLE-4-options [train]
medqa_test         1273 rows  <- GBaker/MedQA-USMLE-4-options [test]
medmcqa_train     20000 rows  <- openlifescienceai/medmcqa [train]
medmcqa_dpo        3000 rows  <- openlifescienceai/medmcqa [train]
medmcqa_val        4183 rows  <- openlifescienceai/medmcqa [validation]
pubmedqa           1000 rows  <- qiaojin/PubMedQA [train]
clinical_notes    12000 rows  <- AGBonnet/augmented-clinical-notes [train]
```
Step 4. List the files:
```
ls -lh ../data/raw
```
You should see: seven .jsonl files (clinical_notes.jsonl is the biggest, roughly 150-250 MB).
Watch out: lines with progress bars ("Generating train split") are normal. "429 Too Many Requests" = run echo ${HF_TOKEN:0:3} (must print hf_) and run the script again.
CHECKPOINT: seven files in data/raw.

## SECTION 4.4 - Create scripts/02_build_sft_jsonl.py (clean + convert)
BEFORE YOU START: what this script does. Raw datasets all look different (MedQA has an options dict, MedMCQA has opa/opb/opc/opd columns, PubMedQA has abstracts). This script converts every row into the same "messages" format, removes broken rows, removes duplicates, makes sure no test question is in training, and writes train/val/test files. This "boring" step decides most of the final quality.
WHERE: VS Code Explorer, folder scripts
GOAL: the cleaning and conversion script.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 02_build_sft_jsonl.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/02_build_sft_jsonl.py
| Line | What it does (plain English) |
| `random.seed(42)` | fixes the random generator: the split is identical every run (reproducible) |
| `MAX_CHARS = 7000` | about 1,750 tokens (1 token is roughly 4 characters of English): every example fits max_seq_length 2048 |
| `LETTERS = "ABCD"` | maps answer index 0-3 to a letter |
| `def clean(text)` | None -> ""; 3+ newlines -> 2; repeated spaces/tabs -> 1; trims both ends |
| `def key(text)` | a fingerprint for finding duplicates: lower-case, letters and digits only, first 300 characters |
| `def mcq_prompt(question, options)` | builds the user message: question, blank line, "Options:", then "A) ..." lines. The API expects users to send this same shape |
| `def short(text, limit=900)` | cuts long explanations at the last full stop before 900 characters |
| `def from_medqa(row)` | one MedQA row -> one example; `sorted(row["options"].items())` gives [("A", ...), ("B", ...)] |
| `answer = f"Answer: {letter}) {clean(row['answer'])}"` | the target, e.g. Answer: D) Nitrofurantoin (no explanation yet; Part 5 adds one) |
| `"answer_letter"`, `"question_key"` | kept for scoring accuracy later and for de-duplication |
| `def from_medmcqa(row)` | one MedMCQA row; `if row.get("choice_type") != "single" ... return None` drops multi-answer and broken rows |
| `opts = list(zip(LETTERS, [row["opa"], ...]))` | pairs A-D with the four option texts |
| `if row.get("exp") and len(clean(row["exp"])) > 40` | adds the expert explanation only if it is real text |
| `def from_pubmedqa(row)` | the abstract becomes "Context:", options A) yes B) no C) maybe, long_answer becomes the explanation |
| `def compact(value)` | removes empty fields ("None", "", [], {}) from the summary JSON (about 70% of the raw fields are "None") |
| `def from_clinical(row)` | one clinical row -> TWO examples: notes (conversation -> note) and summary (note -> JSON) |
| `json.loads(row["summary"])` inside try/except | about 25% of the raw summaries are not valid JSON; those rows are skipped |
| `def fits(example)` | True if all messages together are at most MAX_CHARS characters |
| `def build(rows, fn)` | apply a converter to every row, drop the None results |
| `def sample(rows, n)` | n random rows (or all if fewer) |
| `qa_test = (sample(... medqa_test ..., 200) + sample(... medmcqa_val ..., 200))` | the QA test set comes ONLY from the official test/validation splits |
| `qa_test += pubmed[:100]` | 100 PubMedQA rows for testing; the other 900 may be used for training |
| `test_keys = {...}` and `if k in seen or k in test_keys or not fits(ex): continue` | skip duplicates, skip anything identical to a TEST question (decontamination), skip too-long examples |
| `write_jsonl(SFT / "qa_val.jsonl", qa_clean[:300])` | 300 validation rows; the rest is training |
| `dpo_qa = [...]` -> `PREF / "qa_dpo.jsonl"` | unseen MedMCQA prompts for DPO, also checked against train and test |
| `for p in build(... from_clinical): k = key(p[1]["messages"][1]["content"])` | the full-data test found 927 notes repeated up to 5 times; this keeps only the first copy of each note, so a test note can never also be a training note |
| `splits = {"test": pairs[:200], "val": pairs[200:400], "dpo": pairs[400:1400], "train": pairs[1400:]}` | ONE shared split for notes and summary |
| `folder = PREF if split == "dpo" else SFT` | DPO prompts to data/pref, the rest to data/sft |
CHECKPOINT: ls scripts shows 02_build_sft_jsonl.py

## SECTION 4.5 - Run the conversion
WHERE: VS Code Terminal, folder ~/medscriptai/scripts, (.venv-train) active
GOAL: 9 files in data/sft and 3 in data/pref.
Step 1. Run:
```
python 02_build_sft_jsonl.py
```
You should see (real output, about 1 minute):
```
wrote    300 rows -> data/sft/qa_val.jsonl
wrote  23669 rows -> data/sft/qa_train.jsonl
wrote    500 rows -> data/sft/qa_test.jsonl
wrote   1891 rows -> data/pref/qa_dpo.jsonl
wrote    200 rows -> data/sft/notes_test.jsonl
wrote    200 rows -> data/sft/summary_test.jsonl
wrote    200 rows -> data/sft/notes_val.jsonl
wrote    200 rows -> data/sft/summary_val.jsonl
wrote   1000 rows -> data/pref/notes_dpo.jsonl
wrote   1000 rows -> data/pref/summary_dpo.jsonl
wrote   6367 rows -> data/sft/notes_train.jsonl
wrote   6367 rows -> data/sft/summary_train.jsonl
```
Why 1891 and not 3000 DPO prompts: many MedMCQA questions are repeats or multi-answer; the filters removed them. That is correct behaviour.
Step 2. Look at one QA training example in readable form:
```
head -n 1 ../data/sft/qa_train.jsonl | python -m json.tool
```
You should see: the JSON shown in "BEFORE YOU START" (task, source, answer_letter, question_key, messages).
Step 3. Look at one notes example (the user message is a conversation, the assistant message is the note):
```
head -n 1 ../data/sft/notes_train.jsonl | python -c "import sys,json; m=json.load(sys.stdin)['messages']; print(m[1]['content'][:400]); print('-----'); print(m[2]['content'][:400])"
```
You should see: lines starting "Doctor: Good morning..." and "Patient: ...", then "-----", then a clinical note such as "A 24-year-old unmarried woman was admitted to the Department of Neurology ...".
Step 4. Look at one summary target (JSON):
```
head -n 1 ../data/sft/summary_train.jsonl | python -c "import sys,json; print(json.load(sys.stdin)['messages'][2]['content'][:400])"
```
You should see: {"visit motivation": "...", "admission": [...], "patient information": {"age": "24", "sex": "Female"}, ...
CHECKPOINT: the 12 "wrote" lines appeared and the three examples look sensible.
If not: FileNotFoundError data/raw/... = Section 4.3 did not finish; re-run it.

## SECTION 4.6 - Create and run scripts/03_validate_jsonl.py
WHERE: VS Code Explorer (create), then VS Code Terminal (run)
GOAL: proof that every line is valid chat data and fits in 2048 tokens.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 03_validate_jsonl.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/03_validate_jsonl.py
| Line | What it does (plain English) |
| `ROLES = ["system", "user", "assistant"]` | the only allowed message order |
| `AutoTokenizer.from_pretrained(BASE_MODEL)` inside try | loads Llama-3.1's real tokenizer to count tokens; if that fails, structure is still checked |
| `for path in sorted(Path(SFT).glob("*.jsonl"))` | every .jsonl file in data/sft |
| `row = json.loads(line)` | check 1: valid JSON |
| `assert roles == ROLES` | check 2: exactly system -> user -> assistant |
| `assert all(m["content"].strip() ...)` | check 3: no empty message |
| `if row["task"] == "summary": json.loads(...)` | check 4: the summary target is valid JSON |
| `tok.apply_chat_template(row["messages"], tokenize=True)` | check 5: the real token count after Llama's chat template |
| `ids = ids["input_ids"] if ... else ids` | newer transformers return a dict, older a list; this handles both |
| `print("ALL_OK" ...); sys.exit(1 if errors else 0)` | exit code 0 = success, 1 = failure |
Step 6. Run it (the first run downloads the tokenizer, a few MB):
```
python 03_validate_jsonl.py
```
You should see (real output):
```
notes_test.jsonl       rows=   200 tokens median=1146 max=1417
notes_train.jsonl      rows=  6367 tokens median=1138 max=1605
notes_val.jsonl        rows=   200 tokens median=1144 max=1540
qa_test.jsonl          rows=   500 tokens median=270 max=810
qa_train.jsonl         rows= 23669 tokens median=253 max=984
qa_val.jsonl           rows=   300 tokens median=245 max=582
summary_test.jsonl     rows=   200 tokens median=1095 max=1493
summary_train.jsonl    rows=  6367 tokens median=1083 max=1659
summary_val.jsonl      rows=   200 tokens median=1090 max=1663
ALL_OK
```
CHECKPOINT: the last line is ALL_OK and every max is below 2048.
If not: an ERROR line names the file and line number; "apply_chat_template requires jinja2" = run uv pip install --python ../.venv-train jinja2.

## SECTION 4.7 - Create scripts/10_train_sft.py (the training script)
BEFORE YOU START: what happens when you run it. (1) The 4-bit Llama model (5.7 GB) is downloaded and loaded on the GPU. (2) LoRA adapters (about 42 million trainable numbers) are attached. (3) Each training example is turned into one long text with the chat template. (4) The trainer runs 125 steps (2,000 examples / 16 per step); every 10 steps it prints the loss, every 100 steps the validation loss. (5) Only the adapter (about 80-170 MB) is saved to outputs/adapters/sft-qa. GPU memory used: roughly 10-16 GB of the 80 GB.
WHERE: VS Code Explorer, folder scripts
GOAL: the QLoRA SFT training script.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 10_train_sft.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/10_train_sft.py
| Line | What it does (plain English) |
| `from unsloth import FastLanguageModel` (first import) | Unsloth must be imported BEFORE trl/transformers so it can patch them with its fast code |
| `from unsloth.chat_templates import train_on_responses_only` | masks system/user tokens so the loss is only on the answer |
| `from trl import SFTConfig, SFTTrainer` | Hugging Face TRL's supervised fine-tuning trainer and its settings |
| `--task` with `choices=[...]` | which adapter to train; choices prevents typos |
| `--max-train` | use only the first N training rows (2,000 in this part) |
| `--epochs 1.0`, `--max-steps -1` | one pass over the data; -1 = stop by epochs |
| `--lr 2e-4` | learning rate: the standard starting point for QLoRA |
| `--rank 16` | LoRA rank r |
| `--batch 8`, `--grad-accum 2` | 8 examples per GPU step, gradients added over 2 steps = 16 per update |
| `--train-file` | train on another file (Part 5 uses the synthetic one) |
| `FastLanguageModel.from_pretrained(model_name=BASE_MODEL_4BIT, max_seq_length=2048, load_in_4bit=True, dtype=None)` | load Llama-3.1-8B-Instruct in 4-bit with its tokenizer; dtype None = automatic bfloat16 |
| `FastLanguageModel.get_peft_model(model, r=..., lora_alpha=..., lora_dropout=0.0, target_modules=[...], use_gradient_checkpointing="unsloth", random_state=3407)` | attach LoRA to all 7 linear layer types; gradient checkpointing recomputes activations to save memory |
| `model.print_trainable_parameters()` | prints about 42M trainable of 8B |
| `def to_text(batch)` | turns each messages list into one text with Llama's chat template |
| `.removeprefix(tokenizer.bos_token)` | removes the leading <|begin_of_text|> because the trainer adds it again; without this every example would start with TWO begin tokens (this bug was found while testing this manual) |
| `load_dataset("json", data_files=train_file, split="train")` | read the .jsonl into a table |
| `train.select(range(...))` | keep the first N rows |
| `val = ... .select(range(100))` | 100 validation rows are enough to watch the validation loss |
| `train.map(to_text, batched=True, remove_columns=...)` | add the "text" column, drop the columns the trainer does not need |
| `print(train[0]["text"][:600])` | shows one formatted example before training starts |
| `SFTConfig(output_dir=..., dataset_text_field="text", max_length=2048, ...)` | where checkpoints go, which column has the text, maximum length |
| `lr_scheduler_type="cosine"`, `warmup_steps=10` | LR rises for 10 steps, then decreases smoothly to almost 0 |
| `optim="adamw_8bit"`, `weight_decay=0.01`, `bf16=True` | memory-saving 8-bit optimizer, mild regularisation, 16-bit maths |
| `logging_steps=10`, `eval_strategy="steps"`, `eval_steps=100` | print loss every 10 steps, validation loss every 100 |
| `save_steps=200`, `save_total_limit=2` | checkpoint every 200 steps, keep only 2 |
| `report_to="none"`, `seed=3407` | no external tracker; fixed data order |
| `SFTTrainer(model=model, processing_class=tokenizer, ...)` | processing_class = the tokenizer (new name in TRL 1.x) |
| `train_on_responses_only(trainer, instruction_part=..., response_part=...)` | finds the user and assistant headers and sets labels to -100 everywhere except the answer |
| `stats = trainer.train()` | the training loop |
| `model.save_pretrained(str(out_dir))`, `tokenizer.save_pretrained(...)` | save ONLY the adapter + tokenizer |
| `train_log.json` | every logged loss and LR value, for your report |
CHECKPOINT: ls scripts shows 10_train_sft.py

## SECTION 4.8 - Run your first training
WHERE: VS Code Terminal, new tmux session
GOAL: outputs/adapters/sft-qa with a trained adapter.
Step 1. Start a tmux session:
```
tmux new -s train
```
Step 2. Activate and go to scripts:
```
cd ~/medscriptai/scripts && source ../.venv-train/bin/activate
```
Step 3. Start training on 2,000 QA examples:
```
python 10_train_sft.py --task qa --max-train 2000
```
You should see, in this order:
```
==((====))==  Unsloth 2026.10.7: Fast Llama patching. ... GPU: NVIDIA A100 80GB PCIe ...
trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196   (numbers may differ slightly)
<|start_header_id|>system<|end_header_id|> ... (one formatted example)
  1%|          | 1/125 [...]
{'loss': 1.9..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.08}
...
{'eval_loss': 0.9..., ...}
DONE adapter=/home/ubuntu/medscriptai/outputs/adapters/sft-qa train_loss=0.8...
```
(The loss numbers are illustrative: a healthy run starts around 1.5-2.5 and ends around 0.6-1.0.)
Step 4. Detach while it runs: Ctrl+B, then D.
Step 5. Open a second VS Code terminal (click + in the terminal panel) and watch the GPU:
```
watch -n 5 nvidia-smi
```
You should see: a python process using roughly 10-16 GB and GPU-Util 90-100%. Press Ctrl+C to stop watching.
Step 6. Re-attach to read the end: tmux attach -t train
How to read the log:
| What you see | Meaning | Action |
| loss falls steadily | learning | none |
| loss is "nan" or jumps above 5 | learning rate too high or broken data | stop (Ctrl+C), rerun with --lr 1e-4 |
| eval_loss rises twice in a row while loss falls | overfitting | use fewer epochs or more data |
| CUDA out of memory | batch too large for the GPU | add --batch 4 --grad-accum 4 |
CHECKPOINT: ls ../outputs/adapters/sft-qa shows adapter_config.json, adapter_model.safetensors, tokenizer files and train_log.json.
If not: re-attach (tmux attach -t train) and read the last error line; see Part 10.

## SECTION 4.9 - Create and run scripts/11_chat_test.py (base vs fine-tuned)
WHERE: VS Code Explorer (create), then VS Code Terminal (run) with (.venv-train)
GOAL: see with your own eyes what fine-tuning changed.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 11_chat_test.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/11_chat_test.py
| Line | What it does (plain English) |
| `FastLanguageModel.from_pretrained(adapter, ...)` | pointing at the ADAPTER folder: Unsloth reads adapter_config.json, loads the 4-bit base and attaches your LoRA |
| `FastLanguageModel.for_inference(model)` | switch Unsloth to its fast generation mode |
| `apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)` | format system + user and end with the assistant header so the model starts answering |
| `model.generate(**ids, max_new_tokens=400, do_sample=False)` | greedy decoding: always the most likely token, repeatable |
| `out[0][ids["input_ids"].shape[1]:]` | keep only the new tokens (cut the prompt) |
| `prompt = row["messages"][:2]` | system + user; the reference answer is hidden |
| `with model.disable_adapter():` | temporarily switch LoRA OFF = the plain base model |
| `torch.cuda.empty_cache()` | free cached GPU memory between examples |
Step 6. Run:
```
python 11_chat_test.py --task qa --n 3
```
You should see: three blocks with USER, BASE, FINE-TUNED and REFERENCE. Usually BASE writes a long free answer that often does not start with "Answer:", and FINE-TUNED starts exactly with "Answer: <letter>) <option>".
Watch out: three examples prove the adapter works, not that it is better. Real measurement is Part 5.
CHECKPOINT: the FINE-TUNED answers follow the "Answer: X) ..." format.

## SECTION 4.10 - Create and run scripts/12_push_to_hub.py (publish)
WHERE: VS Code Explorer (create), then VS Code Terminal (run) with (.venv-train)
GOAL: your adapter in a private repo on huggingface.co.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 12_push_to_hub.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/12_push_to_hub.py
| Line | What it does (plain English) |
| `from huggingface_hub import HfApi` | the official Hub client; it reads HF_TOKEN from your environment |
| `--folder`, `--repo`, `--type`, `--public` | what to upload, where, model or dataset, visibility (private by default) |
| `api.create_repo(..., exist_ok=True)` | create the repo; no error if it exists |
| `api.upload_folder(..., ignore_patterns=[...])` | upload, skipping training leftovers |
Step 6. Run it (replace YOUR_HF_USERNAME):
```
python 12_push_to_hub.py --folder ../outputs/adapters/sft-qa --repo YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora
```
You should see: https://huggingface.co/YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora
Step 7. Open that URL in your browser.
You should see: adapter_config.json and adapter_model.safetensors, and the label Private.
| Placeholder | Replace with | Example |
| YOUR_HF_USERNAME | your Hugging Face user name | rahulml |
Watch out: the name must start with "Llama" (Llama 3.1 licence) [15].
CHECKPOINT: the Hub page shows your adapter files.
If not: 401 Unauthorized = token missing or read-only (make a Write token, Section 2A.3, and repeat Section 2B.4 Step 3-4).

## SECTION 4.11 - First cloud step: copy the adapter to object storage (S3)
BEFORE YOU START: S3 is AWS's file storage ("buckets" hold "objects" = files). Production systems keep model files in S3 so any server can download them. You practise on FakeCloud first; the blue column is the same on real AWS.
WHERE: VS Code Terminal, folder ~/medscriptai (type cd ~/medscriptai first)
GOAL: the adapter in a bucket.
| Step | FakeCloud (free practice) {w=70} | Real AWS (same step) {blue} |
| 1 Point the CLI | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL` (your aws configure keys are used) |
| 2 Unique bucket name | `export BUCKET=medscriptai-artifacts-$(aws sts get-caller-identity --query Account --output text)-us-east-1` | same command (your account number makes it unique worldwide) |
| 3 Create the bucket | `aws s3 mb s3://$BUCKET` | `aws s3 mb s3://$BUCKET --region us-east-1` |
| 4 Upload the adapter | `aws s3 sync outputs/adapters/sft-qa s3://$BUCKET/adapters/sft-qa` | same command |
| 5 List | `aws s3 ls s3://$BUCKET --recursive --human-readable` | same command |
| You should see | adapters/sft-qa/adapter_model.safetensors with its size | the same; also in AWS console > S3 > your bucket |
| Cost | US$0 | about US$0.023 per GB-month (verify): under 1 cent per month |
What each piece means:
| Piece | What it does |
| `aws s3 mb` | make bucket |
| `aws s3 sync SRC DST` | copy only new or changed files (safe to repeat) |
| `--query Account --output text` | print just one field as plain text |
| `$(...)` | run the inner command and paste its output |
CHECKPOINT: aws s3 ls shows the adapter in your column's bucket.
If not: NoSuchBucket = echo $BUCKET is empty in this terminal; repeat Step 2.

## SECTION 4.12 - Save your progress in Git
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: a commit with the Part 4 scripts.
Step 1. Run:
```
cd ~/medscriptai && git add . && git commit -m "Part 4: data pipeline and first SFT adapter"
```
You should see: "6 files changed" (or similar). data/ and outputs/ are not committed (they are in .gitignore).
CHECKPOINT: git log --oneline shows two commits.

## PART 4 RESULT CHECK
| Check | Command | Expected |
| raw data | ls data/raw \| wc -l | 7 |
| chat data valid | cd scripts && python 03_validate_jsonl.py \| tail -1 | ALL_OK |
| adapter exists | ls outputs/adapters/sft-qa | adapter_config.json adapter_model.safetensors ... |
| adapter changes behaviour | python 11_chat_test.py --task qa --n 3 | FINE-TUNED starts with "Answer:" |
| published | open your Hub URL | adapter files, Private |
| in (Fake)S3 | aws s3 ls s3://$BUCKET --recursive | adapters/sft-qa/... |

## 4.13 Common mistake at this level
The mistake: training on text formatted differently from what the server sends later (two <|begin_of_text|> tokens, a different system prompt, or a missing chat template). The model then underperforms and you blame the hyper-parameters.
How to spot it: print one formatted training example (10_train_sft.py does this), print what your API sends (Part 6A), and compare them. This project avoids it by importing SYSTEM from scripts/common.py everywhere and by the removeprefix fix.
