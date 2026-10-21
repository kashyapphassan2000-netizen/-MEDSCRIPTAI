# PART 4 - BASICS: your first medical fine-tune (data -> .jsonl -> QLoRA SFT -> test -> publish)

## 4.0 Concept
What: download four open medical datasets, clean them, convert them into chat-format .jsonl, and train your first LoRA adapter (qa) on 2,000 exam questions with Unsloth QLoRA.
Why: this is the smallest complete loop of post-training. Every later part reuses these exact files and scripts at larger scale.
When: always start a new domain project this way: a small, fast run proves the data format, the template and the GPU setup before you spend hours of GPU money.
Cost and time: about 1.5 GPU-hours on 1x A100 (about US$2). Training itself takes roughly 10-15 minutes (estimate; your log shows the real speed).

## SECTION 4.1 - Download the datasets
WHERE: VS Code Terminal on the VM, folder ~/-MEDSCRIPTAI
GOAL: seven raw .jsonl files in data/raw, untouched copies of the sources.
Step 1. Start a tmux session so the job survives a disconnect:
```
tmux new -s data
```
Step 2. Activate the training environment:
```
cd ~/-MEDSCRIPTAI && source .venv-train/bin/activate
```
Step 3. Move into the scripts folder:
```
cd scripts
```
Step 4. Run the download script:
```
python 01_download_data.py
```
You should see (after 2-5 minutes) seven lines like: medqa_train 10178 rows <- GBaker/MedQA-USMLE-4-options [train], medmcqa_train 20000 rows, medmcqa_dpo 3000 rows, medmcqa_val 4183 rows, pubmedqa 1000 rows, clinical_notes 12000 rows.
Step 5. Detach from tmux (the job keeps running): press Ctrl+B, release, then press D. Re-attach later with tmux attach -t data.
FILE scripts/common.py (shared settings used by every script):
@@include scripts/common.py
| Line | What it does (plain English) |
| `import json`, `import os`, `from pathlib import Path` | standard Python tools: JSON text, environment variables, file paths that work on every OS |
| `ROOT = Path(__file__).resolve().parent.parent` | the repo folder = two levels above this file (scripts/common.py -> repo) |
| `DATA`, `RAW`, `SFT`, `PREF`, `OUT` | one variable per folder so no script hard-codes paths |
| `BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Llama-3.1-8B-Instruct")` | the bf16 base model; you can override it with an environment variable without editing code |
| `BASE_MODEL_4BIT = ...-bnb-4bit` | the same model already quantised to 4-bit NF4 by Unsloth (5.7 GB download instead of 16 GB) |
| `SYSTEM = {...}` | one system prompt per task. Each task = one adapter. The API (Part 6) imports this SAME dict, so training and serving prompts can never differ |
| `def write_jsonl(path, rows)` | writes a list of dicts, one JSON object per line; `mkdir(parents=True, exist_ok=True)` creates missing folders |
| `json.dumps(row, ensure_ascii=False)` | turns a dict into JSON text; ensure_ascii=False keeps symbols like °C readable |
| `print(f"wrote {len(rows):>6} rows -> ...")` | `:>6` right-aligns the number in 6 characters so the log lines up |
| `def read_jsonl(path)` | reads the file back; `if line.strip()` skips empty lines |
| `def chat(task, user, assistant=None)` | builds the messages list: system, user, and (for training rows) the assistant answer. This is the conversational format TRL and vLLM understand |
FILE scripts/01_download_data.py:
@@include scripts/01_download_data.py
| Line | What it does (plain English) |
| `import argparse` | lets you pass options such as --limit on the command line |
| `from datasets import load_dataset` | Hugging Face's loader: downloads once, caches in HF_HOME, returns a table |
| `from common import RAW` | the data/raw folder from common.py |
| `SOURCES = [...]` | the shopping list: Hub name, config, split, output file name, and which rows to keep |
| `("GBaker/MedQA-USMLE-4-options", None, "train", "medqa_train", None)` | US medical licensing exam questions with 4 options (CC-BY-4.0); None = keep all rows |
| `("openlifescienceai/medmcqa", ..., (0, 20000))` | Indian medical entrance exam questions with explanations (Apache-2.0); keep rows 0-19,999 after shuffling |
| `(..., "medmcqa_dpo", (20000, 23000))` | 3,000 DIFFERENT MedMCQA questions kept aside for DPO in Part 6, so DPO prompts are new to the SFT model |
| `("qiaojin/PubMedQA", "pqa_labeled", ...)` | 1,000 research-abstract questions with yes/no/maybe answers (MIT) |
| `("AGBonnet/augmented-clinical-notes", ..., (0, 12000))` | 12,000 clinical notes with a doctor-patient conversation and a JSON summary (MIT) |
| `parser.add_argument("--limit", ...)` | optional tiny run for testing: keep only N rows per file |
| `RAW.mkdir(parents=True, exist_ok=True)` | create data/raw if needed; no error if it already exists |
| `ds = load_dataset(name, config, split=split)` | download (first time) or load from cache |
| `ds.shuffle(seed=42).select(range(rows[0], min(rows[1], len(ds))))` | shuffle with a fixed seed so the order is identical every run, then take a slice. Same seed = same order = the SFT slice and the DPO slice never overlap |
| `if args.limit: ds = ds.select(range(...))` | only used for quick tests |
| `ds.to_json(str(path), lines=True, force_ascii=False)` | save as .jsonl (lines=True means one object per line) |
| `print(f"{out_name:16} ...")` | progress line; `:16` pads the name to 16 characters |
Watch out: if you see "429 Too Many Requests", you are being rate-limited by the Hub: make sure HF_TOKEN is set (echo $HF_TOKEN) and run the script again; finished files are cached.
CHECKPOINT: ls ../data/raw shows clinical_notes.jsonl, medmcqa_dpo.jsonl, medmcqa_train.jsonl, medmcqa_val.jsonl, medqa_test.jsonl, medqa_train.jsonl, pubmedqa.jsonl.
If not: re-attach with tmux attach -t data and read the last error line.

## SECTION 4.2 - Clean the data and convert it to chat .jsonl
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: train/val/test .jsonl files for qa, notes and summary, plus held-out DPO prompts.
Step 1. Run:
```
python 02_build_sft_jsonl.py
```
You should see lines such as: wrote 300 rows -> data/sft/qa_val.jsonl, wrote ~30000 rows -> data/sft/qa_train.jsonl, wrote 500 rows -> data/sft/qa_test.jsonl, wrote ~2950 rows -> data/pref/qa_dpo.jsonl, then notes/summary test 200, val 200, dpo 1000 and train ~7000 each (exact numbers depend on how many rows pass the filters).
Step 2. Look at one training line in a readable way:
```
head -n 1 ../data/sft/qa_train.jsonl | python -m json.tool
```
You should see a JSON object with "task": "qa", "source", "answer_letter", "question_key" and "messages" holding three dicts with roles system, user and assistant. The assistant content starts with Answer: and a letter.
FILE scripts/02_build_sft_jsonl.py:
@@include scripts/02_build_sft_jsonl.py
| Line | What it does (plain English) |
| `import json, random, re` | JSON parsing, repeatable random sampling, regular expressions for text clean-up |
| `from common import RAW, SFT, PREF, chat, read_jsonl, write_jsonl` | folders and helpers from common.py |
| `random.seed(42)` | fixes the random generator, so the train/test split is identical every time (reproducibility) |
| `MAX_CHARS = 7000` | about 1,750 tokens (rule of thumb: 1 token is about 4 characters of English); keeps every example under max_seq_length 2048 |
| `LETTERS = "ABCD"` | maps answer index 0-3 to a letter |
| `def clean(text)` | turns None into "", squeezes 3+ newlines into 2 and repeated spaces/tabs into 1, trims the ends |
| `def key(text)` | a fingerprint for de-duplication: lower-case, keep only letters and digits, first 300 characters. "What is X?" and "what is x" get the same key |
| `def mcq_prompt(question, options)` | builds the user message: question, blank line, "Options:", then lines like "A) Ampicillin". The API expects users to send this same shape |
| `def short(text, limit=900)` | shortens long explanations at the last full stop before 900 characters, so targets stay focused |
| `def from_medqa(row)` | converts one MedQA row |
| `opts = sorted(row["options"].items())` | the options dict {"A": ..., "B": ...} becomes a sorted list of (letter, text) pairs |
| `answer = f"Answer: {letter}) {clean(row['answer'])}"` | the target answer, e.g. Answer: D) Nitrofurantoin. MedQA has no explanation; Part 5 adds one with a teacher model |
| `return {"task": ..., "answer_letter": ..., "question_key": ..., "messages": chat(...)}` | one training example; answer_letter is kept for scoring accuracy later, question_key for de-duplication |
| `def from_medmcqa(row)` | converts one MedMCQA row |
| `if row.get("choice_type") != "single" or row.get("cop") not in (0, 1, 2, 3): return None` | drops multi-answer questions and rows with a missing correct option (cop = correct option index) |
| `opts = list(zip(LETTERS, [row["opa"], ...]))` | pairs A-D with option texts opa-opd |
| `if row.get("exp") and len(clean(row["exp"])) > 40` | adds the expert explanation only when it is real text (more than 40 characters) |
| `def from_pubmedqa(row)` | converts PubMedQA: the abstract becomes "Context:", the answer options are A) yes B) no C) maybe, and the long_answer becomes the explanation |
| `def compact(value)` | recursively removes empty fields ("None", "", [], {}) from the summary JSON. The raw summaries are about 70% "None" values; removing them makes targets short and teaches the model to output only facts that exist |
| `def from_clinical(row)` | one clinical row gives TWO examples: notes (conversation -> note) and summary (note -> JSON) |
| `summary = compact(json.loads(row["summary"]))` inside try/except | about 25% of the raw summaries are not valid JSON; those rows are skipped (you measured this; never train on broken targets) |
| `json.dumps(summary, ensure_ascii=False)` | the cleaned summary as one compact JSON string (the assistant target) |
| `def fits(example)` | True if the total characters of all messages are at most MAX_CHARS |
| `def build(rows, fn)` | applies a converter to every row and drops the None results |
| `def sample(rows, n)` | random n rows, or all rows if there are fewer |
| `qa_test = (sample(...medqa_test..., 200) + sample(...medmcqa_val..., 200))` | the QA test set comes from the OFFICIAL test/validation splits only: 200 MedQA + 200 MedMCQA |
| `pubmed = build(...); random.shuffle(pubmed); qa_test += pubmed[:100]` | 100 PubMedQA rows go to test; the other 900 can be used for training |
| `qa_pool = (...)` | every training candidate: MedQA train + MedMCQA train + 900 PubMedQA |
| `test_keys = {x["question_key"] for x in qa_test}` | fingerprints of every test question |
| `for ex in qa_pool: if k in seen or k in test_keys or not fits(ex): continue` | skip exact duplicates, skip anything that matches a TEST question (decontamination), skip examples that are too long |
| `write_jsonl(SFT / "qa_val.jsonl", qa_clean[:300])` | 300 rows for validation loss during training; the rest is the training set |
| `dpo_qa = [...] write_jsonl(PREF / "qa_dpo.jsonl", dpo_qa)` | 3,000 unseen MedMCQA prompts for DPO, also checked against train and test |
| `pairs = [p for p in build(... from_clinical) if fits(p[0]) and fits(p[1])]` | (notes, summary) pairs that are valid and short enough |
| `splits = {"test": pairs[:200], "val": pairs[200:400], "dpo": pairs[400:1400], "train": pairs[1400:]}` | ONE shared split for both tasks, so a note that is in the summary test set is never a notes training example |
| `folder = PREF if split == "dpo" else SFT` | DPO prompts go to data/pref, everything else to data/sft |
Watch out: the assistant answer format ("Answer: X) text") is a CONTRACT. The evaluation script and the API parse it. If you change it here, change it everywhere.
CHECKPOINT: ls ../data/sft shows nine files (qa, notes, summary x train, val, test) and ls ../data/pref shows qa_dpo.jsonl, notes_dpo.jsonl, summary_dpo.jsonl.
If not: "FileNotFoundError: data/raw/..." means Section 4.1 did not finish; re-run it.

## SECTION 4.3 - Validate every .jsonl file before paying for GPU time
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: prove every line is valid chat data and fits the 2048-token window.
Step 1. Run:
```
python 03_validate_jsonl.py
```
You should see one line per file such as qa_train.jsonl rows= 30000 tokens median=314 max=934, notes_train.jsonl ... median=1130 max=1520, and finally ALL_OK.
FILE scripts/03_validate_jsonl.py:
@@include scripts/03_validate_jsonl.py
| Line | What it does (plain English) |
| `ROLES = ["system", "user", "assistant"]` | the only allowed message order |
| `from transformers import AutoTokenizer` inside try | loads Llama-3.1's real tokenizer to count tokens; if it cannot load, structure is still checked |
| `for path in sorted(Path(SFT).glob("*.jsonl"))` | every .jsonl file in data/sft |
| `for n, line in enumerate(path.open(...), start=1)` | read line by line with a 1-based line number for error messages |
| `row = json.loads(line)` | check 1: valid JSON |
| `assert roles == ROLES` | check 2: exactly system -> user -> assistant |
| `assert all(m["content"].strip() ...)` | check 3: no empty message |
| `if row["task"] == "summary": json.loads(...)` | check 4: summary targets are valid JSON |
| `ids = tok.apply_chat_template(row["messages"], tokenize=True)` | check 5: the exact token count after Llama's chat template (what the trainer will see) |
| `ids = ids["input_ids"] if ... else ids` | newer transformers return a dict, older return a list; this line handles both |
| `stats = f" tokens median=... max=..."` | median and maximum length per file |
| `print("ALL_OK" ...); sys.exit(1 if errors else 0)` | exit code 0 = success, 1 = failure (useful in scripts and CI) |
Watch out: "apply_chat_template requires jinja2" means jinja2 is missing; run uv pip install jinja2 (it is in requirements/train.txt, so this only happens in a hand-made environment).
CHECKPOINT: the last line is ALL_OK and every max is below 2048.
If not: the ERROR line names the file and line number; open it with sed -n 'LINEp' ../data/sft/FILE.

## SECTION 4.4 - Train your first adapter (QLoRA SFT on 2,000 QA examples)
WHERE: VS Code Terminal, new tmux session, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: outputs/adapters/sft-qa containing a trained LoRA adapter.
Step 1. Start a new tmux session:
```
tmux new -s train
```
Step 2. Activate the environment and go to the scripts folder:
```
cd ~/-MEDSCRIPTAI && source .venv-train/bin/activate && cd scripts
```
Step 3. Start training on the first 2,000 examples:
```
python 10_train_sft.py --task qa --max-train 2000
```
You should see, in order: the Unsloth banner with your GPU name; "Unsloth: Will patch your computer"; a line like trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196; one formatted example starting with <|start_header_id|>system<|end_header_id|>; then a progress bar with about 125 steps and lines like {'loss': 1.9, ...} every 10 steps.
Step 4. In a SECOND VS Code Terminal (click + in the terminal panel), watch the GPU:
```
watch -n 5 nvidia-smi
```
You should see: python using about 10-16 GB of the 80 GB and GPU-Util at 90-100%. Press Ctrl+C to stop watching.
Step 5. When training ends, read the last line:
You should see: DONE adapter=/home/ubuntu/-MEDSCRIPTAI/outputs/adapters/sft-qa train_loss=0.8xxx (your number will differ; a falling loss from about 1.5-2.5 down to about 0.6-1.0 is healthy).
FILE scripts/10_train_sft.py:
@@include scripts/10_train_sft.py
| Line | What it does (plain English) |
| `from unsloth import FastLanguageModel` (first import) | Unsloth must be imported BEFORE trl/transformers so it can patch them with its fast kernels |
| `from unsloth.chat_templates import train_on_responses_only` | the helper that masks system/user tokens so loss is computed only on the answer |
| `from trl import SFTConfig, SFTTrainer` | Hugging Face TRL's supervised fine-tuning trainer and its settings |
| `p.add_argument("--task", required=True, choices=[...])` | which adapter to train; choices prevents typos |
| `--max-train` | use only the first N training rows (2,000 in Part 4) |
| `--epochs 1.0` | one pass over the data; more passes risk memorisation |
| `--max-steps -1` | -1 = stop by epochs; a number = stop after that many optimizer steps (handy for tests) |
| `--lr 2e-4` | learning rate; 2e-4 is the standard starting point for QLoRA (the QLoRA paper and Unsloth defaults) |
| `--rank 16` | LoRA rank r: width of the adapter matrices. 8-16 for focused tasks, 32-64 for big behaviour changes |
| `--batch 8`, `--grad-accum 2` | 8 examples per GPU step, gradients summed over 2 steps = effective batch 16 |
| `--train-file` | train on another file (Part 5 uses qa_train_synth.jsonl) |
| `run = a.run_name or f"sft-{a.task}"` | folder name, e.g. sft-qa |
| `FastLanguageModel.from_pretrained(model_name=BASE_MODEL_4BIT, ...)` | downloads (once) and loads Llama-3.1-8B-Instruct in 4-bit NF4 plus its tokenizer |
| `max_seq_length=2048` | longest sequence the model will be fed; our longest example is about 1,600 tokens |
| `load_in_4bit=True` | QLoRA: the frozen base weights stay 4-bit, saving about 10 GB of VRAM |
| `dtype=None` | auto-detect: bfloat16 on A100/L40/H100 |
| `FastLanguageModel.get_peft_model(model, ...)` | wraps the model with LoRA adapters (PEFT) |
| `r=a.rank`, `lora_alpha=a.rank` | rank 16 and alpha 16: the update is scaled by alpha/r = 1.0 |
| `lora_dropout=0.0` | no dropout inside LoRA; Unsloth is fastest at 0 and short runs rarely overfit through LoRA dropout |
| `target_modules=["q_proj", ..., "down_proj"]` | put adapters on all 4 attention projections and all 3 MLP projections; adapting all linear layers beats attention-only (QLoRA paper) |
| `use_gradient_checkpointing="unsloth"` | do not store all activations; recompute them in the backward pass. Saves VRAM, costs about 20% speed |
| `random_state=3407` | seed for the adapter's random initialisation |
| `model.print_trainable_parameters()` | prints about 42M trainable of 8B: proof that only LoRA is trained |
| `def to_text(batch)` | turns each messages list into one string with Llama's chat template |
| `tokenizer.apply_chat_template(m, tokenize=False)` | produces <|begin_of_text|><|start_header_id|>system<|end_header_id|> ... <|eot_id|> text |
| `.removeprefix(tokenizer.bos_token)` | removes the leading <|begin_of_text|> because the trainer adds it again during tokenisation. Without this you get TWO begin tokens (we caught this bug while testing) |
| `load_dataset("json", data_files=train_file, split="train")` | reads the .jsonl into a Dataset ("train" is just the default split name for a single file) |
| `train.select(range(min(a.max_train, len(train))))` | keep the first N rows |
| `val = ... .select(range(100))` | 100 validation rows are enough to watch eval loss and keep evaluation fast |
| `train.map(to_text, batched=True, remove_columns=train.column_names)` | add the "text" column and drop the original columns (task, source, ...) the trainer does not need |
| `print(train[0]["text"][:600])` | eyeball one example: the template must look right BEFORE you spend GPU time |
| `SFTConfig(output_dir=...)` | where checkpoints are written (outputs/checkpoints/sft-qa) |
| `dataset_text_field="text"` | the column that holds the full formatted conversation |
| `max_length=2048` | truncate anything longer (nothing is, thanks to the validator) |
| `per_device_train_batch_size=a.batch`, `gradient_accumulation_steps=a.grad_accum` | batch 8 x 2 = 16 examples per optimizer step; 2,000 / 16 = 125 steps |
| `num_train_epochs`, `max_steps` | stop condition |
| `learning_rate=a.lr` | 2e-4 |
| `lr_scheduler_type="cosine"` | LR follows half a cosine wave down to about 0 at the end: smooth, stable finish |
| `warmup_steps=10` | the LR ramps up for the first 10 steps so early random updates do not wreck the model |
| `optim="adamw_8bit"` | AdamW with optimizer states stored in 8-bit (bitsandbytes): big memory saving, same quality |
| `weight_decay=0.01` | a small pull of weights towards zero: mild regularisation |
| `bf16=True` | compute in bfloat16 |
| `logging_steps=10` | print loss every 10 steps |
| `eval_strategy="steps", eval_steps=100` | compute validation loss every 100 steps: if it rises while train loss falls, you are overfitting |
| `save_steps=200, save_total_limit=2` | checkpoint every 200 steps, keep only the newest 2 (disk safety) |
| `report_to="none"` | do not send metrics to W&B/Trackio (add later if you like) |
| `seed=3407` | data order seed |
| `SFTTrainer(model=..., processing_class=tokenizer, ...)` | processing_class is the new name for the tokenizer argument in TRL 1.x |
| `train_on_responses_only(trainer, instruction_part=..., response_part=...)` | finds the user header and the assistant header in each tokenised example and sets labels to -100 for everything except the assistant's answer |
| `"<|start_header_id|>assistant<|end_header_id|>\n\n"` | Llama-3.1's exact marker where the answer starts; for another model family these strings are different |
| `stats = trainer.train()` | the training loop: forward pass, loss, backward pass, optimizer step, repeat |
| `model.save_pretrained(str(out_dir))` | saves ONLY adapter_model.safetensors (~80-170 MB) and adapter_config.json, which records the base model name |
| `tokenizer.save_pretrained(str(out_dir))` | saves the tokenizer next to the adapter so the folder is self-contained |
| `(out_dir / "train_log.json").write_text(...)` | every logged loss/LR value, for your report and plots |
Watch out: "CUDA out of memory" means batch too large for your GPU (for example on a 24 GB card): add --batch 4 --grad-accum 4 (same effective batch 16).
Watch out: if the eval loss goes UP for two evaluations in a row while train loss goes down, stop: you are overfitting. Use fewer epochs or more data.
CHECKPOINT: ls ../outputs/adapters/sft-qa shows adapter_config.json, adapter_model.safetensors, tokenizer files and train_log.json.
If not: re-attach with tmux attach -t train and read the error; the Troubleshooting part (Part 10) lists the common ones.

## SECTION 4.5 - Compare the base model and your adapter
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: see with your own eyes what fine-tuning changed.
Step 1. Run:
```
python 11_chat_test.py --task qa --n 3
```
You should see three blocks with USER, BASE, FINE-TUNED and REFERENCE. Typically BASE writes a long free-form answer and often does not start with "Answer:"; FINE-TUNED starts with "Answer: <letter>) <option>" exactly like the training data.
FILE scripts/11_chat_test.py:
@@include scripts/11_chat_test.py
| Line | What it does (plain English) |
| `FastLanguageModel.from_pretrained(adapter, ...)` | pointing at an ADAPTER folder: Unsloth reads adapter_config.json, loads the 4-bit base model, and attaches your LoRA |
| `FastLanguageModel.for_inference(model)` | switches Unsloth to its faster generation mode (about 2x) |
| `def answer(messages)` | generates one reply |
| `apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)` | formats system + user and ends with the assistant header so the model starts answering; returns PyTorch tensors |
| `.to("cuda")` | moves the tensors to the GPU |
| `model.generate(**ids, max_new_tokens=400, do_sample=False)` | greedy decoding: always the most likely token, so results are repeatable |
| `out[0][ids["input_ids"].shape[1]:]` | cut away the prompt tokens, keep only the new answer |
| `skip_special_tokens=True` | hide <|eot_id|> and similar tokens |
| `prompt = row["messages"][:2]` | system + user only; the reference answer is hidden from the model |
| `with model.disable_adapter():` | temporarily switches LoRA OFF: the same object behaves like the plain base model |
| `tuned = answer(prompt)` | outside the with-block LoRA is ON again |
| `torch.cuda.empty_cache()` | frees cached GPU memory between examples |
Watch out: three examples prove nothing statistically; they only prove the adapter loads and changed behaviour. Real measurement comes in Part 5.
CHECKPOINT: the FINE-TUNED answers follow the "Answer: X) ..." format.
If not: if FINE-TUNED looks identical to BASE, check that the adapter path is correct and that train_loss went down.

## SECTION 4.6 - Publish the adapter on the Hugging Face Hub
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: a private model repo holding your adapter (your first production artifact).
Step 1. Run (replace YOUR_HF_USERNAME):
```
python 12_push_to_hub.py --folder ../outputs/adapters/sft-qa --repo YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora
```
You should see: a URL like https://huggingface.co/YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora
Step 2. Open the URL in your browser.
You should see: adapter_config.json and adapter_model.safetensors (about 80-170 MB), and the repo marked Private.
FILE scripts/12_push_to_hub.py:
@@include scripts/12_push_to_hub.py
| Line | What it does (plain English) |
| `from huggingface_hub import HfApi` | the official Hub client; it reads HF_TOKEN from your environment |
| `--folder`, `--repo`, `--type`, `--public` | what to upload, where, model or dataset, and visibility (private by default) |
| `api.create_repo(a.repo, repo_type=a.type, private=not a.public, exist_ok=True)` | creates the repo; exist_ok=True means no error if it already exists |
| `api.upload_folder(..., ignore_patterns=["checkpoint-*", "*.pt", "optimizer*"])` | uploads the folder but skips training leftovers such as optimizer states |
| `print(f"https://huggingface.co/...")` | prints the repo address |
Placeholders:
| Placeholder | Replace with | Example |
| YOUR_HF_USERNAME | your Hugging Face user name | rahulml |
Watch out: the repo name must start with "Llama" (Llama 3.1 licence) [15].
CHECKPOINT: the Hub page shows the adapter files.
If not: "401 Unauthorized" = HF_TOKEN missing or read-only; create a Write token (Section 2.2) and run source ~/.bashrc.

## SECTION 4.7 - Your first cloud step: store the adapter in object storage (S3)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: the adapter copied to a bucket, practised free on FakeCloud, then (optionally) on real AWS.
Step 1. Follow the column you are working in, one row at a time.
| Step {w=70} | FakeCloud (free practice) | Real AWS (same step) {blue} |
| 1 Point the CLI | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL` (your keys from `aws configure` are used) |
| 2 Make a unique bucket name | `export BUCKET=medscriptai-artifacts-$(aws sts get-caller-identity --query Account --output text)-us-east-1` | same command (your real account number makes the name globally unique) |
| 3 Create the bucket | `aws s3 mb s3://$BUCKET` | `aws s3 mb s3://$BUCKET --region us-east-1` |
| 4 Upload the adapter | `aws s3 sync outputs/adapters/sft-qa s3://$BUCKET/adapters/sft-qa` | same command |
| 5 List what is there | `aws s3 ls s3://$BUCKET --recursive --human-readable` | same command |
| You should see | adapters/sft-qa/adapter_model.safetensors with its size | the same list; also visible in the AWS console > S3 > your bucket |
| Cost | US$0 | about US$0.023 per GB-month in us-east-1 (verify); a 170 MB adapter costs well under 1 cent per month |
What each command piece means:
| Command piece | What it does |
| `aws s3 mb` | "make bucket" |
| `aws s3 sync SRC DST` | copies only new or changed files (re-running is cheap and safe) |
| `--query Account --output text` | asks the CLI for just one field, as plain text |
| `$(...)` | runs the command inside and pastes its output into the outer command |
Watch out: bucket names are global across all AWS customers; that is why the account number is in the name.
CHECKPOINT: aws s3 ls shows the adapter in the bucket for the column you used.
If not: "NoSuchBucket" = Step 3 failed or BUCKET is empty in this terminal (echo $BUCKET).

## 4.8 Common mistake at this level
The mistake: training on text that is formatted differently from what the server sends at inference. Examples: two <|begin_of_text|> tokens (fixed in 10_train_sft.py), a different system prompt in the API, or a missing chat template. The model then underperforms and you blame the hyper-parameters.
How to spot it: print one formatted training example (10_train_sft.py does this), print the exact messages your API sends, and compare token by token. The repo avoids the problem by importing SYSTEM from one file everywhere.
