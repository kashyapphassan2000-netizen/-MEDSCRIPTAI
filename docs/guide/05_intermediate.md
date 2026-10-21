# PART 5 - INTERMEDIATE: synthetic data, three adapters, real evaluation

## BEFORE YOU START: what you will build in this part
In Part 4 you trained one adapter on 2,000 examples. Now you do what a real team does next: improve the DATA, scale the training, and MEASURE.
```
Section 5.1-5.2   start the TEACHER model server (Qwen2.5-14B-Instruct) on port 8001
Section 5.3-5.6   synthetic data: the teacher writes explanations for 10,000 MedQA answers (distilabel)
Section 5.7       train three adapters: sft-qa (with the explanations), sft-notes, sft-summary
Section 5.8       serve base + 3 adapters with vLLM on port 8000
Section 5.9-5.11  evaluate: accuracy, ROUGE-L, BERTScore, LLM judge
Section 5.12-5.14 publish datasets, store reports and secrets in (Fake)AWS, commit
```
New words in this part:
| Word | Plain meaning |
| model server | a program that keeps a model loaded on the GPU and answers requests over HTTP, like a website for a model |
| vLLM | the fastest popular open-source model server; it batches many requests together on the GPU |
| OpenAI-compatible API | vLLM accepts the same request format as OpenAI's API, so any OpenAI client library works with it |
| port | a numbered "door" on a computer; 8000 = your model, 8001 = the teacher, 8080 = your API (Part 6A) |
| teacher model | a bigger model whose answers are used to teach the smaller one |
| synthetic data | training examples written by a model instead of humans (see Part 3, section 3.7) |
| distilabel | a library that runs synthetic-data pipelines: load rows -> send to a model -> collect -> save |
| Docker container | a program running in an isolated box with its own files; vLLM runs in one |
Harder than Part 4 because: two model servers, a pipeline framework, three training runs, and metrics you must interpret correctly. Budget: about 6 GPU-hours (about US$8).

## SECTION 5.1 - Create serve/start_teacher.sh
WHERE: VS Code Explorer, folder serve
GOAL: a script that starts the teacher model server.
Step 1. Right-click the folder serve.
Step 2. Click New File...
Step 3. Type start_teacher.sh and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include serve/start_teacher.sh
| Line | What it does (plain English) |
| `#!/usr/bin/env bash` | "run this file with bash" |
| `set -euo pipefail` | stop at the first error, treat unset variables as errors, fail a pipeline if any part fails |
| `VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0-cu129}"` | the official vLLM image (version 0.32.0, CUDA 12.9 build). `${X:-default}` = use X if it is set, otherwise the default |
| `GPU_UTIL="${GPU_UTIL:-0.85}"` | fraction of GPU memory this server may use |
| `docker rm -f medscriptai-teacher 2>/dev/null \|\| true` | remove an old container with the same name; ignore the error if there is none |
| `docker run -d --name medscriptai-teacher` | start a container in the background with a fixed name |
| `--gpus all --ipc=host` | give it the GPU; share memory with the host (vLLM needs this) |
| `-p 8001:8000` | the server listens on 8000 inside the container; you reach it on port 8001 |
| `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` | your driver (570) supports CUDA 12.8 but the image uses 12.9; NVIDIA's compatibility libraries inside the image bridge this on data-centre GPUs (A100, L40, H100). We read this in vLLM's own Dockerfile |
| `-v ${HF_HOME}:/root/.cache/huggingface` | share the download cache, so the model downloads only once |
| `--model Qwen/Qwen2.5-14B-Instruct` | the teacher: 14.7B parameters, Apache-2.0 licence |
| `--served-model-name teacher` | clients ask for "teacher" instead of the long name |
| `--dtype bfloat16` | 16-bit weights, about 29.5 GB |
| `--max-model-len 8192` | longest prompt + answer in tokens |
| `--gpu-memory-utilization "$GPU_UTIL"` | vLLM reserves this share of the GPU and fills the spare part with the KV cache (its memory for running requests) |
Step 6. Make the script executable:
```
chmod +x ~/medscriptai/serve/start_teacher.sh
```
CHECKPOINT: ls -l ~/medscriptai/serve shows start_teacher.sh with -rwx at the start.

## SECTION 5.2 - Start the teacher and test it
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: the teacher answers on port 8001.
Step 1. Start it:
```
cd ~/medscriptai && bash serve/start_teacher.sh
```
You should see: a long container ID and "Started. Follow the logs with: docker logs -f medscriptai-teacher". The first start downloads the vLLM image (about 11 GB) and the model (about 30 GB): 10-15 minutes.
Step 2. Follow the start-up log:
```
docker logs -f medscriptai-teacher
```
You should see, at the end: "Application startup complete." Press Ctrl+C to stop FOLLOWING (the server keeps running).
Step 3. List the models it serves:
```
curl -s http://localhost:8001/v1/models | jq -r '.data[].id'
```
You should see: teacher
Step 4. Ask a test question:
```
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"teacher","messages":[{"role":"user","content":"In one sentence: why is nitrofurantoin used for cystitis in pregnancy?"}],"max_tokens":80}' | jq -r '.choices[0].message.content'
```
You should see: one sensible medical sentence.
| Piece | What it does |
| curl -s URL | send an HTTP request quietly |
| -H "Content-Type: application/json" | "my body is JSON" |
| -d '{...}' | the JSON body: which model, the messages, max answer length |
| jq -r '.choices[0].message.content' | pull the answer text out of the JSON reply |
| If you see... | It means... | Do this next |
| "CUDA driver version is insufficient" in the log | the compatibility layer was not used | check the VLLM_ENABLE_CUDA_COMPATIBILITY line, or use the R595 OS image with VLLM_IMAGE=vllm/vllm-openai:v0.32.0 |
| "No available memory for the cache blocks" | GPU memory is used by something else | nvidia-smi; stop other containers or training |
| curl: (7) Failed to connect | still loading | wait for "Application startup complete" |
CHECKPOINT: Step 3 prints teacher and Step 4 prints a sentence.

## SECTION 5.3 - Create scripts/20_synth_explanations.py (synthetic data)
BEFORE YOU START: how the distilabel pipeline works.
```
qa_train.jsonl (MedQA rows with only "Answer: D) Nitrofurantoin")
      |
      v
LoadDataFromDicts  -> sends rows in batches of 64
      |
      v
TextGeneration     -> for each row asks the teacher: "<question + options> The correct answer is: D) Nitrofurantoin.
      |               Explain step by step why it is correct and briefly why the other options are wrong."
      v
quality filter     -> keep only explanations of 80-1,500 characters that mention the correct option
      |
      v
qa_train_synth.jsonl (same rows, MedQA answers now end with "Explanation: ...")
```
The teacher is GIVEN the correct answer (rationale generation), so it explains rather than guesses. That is much more reliable than letting it answer by itself.
WHERE: VS Code Explorer, folder scripts
GOAL: the synthetic-data script.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 20_synth_explanations.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/20_synth_explanations.py
| Line | What it does (plain English) |
| `from distilabel.models import OpenAILLM` | distilabel's client for any OpenAI-compatible server (our vLLM teacher) |
| `from distilabel.pipeline import Pipeline` | a pipeline = connected steps that pass batches of rows along |
| `from distilabel.steps import LoadDataFromDicts` | first step: feeds a Python list of dicts into the pipeline |
| `from distilabel.steps.tasks import TextGeneration` | sends each row's "instruction" to the model and writes the reply into "generation" |
| `TEACHER_URL`, `TEACHER` | where the teacher listens (port 8001) and its served name |
| `TEACHER_SYSTEM = (...)` | instructions for the teacher: expert tone, at most 120 words, do not change the answer |
| `todo = [r for r in rows if r["source"] == "medqa" and "Explanation:" not in ...]` | only MedQA rows without an explanation (MedMCQA and PubMedQA already have one) |
| `"gold": r["messages"][2]["content"]` | the correct answer line, used by the quality filter |
| `"instruction": (... "The correct answer is: " ...)` | what the teacher receives: question + options + the correct answer + the request to explain |
| `with Pipeline(name=...) as pipeline:` | steps created inside this block belong to the pipeline |
| `LoadDataFromDicts(data=inputs, batch_size=64)` | sends 64 rows at a time |
| `OpenAILLM(model=TEACHER, base_url=TEACHER_URL, api_key="EMPTY", generation_kwargs={"temperature": 0.3, "max_new_tokens": 300})` | vLLM ignores the key; temperature 0.3 = focused; at most 300 new tokens |
| `system_prompt=TEACHER_SYSTEM` | added as the system message of every request |
| `input_batch_size=32` | 32 requests in flight at once; vLLM processes them together on the GPU |
| `load >> explain` | connect the two steps |
| `distiset = pipeline.run(use_cache=False)` | run everything; use_cache=False forces a fresh run |
| `distiset["default"]["train"]` | the output table of the last step |
| `def good(text, gold)` | the quality filter: 80-1,500 characters and it mentions the correct option text or letter |
| `explanations = {...}` | question_key -> explanation, only for rows that passed |
| `write_jsonl(DATA / "synth" / "medqa_explanations.jsonl", ...)` | the raw synthetic data, kept separately (you publish it in 5.12) |
| `r["messages"][2]["content"] += f"\nExplanation: ..."` | append the explanation to the answer |
| `write_jsonl(SFT / "qa_train_synth.jsonl", rows)` | the improved training file; qa_train.jsonl stays untouched for comparison |
CHECKPOINT: ls scripts shows 20_synth_explanations.py

## SECTION 5.4 - Dry run on 50 rows and inspect quality
WHERE: VS Code Terminal, tmux, folder ~/medscriptai/scripts, (.venv-synth) active
GOAL: prove the pipeline works and the explanations are good BEFORE the full run.
Step 1. Start tmux:
```
tmux new -s synth
```
Step 2. Activate the synthetic-data environment (NOT .venv-train):
```
cd ~/medscriptai/scripts && source ../.venv-synth/bin/activate
```
Step 3. Run on 50 rows (1-2 minutes):
```
python 20_synth_explanations.py --limit 50
```
You should see: "50 MedQA rows need an explanation", distilabel's colourful pipeline log, then "kept 4x/50 explanations after the quality filter" (most rows should pass).
Step 4. Read three random explanations:
```
shuf -n 3 ../data/synth/medqa_explanations.jsonl | python -m json.tool
```
You should see: three {"question_key": ..., "explanation": ...} objects with sensible medical reasoning that mentions the correct option.
Step 5. Judge them with this checklist (write yes/no for each):
| Question | Good sign |
| Does it support the GIVEN answer (not argue for another option)? | yes |
| Is it medically plausible (check one claim on Wikipedia or a textbook)? | yes |
| Is it under about 120 words and to the point? | yes |
| Does it avoid personal medical advice ("you should take...")? | yes |
Watch out: "No module named 'requests'" or datasets errors = wrong environment; the prompt must start with (.venv-synth).
CHECKPOINT: most of the 50 passed and your checklist is mostly "yes".
If not: if explanations argue for another option, the teacher is not following instructions: make TEACHER_SYSTEM stricter and repeat Step 3.

## SECTION 5.5 - Full synthetic run
WHERE: VS Code Terminal, tmux session synth, (.venv-synth) active
GOAL: explanations for all MedQA training rows.
Step 1. Run without --limit (roughly 15-30 minutes on an A100, estimate):
```
python 20_synth_explanations.py
```
You should see: "N MedQA rows need an explanation" (about 9,500-10,000), then "kept .../... explanations after the quality filter" and "wrote 23669 rows -> data/sft/qa_train_synth.jsonl".
Step 2. Count how many answers now have an explanation, before and after:
```
grep -c "Explanation:" ../data/sft/qa_train.jsonl && grep -c "Explanation:" ../data/sft/qa_train_synth.jsonl
```
You should see: the second number is about 9,000-10,000 larger than the first.
CHECKPOINT: qa_train_synth.jsonl exists with 23669 lines (wc -l ../data/sft/qa_train_synth.jsonl).

## SECTION 5.6 - Free the GPU
WHERE: VS Code Terminal
GOAL: no model server using GPU memory (training needs it).
Step 1. Stop the teacher:
```
docker rm -f medscriptai-teacher
```
Step 2. Check the GPU is free:
```
nvidia-smi --query-gpu=memory.used --format=csv
```
You should see: a small number such as 1 MiB.
CHECKPOINT: memory.used is near 0.

## SECTION 5.7 - Train the three adapters
WHERE: VS Code Terminal, new tmux session, folder ~/medscriptai/scripts, (.venv-train) active
GOAL: outputs/adapters/sft-qa (retrained with explanations), sft-notes, sft-summary.
Step 1. Start tmux:
```
tmux new -s train3
```
Step 2. Activate the TRAINING environment:
```
cd ~/medscriptai/scripts && source ../.venv-train/bin/activate
```
Step 3. Train all three, one after another (about 2.5-3.5 hours in total on an A100, estimate):
```
python 10_train_sft.py --task qa --train-file ../data/sft/qa_train_synth.jsonl --max-train 12000 && python 10_train_sft.py --task notes && python 10_train_sft.py --task summary
```
Step 4. Detach (Ctrl+B, then D). Come back with tmux attach -t train3.
| Run | Training rows | Steps (batch 16) | Rough time on A100 |
| qa (synthetic) | 12,000 | 750 | 45-70 min |
| notes | 6,367 | 398 | 40-60 min |
| summary | 6,367 | 398 | 40-60 min |
You should see: three "DONE adapter=..." lines. The qa run replaces the Part 4 sft-qa adapter (the Part 4 one is still on the Hugging Face Hub).
Watch out: && means "run the next command only if the previous one succeeded". If one fails, fix it and run only the missing ones.
CHECKPOINT: ls ../outputs/adapters shows sft-qa, sft-notes, sft-summary, each with adapter_model.safetensors.

## SECTION 5.8 - Create serve/start_vllm.sh and serve all adapters
BEFORE YOU START: multi-LoRA serving. vLLM loads the 16 GB base model ONCE and keeps your small adapters next to it. Each request names a model: "base" (no adapter), "qa", "notes" or "summary". vLLM applies the right adapter for each request, even inside the same GPU batch. One GPU, many specialised models.
WHERE: VS Code Explorer (create), then VS Code Terminal (run)
GOAL: base + adapters on port 8000, and the judge on port 8001, both on the A100.
Step 1. Right-click the folder serve.
Step 2. Click New File...
Step 3. Type start_vllm.sh and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include serve/start_vllm.sh
| Line | What it does (plain English) |
| `GPU_UTIL="${GPU_UTIL:-0.85}"` | share of GPU memory; you lower it when the judge runs next to it |
| `ADAPTERS="$(pwd)/outputs/adapters"` | your adapters folder (run the script from ~/medscriptai) |
| `for pair in qa=sft-qa notes=sft-notes ... ; do ... done` | for each name=folder pair, register the adapter only if its folder exists; so the script works after Part 4 (qa only) and after Part 6A (all five) |
| `name="${pair%%=*}"; dir="${pair#*=}"` | bash string cutting: the part before "=" and the part after "=" |
| `EXTRA=(--lora-modules "${LORA_ARGS[@]}")` | builds "--lora-modules qa=/adapters/sft-qa ..." only when adapters exist |
| `-e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` | allows adding/removing adapters while running (Part 6A). vLLM warns this is risky in production, so the production Compose file in Part 6B does NOT set it |
| `-v "$ADAPTERS":/adapters:ro` | your adapters appear inside the container at /adapters, read-only |
| `--model unsloth/Llama-3.1-8B-Instruct --served-model-name base` | the 16-bit base model, called "base" |
| `--max-model-len 4096` | longest context per request |
| `--enable-lora` | turn on LoRA support |
| `--max-loras 4` | how many DIFFERENT adapters can be active in one GPU batch. The default is 1, which makes requests for different adapters wait for each other: a classic production mistake |
| `--max-lora-rank 16` | must be at least the rank of your adapters (16); allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512 |
| `--max-cpu-loras 8` | how many adapters are cached in normal memory for fast swapping |
| `"${EXTRA[@]}"` | adds the --lora-modules list |
Step 6. Make it executable:
```
chmod +x ~/medscriptai/serve/start_vllm.sh
```
Step 7. Start the student server with 35% of the GPU:
```
cd ~/medscriptai && GPU_UTIL=0.35 bash serve/start_vllm.sh
```
You should see: Adapters: qa=/adapters/sft-qa notes=/adapters/sft-notes summary=/adapters/sft-summary
Step 8. Wait until it is ready, then list its models:
```
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done; curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary (one per line).
Step 9. Start the judge (the teacher) with 50% of the GPU:
```
GPU_UTIL=0.50 bash serve/start_teacher.sh
```
Step 10. Wait for it:
```
until curl -s localhost:8001/v1/models > /dev/null; do sleep 10; done; echo judge ready
```
Watch out: the two GPU_UTIL values must add up to about 0.9 or less, otherwise the second server says "insufficient GPU memory".
CHECKPOINT: Step 8 lists four models and Step 10 prints "judge ready".

## SECTION 5.9 - Create and run scripts/30_generate_preds.py (answers on the test set)
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: one predictions file per task and model in outputs/preds.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 30_generate_preds.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/30_generate_preds.py
| Line | What it does (plain English) |
| `from concurrent.futures import ThreadPoolExecutor` | send many requests at the same time; vLLM batches them |
| `from openai import OpenAI` | the official OpenAI client, pointed at vLLM |
| `--models` | names known to vLLM: "base" = no adapter, "qa" = base + qa adapter |
| `client = OpenAI(base_url=a.url, api_key="EMPTY")` | vLLM ignores the key, the client needs one |
| `rows = read_jsonl(SFT / f"{a.task}_test.jsonl")[: a.n]` | the held-out TEST set |
| `max_tokens = 400 if a.task == "qa" else 900` | answer length limit per task |
| `messages=row["messages"][:2]` | system + user only; the reference is never sent |
| `temperature=0.0` | greedy decoding: reproducible scores |
| `pool.map(lambda r: ask(model, r), rows)` | 16 parallel workers, results in the original order |
| `out = [{"i", "source", "answer_letter", "prompt", "reference", "prediction"}]` | everything the metrics and the judge need |
| `write_jsonl(OUT / "preds" / a.task / f"{model}.jsonl", out)` | e.g. outputs/preds/qa/base.jsonl |
Step 6. Activate the EVALUATION environment:
```
cd ~/medscriptai/scripts && source ../.venv-eval/bin/activate
```
Step 7. Generate (about 5-10 minutes):
```
python 30_generate_preds.py --task qa --models base qa && python 30_generate_preds.py --task notes --models base notes && python 30_generate_preds.py --task summary --models base summary
```
You should see: six lines such as "wrote    500 rows -> outputs/preds/qa/base.jsonl" and "wrote    200 rows -> outputs/preds/notes/notes.jsonl".
CHECKPOINT: ls ../outputs/preds/qa ../outputs/preds/notes ../outputs/preds/summary shows two files each.

## SECTION 5.10 - Create and run scripts/31_eval_metrics.py (automatic metrics)
BEFORE YOU START: the metrics with a tiny example.
| Metric | Reference | Prediction | Score (roughly) | Why |
| accuracy | Answer: C) ... | Answer: C) ... | 1 (correct) | same letter |
| ROUGE-L | "patient presented with acute kidney injury" | "patient had acute kidney injury" | about 0.8 | the longest common word sequence "patient ... acute kidney injury" is most of both |
| BERTScore F1 | "the patient had a heart attack" | "the patient suffered a myocardial infarction" | high (about 0.9) | different words, same meaning |
| json_valid | (summary task) | {"age": "40"} | 1 | it parses as JSON |
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: outputs/reports/metrics.json with every metric for every model.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 31_eval_metrics.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/31_eval_metrics.py
| Line | What it does (plain English) |
| `from rouge_score import rouge_scorer` | Google's reference ROUGE implementation |
| `from bert_score import score as bert_score` | BERTScore (Zhang et al., 2020) |
| `--bert-model roberta-large` | the standard BERTScore model for English (1.4 GB download on first use) |
| `RougeScorer(["rougeL"], use_stemmer=True)` | ROUGE-L; the stemmer treats "infections" and "infection" as the same |
| `def letter(text)` | the letter after "Answer:" |
| `def is_json(text)` | True if the text parses as JSON (also inside a ```json block) |
| `rouge.score(ref, pred)["rougeL"].fmeasure` | ROUGE-L F1 per row, then averaged |
| `bert_score(preds, refs, model_type=..., lang="en")` | BERTScore precision, recall, F1; we keep the mean F1 |
| `m["accuracy"]`, `m["format_ok"]` | share of correct letters; share of answers that used the "Answer: X)" format at all |
| `m["json_valid"]` | share of summary outputs that are valid JSON |
| `report.write_text(...)` | saves outputs/reports/metrics.json |
Step 6. Run:
```
python 31_eval_metrics.py
```
You should see: six lines such as
```
qa       base         {'n': 500, 'rougeL': ..., 'bertscore_f1': ..., 'accuracy': ..., 'format_ok': ...}
qa       qa           {'n': 500, 'rougeL': ..., 'bertscore_f1': ..., 'accuracy': ..., 'format_ok': ...}
...
saved /home/ubuntu/medscriptai/outputs/reports/metrics.json
```
How to read your numbers:
| Metric | What a good change looks like | Trap |
| format_ok (qa) | fine-tuned close to 1.0 | read it BEFORE accuracy: unparseable answers count as wrong |
| accuracy (qa) | fine-tuned higher than base | a few points can be noise: test it in Part 7 |
| ROUGE-L | fine-tuned higher on notes/summary | copying the reference style raises it even when facts are wrong |
| BERTScore F1 | values cluster around 0.80-0.90 | compare models with each other, never with a fixed "good" value |
| json_valid (summary) | fine-tuned close to 1.0 | valid JSON can still contain invented facts (the judge checks) |
CHECKPOINT: cat ../outputs/reports/metrics.json shows six entries.

## SECTION 5.11 - Create and run scripts/32_judge.py (LLM-as-a-judge)
BEFORE YOU START: an LLM judge is a strong model asked to grade answers against a rubric. In score mode it gives each answer 1-5. In pairwise mode (Part 6A) it compares two answers; because judges prefer whichever answer comes first, we ask twice with the order swapped and only count a win if it wins both times.
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: a 1-5 quality score per model on 100 test items per task.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 32_judge.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/32_judge.py
| Line | What it does (plain English) |
| `judge = OpenAI(base_url=os.getenv("JUDGE_URL", "http://localhost:8001/v1"), ...)` | the judge is the teacher server on port 8001 |
| `RUBRIC = {...}` | what "good" means per task; for notes, faithfulness (no invented facts) comes first |
| `def ask_json(prompt)` | asks at temperature 0 and pulls the first {...} out of the reply; {} if the judge broke the format |
| `def score_prompt(row)` | task input + reference + model answer + "Return ONLY a JSON object {score, reason}"; long texts are cut at 3,000-4,000 characters |
| `def pair_prompt(row, first, second)` | the pairwise version with ANSWER A and ANSWER B |
| `if a.mode == "score":` | grade each answer alone and average the valid scores |
| `def duel(i)` | pairwise: ask A-then-B and B-then-A |
| `if first == "A" and second == "B": return ma` | a model wins only if it is preferred in BOTH orders, otherwise "tie": this cancels position bias (Zheng et al., 2023) |
| `report = {... "per_item": outcomes}` | win rates plus each verdict (Part 7 compares them with your own labels) |
| `print(json.dumps({... if k != "per_item"}))` | prints only the summary |
| `path = OUT / "reports" / f"judge_..."` | one report file per run |
Step 6. Run score mode for notes and summary:
```
python 32_judge.py --task notes --mode score --models base notes --n 100 && python 32_judge.py --task summary --mode score --models base summary --n 100
```
You should see: two JSON blocks like {"base": {"mean_score": ..., "judged": 100}, "notes": {"mean_score": ..., "judged": 100}}.
Watch out: the judge is also the teacher that wrote the synthetic explanations, so it may favour its own style (self-preference bias). Say this in interviews; Part 7 measures judge reliability.
CHECKPOINT: outputs/reports contains judge_notes_score_base_notes.json and judge_summary_score_base_summary.json.

## SECTION 5.12 - Publish your datasets
WHERE: VS Code Terminal, folder ~/medscriptai/scripts, (.venv-train) active
GOAL: two private dataset repos on the Hub.
Step 1. Activate the training environment:
```
source ../.venv-train/bin/activate
```
Step 2. Publish the SFT data:
```
python 12_push_to_hub.py --type dataset --folder ../data/sft --repo YOUR_HF_USERNAME/medscriptai-sft
```
Step 3. Publish the synthetic explanations:
```
python 12_push_to_hub.py --type dataset --folder ../data/synth --repo YOUR_HF_USERNAME/medscriptai-medqa-explanations
```
Step 4. On each dataset page click Edit dataset card and write: the sources and licences (MedQA CC-BY-4.0, MedMCQA Apache-2.0, PubMedQA MIT, augmented-clinical-notes MIT), how you cleaned and de-duplicated, that the explanations are synthetic from Qwen2.5-14B-Instruct, and "not for clinical use".
Watch out: CC-BY-4.0 requires attribution: credit the MedQA authors in the card.
CHECKPOINT: both dataset pages show your files.

## SECTION 5.13 - Cloud step: reports to S3, token into Secrets Manager
BEFORE YOU START: Secrets Manager is AWS's safe for passwords and tokens. Servers read secrets from it at start-up instead of having them written in files or code.
WHERE: VS Code Terminal, folder ~/medscriptai (cd ~/medscriptai)
GOAL: practise storing artifacts and secrets like production systems.
| Step | FakeCloud (free practice) {w=70} | Real AWS (same step) {blue} |
| 1 Point the CLI | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL` |
| 2 Bucket name | `export BUCKET=medscriptai-artifacts-$(aws sts get-caller-identity --query Account --output text)-us-east-1` | same |
| 3 Upload reports | `aws s3 sync outputs/reports s3://$BUCKET/reports` | same |
| 4 Upload all adapters | `aws s3 sync outputs/adapters s3://$BUCKET/adapters --exclude "*/checkpoint-*"` | same |
| 5 Store the HF token | `aws secretsmanager create-secret --name medscriptai/HF_TOKEN --secret-string "$HF_TOKEN"` | same (about US$0.40 per secret per month, verify; deleted in teardown) |
| 6 Read it back | `aws secretsmanager get-secret-value --secret-id medscriptai/HF_TOKEN --query SecretString --output text` | same |
| You should see | your hf_ token | your hf_ token; also listed (masked) in AWS console > Secrets Manager |
What each piece means:
| Piece | What it does |
| `--exclude "*/checkpoint-*"` | skip training checkpoints |
| `create-secret --name medscriptai/HF_TOKEN` | store an encrypted secret; the slash only groups names |
| `--query SecretString --output text` | print only the value |
CHECKPOINT: aws s3 ls s3://$BUCKET/adapters/ lists sft-qa, sft-notes, sft-summary.

## SECTION 5.14 - Commit
WHERE: VS Code Terminal
GOAL: Part 5 scripts saved in Git.
Step 1. Run:
```
cd ~/medscriptai && git add . && git commit -m "Part 5: synthetic data, three adapters, evaluation"
```
CHECKPOINT: git log --oneline shows the new commit.

## PART 5 RESULT CHECK
| Check | Command | Expected |
| synthetic file | wc -l data/sft/qa_train_synth.jsonl | 23669 |
| three adapters | ls outputs/adapters | sft-notes sft-qa sft-summary |
| six prediction files | ls outputs/preds/* | base.jsonl + adapter file per task |
| metrics | cat outputs/reports/metrics.json | six task/model entries; fine-tuned format_ok and json_valid near 1.0 |
| judge | ls outputs/reports/judge_* | two score files |
| artifacts in (Fake)S3 | aws s3 ls s3://$BUCKET --recursive \| head | reports/... and adapters/... |

## 5.15 Common mistake at this level
The mistake: celebrating a metric that does not measure what you think: ROUGE-L rises because the model copies the reference style while facts get worse; accuracy rises because test questions leaked into training; the judge prefers the fine-tuned model because it writes like the judge.
How to spot it: read 10 random predictions next to their references; check format_ok before accuracy; keep the de-duplication (Section 4.4) and run the contamination check (Part 7); compare at least two independent metrics; run the significance test before claiming a win.
