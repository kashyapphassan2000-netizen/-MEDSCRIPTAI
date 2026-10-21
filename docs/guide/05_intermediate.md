# PART 5 - INTERMEDIATE: synthetic data, three adapters, real evaluation

## 5.0 Concept
What: serve a stronger open model (Qwen2.5-14B-Instruct) as a teacher, use distilabel to write explanations for the 10,000 MedQA answers that have none, train all three adapters (qa, notes, summary) on the full data, and evaluate base vs fine-tuned with accuracy, ROUGE-L, BERTScore and an LLM judge.
Why: Part 4 proved the loop; Part 5 is what a real team does next: improve the data (synthetic augmentation with quality filters), scale training, and measure. "It looks better" is not evidence; numbers on a held-out test set are.
When: whenever the labels you have are thinner than the behaviour you want (here: letters without reasoning), and before any claim about quality.
Harder than Part 4 because: two model servers, a pipeline framework, three training runs, and metrics you must interpret correctly. Budget: about 6 GPU-hours (about US$8).

## SECTION 5.1 - Start the teacher model (vLLM, port 8001)
WHERE: VS Code Terminal on the VM, folder ~/-MEDSCRIPTAI
GOAL: Qwen2.5-14B-Instruct answering OpenAI-style requests under the name "teacher".
Step 1. Start it:
```
bash serve/start_teacher.sh
```
Step 2. Follow the start-up log (the first start downloads about 30 GB, 5-10 minutes):
```
docker logs -f medscriptai-teacher
```
You should see, at the end: "Application startup complete." and "Uvicorn running on http://0.0.0.0:8000". Press Ctrl+C to stop following (the server keeps running).
Step 3. Ask it a test question:
```
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"teacher","messages":[{"role":"user","content":"In one sentence: why is nitrofurantoin used for cystitis in pregnancy?"}],"max_tokens":80}' | jq -r '.choices[0].message.content'
```
You should see: one sensible medical sentence.
FILE serve/start_teacher.sh:
@@include serve/start_teacher.sh
| Line | What it does (plain English) |
| `set -euo pipefail` | stop at the first error (-e), treat unset variables as errors (-u), fail a pipeline if any part fails |
| `VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0-cu129}"` | the official vLLM server image, CUDA 12.9 build. `${X:-default}` = use X if set, else the default |
| `GPU_UTIL="${GPU_UTIL:-0.85}"` | fraction of the GPU's memory this server may take (weights + KV cache) |
| `docker rm -f medscriptai-teacher ... \|\| true` | remove an old container with the same name; `\|\| true` ignores "no such container" |
| `docker run -d --name medscriptai-teacher` | start a container in the background (-d) with a fixed name |
| `--gpus all --ipc=host` | give the container the GPU; share host memory for fast inter-process communication (vLLM needs it) |
| `-p 8001:8000` | the server listens on 8000 inside the container; you reach it on 8001 on the VM |
| `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` | your VM driver (R570) supports CUDA 12.8 but the image uses 12.9; NVIDIA's forward-compatibility libraries inside the image bridge the gap on data-centre GPUs such as A100/L40/H100 |
| `-v ${HF_HOME}:/root/.cache/huggingface` | share the model cache, so the 30 GB download happens once |
| `--model Qwen/Qwen2.5-14B-Instruct` | the teacher: 14.7B parameters, Apache-2.0, strong at medical reasoning for its size |
| `--served-model-name teacher` | clients ask for model "teacher" instead of the long Hub name |
| `--dtype bfloat16` | 16-bit weights: about 29.5 GB |
| `--max-model-len 8192` | longest prompt+answer in tokens; smaller = more room for parallel requests |
| `--gpu-memory-utilization "$GPU_UTIL"` | vLLM pre-allocates this fraction and fills the rest of it with KV cache (PagedAttention) |
If something goes wrong:
| If you see... | It means... | Do this next |
| "CUDA driver version is insufficient" | the compatibility layer was not used | check the -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 line, or use the R595 OS image with VLLM_IMAGE=vllm/vllm-openai:v0.32.0 |
| "No available memory for the cache blocks" | GPU memory is taken by something else | run nvidia-smi; stop other containers or training jobs |
| curl: (7) Failed to connect | the server is still loading | wait for "Application startup complete" |
CHECKPOINT: curl http://localhost:8001/v1/models lists "teacher".
If not: docker logs medscriptai-teacher | tail -40 shows the reason.

## SECTION 5.2 - Generate synthetic explanations with distilabel
WHERE: VS Code Terminal, tmux session, folder ~/-MEDSCRIPTAI/scripts, (.venv-synth) active
GOAL: data/sft/qa_train_synth.jsonl where MedQA answers also have a teacher explanation.
Step 1. Start tmux:
```
tmux new -s synth
```
Step 2. Activate the synthetic-data environment:
```
cd ~/-MEDSCRIPTAI && source .venv-synth/bin/activate && cd scripts
```
Step 3. Dry run on 50 rows first (1-2 minutes):
```
python 20_synth_explanations.py --limit 50
```
You should see: distilabel's pipeline log, then kept 4x/50 explanations after the quality filter (most should pass).
Step 4. Look at three random explanations:
```
shuf -n 3 ../data/synth/medqa_explanations.jsonl | python -m json.tool
```
You should see: medically sensible explanations that mention the correct option. If they are wrong or rambling, fix the prompt before the full run.
Step 5. Full run (about 10,000 rows; roughly 15-30 minutes on an A100, estimate):
```
python 20_synth_explanations.py
```
You should see: kept ~9xxx/10xxx explanations after the quality filter, and wrote ... rows -> data/sft/qa_train_synth.jsonl.
FILE scripts/20_synth_explanations.py:
@@include scripts/20_synth_explanations.py
| Line | What it does (plain English) |
| `from distilabel.models import OpenAILLM` | distilabel's client for any OpenAI-compatible server; vLLM is one |
| `from distilabel.pipeline import Pipeline` | a pipeline = a graph of steps that pass batches of rows to each other |
| `from distilabel.steps import LoadDataFromDicts` | first step: feeds a Python list of dicts into the pipeline |
| `from distilabel.steps.tasks import TextGeneration` | a task step: sends each row's "instruction" to the LLM, writes the reply to "generation" |
| `TEACHER_URL = os.getenv("TEACHER_URL", "http://localhost:8001/v1")` | where the teacher listens (Section 5.1) |
| `TEACHER = os.getenv("TEACHER_MODEL", "teacher")` | the served model name |
| `TEACHER_SYSTEM = (...)` | instructions for the teacher: expert tone, at most 120 words, never change the given answer |
| `todo = [r for r in rows if r["source"] == "medqa" and "Explanation:" not in ...]` | only MedQA rows that have no explanation yet (MedMCQA and PubMedQA already have one) |
| `"gold": r["messages"][2]["content"]` | the correct answer line, kept for the quality filter |
| `"instruction": (... "The correct answer is: " ...)` | the teacher sees the question, the options and the GOLD answer. It explains a known answer instead of guessing one: this is "rationale generation", much more reliable than letting it answer |
| `with Pipeline(name="medscriptai-explanations") as pipeline:` | steps created inside this block belong to the pipeline |
| `LoadDataFromDicts(data=inputs, batch_size=64)` | sends rows in batches of 64 |
| `OpenAILLM(model=TEACHER, base_url=TEACHER_URL, api_key="EMPTY", generation_kwargs={...})` | vLLM ignores the API key; temperature 0.3 = focused but not robotic; max 300 new tokens |
| `system_prompt=TEACHER_SYSTEM` | added as the system message of every request |
| `input_batch_size=32` | 32 requests in flight: vLLM batches them on the GPU, so this is much faster than one by one |
| `load >> explain` | connects the steps: output of load is input of explain |
| `distiset = pipeline.run(use_cache=False)` | runs everything; use_cache=False forces a fresh run instead of re-using old results |
| `distiset["default"]["train"]` | distilabel returns a DatasetDict; "default" is the leaf step's output, "train" the split |
| `def good(text, gold)` | the quality filter: non-empty, 80-1,500 characters, and it must mention the gold option text or letter |
| `explanations = {...}` | question_key -> explanation, only for rows that passed |
| `write_jsonl(DATA / "synth" / "medqa_explanations.jsonl", ...)` | keep the raw synthetic data separately (you will publish it as a dataset) |
| `r["messages"][2]["content"] += f"\nExplanation: {...}"` | append the explanation to the assistant target |
| `write_jsonl(SFT / "qa_train_synth.jsonl", rows)` | the augmented training file; the original qa_train.jsonl stays untouched for comparison |
Watch out: "No module named 'requests'" or strange datasets errors mean you are in the wrong environment. distilabel 1.5.3 only works with the pins in requirements/synth.txt (we found and fixed this while testing).
Watch out: synthetic data inherits the teacher's mistakes. A filter catches format errors, not subtle medical errors. Read 20 random explanations yourself before training (Step 4). In an interview, say exactly this.
CHECKPOINT: wc -l ../data/sft/qa_train_synth.jsonl equals wc -l ../data/sft/qa_train.jsonl, and grep -c "Explanation:" ../data/sft/qa_train_synth.jsonl is much larger than in qa_train.jsonl.
If not: if almost nothing passes the filter, print a few raw generations from distiset and adjust the prompt or the filter.

## SECTION 5.3 - Free the GPU and train the three adapters
WHERE: VS Code Terminal, tmux session train, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: outputs/adapters/sft-qa (re-trained on synthetic data), sft-notes and sft-summary.
Step 1. Stop the teacher (training needs the GPU memory):
```
docker rm -f medscriptai-teacher
```
Step 2. Open the training tmux session:
```
tmux new -s train3
```
Step 3. Activate and move:
```
cd ~/-MEDSCRIPTAI && source .venv-train/bin/activate && cd scripts
```
Step 4. Train all three, one after another (about 2.5-3.5 hours in total on an A100, estimate):
```
python 10_train_sft.py --task qa --train-file ../data/sft/qa_train_synth.jsonl --max-train 12000 && python 10_train_sft.py --task notes && python 10_train_sft.py --task summary
```
Step 5. Detach with Ctrl+B then D, and come back later with tmux attach -t train3.
| Setting | qa | notes | summary | Why |
| training rows | 12,000 (of ~30,000) | ~7,000 | ~7,000 | 12k QA rows gives most of the gain at 40% of the cost |
| median tokens | ~330 | ~1,130 | ~1,090 | longer examples = more time per step |
| steps (batch 16) | 750 | ~440 | ~440 | rows / 16 |
| rough time on A100 | 45-70 min | 40-60 min | 40-60 min | estimates; read the progress bar for the truth |
You should see: three DONE adapter=... lines. Each train_log.json ends with a lower loss than it started.
Watch out: the && between commands means "run the next only if the previous succeeded". If qa fails, notes and summary do not start; fix and re-run only the missing ones.
CHECKPOINT: ls ../outputs/adapters shows sft-qa, sft-notes, sft-summary, each with adapter_model.safetensors.
If not: tmux attach -t train3 and read the error.

## SECTION 5.4 - Serve all adapters with vLLM (port 8000) and the judge (port 8001)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: one vLLM server with base + 3 adapters, and the teacher back as the judge, both on the same A100.
Step 1. Start the student server with 35% of the GPU:
```
GPU_UTIL=0.35 bash serve/start_vllm.sh
```
You should see: Adapters: qa=/adapters/sft-qa notes=/adapters/sft-notes summary=/adapters/sft-summary
Step 2. Wait until it is ready:
```
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done; curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary (one per line).
Step 3. Start the judge with 50% of the GPU:
```
GPU_UTIL=0.50 bash serve/start_teacher.sh
```
Step 4. Wait for it:
```
until curl -s localhost:8001/v1/models > /dev/null; do sleep 10; done; echo judge ready
```
Watch out: the two GPU_UTIL values must add up to less than about 0.9, or the second server fails with "insufficient GPU memory". The full explanation of start_vllm.sh is in Section 6.4.
CHECKPOINT: both curl commands answer.

## SECTION 5.5 - Generate test-set answers for base and fine-tuned models
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: one predictions file per task and model in outputs/preds.
Step 1. Activate the evaluation environment:
```
cd ~/-MEDSCRIPTAI && source .venv-eval/bin/activate && cd scripts
```
Step 2. Generate (about 5-10 minutes in total):
```
python 30_generate_preds.py --task qa --models base qa && python 30_generate_preds.py --task notes --models base notes && python 30_generate_preds.py --task summary --models base summary
```
You should see: six lines like wrote 500 rows -> outputs/preds/qa/base.jsonl.
FILE scripts/30_generate_preds.py:
@@include scripts/30_generate_preds.py
| Line | What it does (plain English) |
| `from concurrent.futures import ThreadPoolExecutor` | send many requests at the same time; vLLM batches them on the GPU (continuous batching) |
| `from openai import OpenAI` | the official OpenAI Python client; vLLM implements the same API |
| `--models` | list of model names known to vLLM: "base" = no adapter, "qa" = base + qa adapter |
| `--url` | the vLLM address (default http://localhost:8000/v1) |
| `client = OpenAI(base_url=a.url, api_key="EMPTY")` | the client talks to vLLM; the key is required by the SDK but ignored by vLLM |
| `rows = read_jsonl(SFT / f"{a.task}_test.jsonl")[: a.n]` | the held-out TEST set (never used for training) |
| `max_tokens = 400 if a.task == "qa" else 900` | answer length limit per task |
| `def ask(model, row)` | one request |
| `messages=row["messages"][:2]` | only system + user; the reference answer is never sent |
| `temperature=0.0` | greedy decoding: the same input gives the same output, so scores are reproducible |
| `pool.map(lambda r: ask(model, r), rows)` | 16 parallel workers, results in the original order |
| `out = [{"i": i, "source": ..., "answer_letter": ..., "prompt": ..., "reference": ..., "prediction": pred} ...]` | everything the metrics and the judge need in one row |
| `write_jsonl(OUT / "preds" / a.task / f"{model}.jsonl", out)` | e.g. outputs/preds/qa/base.jsonl |
CHECKPOINT: ls ../outputs/preds/* shows base.jsonl and the adapter file for each task.

## SECTION 5.6 - Automatic metrics: accuracy, ROUGE-L, BERTScore, JSON validity
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: outputs/reports/metrics.json with every metric for every model.
Step 1. Run (the first run downloads roberta-large, 1.4 GB):
```
python 31_eval_metrics.py
```
You should see: one line per task/model such as qa base {'n': 500, 'rougeL': ..., 'bertscore_f1': ..., 'accuracy': ..., 'format_ok': ...}.
FILE scripts/31_eval_metrics.py:
@@include scripts/31_eval_metrics.py
| Line | What it does (plain English) |
| `from rouge_score import rouge_scorer` | Google's reference ROUGE implementation |
| `from bert_score import score as bert_score` | BERTScore: compares answers by meaning using contextual embeddings |
| `--bert-model roberta-large` | the standard BERTScore backbone for English |
| `RougeScorer(["rougeL"], use_stemmer=True)` | ROUGE-L = longest common subsequence of words; the stemmer treats "infections" and "infection" as equal |
| `def letter(text)` | reads the letter after "Answer:" |
| `def is_json(text)` | True if the text parses as JSON (also accepts a ```json fenced block) |
| `rouge.score(ref, pred)["rougeL"].fmeasure` | F1 of LCS-precision and LCS-recall, averaged over rows |
| `bert_score(preds, refs, model_type=..., lang="en")` | returns precision, recall and F1 tensors; we keep the mean F1 |
| `m["accuracy"] = sum(letter(p_) == r["answer_letter"] ...)` | the most honest QA metric: fraction of correct letters |
| `m["format_ok"]` | fraction of answers that followed the "Answer: X)" format at all |
| `m["json_valid"]` | fraction of summary outputs that are valid JSON |
| `report.write_text(json.dumps(results, indent=1))` | saves all numbers for the report and the API docs |
How to read the numbers:
| Metric | Good for | Trap |
| accuracy (qa) | multiple choice: exact and meaningful | needs the format to be parseable; check format_ok first |
| ROUGE-L | "did it use the reference's words?" | a correct note with different wording scores low; a fluent wrong note can score high |
| BERTScore F1 | meaning overlap | values cluster high (0.80-0.90 is common); compare models with each other, never to a fixed "good" value |
| json_valid | structured output | valid JSON can still contain invented facts: the judge checks that |
Watch out: do not report BERTScore as "% correct". It is a similarity score.
CHECKPOINT: cat ../outputs/reports/metrics.json shows qa/base, qa/qa, notes/base, notes/notes, summary/base, summary/summary.

## SECTION 5.7 - LLM-as-a-judge (score mode)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: a 1-5 quality score per model from the judge, on 100 test items per task.
Step 1. Run:
```
python 32_judge.py --task notes --mode score --models base notes --n 100 && python 32_judge.py --task summary --mode score --models base summary --n 100
```
You should see: JSON such as {"base": {"mean_score": ..., "judged": 100}, "notes": {...}}.
FILE scripts/32_judge.py:
@@include scripts/32_judge.py
| Line | What it does (plain English) |
| `judge = OpenAI(base_url=os.getenv("JUDGE_URL", "http://localhost:8001/v1"), ...)` | the judge is the teacher server on port 8001 |
| `RUBRIC = {...}` | what "good" means per task: for notes, faithfulness (no invented facts) first |
| `def ask_json(prompt)` | asks at temperature 0 and extracts the first {...} with a regex; returns {} if the judge broke the format |
| `def score_prompt(row)` | task input + reference + model answer + "Return ONLY a JSON object {score, reason}"; [:4000] cuts very long inputs |
| `def pair_prompt(row, first, second)` | the pairwise version: answer A vs answer B (used in Part 6) |
| `preds = {m: read_jsonl(...)[: a.n] for m in a.models}` | the first n predictions of each model |
| `if a.mode == "score":` | grade each answer alone, then average the valid scores |
| `def duel(i)` | pairwise: ask twice, once A-then-B and once B-then-A |
| `if first == "A" and second == "B": return ma` | a model wins only if the judge prefers it in BOTH orders; otherwise it is a tie. This cancels position bias, a well-documented judge weakness (Zheng et al., 2023) |
| `report = {...wins, ties, n, "per_item": outcomes}` | win rates plus each verdict (Part 7 checks the judge against your own labels) |
| `print(json.dumps({... if k != "per_item"}))` | prints the summary only |
| `path = OUT / "reports" / f"judge_{task}_{mode}_{models}.json"` | one report file per run |
Watch out: here the judge is also the teacher that wrote the synthetic explanations, so it may favour answers in its own style (self-preference bias). Say this in interviews. The Elite part shows how to measure judge reliability.
CHECKPOINT: outputs/reports has judge_notes_score_base_notes.json and judge_summary_score_base_summary.json.

## SECTION 5.8 - Publish your datasets
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: two private dataset repos on the Hub.
Step 1. Publish the SFT data:
```
python 12_push_to_hub.py --type dataset --folder ../data/sft --repo YOUR_HF_USERNAME/medscriptai-sft
```
Step 2. Publish the synthetic explanations:
```
python 12_push_to_hub.py --type dataset --folder ../data/synth --repo YOUR_HF_USERNAME/medscriptai-medqa-explanations
```
Step 3. On each dataset page click Edit dataset card and write: sources and licences (MedQA CC-BY-4.0, MedMCQA Apache-2.0, PubMedQA MIT, augmented-clinical-notes MIT), how you cleaned the data, that explanations are synthetic from Qwen2.5-14B-Instruct, and "not for clinical use".
Watch out: CC-BY-4.0 requires attribution: credit the MedQA authors in the card.
CHECKPOINT: both dataset pages load and show your files.

## SECTION 5.9 - Cloud step: reports to S3 and your token into Secrets Manager
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: practise storing artifacts and secrets the way production systems do.
| Step {w=70} | FakeCloud (free practice) | Real AWS (same step) {blue} |
| 1 Point the CLI | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL` |
| 2 Bucket name | `export BUCKET=medscriptai-artifacts-$(aws sts get-caller-identity --query Account --output text)-us-east-1` | same |
| 3 Upload reports | `aws s3 sync outputs/reports s3://$BUCKET/reports` | same |
| 4 Upload all adapters | `aws s3 sync outputs/adapters s3://$BUCKET/adapters --exclude "*/checkpoint-*"` | same |
| 5 Store the HF token | `aws secretsmanager create-secret --name medscriptai/HF_TOKEN --secret-string "$HF_TOKEN"` | same (costs about US$0.40 per secret per month (verify); delete it in teardown) |
| 6 Read it back | `aws secretsmanager get-secret-value --secret-id medscriptai/HF_TOKEN --query SecretString --output text` | same |
| You should see | your hf_ token printed | your hf_ token printed; also visible (masked) in AWS console > Secrets Manager |
What each command piece means:
| Piece | What it does |
| `--exclude "*/checkpoint-*"` | skip training checkpoints (large, not needed for serving) |
| `create-secret --name medscriptai/HF_TOKEN` | stores an encrypted secret; the slash is just a naming convention for grouping |
| `--query SecretString --output text` | print only the secret value |
Watch out: printing a secret is fine for this exercise in your own terminal; never print secrets in logs or CI.
CHECKPOINT: aws s3 ls s3://$BUCKET/adapters/ lists sft-qa, sft-notes, sft-summary.

## 5.10 Common mistake at this level
The mistake: celebrating a metric that does not measure what you think. Examples: ROUGE-L goes up because the fine-tuned model copies the reference style while facts get worse; accuracy goes up because test questions leaked into training; the judge prefers the fine-tuned model because it writes like the judge.
How to spot it: always read 10 random predictions next to the references; check format_ok before accuracy; keep the decontamination step (Section 4.2) and the contamination check (Section 7.1); compare at least two independent metrics; run the significance test (Section 7.2) before claiming a win.
