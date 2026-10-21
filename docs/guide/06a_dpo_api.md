# PART 6A - ADVANCED: DPO, vLLM multi-LoRA hot-swap, FastAPI + LangSmith

## BEFORE YOU START: what you will build in this part
```
Section 6A.1-6A.3  build preference pairs (chosen vs rejected) from your own SFT model, with distilabel
Section 6A.4-6A.5  train DPO on top of the SFT adapters -> dpo-qa, dpo-notes
Section 6A.6       serve 5 adapters on one GPU and add/remove adapters without restarting
Section 6A.7       evaluate base vs SFT vs DPO (metrics + pairwise judge)
Section 6A.8-6A.11 your web API (FastAPI), its tests, and LangSmith tracing
```
DPO in one paragraph (details in Part 3, section 3.8): SFT taught the model to imitate good answers. DPO teaches it to PREFER better answers over worse ones. You give it (prompt, chosen, rejected) triples; it increases the probability of the chosen answer and decreases the rejected one, while a frozen copy of the SFT model (the reference) stops it from drifting too far.
Where the pairs come from (real example shape from this pipeline):
```
prompt   : "Pediatric airway differs from adult - Options: A) Large tongue  B) Short epiglottis ..."
candidate 1 (your SFT model, temperature 0.9): "Answer: B) Short epiglottis ..."   -> wrong letter,  rating 2
candidate 2 (your SFT model, temperature 0.9): "Answer: A) Large tongue ..."       -> right letter,  rating 4
candidate 3 (teacher)                         : "Answer: A) Large tongue ..."       -> right letter,  rating 5
chosen   = candidate 3 (score 10 + 5 = 15)     rejected = candidate 1 (score 0 + 2 = 2)
```
Score = 10 points if the letter is correct + the judge's 1-5 rating. Ground truth always beats the judge.
| Word | Plain meaning |
| preference pair | one prompt with a better (chosen) and a worse (rejected) answer |
| reference model | a frozen copy of the SFT model that DPO compares against |
| beta | how strongly DPO is pulled back towards the reference (0.1 = standard) |
| hot-swap | adding or removing an adapter on a running server without restarting it |
| API endpoint | a URL your program offers, e.g. POST /v1/generate |
| tracing | recording each request's input, output and timing (LangSmith) |
Budget: about 4 GPU-hours (about US$5-6).

## SECTION 6A.1 - Start both servers again
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: student (your SFT adapters) on port 8000 and teacher/judge on port 8001.
Step 1. Check whether they are still running from Part 5:
```
curl -s localhost:8000/v1/models | jq -r '.data[].id'; curl -s localhost:8001/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary and teacher. If you do, skip to Section 6A.2.
Step 2. If not, start the student:
```
cd ~/medscriptai && GPU_UTIL=0.35 bash serve/start_vllm.sh
```
Step 3. Start the teacher:
```
GPU_UTIL=0.50 bash serve/start_teacher.sh
```
Step 4. Wait for both:
```
until curl -s localhost:8000/v1/models >/dev/null && curl -s localhost:8001/v1/models >/dev/null; do sleep 10; done; echo both ready
```
CHECKPOINT: "both ready".

## SECTION 6A.2 - Create scripts/21_build_dpo_pairs.py
WHERE: VS Code Explorer, folder scripts
GOAL: the preference-pair builder.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 21_build_dpo_pairs.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/21_build_dpo_pairs.py
| Line | What it does (plain English) |
| `from distilabel.steps import GroupColumns, LoadDataFromDicts` | GroupColumns merges the outputs of parallel steps into one list |
| `from distilabel.steps.tasks import TextGeneration, UltraFeedback` | UltraFeedback = the rating task from the UltraFeedback paper: the judge rates several answers 1-5 |
| `STUDENT_URL` (port 8000) | vLLM with YOUR SFT adapters; model "qa" = base + sft-qa |
| `rows = read_jsonl(PREF / f"{a.task}_dpo.jsonl")[: a.limit]` | the held-out DPO prompts from Section 4.5 (never used in SFT) |
| `"gold_letter": r.get("answer_letter")` | the correct letter for qa; None for notes |
| `def llm(model, url, temperature)` | a small helper so every step uses the same settings |
| `student_1`, `student_2` at temperature 0.9 | two DIFFERENT samples from your SFT model; high temperature creates variety, including mistakes worth un-learning |
| `teacher` at temperature 0.3 | a third, usually stronger, candidate |
| `GroupColumns(columns=["generation", "model_name"], output_columns=["generations", "generation_models"])` | three single answers -> one list of three |
| `UltraFeedback(llm=..., aspect="overall-rating")` | the teacher rates all three together |
| `load >> [student_1, student_2, teacher] >> group >> rate` | fan out to three generators, join, rate |
| `def letter_of(text)` | the letter after "Answer:" |
| `correct = 1 if ... letter_of(text) == row["gold_letter"] else 0` | the verifiable check: is the letter right? |
| `scored.append((correct * 10 + rating, text))` | correctness dominates; the rating breaks ties |
| `if high - low < 2: continue` | skip near-ties (no real difference = noisy signal) |
| `pairs.append({"prompt": [...], "chosen": [...], "rejected": [...]})` | exactly TRL's conversational preference format |
CHECKPOINT: ls scripts shows 21_build_dpo_pairs.py

## SECTION 6A.3 - Build the pairs
WHERE: VS Code Terminal, tmux, folder ~/medscriptai/scripts, (.venv-synth) active
GOAL: data/pref/qa_pairs.jsonl and data/pref/notes_pairs.jsonl
Step 1. Start tmux:
```
tmux new -s pairs
```
Step 2. Activate the synthetic-data environment:
```
cd ~/medscriptai/scripts && source ../.venv-synth/bin/activate
```
Step 3. Build QA pairs from 1,500 prompts (about 20-40 minutes, estimate):
```
python 21_build_dpo_pairs.py --task qa --limit 1500
```
You should see: "wrote N rows -> data/pref/qa_pairs.jsonl" and "N preference pairs from 1500 prompts". N is usually several hundred: prompts where all three answers were equally good are skipped on purpose.
Step 4. Build notes pairs from 600 prompts:
```
python 21_build_dpo_pairs.py --task notes --limit 600
```
Step 5. Look at one pair:
```
head -n 1 ../data/pref/qa_pairs.jsonl | python -m json.tool
```
You should see: "prompt" (system + user), "chosen" (often the correct letter), "rejected" (often a wrong letter), "score_chosen", "score_rejected".
Step 6. Check the share of pairs where chosen is correct and rejected is not:
```
python -c "import json; p=[json.loads(l) for l in open('../data/pref/qa_pairs.jsonl')]; print(len(p), 'pairs;', sum(x['score_chosen']>=10 and x['score_rejected']<10 for x in p), 'correct-vs-wrong pairs')"
```
CHECKPOINT: at least about 300 QA pairs and 150 notes pairs.
If not: if almost all prompts are skipped, your SFT model and the teacher agree on nearly everything: use more prompts (--limit 1891) or lower the threshold 2 to 1 in the script.

## SECTION 6A.4 - Create scripts/22_train_dpo.py
BEFORE YOU START: what you should see in the DPO log. At step 0 the model equals the reference, so the loss is exactly 0.693 (= ln 2; we verified this on a test run). As training works, loss falls, rewards/margins (how much more the model prefers chosen over rejected) rises, and rewards/accuracies (share of pairs where chosen wins) goes above 0.5.
WHERE: VS Code Explorer, folder scripts
GOAL: the DPO training script.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 22_train_dpo.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/22_train_dpo.py
| Line | What it does (plain English) |
| `from unsloth import FastLanguageModel, PatchDPOTrainer` and `PatchDPOTrainer()` | Unsloth first; in Unsloth 2026.10 PatchDPOTrainer() does nothing (its source just returns) and is kept for older versions |
| `--beta 0.1`, `--lr 5e-6` | standard DPO strength; a learning rate 40x smaller than SFT, because DPO breaks fluency easily |
| `sft_dir = OUT / "adapters" / f"sft-{a.task}"` | start from your SFT adapter |
| `FastLanguageModel.from_pretrained(str(sft_dir), ...)` | loads 4-bit base + SFT LoRA as trainable |
| `tokenizer.padding_side = "left"` | TRL's DPO expects left padding |
| `select_columns(["prompt", "chosen", "rejected"])` | keep only what DPO needs |
| `train_test_split(test_size=0.05, seed=42)` | 5% of pairs held out to measure reward accuracy |
| `loss_type=["sigmoid"]` | the original DPO loss |
| `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` | each pair is 2 sequences; effective batch 16 pairs |
| `DPOTrainer(model=model, ref_model=None, ...)` | with ref_model=None and an existing adapter, TRL 1.15 copies your SFT adapter into a frozen adapter called "ref" and uses it as the reference (we checked this in TRL's source and in a test run) |
| `print("adapters inside the model:", ...)` | must print ['default', 'ref'] |
| `metrics = trainer.evaluate()` | reward metrics on the 5% held-out pairs |
| `model.save_pretrained(str(out_dir), selected_adapters=["default"])` | save only the trained adapter, not the frozen copy |
CHECKPOINT: ls scripts shows 22_train_dpo.py

## SECTION 6A.5 - Train DPO
WHERE: VS Code Terminal, tmux, folder ~/medscriptai/scripts, (.venv-train) active
GOAL: outputs/adapters/dpo-qa and dpo-notes.
Step 1. Free the GPU:
```
docker rm -f medscriptai-vllm medscriptai-teacher
```
Step 2. Start tmux:
```
tmux new -s dpo
```
Step 3. Activate the training environment:
```
cd ~/medscriptai/scripts && source ../.venv-train/bin/activate
```
Step 4. Train (about 20-40 minutes per task, estimate):
```
python 22_train_dpo.py --task qa && python 22_train_dpo.py --task notes
```
You should see:
```
adapters inside the model: ['default', 'ref']
{'loss': 0.6931, ... 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, ...}     <- step 0: exactly ln 2
{'loss': 0.6..., ... 'rewards/accuracies': 0.6..., 'rewards/margins': 0.1..., ...}
...
{'eval_rewards/accuracies': ..., 'eval_rewards/margins': ...}
DONE adapter=/home/ubuntu/medscriptai/outputs/adapters/dpo-qa
```
| What you see | Meaning | Action |
| loss falls below 0.69, accuracies above 0.5, margins rising | DPO is learning the preferences | none |
| rewards/chosen AND rewards/rejected both fall fast | known DPO failure: pushing everything down | rerun with --lr 2e-6 or --beta 0.2 |
| only ['default'] printed | the reference is the base model, not SFT | still works; check your TRL version is 1.15 (pip show trl) |
CHECKPOINT: ls ../outputs/adapters shows dpo-qa and dpo-notes; eval_rewards/accuracies above 0.5.

## SECTION 6A.6 - Serve five adapters and hot-swap them
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: base + sft-qa + sft-notes + sft-summary + dpo-qa + dpo-notes on one GPU; add and remove adapters while it runs.
Step 1. Start the student server with more memory:
```
cd ~/medscriptai && GPU_UTIL=0.60 bash serve/start_vllm.sh
```
You should see: Adapters: qa=/adapters/sft-qa notes=/adapters/sft-notes summary=/adapters/sft-summary qa-dpo=/adapters/dpo-qa notes-dpo=/adapters/dpo-notes
Step 2. Wait and list:
```
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done; curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary, qa-dpo, notes-dpo
Step 3. Ask the SAME question to two different "models" on the same server:
```
for m in base qa-dpo; do curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$m\",\"max_tokens\":120,\"messages\":[{\"role\":\"user\",\"content\":\"A pregnant woman has cystitis. Which is safest?\n\nOptions:\nA) Doxycycline\nB) Nitrofurantoin\nC) Ciprofloxacin\nD) Tetracycline\"}]}" | jq -r '.choices[0].message.content'; echo ---; done
```
You should see: two different answers; the qa-dpo one starts with "Answer: B)".
Step 4. Remove an adapter while the server runs:
```
curl -s -X POST localhost:8000/v1/unload_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo"}'
```
Step 5. Add it back under a new version name:
```
curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo-v2", "lora_path": "/adapters/dpo-notes"}'
```
Step 6. List again:
```
curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: notes-dpo gone, notes-dpo-v2 present: no restart, no downtime.
Step 7. Put the original name back (the evaluation uses it):
```
curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo", "lora_path": "/adapters/dpo-notes"}'
```
How it works: the 16 GB base stays on the GPU once; each adapter is about 80-170 MB; per request vLLM reads the "model" name and applies that adapter's small matrices (the batched-LoRA idea from the Punica and S-LoRA papers). Note: the adapters were trained on the 4-bit base and are served on the 16-bit base; that is standard practice, and it is why you evaluate on THIS server, not inside the training script.
CHECKPOINT: Step 6 showed notes-dpo-v2 and Step 7 succeeded.

## SECTION 6A.7 - Evaluate base vs SFT vs DPO
WHERE: VS Code Terminal, folder ~/medscriptai/scripts, (.venv-eval) active; student server on 8000
GOAL: numbers that show whether DPO helped.
Step 1. Start the judge next to the student (0.60 + 0.30 = 0.90):
```
cd ~/medscriptai && GPU_UTIL=0.30 bash serve/start_teacher.sh
```
Step 2. Activate the evaluation environment:
```
cd ~/medscriptai/scripts && source ../.venv-eval/bin/activate
```
Step 3. Answers from the DPO adapters:
```
python 30_generate_preds.py --task qa --models qa-dpo && python 30_generate_preds.py --task notes --models notes-dpo
```
Step 4. Recompute all metrics:
```
python 31_eval_metrics.py
```
Step 5. Wait for the judge:
```
until curl -s localhost:8001/v1/models > /dev/null; do sleep 10; done; echo ready
```
Step 6. Pairwise judge, DPO vs SFT, with position swap:
```
python 32_judge.py --task qa --mode pairwise --models qa-dpo qa --n 200 && python 32_judge.py --task notes --mode pairwise --models notes-dpo notes --n 100
```
You should see: {"qa-dpo_wins": ..., "qa_wins": ..., "ties": ..., "n": 200} and the same for notes.
| If you see... | It means... | Do this next |
| qa-dpo accuracy higher and qa-dpo_wins > qa_wins | DPO helped | test significance in Part 7 before claiming it |
| same accuracy, more wins | better explanations, same correctness | report both honestly |
| lower accuracy | DPO hurt correctness | --lr 2e-6, --beta 0.2, check pair quality (6A.3 Step 6) |
| ties above 60% | the judge flips with order: weak signal | read 10 duels yourself |
If the judge fails to start ("insufficient GPU memory"): run GPU_UTIL=0.45 bash serve/start_vllm.sh, then repeat Step 1.
CHECKPOINT: metrics.json has qa/qa-dpo and notes/notes-dpo; two pairwise judge files exist.

## SECTION 6A.8 - Create the API files (api/requirements.txt and api/app.py)
BEFORE YOU START: what the API does. Clients (a website, a mobile app, a hospital system) should not talk to vLLM directly. Your API sits in front: it checks an API key, validates the input, picks the adapter for the task (qa -> qa-dpo, notes -> notes, summary -> summary), adds the right system prompt, calls vLLM, measures latency, and returns clean JSON with a disclaimer. Request and response look like this:
```
POST /v1/generate   header X-API-Key: dev-key
{"task": "qa", "input": "A pregnant woman has cystitis. Which is safest?\n\nOptions:\nA) ...\nB) Nitrofurantoin ..."}

{"request_id": "a6f4...", "task": "qa", "model": "qa-dpo", "output": "Answer: B) Nitrofurantoin ...",
 "parsed_json": null, "latency_ms": 640, "disclaimer": "AI-generated. For education/documentation support only. Not medical advice."}
```
WHERE: VS Code Explorer, folder api
GOAL: the two API files.
FILE 1: api/requirements.txt
Step 1. Right-click the folder api.
Step 2. Click New File...
Step 3. Type requirements.txt and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include api/requirements.txt
| Line | Why |
| fastapi | the web framework |
| uvicorn | the web server that runs FastAPI |
| openai | the client that talks to vLLM |
| langsmith | tracing |
FILE 2: api/app.py
Step 6. Right-click the folder api.
Step 7. Click New File...
Step 8. Type app.py and press Enter.
Step 9. Paste the grey box below.
Step 10. Press Ctrl+S.
@@include api/app.py
| Line | What it does (plain English) |
| `from fastapi import FastAPI, Header, HTTPException` | the app object, header reading, HTTP errors |
| `from langsmith import traceable` | a decorator that records a function call as a LangSmith "run" |
| `from langsmith.wrappers import wrap_openai` | wraps the OpenAI client so every model call is recorded with tokens and latency |
| `from pydantic import BaseModel, Field` | typed request/response models with automatic validation and documentation |
| `from common import SYSTEM` | the SAME system prompts as in training |
| `VLLM_URL = os.getenv(...)` | where vLLM is; inside Docker Compose it becomes http://vllm:8000/v1 |
| `API_KEY = os.getenv("API_KEY", "")` | shared secret for clients; empty = no check (local tests only) |
| `TASK_TO_MODEL = {"qa": os.getenv("MODEL_QA", "qa-dpo"), ...}` | which adapter answers which task, changeable without code edits (Part 7 uses this for rollouts) |
| `llm = wrap_openai(OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=120))` | one shared client for all requests; 120 s timeout |
| `app = FastAPI(title="MedScriptAI", version="1.0.0")` | the web app; title and version appear in /docs |
| `class GenerateRequest(BaseModel)` | the request schema |
| `task: str = Field(pattern="^(qa\|notes\|summary)$")` | only these three values pass; anything else gets HTTP 422 automatically |
| `input: str = Field(min_length=1, max_length=20000)` | no empty or huge inputs |
| `max_tokens ... ge=1, le=2000`, `temperature ... ge=0.0, le=1.5` | numeric limits checked automatically |
| `class GenerateResponse(BaseModel)` | the response schema |
| `parsed_json: dict \| None = None` | for summary: the parsed JSON, or null if the model wrote invalid JSON |
| `disclaimer: str = "AI-generated ..."` | always tell users this is not medical advice |
| `@traceable(name="medscriptai.generate", run_type="chain")` | each call becomes a LangSmith trace |
| `def run_generation(...)` | picks the adapter, sends system + user messages to vLLM, returns model name and text |
| `@app.get("/health")` | liveness: the process is alive (Docker uses it) |
| `@app.get("/ready")` | readiness: vLLM answers AND every adapter we route to is loaded; otherwise 503 |
| `served = {m.id for m in llm.models.list().data}` | the names vLLM currently serves |
| `missing = sorted(set(TASK_TO_MODEL.values()) - served)` | routed adapters that are not loaded |
| `@app.post("/v1/generate", response_model=GenerateResponse)` | the main endpoint |
| `x_api_key: str = Header(default="")` | reads the X-API-Key header |
| `if API_KEY and x_api_key != API_KEY: raise HTTPException(401, ...)` | wrong or missing key -> 401 |
| `except Exception: raise HTTPException(502, ...)` | vLLM failed -> 502 Bad Gateway, never a stack trace |
| `if req.task == "summary": parsed = json.loads(...)` | ready-to-use JSON for summary clients |
| `request_id=str(uuid.uuid4())`, `latency_ms=...` | a unique ID per request and the measured time |
CHECKPOINT: ls api shows app.py and requirements.txt

## SECTION 6A.9 - Run the API and call it
WHERE: VS Code Terminal (two terminals), folder ~/medscriptai/api, (.venv-eval) active; student vLLM on port 8000
GOAL: your API answers on port 8080.
Step 1. In terminal 1, activate and start the API:
```
cd ~/medscriptai/api && source ../.venv-eval/bin/activate && VLLM_URL=http://localhost:8000/v1 API_KEY=dev-key PYTHONPATH=../scripts uvicorn app:app --host 0.0.0.0 --port 8080
```
You should see: "Uvicorn running on http://0.0.0.0:8080"
Step 2. Open a second terminal (click + in the terminal panel) and check readiness:
```
curl -s localhost:8080/ready | jq
```
You should see: {"status": "ready", "models": ["base", "notes", "notes-dpo", "qa", "qa-dpo", "summary"]}
Step 3. Call the API:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: dev-key" -d '{"task":"qa","input":"A pregnant woman has cystitis. Which is safest?\n\nOptions:\nA) Doxycycline\nB) Nitrofurantoin\nC) Ciprofloxacin\nD) Tetracycline"}' | jq
```
You should see: request_id, "task": "qa", "model": "qa-dpo", an output starting with "Answer: B)", latency_ms and the disclaimer.
Step 4. Try a wrong key (should be refused):
```
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: wrong" -d '{"task":"qa","input":"x"}'
```
You should see: 401
Step 5. Open the interactive documentation: in VS Code click the PORTS tab (next to TERMINAL), click Forward a Port, type 8080, press Enter, then open http://localhost:8080/docs in your laptop browser.
You should see: Swagger UI with GET /health, GET /ready, POST /v1/generate; you can click "Try it out".
Step 6. Stop the API in terminal 1 with Ctrl+C.
| Piece | What it does |
| PYTHONPATH=../scripts | lets app.py find common.py |
| uvicorn app:app | run the object "app" from the file app.py |
| --host 0.0.0.0 --port 8080 | listen on all network interfaces, port 8080 |
CHECKPOINT: Step 3 returned "model": "qa-dpo" and Step 4 returned 401.

## SECTION 6A.10 - Create the tests and run them without a GPU
BEFORE YOU START: why a fake server. Tests must run anywhere (your laptop, a CI server) without a GPU. tests/mock_openai.py is a tiny fake vLLM that answers in the same SHAPE as the real one. We used exactly these files to test the pipelines and the API while writing this manual.
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: 5 passing API tests.
FILE 1: tests/mock_openai.py
Step 1. Right-click the folder tests.
Step 2. Click New File...
Step 3. Type mock_openai.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include tests/mock_openai.py
| Line | What it does (plain English) |
| `MODELS = [...]` | the names /v1/models reports, like a real vLLM with adapters |
| `def reply_for(model, text)` | returns an answer with the right shape for each kind of request (judge ratings, explanation, JSON, note, "Answer: X)") |
| `re.findall(r"<text \d+>", text)` | detects UltraFeedback rating prompts |
| `@app.post("/v1/chat/completions")` | the OpenAI chat endpoint, returning choices[0].message.content and usage |
FILE 2: tests/test_api.py
Step 6. Right-click the folder tests.
Step 7. Click New File...
Step 8. Type test_api.py and press Enter.
Step 9. Paste the grey box below.
Step 10. Press Ctrl+S.
@@include tests/test_api.py
| Line | What it does (plain English) |
| `sys.path[:0] = ["api", "scripts"]` | lets the test import app.py and common.py |
| `os.environ["API_KEY"] = "test-key"` | set BEFORE importing app (app reads it at import time) |
| `client = TestClient(api.app)` | calls the app in-process, no real network |
| the five `test_...` functions | health, readiness, qa routed to qa-dpo, summary JSON parsed, auth (401) and validation (422) |
Step 11. Start the fake server on port 8009 (8000 is your real vLLM):
```
cd ~/medscriptai && source .venv-eval/bin/activate && (uvicorn tests.mock_openai:app --port 8009 > /tmp/mock.log 2>&1 &) && sleep 2 && curl -s localhost:8009/health
```
You should see: {"status":"ok"}
Step 12. Run the tests:
```
VLLM_URL=http://localhost:8009/v1 python -m pytest -q tests/test_api.py
```
You should see: 5 passed
CHECKPOINT: 5 passed.

## SECTION 6A.11 - LangSmith tracing
WHERE: web browser (smith.langchain.com), then VS Code Terminal
GOAL: every API request visible as a trace.
Step 1. Open https://smith.langchain.com and sign in (or sign up, free Developer plan).
Step 2. Click Settings (gear icon, bottom left).
Step 3. Click API Keys, then Create API Key, then Create.
Step 4. Copy the key (starts with lsv2_) into secrets.txt.
Step 5. Save it on the GPU computer (replace lsv2_YOUR_KEY):
```
echo 'export LANGSMITH_API_KEY=lsv2_YOUR_KEY' >> ~/.bashrc && source ~/.bashrc
```
Step 6. Start the API with tracing on:
```
cd ~/medscriptai/api && source ../.venv-eval/bin/activate && LANGSMITH_TRACING=true LANGSMITH_PROJECT=medscriptai VLLM_URL=http://localhost:8000/v1 API_KEY=dev-key PYTHONPATH=../scripts uvicorn app:app --port 8080
```
Step 7. In the second terminal, repeat the curl from Section 6A.9 Step 3 three times.
Step 8. In LangSmith click Projects (left menu), then medscriptai.
You should see: three runs named medscriptai.generate; click one to see the input, the output, a child ChatOpenAI run with model qa-dpo, tokens and latency.
Step 9. Stop the API with Ctrl+C.
| Variable | What it does |
| LANGSMITH_TRACING=true | turns tracing on (we checked the SDK source: it reads LANGSMITH_TRACING or the older LANGCHAIN_TRACING_V2) |
| LANGSMITH_API_KEY | authenticates the SDK |
| LANGSMITH_PROJECT | the project name (created automatically) |
| LANGSMITH_ENDPOINT | only for EU accounts: https://eu.api.smith.langchain.com |
Watch out: traces contain the full input text. With real patient data that is a privacy problem; here the data is public.
CHECKPOINT: the medscriptai project shows your runs.

## SECTION 6A.12 - Commit
Step 1. Run:
```
cd ~/medscriptai && git add . && git commit -m "Part 6A: DPO, multi-LoRA serving, API and tests"
```
CHECKPOINT: git log --oneline shows the new commit.

## PART 6A RESULT CHECK
| Check | Command | Expected |
| pairs | wc -l data/pref/qa_pairs.jsonl data/pref/notes_pairs.jsonl | hundreds each |
| DPO adapters | ls outputs/adapters | dpo-notes dpo-qa (plus the three sft-*) |
| DPO learned | cat outputs/adapters/dpo-qa/dpo_eval.json | eval_rewards/accuracies above 0.5 |
| five adapters served | curl -s localhost:8000/v1/models \| jq -r '.data[].id' | base qa notes summary qa-dpo notes-dpo |
| DPO vs SFT judged | ls outputs/reports/judge_*pairwise* | two files |
| API | curl localhost:8080/ready (API running) | "status": "ready" |
| tests | pytest -q tests/test_api.py (mock on 8009) | 5 passed |
| traces | LangSmith > Projects > medscriptai | your runs |

## 6A.13 Common mistake at this level
The mistake: calling vLLM directly from clients, leaving runtime LoRA loading open to the internet, or creating a new model client per request.
How to spot it: only the API (port 8080) should ever be reachable from outside; Part 6B's Compose file publishes 8080 only. In app.py the client is created once at the top, not inside the endpoint.
