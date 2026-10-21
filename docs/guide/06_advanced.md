# PART 6 - ADVANCED: DPO, multi-LoRA serving, production API and cloud deployment

## 6.0 Concept
What: build preference pairs from your own SFT model's answers, align the model with DPO, serve base + all adapters from one vLLM process with hot-swapping, put a FastAPI service with LangSmith tracing in front, package it with Docker, push it to ECR and run it on an AWS GPU machine (rehearsed first on FakeCloud).
Why: SFT teaches the model to imitate; DPO teaches it to prefer better answers, which improves correctness, safety and style without a separate reward model or RL loop. Serving and deployment turn a notebook experiment into a system other people can call.
When: DPO after SFT has converged and you can produce comparisons; deployment once the evaluation says the model is worth serving.
Harder than Part 5 because: you now generate data with your own model, train against a reference model, run two GPU servers plus an API, and operate cloud resources that cost real money. Budget: about 5 GPU-hours (about US$7) + about US$3-5 on AWS if you do the blue column.

## SECTION 6.1 - Build DPO preference pairs with distilabel
WHERE: VS Code Terminal, tmux session, folder ~/-MEDSCRIPTAI/scripts, (.venv-synth) active. Both servers from Section 5.4 must be running (student on 8000, teacher on 8001).
GOAL: data/pref/qa_pairs.jsonl and data/pref/notes_pairs.jsonl in TRL's conversational preference format.
Step 1. Check both servers:
```
curl -s localhost:8000/v1/models | jq -r '.data[].id' && curl -s localhost:8001/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary and then teacher.
Step 2. Start tmux and activate:
```
tmux new -s pairs
```
```
cd ~/-MEDSCRIPTAI && source .venv-synth/bin/activate && cd scripts
```
Step 3. Build QA pairs from 1,500 unseen prompts (about 20-40 minutes, estimate):
```
python 21_build_dpo_pairs.py --task qa --limit 1500
```
You should see: wrote ~600-1200 rows -> data/pref/qa_pairs.jsonl and N preference pairs from 1500 prompts (many prompts are skipped because all candidates were equally good: that is expected).
Step 4. Build notes pairs from 600 prompts:
```
python 21_build_dpo_pairs.py --task notes --limit 600
```
Step 5. Look at one pair:
```
head -n 1 ../data/pref/qa_pairs.jsonl | python -m json.tool
```
You should see: "prompt" (system + user), "chosen" (one assistant message, usually the correct letter), "rejected" (one assistant message, often a wrong letter), "score_chosen" and "score_rejected".
FILE scripts/21_build_dpo_pairs.py:
@@include scripts/21_build_dpo_pairs.py
| Line | What it does (plain English) |
| `from distilabel.steps import GroupColumns, LoadDataFromDicts` | GroupColumns merges the outputs of parallel steps into lists |
| `from distilabel.steps.tasks import TextGeneration, UltraFeedback` | UltraFeedback = the rating task from the UltraFeedback paper (Cui et al., 2023): the judge rates several answers 1-5 |
| `STUDENT_URL` (port 8000) | vLLM with YOUR SFT adapters; asking for model "qa" means base + sft-qa adapter |
| `rows = read_jsonl(PREF / f"{a.task}_dpo.jsonl")[: a.limit]` | the held-out DPO prompts written in Section 4.2 (never used in SFT) |
| `"gold_letter": r.get("answer_letter")` | the correct letter for qa; None for notes |
| `def llm(model, url, temperature)` | a small factory so every step uses the same settings |
| `student_1`, `student_2` at temperature 0.9 | two DIFFERENT samples from your SFT model: high temperature creates variety, including mistakes worth un-learning |
| `teacher` at temperature 0.3 | a third, usually stronger candidate |
| `GroupColumns(columns=["generation", "model_name"], output_columns=["generations", "generation_models"])` | turns the three single answers into one list of three |
| `UltraFeedback(llm=..., aspect="overall-rating")` | the teacher rates all three answers together (1-5) with a rationale |
| `load >> [student_1, student_2, teacher] >> group >> rate` | fan-out to three generators in parallel, join, then rate |
| `def letter_of(text)` | extracts the letter after "Answer:" |
| `correct = 1 if row["gold_letter"] and letter_of(text) == row["gold_letter"] else 0` | the verifiable signal: is the letter right? |
| `scored.append((correct * 10 + rating, text))` | correctness dominates (10 points), the judge's rating breaks ties. Never let a judge outvote ground truth |
| `if high - low < 2: continue` | skip near-ties: a pair with no real difference teaches noise |
| `pairs.append({"prompt": [system, user], "chosen": [...], "rejected": [...]})` | exactly the conversational "explicit prompt" format TRL's DPOTrainer expects |
| `"score_chosen": high, "score_rejected": low` | kept for analysis (TRL ignores extra columns after select_columns in Section 6.2) |
Watch out: this is "on-policy" preference data (the rejected answers come from your own model). It is more useful than random pairs because it targets the mistakes YOUR model actually makes (the idea behind iterative/on-policy DPO, e.g. Tulu 3).
CHECKPOINT: wc -l ../data/pref/qa_pairs.jsonl is at least 300 and notes_pairs.jsonl at least 150.
If not: if almost every prompt is skipped, your SFT model and the teacher agree on nearly everything: lower the threshold from 2 to 1, or use more prompts.

## SECTION 6.2 - Train DPO on top of the SFT adapter
WHERE: VS Code Terminal, tmux session, folder ~/-MEDSCRIPTAI/scripts, (.venv-train) active
GOAL: outputs/adapters/dpo-qa and dpo-notes.
Step 1. Free the GPU (stop both servers):
```
docker rm -f medscriptai-vllm medscriptai-teacher
```
Step 2. Start tmux and activate:
```
tmux new -s dpo
```
```
cd ~/-MEDSCRIPTAI && source .venv-train/bin/activate && cd scripts
```
Step 3. Train (about 20-40 minutes per task, estimate):
```
python 22_train_dpo.py --task qa && python 22_train_dpo.py --task notes
```
You should see: adapters inside the model: ['default', 'ref'], then log lines with loss starting at about 0.693, and rewards/accuracies rising above 0.5. At the end a dict such as {'eval_rewards/accuracies': 0.7, 'eval_rewards/margins': 0.4, ...}.
FILE scripts/22_train_dpo.py:
@@include scripts/22_train_dpo.py
| Line | What it does (plain English) |
| `from unsloth import FastLanguageModel, PatchDPOTrainer` | Unsloth first. In Unsloth 2026.10 PatchDPOTrainer() is a no-op (the source simply returns); it is kept so the script also works with older Unsloth versions |
| `--beta 0.1` | strength of the "stay close to the reference" pull. Higher beta = smaller changes |
| `--lr 5e-6` | 40x smaller than SFT. DPO with a large LR quickly breaks fluency |
| `sft_dir = OUT / "adapters" / f"sft-{a.task}"` | start from the SFT adapter, not from the raw base model |
| `FastLanguageModel.from_pretrained(str(sft_dir), ...)` | loads 4-bit base + SFT LoRA; Unsloth loads adapters as trainable (is_trainable=True in its loader) |
| `tokenizer.padding_side = "left"` | TRL's DPO collator pads on the left |
| `load_dataset("json", data_files=...)` then `select_columns(["prompt", "chosen", "rejected"])` | keep only the three columns DPO needs |
| `train_test_split(test_size=0.05, seed=42)` | 5% of pairs held out to measure reward accuracy on unseen pairs |
| `beta=a.beta` | see above |
| `loss_type=["sigmoid"]` | the original DPO loss: -log sigmoid(beta x (chosen log-ratio - rejected log-ratio)). TRL 1.x also offers IPO, SimPO-style sigmoid_norm, APO, etc. |
| `max_length=2048` | prompt + answer truncated to 2048 tokens |
| `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` | each pair = 2 sequences (chosen + rejected), so batch 2 pairs; x8 = effective 16 pairs |
| `learning_rate`, `lr_scheduler_type="cosine"`, `warmup_steps=10`, `optim="adamw_8bit"`, `bf16=True` | same meaning as in SFT (Section 4.4) |
| `eval_steps=50` | reward accuracy on held-out pairs every 50 steps |
| `DPOTrainer(model=model, ref_model=None, ...)` | ref_model=None + a model that already has an adapter: TRL 1.15 COPIES the SFT adapter into a frozen adapter named "ref" and uses it as the reference policy. Textbook DPO (reference = SFT model) without loading a second 8B model |
| `print("adapters inside the model:", ...)` | proof: you must see ['default', 'ref']. If you only see ['default'], the reference is the base model (still works, different maths) |
| `trainer.train()` | at step 0 the policy equals the reference, so every log-ratio is 0 and loss = -log sigmoid(0) = ln 2 = 0.693 |
| `metrics = trainer.evaluate()` | eval loss and rewards on the 5% split |
| `rewards/accuracies` | fraction of pairs where the model now gives the chosen answer a higher implicit reward; above 0.5 = it learned the preference |
| `rewards/margins` | average reward gap chosen - rejected; should be positive and growing |
| `model.save_pretrained(str(out_dir), selected_adapters=["default"])` | save only the trained adapter, not the frozen "ref" copy |
| `(out_dir / "dpo_eval.json").write_text(...)` | keep the DPO metrics for your report |
Watch out: if rewards/chosen AND rewards/rejected both fall fast and generated answers get strange, the LR is too high or beta too low (a known DPO failure: it pushes everything down). Use --lr 2e-6 or --beta 0.2.
CHECKPOINT: ls ../outputs/adapters shows dpo-qa and dpo-notes; eval_rewards/accuracies is above 0.5 for both.
If not: fewer than ~200 pairs often gives noisy results; build more pairs (Section 6.1) and retrain.

## SECTION 6.3 - vLLM multi-LoRA: one base model, five adapters, hot-swapping
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: serve base + sft-qa + sft-notes + sft-summary + dpo-qa + dpo-notes from one GPU process, and add/remove adapters without restarting.
Step 1. Start the student server alone with most of the GPU:
```
GPU_UTIL=0.60 bash serve/start_vllm.sh
```
You should see: Adapters: qa=/adapters/sft-qa notes=/adapters/sft-notes summary=/adapters/sft-summary qa-dpo=/adapters/dpo-qa notes-dpo=/adapters/dpo-notes
Step 2. Wait and list the models:
```
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done; curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: base, qa, notes, summary, qa-dpo, notes-dpo.
Step 3. Send the SAME question to two adapters:
```
for m in base qa-dpo; do curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$m\",\"max_tokens\":120,\"messages\":[{\"role\":\"user\",\"content\":\"A pregnant woman has cystitis. Which is safest?\n\nOptions:\nA) Doxycycline\nB) Nitrofurantoin\nC) Ciprofloxacin\nD) Tetracycline\"}]}" | jq -r '.choices[0].message.content'; echo ---; done
```
You should see: two different answers from the same server; the qa-dpo one starts with "Answer: B)".
Step 4. Hot-unload an adapter at runtime:
```
curl -s -X POST localhost:8000/v1/unload_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo"}'
```
Step 5. Hot-load it again under a new name (for example a new version):
```
curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo-v2", "lora_path": "/adapters/dpo-notes"}'
```
Step 6. List again:
```
curl -s localhost:8000/v1/models | jq -r '.data[].id'
```
You should see: notes-dpo is gone and notes-dpo-v2 is present, with no restart and no downtime.
FILE serve/start_vllm.sh:
@@include serve/start_vllm.sh
| Line | What it does (plain English) |
| `VLLM_IMAGE=...v0.32.0-cu129` | official vLLM server image (CUDA 12.9 build, works on R570 with the compatibility flag) |
| `ADAPTERS="$(pwd)/outputs/adapters"` | adapters on the VM disk |
| `for pair in qa=sft-qa notes=sft-notes ... ; do ... done` | for each name=folder pair: if the folder has adapter_config.json, register it. So the script works after Part 4 (only qa) and after Part 6 (all five) |
| `name="${pair%%=*}"; dir="${pair#*=}"` | bash string cutting: text before "=" and text after "=" |
| `EXTRA=(--lora-modules "${LORA_ARGS[@]}")` | builds the argument list only when at least one adapter exists |
| `-e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` | enables /v1/load_lora_adapter and /v1/unload_lora_adapter. vLLM's docs warn this is risky in production: anyone who can reach port 8000 could load files. That is why docker-compose.yml never publishes vLLM's port |
| `-v "$ADAPTERS":/adapters:ro` | the adapters folder appears inside the container at /adapters, read-only |
| `--model unsloth/Llama-3.1-8B-Instruct --served-model-name base` | the bf16 base model, called "base" by clients |
| `--max-model-len 4096` | longest context per request (our longest is about 1,600 tokens + answer) |
| `--enable-lora` | turns on LoRA support (vLLM's batched LoRA kernels, the Punica/S-LoRA idea: many adapters in one batch) |
| `--max-loras 4` | how many DIFFERENT adapters can be active in the same GPU batch. The default is 1: with the default, requests for different adapters queue behind each other. A classic production gotcha |
| `--max-lora-rank 16` | must be at least the largest adapter rank (ours is 16); allowed values are 1, 8, 16, 32, 64, 128, 256, 320, 512 |
| `--max-cpu-loras 8` | how many adapters are cached in CPU RAM for fast swapping into the GPU |
| `--lora-modules qa=/adapters/sft-qa ...` | name=path pairs: the name is what clients put in the "model" field |
How hot-swapping works: the 16 GB base weights stay on the GPU once. Each adapter is about 80-170 MB. For each request vLLM looks at the "model" name and applies that adapter's small matrices. Three adapters cost almost the same memory as one model, and one GPU can serve many specialised "models".
Watch out: an adapter trained against the 4-bit base is served here on the bf16 base. This is standard practice and works well; the small mismatch is one reason evaluation must run on the SERVED setup (this server), not inside the training script.
CHECKPOINT: Step 6 shows notes-dpo-v2.
If not: "Loading lora ... failed" with a rank error means --max-lora-rank is smaller than the adapter rank.

## SECTION 6.4 - Evaluate base vs SFT vs DPO
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active; student server on 8000
GOAL: numbers that show whether DPO helped.
Step 1. Load the adapter back under its original name (Section 6.3 unloaded it):
```
curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "notes-dpo", "lora_path": "/adapters/dpo-notes"}'
```
Step 2. Start the judge next to it:
```
GPU_UTIL=0.30 bash ../serve/start_teacher.sh
```
Step 3. Activate the evaluation environment:
```
cd ~/-MEDSCRIPTAI && source .venv-eval/bin/activate && cd scripts
```
Step 4. Generate predictions for the DPO adapters:
```
python 30_generate_preds.py --task qa --models qa-dpo && python 30_generate_preds.py --task notes --models notes-dpo
```
Step 5. Recompute all metrics:
```
python 31_eval_metrics.py
```
Step 6. Wait for the judge:
```
until curl -s localhost:8001/v1/models > /dev/null; do sleep 10; done; echo ready
```
Step 7. Pairwise judge with position swap (DPO vs SFT):
```
python 32_judge.py --task qa --mode pairwise --models qa-dpo qa --n 200 && python 32_judge.py --task notes --mode pairwise --models notes-dpo notes --n 100
```
You should see: JSON like {"qa-dpo_wins": ..., "qa_wins": ..., "ties": ..., "n": 200}.
How to read the result:
| If you see... | It means... | Do this next |
| qa-dpo accuracy higher and qa-dpo_wins > qa_wins | DPO helped | run the significance test (Section 7.2) before you claim it |
| accuracy equal, win rate higher | better explanations/style, same correctness | report both honestly |
| accuracy lower | DPO hurt correctness | lower LR, raise beta, check pair quality (Section 6.1 Step 5) |
| ties above 60% | the judge flips with order: weak preference signal | look at 10 duels yourself |
Watch out: the student server was started with GPU_UTIL=0.60 in Section 6.3 and the judge uses 0.30, so together 0.90 of the GPU. If the judge does not start, restart the student with GPU_UTIL=0.45 bash serve/start_vllm.sh.
CHECKPOINT: metrics.json has qa/qa-dpo and notes/notes-dpo; two pairwise judge files exist.

## SECTION 6.5 - The FastAPI service
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI, (.venv-eval) active; student vLLM on 8000
GOAL: one HTTP endpoint that routes each task to the right adapter.
Step 1. Start the API in the foreground (replace dev-key with any string):
```
cd ~/-MEDSCRIPTAI/api && VLLM_URL=http://localhost:8000/v1 API_KEY=dev-key PYTHONPATH=../scripts uvicorn app:app --host 0.0.0.0 --port 8080
```
You should see: Uvicorn running on http://0.0.0.0:8080
Step 2. In a SECOND VS Code Terminal, check readiness:
```
curl -s localhost:8080/ready | jq
```
You should see: {"status": "ready", "models": ["base", "notes", "notes-dpo", "qa", "qa-dpo", "summary"]}
Step 3. Call the API:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: dev-key" -d '{"task":"qa","input":"A pregnant woman has cystitis. Which is safest?\n\nOptions:\nA) Doxycycline\nB) Nitrofurantoin\nC) Ciprofloxacin\nD) Tetracycline"}' | jq
```
You should see: request_id, task "qa", model "qa-dpo", output starting with "Answer: B)", latency_ms and the disclaimer.
Step 4. Open the interactive docs: in VS Code click the PORTS tab next to TERMINAL, click Forward a Port, type 8080, then open http://localhost:8080/docs in your laptop browser.
You should see: the Swagger UI with GET /health, GET /ready and POST /v1/generate.
Step 5. Stop the API with Ctrl+C in the first terminal.
FILE api/app.py:
@@include api/app.py
| Line | What it does (plain English) |
| `from fastapi import FastAPI, Header, HTTPException` | FastAPI app object, header reading, HTTP errors |
| `from langsmith import traceable` | a decorator that records a function call as a LangSmith "run" |
| `from langsmith.wrappers import wrap_openai` | wraps the OpenAI client so every LLM call inside a trace becomes a child run with tokens and latency |
| `from pydantic import BaseModel, Field` | typed request/response models with automatic validation and docs |
| `from common import SYSTEM` | the SAME system prompts used in training (no train/serve skew) |
| `VLLM_URL = os.getenv(...)` | where vLLM is; in Docker Compose this becomes http://vllm:8000/v1 |
| `API_KEY = os.getenv("API_KEY", "")` | shared secret for clients; empty disables auth (only for local tests) |
| `TASK_TO_MODEL = {"qa": os.getenv("MODEL_QA", "qa-dpo"), ...}` | routing table task -> adapter, configurable without code changes (canary: point MODEL_QA to a new adapter, watch metrics, roll back by changing one variable) |
| `llm = wrap_openai(OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=120))` | one shared client for all requests; 120 s timeout |
| `app = FastAPI(title="MedScriptAI", version="1.0.0")` | the web app; title/version appear in /docs |
| `class GenerateRequest(BaseModel)` | the request body schema |
| `task: str = Field(pattern="^(qa\|notes\|summary)$")` | only these three values pass; anything else gets HTTP 422 automatically |
| `input: str = Field(min_length=1, max_length=20000)` | rejects empty and huge inputs (cost and abuse protection) |
| `max_tokens ... ge=1, le=2000`, `temperature ... ge=0.0, le=1.5` | numeric bounds checked by Pydantic |
| `class GenerateResponse(BaseModel)` | the response schema shown in /docs |
| `parsed_json: dict \| None = None` | for task summary, the parsed JSON object (or null if the model produced invalid JSON) |
| `disclaimer: str = "AI-generated ..."` | always tell users this is not medical advice |
| `@traceable(name="medscriptai.generate", run_type="chain")` | each call of run_generation becomes a LangSmith trace named medscriptai.generate |
| `def run_generation(task, text, max_tokens, temperature)` | picks the adapter and calls vLLM with system + user messages |
| `@app.get("/health")` | liveness probe: the process is alive (Docker HEALTHCHECK and load balancers use it) |
| `@app.get("/ready")` | readiness probe: vLLM answers AND every routed adapter is loaded; returns 503 otherwise, so traffic is not sent too early |
| `served = {m.id for m in llm.models.list().data}` | the model names vLLM currently serves |
| `missing = sorted(set(TASK_TO_MODEL.values()) - served)` | adapters we route to but vLLM does not have |
| `@app.post("/v1/generate", response_model=GenerateResponse)` | the main endpoint; response_model enforces the output schema |
| `x_api_key: str = Header(default="")` | FastAPI reads the X-API-Key header into this argument |
| `if API_KEY and x_api_key != API_KEY: raise HTTPException(401, ...)` | reject wrong or missing keys |
| `start = time.perf_counter()` | high-resolution timer for latency |
| `except Exception: raise HTTPException(502, ...)` | if vLLM fails, return 502 Bad Gateway (an upstream error), never a stack trace |
| `if req.task == "summary": parsed = json.loads(...)` | gives clients a ready JSON object; invalid JSON becomes null instead of an error |
| `request_id=str(uuid.uuid4())` | a unique ID per request for logs and support tickets |
| `latency_ms=int(...)` | measured server-side latency |
Step 6. Run the API tests without a GPU, against the mock server (useful on any laptop and in CI):
```
cd ~/-MEDSCRIPTAI && (uvicorn tests.mock_openai:app --port 8009 > /tmp/mock.log 2>&1 &) && sleep 2 && VLLM_URL=http://localhost:8009/v1 python -m pytest -q tests/test_api.py
```
You should see: 5 passed.
FILE tests/test_api.py:
@@include tests/test_api.py
| Line | What it does (plain English) |
| `sys.path[:0] = ["api", "scripts"]` | lets the test import app.py and common.py |
| `os.environ["API_KEY"] = "test-key"` | set BEFORE importing app, because app reads it at import time |
| `client = TestClient(api.app)` | calls the app in-process, no real network needed |
| `test_health` | /health returns {"status": "ok"} |
| `test_ready_lists_adapters` | /ready says ready and lists qa-dpo |
| `test_qa_routes_to_dpo_adapter` | task qa is routed to qa-dpo and the output follows the Answer format |
| `test_summary_returns_parsed_json` | summary output is parsed into parsed_json |
| `test_auth_and_validation` | missing key -> 401; invalid task -> 422 |
FILE tests/mock_openai.py (a fake vLLM for tests, no GPU):
@@include tests/mock_openai.py
| Line | What it does (plain English) |
| `MODELS = [...]` | the names /v1/models reports, like a real vLLM with adapters |
| `def reply_for(model, text)` | returns a reply with the same SHAPE as the real one for each kind of request (judge ratings, explanations, JSON, notes, Answer: X) |
| `re.findall(r"<text \d+>", text)` | detects UltraFeedback rating prompts, which number the candidate texts |
| `@app.post("/v1/chat/completions")` | the OpenAI chat endpoint, returning choices[0].message.content and usage |
Watch out: every request loads the shared client; never create an OpenAI client per request (slow, exhausts connections).
CHECKPOINT: Step 3 returns model "qa-dpo" and Step 6 shows 5 passed.
If not: 503 from /ready = an adapter in TASK_TO_MODEL is not loaded in vLLM; check curl localhost:8000/v1/models.

## SECTION 6.6 - LangSmith tracing
WHERE: web browser (smith.langchain.com) and VS Code Terminal
GOAL: every API request visible as a trace with input, output, adapter and latency.
Step 1. Open https://smith.langchain.com and sign in.
Step 2. Click Settings (gear icon) > API Keys > Create API Key.
Step 3. Copy the key (starts with lsv2_).
Step 4. In the VS Code Terminal save it:
```
echo 'export LANGSMITH_API_KEY=lsv2_YOUR_KEY' >> ~/.bashrc && source ~/.bashrc
```
Step 5. Start the API with tracing on:
```
cd ~/-MEDSCRIPTAI/api && LANGSMITH_TRACING=true LANGSMITH_PROJECT=medscriptai VLLM_URL=http://localhost:8000/v1 API_KEY=dev-key PYTHONPATH=../scripts uvicorn app:app --port 8080
```
Step 6. In the second terminal, repeat the curl from Section 6.5 Step 3 three times.
Step 7. In LangSmith click Projects (left menu) > medscriptai.
You should see: three runs named medscriptai.generate; click one to see the input, the output, a child ChatOpenAI run with the model name (qa-dpo), token counts and latency.
| Variable | What it does |
| LANGSMITH_TRACING=true | turns tracing on (the SDK checks LANGSMITH_TRACING, or the older LANGCHAIN_TRACING_V2) |
| LANGSMITH_API_KEY | authenticates your SDK |
| LANGSMITH_PROJECT | which project the traces go to (created automatically) |
| LANGSMITH_ENDPOINT | only if your account is in the EU region: https://eu.api.smith.langchain.com |
Watch out: traces contain the full input text. With real patient data that is a privacy problem (PHI). For this project the data is public and synthetic-derived; in a real hospital you would self-host or redact inputs.
CHECKPOINT: the medscriptai project shows your runs.
If not: no project appears = tracing is off; check echo $LANGSMITH_API_KEY and that LANGSMITH_TRACING=true is on the same command line.

## SECTION 6.7 - Package the API with Docker and run the whole stack with Compose
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: a medscriptai-api:v1 image, then vLLM + API started together by one command.
Step 1. Build the image (from the repo root):
```
docker build -f api/Dockerfile -t medscriptai-api:v1 .
```
You should see: "naming to docker.io/library/medscriptai-api:v1" (about 290 MB).
Step 2. Run it against the vLLM that is already running on the VM:
```
docker run -d --name api-test --network host -e VLLM_URL=http://localhost:8000/v1 -e API_KEY=dev-key medscriptai-api:v1
```
Step 3. Test it:
```
sleep 5 && curl -s localhost:8080/ready | jq -r .status
```
You should see: ready
Step 4. Remove the test container:
```
docker rm -f api-test
```
FILE api/Dockerfile:
@@include api/Dockerfile
| Line | What it does (plain English) |
| `ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim` | the official Python image through AWS's ECR Public mirror (avoids Docker Hub's anonymous pull limits; we hit a 429 while testing). Switch with --build-arg BASE_IMAGE=python:3.12-slim |
| `FROM ${BASE_IMAGE}` | start from that small Debian image with Python 3.12 |
| `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1` | no .pyc files; logs appear immediately (important for docker logs and CloudWatch) |
| `WORKDIR /app` | all following commands run in /app |
| `COPY api/requirements.txt .` then `RUN pip install --no-cache-dir -r requirements.txt` | install pinned dependencies first, as a separate layer: code changes do not re-install packages (faster rebuilds) |
| `COPY scripts/common.py api/app.py ./` | only the two files the API needs; that is why you build from the repo root |
| `RUN useradd --create-home appuser` and `USER appuser` | do not run as root inside the container (security best practice) |
| `EXPOSE 8080` | documents the port |
| `HEALTHCHECK ... urllib.request.urlopen('http://localhost:8080/health')` | Docker marks the container healthy/unhealthy (no curl needed in the slim image) |
| `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]` | start 2 worker processes listening on all interfaces |
Step 5. Stop the standalone servers (Compose will start its own vLLM):
```
docker rm -f medscriptai-vllm medscriptai-teacher
```
Step 6. Create your settings file:
```
cp deploy/.env.example deploy/.env && nano deploy/.env
```
Step 7. In nano set HF_TOKEN, a long random API_KEY (run openssl rand -hex 24 in another terminal to make one) and LANGSMITH_API_KEY; keep the other lines; press Ctrl+O, Enter, Ctrl+X.
Step 8. Start the stack:
```
cd deploy && docker compose --env-file .env up -d && cd ..
```
Step 9. Watch until both are healthy (2-4 minutes):
```
watch -n 10 docker compose -f deploy/docker-compose.yml ps
```
You should see: vllm "healthy" and api "running"/"healthy". Press Ctrl+C.
Step 10. Test through the stack:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: $(grep ^API_KEY deploy/.env | cut -d= -f2)" -d '{"task":"summary","input":"A 54-year-old man with type 2 diabetes presented with chest pain for 2 hours. ECG showed ST elevation. He underwent primary PCI."}' | jq
```
You should see: model "summary" and parsed_json with fields such as "patient information".
FILE deploy/docker-compose.yml:
@@include deploy/docker-compose.yml
| Line | What it does (plain English) |
| `services: vllm: ... api: ...` | two containers that Compose starts, networks and restarts together |
| `image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0}` | the vLLM image; .env sets the -cu129 build for your R570 driver |
| `command: > --model ... --lora-modules ...` | the same server arguments as start_vllm.sh, written in YAML ("> " folds the lines into one) |
| `environment: HF_TOKEN, VLLM_ENABLE_CUDA_COMPATIBILITY` | from .env |
| `volumes: ${ADAPTERS_DIR:?...}:/adapters:ro` | `:?message` makes Compose STOP with that message if the variable is missing (no silent empty paths) |
| `ipc: host` | same as --ipc=host |
| `deploy.resources.reservations.devices: driver nvidia, count all, capabilities [gpu]` | Compose's way to give a container the GPU |
| `healthcheck: test: python3 -c "...urlopen('http://localhost:8000/health')"` | vLLM is healthy when its /health answers; start_period 120s gives it time to load |
| `restart: unless-stopped` | restart after crashes and reboots |
| no `ports:` under vllm | vLLM is reachable only inside the Compose network (by the name vllm), never from the internet |
| `api: image: ${API_IMAGE:-medscriptai-api:v1}` | on EC2 this becomes the ECR image address |
| `VLLM_URL: http://vllm:8000/v1` | Compose's internal DNS: the service name is the host name |
| `ports: "8080:8080"` | only the API is published |
| `depends_on: vllm: condition: service_healthy` | start the API only after vLLM is healthy |
FILE deploy/compose.aws.yml (extra settings used only on AWS):
@@include deploy/compose.aws.yml
| Line | What it does (plain English) |
| `logging: driver: awslogs` | Docker sends container logs straight to CloudWatch Logs |
| `awslogs-group: /medscriptai/api` | the log group name |
| `awslogs-create-group: "true"` | create the group if it does not exist (the EC2 role allows logs:CreateLogGroup) |
| `docker compose -f docker-compose.yml -f compose.aws.yml` | the second file is merged on top of the first |
Watch out: never commit deploy/.env (it is in .gitignore).
CHECKPOINT: docker compose ps shows both services healthy and Step 10 returns parsed JSON.
If not: docker compose -f deploy/docker-compose.yml logs vllm | tail -50 shows vLLM's error.

## SECTION 6.8 - Deploy step by step: registry, storage, secrets, IAM, logs, network, GPU machine
WHERE: VS Code Terminal on the Hyperstack VM, folder ~/-MEDSCRIPTAI
GOAL: the same production deployment done twice: first free on FakeCloud, then for real on AWS.
Before you start (both columns): export API_KEY=$(openssl rand -hex 24) and make sure LANGSMITH_API_KEY and HF_TOKEN are set (echo them). For the blue column your GPU quota request (Section 2.11) must be approved.
| Step {w=60} | FakeCloud (free practice) | Real AWS (same step) {blue} |
| 1 Target | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL && export AWS_DEFAULT_REGION=us-east-1` |
| 2 Who am I | `export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)` | same (your real 12-digit ID) |
| 3 Registry address | `export REGISTRY=localhost:4566` | `export REGISTRY=$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com` |
| 4 Create the image repository | `aws ecr create-repository --repository-name medscriptai-api --image-scanning-configuration scanOnPush=true` | same |
| 5 Log Docker in | `aws ecr get-login-password \| docker login --username AWS --password-stdin $REGISTRY` | same |
| 6 Tag the image | `docker tag medscriptai-api:v1 $REGISTRY/medscriptai-api:v1` | same |
| 7 Push | `docker push $REGISTRY/medscriptai-api:v1` | same (about 290 MB upload) |
| 8 Verify | `aws ecr describe-images --repository-name medscriptai-api --query 'imageDetails[].imageTags'` | same; console: ECR > Repositories > medscriptai-api |
| 9 Bucket | `export BUCKET=medscriptai-artifacts-$ACCOUNT_ID-us-east-1 && aws s3 mb s3://$BUCKET` | same (skip mb if you created it in Part 4) |
| 10 Adapters + compose files to S3 | `aws s3 sync outputs/adapters s3://$BUCKET/adapters --exclude "*/checkpoint-*" && aws s3 cp deploy/docker-compose.yml s3://$BUCKET/deploy/ && aws s3 cp deploy/compose.aws.yml s3://$BUCKET/deploy/` | same |
| 11 Secrets | `for n in HF_TOKEN LANGSMITH_API_KEY API_KEY; do aws secretsmanager create-secret --name medscriptai/$n --secret-string "${!n}"; done` | same |
| 12 IAM role for the machine | `aws iam create-role --role-name medscriptai-ec2-role --assume-role-policy-document file:///tmp/medscriptai-trust.json` (write the trust JSON shown below first) | same |
| 13 Least-privilege policy | `aws iam put-role-policy --role-name medscriptai-ec2-role --policy-name medscriptai-runtime --policy-document file:///tmp/medscriptai-policy.json` | same |
| 14 Instance profile | `aws iam create-instance-profile --instance-profile-name medscriptai-ec2-profile && aws iam add-role-to-instance-profile --instance-profile-name medscriptai-ec2-profile --role-name medscriptai-ec2-role` | same, then `sleep 10` (real IAM needs seconds to propagate) |
| 15 Logs | `aws logs create-log-group --log-group-name /medscriptai/api && aws logs put-retention-policy --log-group-name /medscriptai/api --retention-in-days 14` | same |
| 16 Alarm | `aws cloudwatch put-metric-alarm --alarm-name medscriptai-cpu-high --namespace AWS/EC2 --metric-name CPUUtilization --statistic Average --period 300 --evaluation-periods 2 --threshold 90 --comparison-operator GreaterThanThreshold` | same (add --alarm-actions with an SNS topic to get e-mails) |
| 17 Firewall (security group) | `export SG_ID=$(aws ec2 create-security-group --group-name medscriptai-sg --description "MedScriptAI API" --query GroupId --output text)` | same (uses your default VPC) |
| 18 Allow only your IP | `export MY_IP=$(curl -s https://checkip.amazonaws.com) && for p in 22 8080; do aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port $p --cidr $MY_IP/32; done` | same. Note: MY_IP here is the Hyperstack VM's IP; to call the API from your laptop, add your laptop's IP as well |
| 19 SSH key pair | `aws ec2 create-key-pair --key-name medscriptai-key --query KeyMaterial --output text > ~/.ssh/medscriptai-key.pem && chmod 400 ~/.ssh/medscriptai-key.pem` | same |
| 20 Machine image (AMI) | `export AMI=$(aws ec2 describe-images --owners amazon --query 'Images[0].ImageId' --output text)` (FakeCloud ships demo AMIs) | `export AMI=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)` (driver 595, CUDA 13.2, Docker included [13]) |
| 21 Boot script | `sed -e "s\|__REGION__\|us-east-1\|g" -e "s\|__REGISTRY__\|$REGISTRY\|g" -e "s\|__BUCKET__\|$BUCKET\|g" -e "s\|__TAG__\|v1\|g" deploy/ec2_user_data.sh > /tmp/user-data.sh` | same |
| 22 Launch the GPU machine | `aws ec2 run-instances --image-id $AMI --instance-type g6.xlarge --key-name medscriptai-key --security-group-ids $SG_ID --iam-instance-profile Name=medscriptai-ec2-profile --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' --user-data file:///tmp/user-data.sh --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=medscriptai-gpu}]'` | same. BILLING STARTS NOW (about US$0.805/h + EBS) |
| 23 Get its IP | `aws ec2 describe-instances --filters Name=tag:Name,Values=medscriptai-gpu --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text` | same |
| You should see | an i-... id, "running" or "pending", and a fake IP such as 52.0.0.10. FakeCloud starts a small container to represent the instance; there is no GPU and the boot script does not serve the model | a real public IP. The boot script needs 10-15 minutes (downloads about 30 GB) |
| 24 Watch the boot | not applicable | `ssh -i ~/.ssh/medscriptai-key.pem ubuntu@EC2_IP` then `sudo tail -f /var/log/cloud-init-output.log` (wait for docker compose to report started) |
| 25 Test the API | not applicable (no model behind FakeCloud) | `curl -s http://EC2_IP:8080/ready` then the /v1/generate curl from Section 6.5 with your API_KEY |
| 26 Read the logs | `aws logs describe-log-groups --query 'logGroups[].logGroupName'` | console: CloudWatch > Log groups > /medscriptai/api, or `aws logs tail /medscriptai/api --follow` |
| 27 TEAR DOWN | `TARGET=fakecloud bash deploy/teardown.sh` | `TARGET=aws bash deploy/teardown.sh` (then check EC2 > Instances shows terminated and Billing shows no running resources) |
Trust policy for Step 12 (write it with this command first):
```
cat > /tmp/medscriptai-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
```
Permissions policy for Step 13 (least privilege: only our repo, our bucket, our secrets, logs):
```
cat > /tmp/medscriptai-policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],"Resource":"arn:aws:ecr:us-east-1:$ACCOUNT_ID:repository/medscriptai-api"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Effect":"Allow","Action":["secretsmanager:GetSecretValue"],"Resource":"arn:aws:secretsmanager:us-east-1:$ACCOUNT_ID:secret:medscriptai/*"},
 {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}
EOF
```
Command pieces that are new here:
| Piece | What it does |
| `aws ecr get-login-password \| docker login --password-stdin` | ECR gives a 12-hour token; piping it avoids typing it or saving it in shell history |
| `docker tag LOCAL REGISTRY/REPO:TAG` | a second name for the same image that says where it will be pushed |
| `"${!n}"` | bash indirect expansion: the value of the variable whose NAME is in n (HF_TOKEN, ...) |
| trust policy (Principal ec2.amazonaws.com) | WHO may use the role: EC2 machines |
| permissions policy | WHAT the role may do; every Resource is limited to MedScriptAI's own repo, bucket and secrets |
| instance profile | the container that attaches a role to an EC2 machine; programs on the machine get temporary credentials automatically, no keys on disk |
| `--cidr $MY_IP/32` | /32 = exactly one IP address |
| `--block-device-mappings ... VolumeSize 150, gp3` | a 150 GB SSD root disk: the vLLM image (about 11 GB compressed) + the model (16 GB) need room |
| `--user-data file://...` | a script cloud-init runs once at first boot, as root |
| g6.xlarge | 1x NVIDIA L4 (24 GB), 4 vCPU, 16 GB RAM: the cheapest current AWS GPU that fits an 8B bf16 model with LoRA |
Placeholders:
| Placeholder | Replace with | Example |
| EC2_IP | the PublicIpAddress from Step 23 | 3.91.20.14 |
FILE deploy/ec2_user_data.sh (what the AWS machine does on first boot):
@@include deploy/ec2_user_data.sh
| Line | What it does (plain English) |
| `set -euxo pipefail` | stop on errors, and print every command into /var/log/cloud-init-output.log (-x) so you can debug |
| `REGION=__REGION__` etc. | placeholders that Step 21 (or deploy.sh) replaced with real values |
| `aws s3 sync "s3://$BUCKET/adapters" $APP_DIR/adapters` | downloads the adapters using the machine's role (no keys) |
| `secret() { aws secretsmanager get-secret-value ... }` | a small function that reads one secret by name |
| `cat > $APP_DIR/.env <<EOF ... EOF` and `chmod 600` | writes the settings file readable only by root |
| `VLLM_IMAGE=vllm/vllm-openai:v0.32.0` and `VLLM_ENABLE_CUDA_COMPATIBILITY=0` | the Deep Learning AMI has driver 595 / CUDA 13.2, so the default CUDA 13 image runs natively, no compatibility layer |
| `API_IMAGE=$REGISTRY/medscriptai-api:$TAG` | Compose pulls YOUR image from ECR |
| `aws ecr get-login-password ... \| docker login ...` | lets Docker pull from your private registry |
| `aws s3 cp "s3://$BUCKET/deploy/docker-compose.yml" ...` | the same compose files you tested on Hyperstack |
| `docker compose -f docker-compose.yml -f compose.aws.yml --env-file .env up -d` | starts vLLM + API with CloudWatch logging |
The same deployment in one command (after you have done it by hand once, this is how you work):
```
TARGET=fakecloud bash deploy/deploy.sh
```
FILE deploy/deploy.sh:
@@include deploy/deploy.sh
| Line | What it does (plain English) |
| `TARGET="${TARGET:-fakecloud}"` | the default target is the free emulator; you must type TARGET=aws on purpose |
| `MY_IP=...checkip.amazonaws.com` | your public IP, used to lock the firewall |
| `if [ "$TARGET" = "fakecloud" ]; then export AWS_ENDPOINT_URL=...` | the ONLY difference between the two targets |
| `aws ecr describe-repositories ... \|\| aws ecr create-repository ...` | "create if missing": the script is idempotent (we ran it twice in testing; the second run succeeds) |
| `aws s3 mb ... 2>/dev/null \|\| true` | ignore "bucket already exists" |
| `value="${!name:-}"` and `continue` | skip secrets you have not exported |
| `create-secret ... \|\| put-secret-value ...` | create the secret, or update it if it exists |
| `aws iam get-role ... \|\| aws iam create-role ...` | create the role only once |
| `SG_ID=$(aws ec2 describe-security-groups --filters ...)` | re-use the security group on a second run |
| `sort_by(Images,&CreationDate)[-1].ImageId` | JMESPath: sort the matching AMIs by date and take the newest |
| `sed -e "s\|__REGION__\|$AWS_REGION\|g" ...` | fills the placeholders in the boot script |
| `aws ec2 describe-key-pairs ... \|\| { aws ec2 create-key-pair ... }` | create and save the SSH key once |
| `timeout 300 aws ec2 wait instance-running` | wait (at most 5 minutes) until the machine runs |
FILE deploy/teardown.sh:
@@include deploy/teardown.sh
| Line | What it does (plain English) |
| `set -uo pipefail` (no -e) | keep going if one resource is already gone |
| `describe-instances --filters "Name=tag:Name,Values=$APP-gpu"` | find our machines by their Name tag |
| `terminate-instances` then `wait instance-terminated` | delete the machine (stops compute billing) and wait |
| `for try in 1 2 3 4 5 6; do delete-security-group ...` | a security group can only be deleted after the instance is fully gone; we retry (found while testing) |
| `delete-repository --force` | delete the repo including its images |
| `aws s3 rb ... --force` | delete the bucket and everything in it |
| `delete-secret --force-delete-without-recovery` | delete secrets immediately (no 7-30 day recovery window that is still billed) |
| `remove-role-from-instance-profile`, `delete-instance-profile`, `delete-role-policy`, `delete-role` | IAM pieces must be removed in this order |
Watch out: the most expensive mistake in this whole manual is forgetting Step 27 on real AWS. Set a phone reminder when you launch the instance.
CHECKPOINT (FakeCloud): deploy.sh ends with DONE and teardown.sh ends with TEARDOWN DONE (fakecloud).
CHECKPOINT (AWS): curl http://EC2_IP:8080/v1/generate returns an answer from your adapter; after teardown, EC2 shows the instance terminated.
If not: on AWS, "VcpuLimitExceeded" = GPU quota not approved yet (Section 2.11); "InvalidAMIID" = run Step 20 in us-east-1; API not answering after 20 minutes = read /var/log/cloud-init-output.log on the machine.

## SECTION 6.9 - (Optional) SageMaker endpoint: the managed alternative
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI
GOAL: understand the SageMaker deployment objects; practise their API calls for free.
| Step {w=60} | FakeCloud (free practice) | Real AWS (same step) {blue} |
| 1 Model | `aws sagemaker create-model --model-name medscriptai-model --primary-container Image=$REGISTRY/medscriptai-api:v1 --execution-role-arn arn:aws:iam::$ACCOUNT_ID:role/medscriptai-ec2-role` | needs a SageMaker-compatible container (answers GET /ping and POST /invocations on port 8080) and a role trusted by sagemaker.amazonaws.com. Our API image is NOT that; AWS's Large Model Inference (LMI) containers with a vLLM backend are the usual choice (verify current names in the SageMaker docs) |
| 2 Endpoint config | `aws sagemaker create-endpoint-config --endpoint-config-name medscriptai-cfg --production-variants VariantName=AllTraffic,ModelName=medscriptai-model,InstanceType=ml.g6.xlarge,InitialInstanceCount=1` | same shape |
| 3 Endpoint | `aws sagemaker create-endpoint --endpoint-name medscriptai-endpoint --endpoint-config-name medscriptai-cfg` | same; billing per hour while InService |
| 4 Status | `aws sagemaker describe-endpoint --endpoint-name medscriptai-endpoint --query EndpointStatus` | same |
| You should see | "InService" (FakeCloud emulates the control plane only; invoke-endpoint returns NotFound, we tested it) | "Creating" for 5-15 minutes, then "InService" |
| 5 Delete | `aws sagemaker delete-endpoint --endpoint-name medscriptai-endpoint` | same, plus delete-endpoint-config and delete-model |
Watch out: honest comparison for interviews: EC2 + Docker Compose (what you built) gives full control and is cheapest for one always-on model; SageMaker adds managed autoscaling, deployment guardrails and monitoring, at a higher per-hour price and with more container rules.
CHECKPOINT: on FakeCloud, describe-endpoint returned InService and delete-endpoint succeeded; on AWS, every SageMaker endpoint you created is deleted (console: SageMaker > Inference > Endpoints is empty).

## 6.10 Common mistake at this level
The mistake: exposing the model server directly (port 8000 open with runtime LoRA loading on), keeping secrets in .env files baked into images or Git, and leaving GPU machines running.
How to spot it: from your laptop, curl http://EC2_IP:8000/v1/models must FAIL (only 8080 is published, and only to your IP); docker history medscriptai-api:v1 must show no secrets; aws ec2 describe-instances after teardown must show nothing running.
