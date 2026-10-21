# MedScriptAI

Domain-specific medical assistant built on **Llama-3.1-8B-Instruct** with a full post-training pipeline:
open medical data → `.jsonl` → synthetic data (distilabel) → **QLoRA SFT** (Unsloth + TRL + PEFT) → **DPO** →
evaluation (accuracy, ROUGE-L, BERTScore, LLM-as-a-judge) → **vLLM multi-LoRA** serving → **FastAPI + LangSmith** →
**Docker** → **AWS ECR / EC2** (practised for free on FakeCloud first).

One base model, three hot-swappable LoRA adapters:

| adapter | task | data |
|---|---|---|
| `qa` / `qa-dpo` | medical MCQ answer + explanation | MedQA (CC-BY-4.0), MedMCQA (Apache-2.0), PubMedQA (MIT) |
| `notes` | doctor–patient conversation → clinical note | AGBonnet/augmented-clinical-notes (MIT) |
| `summary` | clinical note → structured JSON | AGBonnet/augmented-clinical-notes (MIT) |

## Layout

```
scripts/   01-04 data · 10-12 SFT · 20-22 synthetic data + DPO · 30-34 evaluation
serve/     vLLM start scripts (student with LoRAs on :8000, teacher/judge on :8001)
api/       FastAPI app + Dockerfile (CPU image, talks to vLLM)
deploy/    docker-compose, deploy.sh / teardown.sh (TARGET=fakecloud | aws), EC2 user-data
tests/     mock OpenAI/vLLM server + API tests (run without a GPU)
docs/      the step-by-step lab manual (PDF) and plan images
```

## Quick start (GPU VM)

```bash
uv venv .venv-train && . .venv-train/bin/activate
uv pip install -r requirements/train.txt --torch-backend cu126
cd scripts && python 01_download_data.py && python 02_build_sft_jsonl.py && python 03_validate_jsonl.py
python 10_train_sft.py --task qa --max-train 2000
```

The full walkthrough (Parts 0–14) is in `docs/`.

Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc.
Educational project: not a medical device and not medical advice.
