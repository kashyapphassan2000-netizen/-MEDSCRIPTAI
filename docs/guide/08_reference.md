# PART 8 - Tools and Resources Map (checked October 2026)
GitHub stars were read on 21 October 2026 and change daily. Prices marked (verify) could not be confirmed on the official page.
| Tool | What it does | Free / Open-source / Paid | Exact cost + free-tier limits | Who uses it | Platform | Official link (+ stars) | Best for | My verdict |
| Unsloth 2026.10.7 | 2x faster, lower-VRAM LoRA/QLoRA/GRPO training | open source (Apache-2.0) + paid Pro | free | individuals, startups, Kaggle winners | Linux/Windows, NVIDIA (AMD/Apple partial) | https://github.com/unslothai/unsloth (78k) | single-GPU QLoRA | default choice for one GPU |
| TRL 1.15/1.16 | SFT, DPO, GRPO, reward trainers | open source (Apache-2.0) | free | Hugging Face, AI2, most open post-training papers | Python | https://github.com/huggingface/trl (19.6k) | the standard post-training API | learn its configs deeply |
| PEFT 0.21 | LoRA and other adapters | open source | free | industry-wide | Python | https://github.com/huggingface/peft (21.8k) | adapters | always used under TRL/Unsloth |
| Transformers 5.x | model and tokenizer library | open source | free | everyone | Python | https://github.com/huggingface/transformers (168k) | loading any model | core skill |
| bitsandbytes 0.50 | 4-bit/8-bit quantisation, 8-bit optimizers | open source | free | QLoRA users | Linux/Windows | https://github.com/bitsandbytes-foundation/bitsandbytes (8.5k) | QLoRA | needed, rarely touched |
| distilabel 1.5.3 | synthetic data and AI-feedback pipelines | open source (Apache-2.0) | free | Argilla/HF community, Zephyr-style datasets | Python | https://github.com/argilla-io/distilabel (3.4k) | generate + rate pipelines | last release Jan 2025: pin versions or write plain async OpenAI calls |
| vLLM 0.32 | high-throughput inference, multi-LoRA | open source (Apache-2.0) | free | Meta, Mistral, AWS, many startups | Linux, NVIDIA/AMD/TPU | https://github.com/vllm-project/vllm (94.6k) | serving open models | industry default |
| SGLang | inference engine (RadixAttention, structured output) | open source | free | xAI, LMSYS, cloud providers | Linux | https://github.com/sgl-project/sglang (37.6k) | very high throughput, agents | strong alternative to vLLM |
| Axolotl / LLaMA-Factory | config-driven fine-tuning frameworks | open source | free | researchers, startups | Linux | https://github.com/axolotl-ai-cloud/axolotl (12.6k), https://github.com/hiyouga/LLaMA-Factory (75.6k) | multi-GPU / no-code training | good once you understand TRL |
| FastAPI 0.156 | Python web API framework | open source (MIT) | free | Microsoft, Netflix, Uber, countless startups | Python | https://github.com/fastapi/fastapi (103k) | ML APIs | default for Python services |
| LangSmith | tracing, evaluation, prompt management | paid SaaS with free tier | Developer plan US$0: 1 seat, about 5k base traces/month; overage about US$2.50-5.00 per 1k traces (sources disagree, verify) [11] | LangChain users, many enterprises | web + SDK | https://smith.langchain.com (SDK 1.1k) | quick tracing | fine for demos; check data-privacy terms |
| Langfuse | open-source LLM tracing/evals | open source (MIT) + cloud | self-host free; cloud free tier (verify) | startups wanting self-hosting | web + SDK | https://github.com/langfuse/langfuse (36.1k) | self-hosted tracing | best when data must stay inside |
| Docker / Compose | containers | free (Docker Desktop paid for large companies) | Engine free | everyone | Linux/Win/Mac | https://docs.docker.com | packaging | must-know |
| FakeCloud 0.51 | local AWS emulator, 106 services incl. ECR push/pull, EC2, SageMaker control plane | open source (AGPL-3.0) | free, no account [5] | developers testing AWS code | Linux/macOS/Docker | https://github.com/faiscadev/fakecloud (761) | free AWS practice and integration tests | perfect for learning; not a GPU |
| LocalStack | the best-known AWS emulator | open source core + paid tiers | free tier needs an account/auth token (verify) | many companies' CI | Docker | https://github.com/localstack/localstack (65k) | team CI | industry standard alternative |
| moto | Python library that mocks AWS | open source | free | Python unit tests | Python | https://github.com/getmoto/moto (8.7k) | unit tests | lightweight |
| AWS (ECR, S3, EC2, SageMaker, CloudWatch) | production cloud | paid | g6.xlarge about US$0.805/h on demand in us-east-1 [12]; ECR storage about US$0.10/GB-month (verify) | most enterprises | web/CLI | https://aws.amazon.com | real deployment | learn the core 6 services |
| Hyperstack | GPU cloud VMs | paid | A100-80G-PCIe about US$1.35/h, L40 about US$1.00/h, RTX A6000 about US$0.50/h on demand [1] (verify) | startups, researchers | web/API | https://www.hyperstack.cloud | cheap single-GPU training | good value; remember hibernate |
| lm-evaluation-harness | standard benchmark runner (MedQA, MMLU, ...) | open source (MIT) | free | EleutherAI, HF Open LLM Leaderboard | Python | https://github.com/EleutherAI/lm-evaluation-harness (14.3k) | comparable benchmark numbers | add it after this project |
| Stanford HELM / MedHELM | holistic evaluation framework | open source | free | Stanford, research | Python | https://github.com/stanford-crfm/helm (3k) | rigorous medical evaluation | read the MedHELM paper |
| Prometheus-eval | open judge models | open source | free | researchers | Python | https://github.com/prometheus-eval/prometheus-eval (1.1k) | judging without paid APIs | good second judge |
| DeepEval | LLM test framework (pytest-style) | open source + cloud | free core | startups | Python | https://github.com/confident-ai/deepeval (19k) | CI tests for LLM apps | nice for API regression tests |
| uv | fast Python package/env manager | open source | free | fast-growing industry default | all | https://github.com/astral-sh/uv (91k) | environments | use everywhere |
| VS Code + Remote-SSH | editor on remote machines | free | free | everyone | all | https://code.visualstudio.com (194k) | working on GPU VMs | default |
| WinSCP | SFTP/SCP file manager | free (GPL) | free | Windows users | Windows | https://winscp.net (3.9k) | moving files | simple and reliable |
Best free setup: Google Colab or Kaggle free GPU (T4, 15-16 GB; no bf16, slower) + Unsloth + TRL + FakeCloud + LangSmith Developer + Hugging Face Hub. You can do Parts 4-5 on a T4 with --batch 2 --grad-accum 8, but not the 14B teacher.
Best paid setup: 1x H100 80 GB (Hyperstack about US$1.90-3.00/h [1]) for everything; AWS or GCP for deployment with Terraform; Langfuse self-hosted or LangSmith Plus for tracing; W&B or MLflow for experiment tracking.
Best setup within this manual's budget (about US$30-40): Hyperstack 1x A100-80G-PCIe only while working + hibernate + FakeCloud for every AWS rehearsal + one 2-3 hour real AWS deployment with immediate teardown.
Five open-source projects worth studying:
- **huggingface/alignment-handbook** (5.7k stars): the exact Zephyr SFT + DPO recipes and configs; compare them with yours. https://github.com/huggingface/alignment-handbook
- **allenai/open-instruct** (3.9k stars): AI2's Tulu 3 post-training code (SFT, DPO, RLVR) with decontamination tooling. https://github.com/allenai/open-instruct
- **epfLLM/meditron** (2.2k stars): an open medical LLM with documented data and evaluation. https://github.com/epfLLM/meditron
- **openai/simple-evals** (4.7k stars): includes the HealthBench grading code and rubric approach. https://github.com/openai/simple-evals
- **vllm-project/vllm** (94.6k stars): read docs/features/lora.md and the LoRA code path to understand multi-adapter batching. https://github.com/vllm-project/vllm

# PART 9 - Keep It Working
| Task | How |
| Save your work | push adapters and datasets to the Hub (Sections 4.10, 5.12); git add/commit/push your code changes; aws s3 sync outputs/reports to S3 |
| Push your code to GitHub (portfolio) | 1) github.com > New repository > name medscriptai > Create (no README). 2) on the VM: git remote add origin https://github.com/YOUR_GITHUB_USER/medscriptai.git. 3) git branch -M main. 4) git push -u origin main (user name + a GitHub personal access token as password: github.com > Settings > Developer settings > Personal access tokens) |
| Back up the VM | your important files are on the Hub and in Git; the VM is disposable. To be extra safe, Hyperstack > Volumes > create a Shared Storage Volume and copy outputs/ there (see Hyperstack docs) |
| Stop costs | Hibernate the Hyperstack VM (Section 2A.13); TARGET=aws bash deploy/teardown.sh; delete unused Hub repos is optional (free) |
| Restart after hibernate | Restore the VM; update the IP in ~/.ssh/config and WinSCP; on the VM: mkdir -p /ephemeral/hf-cache; restart FakeCloud (Section 2B.11 Step 6); cd deploy && docker compose --env-file .env up -d |
| Resume an interrupted training run | checkpoints are in outputs/checkpoints/RUN; change trainer.train() to trainer.train(resume_from_checkpoint=True) in 10_train_sft.py and run the same command again |
| Update libraries safely | never upgrade in place: create .venv-train2, install the new versions, run 03_validate_jsonl.py and a --max-steps 20 training test, compare, then switch. Unsloth releases often pin new transformers/trl versions |
| Update vLLM | change VLLM_IMAGE to the new tag, restart, re-run 30_generate_preds.py on 50 rows and compare outputs with the old version |
| Undo a code change | git diff (see changes), git restore FILE (undo one file), git log + git revert COMMIT (undo a pushed commit) |
| Free disk space | docker system prune (removes stopped containers and dangling images); du -sh $HF_HOME/hub/* to find big cached models, then delete only the folders you no longer need |
| Rotate secrets | create a new HF/LangSmith key, aws secretsmanager put-secret-value, restart the API; delete the old key on the website |

# PART 10 - Troubleshooting
| Symptom | Likely cause | Exact fix (steps) |
| Hyperstack connection problems (timeouts, publickey, VS Code stuck, WinSCP key) | see the dedicated table | MS Part 2A, Section 2A.14 |
| SyntaxError / invalid character when running a script you created | the paste was cut, or curly quotes came from an edited copy | open the file, Ctrl+A, Delete, paste again from the grey box, Ctrl+S; check with python -m py_compile scripts/FILE.py |
| ModuleNotFoundError: No module named 'common' | script run from the wrong folder | cd ~/medscriptai/scripts first (scripts import common.py from their own folder) |
| ssh: Connection timed out | SSH access/firewall not enabled or wrong IP after restore | 1) Hyperstack > VM > Firewall: Enable SSH access. 2) Copy the current PUBLIC IP. 3) Update HostName in ~/.ssh/config |
| torch.cuda.is_available() is False | CPU build of PyTorch installed | cd ~/medscriptai && uv pip install --python .venv-train -r requirements/train.txt --torch-backend cu126 --reinstall |
| CUDA out of memory during training | batch too big, or vLLM/teacher still running on the GPU | 1) nvidia-smi to see who uses memory. 2) docker rm -f medscriptai-vllm medscriptai-teacher. 3) or add --batch 4 --grad-accum 4 |
| Two <|begin_of_text|> tokens / quality worse than expected | chat template applied twice | use 10_train_sft.py as given (removeprefix(tokenizer.bos_token)); print train[0]["text"] and check it starts with <|start_header_id|> |
| vLLM: "CUDA driver version is insufficient for CUDA runtime version" | image CUDA newer than the driver | 1) keep -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 with the -cu129 image (data-centre GPU). 2) or deploy with the R595 / CUDA 13.2 OS image and VLLM_IMAGE=vllm/vllm-openai:v0.32.0 |
| vLLM returns 404 "The model `qa` does not exist" | adapter not registered or wrong name | 1) curl localhost:8000/v1/models. 2) check outputs/adapters/sft-qa/adapter_config.json exists. 3) restart serve/start_vllm.sh |
| Requests for different adapters are slow / queue | --max-loras left at the default 1 | start vLLM with --max-loras 4 (already in start_vllm.sh and docker-compose.yml) |
| distilabel: No module named 'requests' / datasets errors | wrong environment or unpinned versions | cd ~/medscriptai && uv pip install --python .venv-synth -r requirements/synth.txt, then source .venv-synth/bin/activate |
| Docker build: 429 Too Many Requests pulling the base image | anonymous pull rate limit | 1) wait 10 minutes, or 2) docker build --build-arg BASE_IMAGE=python:3.12-slim ... (other registry), or 3) docker login to Docker Hub |
| AWS: VcpuLimitExceeded when launching g6.xlarge | GPU quota still 0 | Service Quotas > Amazon EC2 > Running On-Demand G and VT instances > request 8 vCPUs (Section 2B.12); wait for approval |
| aws CLI: Could not connect to the endpoint URL http://localhost:4566 | FakeCloud not running (e.g. after hibernate) | 1) nohup fakecloud --storage-mode persistent --data-path ~/fakecloud-data > ~/fakecloud.log 2>&1 &. 2) curl localhost:4566/_fakecloud/health |
| Real AWS command accidentally went to FakeCloud (or the opposite) | AWS_ENDPOINT_URL still set / not set | echo $AWS_ENDPOINT_URL before every blue-column step; unset AWS_ENDPOINT_URL for real AWS |
| teardown: DependencyViolation deleting the security group | instance still shutting down | run TARGET=aws bash deploy/teardown.sh again after 1-2 minutes (the script already retries 6 times) |

# PART 11 - Practice Projects (goals and success criteria only, no solutions)
## 11.1 Easy - a fourth adapter: patient-friendly explanations
Goal: an adapter "explain" that rewrites a clinical note in plain language for patients (reading age about 12).
Requirements: generate 2,000 training pairs with distilabel from the notes data (teacher writes the plain version); add a SYSTEM["explain"] prompt; train with 10_train_sft.py; serve it as a fifth adapter; add task "explain" to the API.
Success criteria: the API returns explanations for 20 unseen notes; a readability score (for example Flesch-Kincaid grade) is at least 3 grades lower than the source note; the judge rates faithfulness at least 4/5 on 80% of items.
## 11.2 Medium - RL with verifiable rewards for QA
Goal: improve qa-dpo with GRPO using the correct letter as the reward.
Requirements: use TRL's GRPOTrainer (or Unsloth's GRPO support) on the 1,891 DPO prompts; reward = 1 for the correct letter, 0 otherwise, plus a small format reward; evaluate exactly as in Sections 6A.7 and 7.3.
Success criteria: a QA accuracy gain over qa-dpo whose 95% paired bootstrap interval excludes 0, with format_ok still above 99%.
## 11.3 Hard - production hardening on AWS
Goal: a deployment a security reviewer would accept.
Requirements: infrastructure as code (Terraform or AWS CDK) for everything in deploy.sh; HTTPS with an Application Load Balancer and an ACM certificate; the API in a private subnet; secrets only from Secrets Manager; CloudWatch alarm on 5xx rate and p95 latency with e-mail alerts; a canary that sends 10% of qa traffic to a new adapter; a load test (for example locust) report; teardown with one command; everything rehearsed on FakeCloud first.
Success criteria: terraform apply and destroy both succeed on FakeCloud and AWS; p95 latency under 3 s at 5 requests/second on g6.xlarge for qa; zero secrets in Git, images or user data; the canary can be rolled back in under 1 minute.

# PART 12 - Cheat Sheet
## Part 2 - workspace
```
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\hyperstack_ed25519 -C "medscriptai"
ssh -i $env:USERPROFILE\.ssh\hyperstack_ed25519 ubuntu@VM_IP
nvidia-smi
echo 'export HF_HOME=/ephemeral/hf-cache' >> ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/medscriptai/{scripts,serve,api,deploy,tests,requirements,data,outputs}
uv venv .venv-train --python 3.12
uv pip install --python .venv-train -r requirements/train.txt --torch-backend cu126
uv venv .venv-synth --python 3.12 && uv pip install --python .venv-synth -r requirements/synth.txt
uv venv .venv-eval --python 3.12 && uv pip install --python .venv-eval -r requirements/eval.txt --torch-backend cu126
curl -fsSL https://fakecloud.dev/install.sh | bash -s -- --version v0.51.0
nohup fakecloud --storage-mode persistent --data-path ~/fakecloud-data > ~/fakecloud.log 2>&1 &
export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
unset AWS_ENDPOINT_URL        # back to real AWS
tmux new -s NAME   |   Ctrl+B then D (detach)   |   tmux attach -t NAME
```
## Part 4 - data and first SFT (.venv-train, folder scripts)
```
python 01_download_data.py
python 02_build_sft_jsonl.py
python 03_validate_jsonl.py
python 10_train_sft.py --task qa --max-train 2000
python 11_chat_test.py --task qa --n 3
python 12_push_to_hub.py --folder ../outputs/adapters/sft-qa --repo YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora
aws s3 sync outputs/adapters/sft-qa s3://$BUCKET/adapters/sft-qa
```
## Part 5 - synthetic data, three adapters, evaluation
```
bash serve/start_teacher.sh                                  # teacher/judge on :8001
python 20_synth_explanations.py --limit 50                   # (.venv-synth)
python 10_train_sft.py --task qa --train-file ../data/sft/qa_train_synth.jsonl --max-train 12000
python 10_train_sft.py --task notes && python 10_train_sft.py --task summary
GPU_UTIL=0.35 bash serve/start_vllm.sh                       # student + adapters on :8000
python 30_generate_preds.py --task qa --models base qa       # (.venv-eval)
python 31_eval_metrics.py
python 32_judge.py --task notes --mode score --models base notes --n 100
```
## Part 6A/6B - DPO, serving, API, Docker, deployment
```
python 21_build_dpo_pairs.py --task qa --limit 1500          # (.venv-synth)
python 22_train_dpo.py --task qa                             # (.venv-train)
curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name":"NAME","lora_path":"/adapters/FOLDER"}'
curl -s -X POST localhost:8000/v1/unload_lora_adapter -H "Content-Type: application/json" -d '{"lora_name":"NAME"}'
python 32_judge.py --task qa --mode pairwise --models qa-dpo qa --n 200
VLLM_URL=http://localhost:8000/v1 API_KEY=dev-key PYTHONPATH=../scripts uvicorn app:app --port 8080
docker build -f api/Dockerfile -t medscriptai-api:v1 .
cd deploy && docker compose --env-file .env up -d
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
docker tag medscriptai-api:v1 $REGISTRY/medscriptai-api:v1 && docker push $REGISTRY/medscriptai-api:v1
TARGET=fakecloud bash deploy/deploy.sh      |   TARGET=aws bash deploy/deploy.sh
TARGET=fakecloud bash deploy/teardown.sh    |   TARGET=aws bash deploy/teardown.sh
```
## Part 7 - elite checks
```
python 04_contamination_check.py
python 33_significance.py --a qa --b qa-dpo
python 35_judge_agreement.py --judge-file ../outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
python 36_summary_field_scores.py --models base summary
python 34_report.py
```
## Key numbers and settings
| Setting | Value | Why |
| LoRA rank / alpha / dropout | 16 / 16 / 0 | scale 1.0; fast in Unsloth |
| target modules | q,k,v,o,gate,up,down | all linear layers |
| SFT LR / epochs / batch | 2e-4 / 1 / 16 effective | QLoRA standard |
| DPO beta / LR / batch | 0.1 / 5e-6 / 16 pairs | stable DPO |
| max sequence length | 2048 (train), 4096 (serve) | data max about 1,600 tokens |
| DPO loss at step 0 | 0.693 = ln 2 | policy equals reference |
| vLLM LoRA flags | --enable-lora --max-loras 4 --max-lora-rank 16 --max-cpu-loras 8 | default max-loras is 1 |
| Memory rule of thumb | 2 bytes/param (bf16), ~0.5-0.7 bytes/param (4-bit) | 8B: 16 GB vs ~6 GB |
| Hyperstack states | SHUTOFF = still billed; HIBERNATED = GPU not billed, /ephemeral may be wiped | Section 2A.13 |
| FakeCloud | http://localhost:4566, account 123456789012, any keys | Section 2B.11 |

# PART 13 - Self-Test
1. What happens if you remove .removeprefix(tokenizer.bos_token) from 10_train_sft.py?
2. Do this: compute how many optimizer steps one epoch takes with 7,040 notes examples, --batch 8 and --grad-accum 2.
3. What happens if you start vLLM without --max-loras and three users call qa, notes and summary at the same moment?
4. What happens to the DPO loss at step 0, and why?
5. Do this: write the aws command that lists the images in the medscriptai-api ECR repository, for FakeCloud.
6. What happens if you hibernate the Hyperstack VM with the Hugging Face cache on /ephemeral, and what must you do after Restore?
7. Your QA accuracy goes from 0.612 to 0.628 on 500 questions and the paired 95% CI is [-0.010, +0.042]. What may you claim?
8. What happens if the LLM judge always answers "A" in pairwise mode with position swapping?
9. Do this: change the API so task qa is answered by the SFT adapter instead of the DPO adapter, without editing Python code.
10. Why does the EC2 instance profile policy list specific ARNs instead of "Resource": "*" for ECR images, S3 and secrets?

# PART 14 - Sources
[1] Hyperstack GPU pricing: https://www.hyperstack.cloud/gpu-pricing (and https://computeprices.com/providers/hyperstack/gpus/a100pcie)
[2] Hyperstack getting started (billing starts when ACTIVE; console flow): https://docs.hyperstack.cloud/docs/getting-started/
[3] Hyperstack VM states and billing; hibernation: https://docs.hyperstack.cloud/docs/billing/states-and-billing and https://docs.hyperstack.cloud/docs/virtual-machines/hibernation
[4] Hyperstack operating system images: https://docs.hyperstack.cloud/docs/virtual-machines/images
[5] FakeCloud: https://fakecloud.dev and https://github.com/faiscadev/fakecloud
[6] Hyperstack ephemeral storage (/ephemeral, wiped on hibernate): https://docs.hyperstack.cloud/docs/storage/ephemeral
[7] FakeCloud install and ECR docs: https://fakecloud.dev/docs/getting-started/install and https://fakecloud.dev/docs/services/ecr
[8] vLLM LoRA adapters (flags, runtime load/unload): https://docs.vllm.ai/en/latest/features/lora.html
[9] TRL SFTTrainer: https://huggingface.co/docs/trl/main/en/sft_trainer
[10] TRL DPOTrainer: https://huggingface.co/docs/trl/main/en/dpo_trainer
[11] LangSmith pricing (secondary sources, verify on the official page): https://www.langchain.com/pricing, https://www.zenml.io/blog/langsmith-pricing, https://www.metacto.com/blogs/the-true-cost-of-langsmith-a-comprehensive-pricing-integration-guide
[12] AWS g6.xlarge / g5.xlarge pricing (aggregators): https://calculator.holori.com/aws/ec2/g6.xlarge/us-east-1, https://www.doit.com/compute/spot/us-east-1/g6.xlarge
[13] AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) release notes: https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-base-gpu-ami-ubuntu-24-04.html
[14] Unsloth documentation: https://unsloth.ai/docs and https://github.com/unslothai/unsloth
[15] Llama 3.1 Community License: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and https://www.llama.com/llama3_1/license/
[16] distilabel on PyPI (1.5.3, January 2025): https://pypi.org/project/distilabel/
[17] Datasets: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options, https://huggingface.co/datasets/openlifescienceai/medmcqa, https://huggingface.co/datasets/qiaojin/PubMedQA, https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes
[18] Qwen2.5-14B-Instruct: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
[19] LangSmith SDK: https://github.com/langchain-ai/langsmith-sdk and https://docs.smith.langchain.com
[20] Papers: QLoRA https://arxiv.org/abs/2305.14314, LoRA https://arxiv.org/abs/2106.09685, DPO https://arxiv.org/abs/2305.18290, Zephyr https://arxiv.org/abs/2310.16944, Tulu 3 https://arxiv.org/abs/2411.15124, UltraFeedback https://arxiv.org/abs/2310.01377, LLM-as-a-judge https://arxiv.org/abs/2306.05685, Prometheus 2 https://arxiv.org/abs/2405.01535, PagedAttention https://arxiv.org/abs/2309.06180, S-LoRA https://arxiv.org/abs/2311.03285, GPT-3 https://arxiv.org/abs/2005.14165, BERTScore https://arxiv.org/abs/1904.09675
[21] Medical evaluation: Med-PaLM 2 https://doi.org/10.1038/s41591-024-03423-7, HealthBench https://arxiv.org/abs/2505.08775, MedHELM https://arxiv.org/abs/2505.23802
[22] AWS documentation: ECR https://docs.aws.amazon.com/ecr/, EC2 user data https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html, Service Quotas https://docs.aws.amazon.com/servicequotas/, Secrets Manager https://docs.aws.amazon.com/secretsmanager/
[23] uv: https://docs.astral.sh/uv/  ·  VS Code Remote-SSH: https://code.visualstudio.com/docs/remote/ssh  ·  WinSCP: https://winscp.net/eng/docs/start

# ANSWER KEY
1. The trainer adds <|begin_of_text|> again, so every example starts with two BOS tokens. Training still runs, but the model learns from a sequence it never sees at inference (vLLM sends one BOS), which can hurt quality. The CPU smoke test caught exactly this.
2. 7,040 / (8 x 2) = 440 optimizer steps.
3. The default --max-loras is 1, so only one adapter can be active in a GPU batch; requests for the other two adapters wait for the next batches. Latency rises even though the GPU has capacity. With --max-loras 4 all three run in the same batch.
4. About 0.693 (= ln 2). At step 0 the policy equals the reference (the frozen "ref" copy of the SFT adapter), so both log-ratios are 0, the margin is 0 and -log(sigmoid(0)) = ln 2.
5. aws --endpoint-url http://localhost:4566 ecr describe-images --repository-name medscriptai-api (or set AWS_ENDPOINT_URL and drop the flag).
6. GPU billing stops, the root disk is kept, but /ephemeral is wiped, so the cache is lost (it re-downloads automatically). After Restore: run mkdir -p /ephemeral/hf-cache, update the IP if it changed, and restart FakeCloud.
7. Only that the observed difference is +1.6 points and NOT statistically significant: the interval contains 0. Do not claim an improvement; collect more test data or a bigger effect.
8. With swapping, A-then-B gives "A" (model 1) and B-then-A gives "A" (model 2), so the orders disagree and every item becomes a tie. The win rates are 0/0 and ties 100%: a clear signal of a broken or position-biased judge.
9. Set MODEL_QA=qa in deploy/.env and run docker compose --env-file .env up -d api in the deploy folder (only the API container restarts).
10. Least privilege: if the machine is compromised, the attacker can only read MedScriptAI's own image, bucket and secrets, not every image, bucket and secret in the account. "Resource": "*" is kept only for ecr:GetAuthorizationToken (it has no resource) and logs.
