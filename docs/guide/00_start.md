# MEDSCRIPTAI - Zero-to-Elite Lab Manual (beginner edition, October 2026)
Who this is for: a complete beginner in fine-tuning who knows a little Python. Every step is one action. Every file you need is created by you, by hand, in a folder you create. Nothing is skipped.
Assumptions (change anything that is wrong for you): Windows 11 laptop with no usable GPU; goal = a portfolio project you can build, show and defend in AI-engineer interviews; budget = pay-as-you-go GPU on Hyperstack (about US$25-40 for the whole manual), FakeCloud is free, real AWS is optional (about US$3-10 if you follow the teardown steps).

## WHERE TO FIND WHAT (read this first)
This document is split into tabs. Click a tab name in the left panel of Google Docs (Document tabs) to open it.
| If you are looking for... | Open this tab | Section |
| What the project is, cost, tools, accounts to create | MEDSCRIPT AI (this tab) | Part 0, Part 1 |
| Hyperstack account, GPU machine, SSH key, connecting VS Code and WinSCP | MS Part 2A - Hyperstack + SSH + VS Code + WinSCP | 2A.1 - 2A.14 |
| Creating the project folder, Python environments, FakeCloud, AWS CLI | MS Part 2B - Project Folder + Tools on the VM | 2B.1 - 2B.12 |
| The ideas behind everything (read before coding) | MS Part 3 - Core Idea | 3.1 - 3.10 |
| Data -> .jsonl -> first fine-tune -> test -> publish | MS Part 4 - Basics | 4.1 - 4.12 |
| Synthetic data (distilabel), three adapters, evaluation | MS Part 5 - Intermediate | 5.1 - 5.14 |
| DPO, vLLM multi-LoRA hot-swap, FastAPI, LangSmith | MS Part 6A - DPO + vLLM + API | 6A.1 - 6A.12 |
| Docker, ECR, S3, IAM, EC2 deployment (FakeCloud and AWS side by side) | MS Part 6B - Docker + Cloud Deploy | 6B.1 - 6B.12 |
| Contamination, significance tests, judge calibration, rollout | MS Part 7 - Elite | 7.1 - 7.8 |
| Tools map, maintenance, troubleshooting, projects, cheat sheet, self-test, sources | MS Parts 8-14 - Reference | 8 - 14 |
How every practical section is written:
| Label | Meaning |
| WHERE: | the exact place you do the work (which app, which window, which folder) |
| GOAL: | what this section achieves, in one line |
| Step 1. Step 2. ... | one action per step; do them in order |
| grey box | text you type or paste EXACTLY (never type the prompt sign $) |
| You should see: | what appears if the step worked |
| Watch out: | the mistake beginners make here, and how to avoid it |
| CHECKPOINT: | what must be true before you continue; If not: the fix |
| Real AWS (same step) column, light blue | the same cloud step on real AWS; the white column is FakeCloud (free practice) |
Golden rules: (1) never skip a CHECKPOINT; (2) if a step fails, stop and fix it before the next step; (3) read the "BEFORE YOU START" box of every part before typing anything.

# PART 0 - Start Here

## 0.1 What MedScriptAI is (in plain words)
MedScriptAI is your own medical AI assistant. You take Meta's Llama-3.1-8B-Instruct (a general chatbot model with 8 billion parameters) and teach it three medical jobs: answer medical exam questions with an explanation (qa), turn a doctor-patient conversation into a clinical note (notes), and turn a clinical note into structured JSON data (summary).
Teaching an existing model new behaviour is called post-training. You will do the two most important kinds: SFT (supervised fine-tuning: show the model thousands of examples of correct answers) and DPO (direct preference optimisation: show the model pairs of answers and tell it which one is better).
Real-world use: AI medical scribes (Nuance DAX, Abridge, Suki), exam-prep tutors, extraction of structured data from health records, hospital copilots. Important: real clinical use needs regulatory approval; this project is an educational portfolio system, never a medical device.

## 0.2 The whole journey in one picture
laptop (VS Code + WinSCP) --SSH--> Hyperstack GPU computer --> download data --> clean to .jsonl --> synthetic data with a teacher model --> SFT training (3 small "adapters") --> DPO training --> evaluation --> serve with vLLM --> FastAPI web API --> Docker image --> FakeCloud (free AWS practice) --> real AWS (optional) --> tear down.
| Part | What you will do | Time estimate |
| 0 Start Here | understand the project, tools and costs | 20 min |
| 1 Prerequisites | check your skills, create accounts | 1-2 h |
| 2A Hyperstack + connection | GPU machine, SSH key, VS Code, WinSCP | 1-2 h |
| 2B Project folder + tools | folders, Python environments, FakeCloud, AWS CLI | 1-2 h |
| 3 Core Idea | the ideas behind every step | 45 min |
| 4 Basics | data -> .jsonl -> first fine-tuned adapter -> test -> publish | 3-4 h (about 1.5 GPU-h) |
| 5 Intermediate | synthetic data, three adapters, evaluation | 1 day (about 6 GPU-h) |
| 6A Advanced | DPO, vLLM multi-LoRA serving, FastAPI, LangSmith | 1 day (about 4 GPU-h) |
| 6B Advanced | Docker, ECR, S3, IAM, EC2, teardown | half a day (about 1 GPU-h + US$3-5 AWS) |
| 7 Elite | prove your results are real | 4-6 h |
| 8-14 Reference | tools, maintenance, troubleshooting, projects, cheat sheet, self-test, sources | reference |

## 0.3 What you will be able to do at the end
- Download open medical datasets, clean them and convert them to the chat .jsonl format that training libraries expect.
- Fine-tune Llama-3.1-8B-Instruct with QLoRA (Unsloth + TRL + PEFT), then improve it with DPO, and explain every setting.
- Create synthetic training data and preference pairs with distilabel and a teacher model, with quality filters.
- Evaluate honestly: accuracy, ROUGE-L, BERTScore, LLM-as-a-judge with position swap, confidence intervals, contamination checks.
- Serve one base model with several hot-swappable adapters in vLLM behind a FastAPI web API traced in LangSmith.
- Package it with Docker, push it to (Fake)ECR, run it on an AWS GPU machine, and delete everything so the bill stops.

## 0.4 Every place you will work
| Place | What it is | What you do there |
| Your laptop: PowerShell | the Windows terminal | create the SSH key, first SSH login |
| Your laptop: VS Code + Remote-SSH | code editor connected to the GPU computer | create every folder and file, run every command |
| Your laptop: WinSCP | drag-and-drop file transfer | copy reports and files between laptop and GPU computer |
| Your laptop: web browser | websites | Hyperstack console, Hugging Face, LangSmith, AWS console |
| Hyperstack GPU computer (Ubuntu 24.04 + NVIDIA A100 80 GB) | a computer you rent per hour | all training, serving, Docker and FakeCloud |
| FakeCloud (on the GPU computer, port 4566) | free copy of AWS for practice | practise every AWS command at no cost |
| Real AWS (region us-east-1) | Amazon's cloud | registry, storage, permissions, secrets, logs, a GPU server |
| Hugging Face Hub | website hosting models and datasets | download Llama and data; publish your adapters |
| LangSmith | tracing website | see every request to your API |

## 0.5 What each piece does
| Component | Its job |
| Llama-3.1-8B-Instruct | the base model you improve |
| Hugging Face datasets library | downloads MedQA, MedMCQA, PubMedQA and clinical notes |
| .jsonl file | a text file with one JSON object per line: the standard training input |
| distilabel | builds synthetic-data pipelines (a teacher model writes, a judge model rates) |
| Qwen2.5-14B-Instruct | the teacher and judge model (bigger and stronger than your 8B model; Apache-2.0 licence) |
| bitsandbytes | stores the big model in 4-bit so it fits in GPU memory (the "Q" in QLoRA) |
| PEFT (LoRA) | adds small trainable "adapter" matrices; only about 0.5% of the parameters are trained |
| Unsloth | makes LoRA training about 2x faster and lighter |
| TRL | Hugging Face's training library: SFTTrainer and DPOTrainer |
| vLLM | a very fast model server; serves many adapters on one base model |
| FastAPI | your web API (POST /v1/generate) in front of vLLM |
| LangSmith | records every API request (input, output, time) |
| Docker / Docker Compose | packages the API into an image; starts vLLM + API together |
| FakeCloud | free local AWS emulator: same aws commands, address http://localhost:4566 |
| AWS ECR / S3 / IAM / Secrets Manager / CloudWatch / EC2 | image registry, file storage, permissions, secrets, logs and alarms, the cloud GPU server |

## 0.6 What it costs
| Item | Price (October 2026) | How much you need | Approx. cost |
| Hyperstack 1x A100-80G-PCIe | about US$1.35 per hour [1] (verify on the pricing page) | about 13-15 hours in total | US$18-21 |
| Hyperstack storage while hibernated | about US$0.10 per TB per hour [3] | a few days | under US$3 |
| FakeCloud | free and open source [5] | always | US$0 |
| AWS g6.xlarge (1x NVIDIA L4 24 GB) | about US$0.805 per hour in us-east-1 [12] | 2-3 hours | about US$2-3 |
| AWS ECR, S3, Secrets Manager, CloudWatch | cents for this project | Part 6B | under US$2 |
| Hugging Face, LangSmith Developer plan | free (LangSmith: about 5k traces per month [11]) | always | US$0 |
Watch out: Hyperstack keeps billing a machine that is only "stopped" (SHUTOFF). Only HIBERNATED or deleted machines stop the GPU bill [3]. Section 2A.13 shows how.

# PART 1 - Prerequisites

## 1.1 Checklist
| Item | MUST or NICE-TO-HAVE |
| Python basics: variables, lists, dicts, functions, f-strings, reading an error message | MUST |
| JSON: objects, arrays, double-quoted strings | MUST |
| Terminal basics: cd, ls, pwd, cat, mkdir | MUST |
| What a GPU and VRAM are | MUST |
| HTTP APIs: URL, POST, JSON body, status codes | MUST |
| Git basics | NICE-TO-HAVE |
| Docker basics | NICE-TO-HAVE (taught in Part 6B) |
| Statistics: mean, confidence interval | NICE-TO-HAVE (taught in Part 7) |

## 1.2 Refreshers and 1-minute checks
Python. A dict maps keys to values; a list keeps order. Check: what does this print?
```
row = {"task": "qa", "messages": [1, 2, 3]}
print(len(row["messages"]), row.get("source", "none"))
```
Answer: 3 none.
JSON. Text format: {} objects, [] arrays, strings MUST use double quotes. A .jsonl file has one JSON object per line. Check: which line is NOT valid JSON?
```
{"role": "user", "content": "hi"}
{'role': 'user', 'content': 'hi'}
```
Answer: the second (single quotes are Python, not JSON).
Terminal. pwd = where am I; ls = list files; cd folder = go into folder; cd .. = go up one folder; mkdir -p a/b = make folders; cat file = print a file. Check: after mkdir -p test/one and ls test you should see one.
GPU and VRAM. The GPU does the maths of the model in parallel; VRAM is the GPU's memory. 8 billion parameters x 2 bytes (16-bit) = 16 GB just for the weights; in 4-bit about 5-6 GB. Check: how much VRAM do 14 billion parameters need in 16-bit? Answer: about 28 GB.
HTTP APIs. A client sends a request (method, URL, JSON body); the server answers with a status code (200 OK, 401 not authorised, 422 invalid input, 500 server error) and JSON. Check: which code means "your API key is missing"? Answer: 401.

## 1.3 Accounts to create now (some approvals take a day)
| What | Where | Cost | Which option |
| Hugging Face account | https://huggingface.co/join | free | normal account |
| Hugging Face token | https://huggingface.co/settings/tokens | free | Write token (Section 2A.3) |
| Llama 3.1 licence | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | free | fill the form; approval can take hours |
| Hyperstack account | https://console.hyperstack.cloud | pay as you go | add US$30-40 credit (Section 2A.4) |
| LangSmith account | https://smith.langchain.com | free Developer plan | you create the API key in Part 6A |
| AWS account (optional) | https://aws.amazon.com | pay as you go | region us-east-1; request GPU quota early (Section 2B.12) |
Watch out: the Llama licence says any model you publish that was fine-tuned from Llama must have a name starting with "Llama" and you must show "Built with Llama" [15]. Your adapters will be called Llama-3.1-8B-MedScriptAI-... for this reason.
CHECKPOINT: you have a Hugging Face account, the Llama licence is requested, and you can log in to Hyperstack.
