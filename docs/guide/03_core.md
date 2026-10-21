# PART 3 - The Core Idea (read this before you write any code)

## 3.1 One analogy for the whole project
Llama-3.1-8B-Instruct is a smart new graduate doctor who knows a lot but does not write in your hospital's style. SFT (supervised fine-tuning) is residency: you show thousands of worked examples and the doctor imitates them. DPO (preference optimisation) is a senior consultant who looks at two of the doctor's notes and says "this one is better"; the doctor shifts towards the better habits. LoRA means you do not rewrite the doctor's whole brain; you hand them a slim specialty notebook (an adapter). QLoRA compresses the textbook to fit in a backpack (4-bit) while the notebook stays sharp. vLLM multi-LoRA is one doctor carrying three notebooks (qa, notes, summary) and opening the right one for each patient in milliseconds.

## 3.2 Tokens and the chat template
A model never sees letters. A tokenizer cuts text into tokens (word pieces) and turns each into a number. Llama-3.1 has 128,256 different tokens. "Nitrofurantoin" becomes several tokens; "the" is one.
A chat model also needs to know WHO is speaking. Llama-3.1 uses special tokens around every message. This is the exact text the model sees for one of your training examples (shortened):
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are MedScriptAI, a careful medical education assistant. ...<|eot_id|><|start_header_id|>user<|end_header_id|>

A patient presents with symptoms of Hypoglycemia. ... Options:
A) Insulinoma
B) Accidental sulfonylurea ingestion
C) Accidental exogenous Insulin administration
D) Accidental Metformin ingestion<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: C) Accidental exogenous Insulin administration
Explanation: ...<|eot_id|>
```
| Token | Meaning |
| <|begin_of_text|> | start of the whole conversation (must appear exactly once) |
| <|start_header_id|>role<|end_header_id|> | "the next message is from system / user / assistant" |
| <|eot_id|> | end of this message |
| Cutting Knowledge Date ... | Llama-3.1's template adds this to every system message automatically |
Rule: training and serving must produce EXACTLY this format. That is why every script imports the same system prompts from one file (scripts/common.py).

## 3.3 What "training" actually does
The model predicts the next token, one at a time. Training shows it a correct example and measures how surprised it was by each real next token. That surprise is the loss (cross-entropy). An optimizer nudges the weights so that next time the loss is a bit lower.
Tiny example: after "Answer: C) Accidental exogenous", the correct next token is "Insulin". If the model gave "Insulin" a probability of 0.05, the loss for that token is -ln(0.05) = 3.0. After training it gives 0.9, loss -ln(0.9) = 0.1.
Loss masking: we only want the model to learn to WRITE ANSWERS, not to write questions. So the loss is computed only on the assistant part; the system and user tokens are masked (their label is set to -100, which means "ignore").
| Word | Plain meaning |
| step | one update of the weights |
| batch | how many examples are used for one step (we use 16) |
| epoch | one full pass through the training data |
| learning rate (LR) | how big each nudge is (too big = chaos, too small = no learning) |
| validation loss | the loss on examples the model does NOT train on; if it goes up while training loss goes down, the model is memorising (overfitting) |

## 3.4 LoRA in numbers
One attention matrix in Llama-3.1-8B (q_proj) is 4,096 x 4,096 = 16.8 million numbers. Full fine-tuning would change all of them, in every one of the 32 layers, plus all the other matrices: 8 billion numbers, about 16 GB of weights plus optimizer state that is several times bigger.
LoRA freezes the big matrix W and learns two thin matrices B (4,096 x 16) and A (16 x 4,096) instead. Their product B x A has the same shape as W, but only 2 x 4,096 x 16 = 131,072 trainable numbers. With adapters on 7 matrices in 32 layers you train about 42 million numbers (0.5% of the model). The saved adapter file is only about 80-170 MB.
| LoRA setting | Our value | Meaning |
| r (rank) | 16 | width of the thin matrices; 8-16 for focused tasks, 32-64 for big behaviour changes |
| lora_alpha | 16 | scaling: the update is multiplied by alpha / r = 1.0 |
| target_modules | q, k, v, o, gate, up, down | which matrices get an adapter (all linear layers works best) |

## 3.5 QLoRA: making an 8B model fit
QLoRA stores the frozen base weights in 4-bit (the NF4 format) instead of 16-bit: 16 GB becomes about 5.7 GB. The small LoRA matrices stay in 16-bit and are the only part trained. Quality stays close to full 16-bit LoRA (QLoRA paper, Dettmers et al. 2023). Unsloth adds fast custom GPU code on top, about 2x faster with less memory.

## 3.6 Data: train, validation, test, and leakage
| Split | Used for | Rule |
| train | the model learns from it | the biggest part |
| validation (val) | watch the loss during training | never trained on |
| test | final scores you report | never seen until evaluation; never used to tune |
Leakage (contamination) means a test question, or a near-copy of it, is also in the training data. Then the score measures memory, not skill. While building this manual, the full-data run found that the clinical-notes dataset repeats 927 notes up to 5 times under different IDs, so 14-18% of the notes/summary test items had a copy in training. The data script now removes repeated notes before splitting (result: 0.0%). Lesson: always check.

## 3.7 Synthetic data (Part 5): what it is and why
Synthetic data = training examples written by another AI model instead of by humans. Our problem: MedQA gives only the correct letter, for example "Answer: D) Nitrofurantoin", with no reasoning. A model trained on letters learns to guess. So we ask a bigger, stronger model (the teacher, Qwen2.5-14B-Instruct) to explain WHY the given correct answer is right. This is called rationale generation: the teacher is told the answer, so it explains instead of guessing.
Before and after (one MedQA training target):
```
BEFORE:  Answer: D) Nitrofurantoin
AFTER:   Answer: D) Nitrofurantoin
         Explanation: Nitrofurantoin is a first-line choice for uncomplicated cystitis in pregnancy because ...
```
distilabel is the library that runs this as a pipeline: load rows -> send each to the teacher -> collect the replies -> save. A quality filter then throws away bad explanations (too short, too long, or not mentioning the correct option). Risk: the teacher can be wrong in subtle ways, so you always read a random sample yourself.

## 3.8 DPO (Part 6A): learning from "better vs worse"
DPO needs triples: a prompt, a chosen answer (better) and a rejected answer (worse). We create them from your own SFT model: it answers each prompt twice at high temperature (so the answers differ), the teacher answers once, the teacher rates all three (1-5), and for qa we also check whether the letter is correct. Best = chosen, worst = rejected.
DPO then raises the probability of chosen answers and lowers rejected ones, while a reference model (a frozen copy of your SFT adapter) keeps it from drifting too far. beta = 0.1 controls the strength of that leash. At step 0 the model equals the reference, so the DPO loss is exactly ln 2 = 0.693; it falls as the model learns the preferences.

## 3.9 Evaluation (Parts 5-7): five kinds of evidence
| Method | Question it answers | Example |
| accuracy | did it pick the correct letter? | 312 correct of 500 = 0.624 |
| ROUGE-L | how much of the reference's word order does the answer share? | reference "acute kidney injury", answer "kidney injury" -> high overlap |
| BERTScore | is the MEANING similar, even with different words? | "heart attack" vs "myocardial infarction" scores high |
| LLM-as-a-judge | which answer would an expert prefer? | the teacher compares two notes; we ask twice with the order swapped to remove position bias |
| statistics | is a difference real or luck? | a 95% confidence interval that does not contain 0 |

## 3.10 Serving and production (Parts 6A-6B)
vLLM loads the base model once on the GPU and applies a small adapter per request. Clients talk to it with the OpenAI API format; the "model" field chooses the adapter:
```
{"model": "qa-dpo", "messages": [{"role": "user", "content": "Which drug is safe ...?"}]}
```
| Piece | Job |
| FastAPI app | your public endpoint: checks the API key, picks the adapter for the task, adds the system prompt, calls vLLM, returns JSON |
| LangSmith | records each request so you can debug latency and bad answers |
| Docker image | the API packaged with its exact dependencies, runs the same everywhere |
| ECR | AWS's private registry where images are stored |
| EC2 g6.xlarge | the AWS GPU computer that runs vLLM + API |
| IAM role, security group, Secrets Manager, CloudWatch | who may do what, which network ports are open, where passwords live, where logs go |
| FakeCloud | practise all AWS commands for free before paying for real AWS |
CHECKPOINT: you can explain in your own words: what a token is, why the loss is masked, what r=16 means, why the test set must not leak, what synthetic data is for, what chosen/rejected mean, and why we ask the judge twice.
