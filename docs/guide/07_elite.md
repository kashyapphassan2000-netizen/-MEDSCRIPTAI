# PART 7 - ELITE: how top labs and practitioners do it

## 7.0 Concept
What: the habits that separate a demo from work a frontier lab, a health-AI company or a strong reviewer would trust: proving there is no test leakage, proving a gain is not noise, proving the judge agrees with humans, scoring against explicit criteria, rolling out model versions without downtime, and documenting everything in a model card.
Why: in interviews and in real teams, "accuracy went from 61% to 64%" is worthless until you can answer: was the test set contaminated, is 3 points significant on 500 questions, who graded it and how reliable is the grader, and can you roll it back?
When: before you claim any result anywhere (CV, README, interview, paper).
Harder than Part 6 because: the work is statistical and methodological, not just engineering, and it can (and should) prove your own earlier claims wrong.

## 7.1 Who defines the top level today
| Who / what | Contribution you should know | Where |
| Tim Dettmers et al. (University of Washington), 2023 | QLoRA: 4-bit NF4 + LoRA, double quantisation, paged optimizers; fine-tuning 65B models on one GPU | https://arxiv.org/abs/2305.14314 |
| Edward Hu et al. (Microsoft), 2021 | LoRA: low-rank adapters | https://arxiv.org/abs/2106.09685 |
| Rafael Rafailov, Archit Sharma, Eric Mitchell et al. (Stanford), 2023 | DPO: preference optimisation without a reward model or RL | https://arxiv.org/abs/2305.18290 |
| Lewis Tunstall et al. (Hugging Face H4), 2023 | Zephyr: distilled SFT + DPO on AI-feedback (UltraFeedback) for a 7B model; the recipe this project follows | https://arxiv.org/abs/2310.16944 |
| Nathan Lambert et al. (AI2), 2024 | Tulu 3: open post-training recipe (SFT, DPO with on-policy data, RL with verifiable rewards), with decontamination | https://arxiv.org/abs/2411.15124 |
| Karan Singhal et al. (Google), Nature Medicine 2025 | Med-PaLM 2: expert-level MedQA (86.5%) and physician pairwise evaluation on many axes | https://doi.org/10.1038/s41591-024-03423-7 |
| Rahul K. Arora et al. (OpenAI), 2025 | HealthBench: 5,000 health conversations graded with 48,562 physician-written rubric criteria | https://arxiv.org/abs/2505.08775 |
| Stanford CRFM / Stanford Medicine, 2025 | MedHELM: holistic evaluation across 121 real clinical tasks (notes, decision support, communication, admin) | https://arxiv.org/abs/2505.23802 |
| Lianmin Zheng et al. (LMSYS), 2023 | LLM-as-a-judge: position, verbosity and self-preference bias, and the swap trick you used | https://arxiv.org/abs/2306.05685 |
| Seungone Kim et al., 2024 | Prometheus 2: open evaluator models for rubric and pairwise grading | https://arxiv.org/abs/2405.01535 |
| Woosuk Kwon et al. (vLLM, UC Berkeley), 2023 | PagedAttention: the KV-cache memory manager behind vLLM | https://arxiv.org/abs/2309.06180 |
| Ying Sheng et al., 2023 (S-LoRA) and Lequn Chen et al., 2023 (Punica) | serving thousands of LoRA adapters on one base model | https://arxiv.org/abs/2311.03285 |
| Daniel and Michael Han (Unsloth) | hand-written Triton kernels that make LoRA/QLoRA about 2x faster with less VRAM | https://github.com/unslothai/unsloth |
| Brown et al. (OpenAI), 2020, GPT-3 paper appendix | 13-gram overlap contamination analysis, the method in Section 7.2 | https://arxiv.org/abs/2005.14165 |

## SECTION 7.2 - Prove there is no test leakage (near-duplicate contamination check)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: the number of test questions that are near-copies of training questions (should be 0).
Step 1. Run:
```
python 04_contamination_check.py
```
You should see: qa test rows= 500 near-duplicates of training rows= 0 (0.0%) and the same for notes and summary.
FILE scripts/04_contamination_check.py:
@@include scripts/04_contamination_check.py
| Line | What it does (plain English) |
| `N = 13` | n-gram length, as in the GPT-3 contamination analysis |
| `THRESHOLD = 0.5` | flag a test question if at least half of its 13-grams come from ONE training question |
| `def grams(text)` | lower-case words, cut the "Options:" block, return the set of all 13-word windows |
| `index = defaultdict(set)` | inverted index: 13-gram -> ids of training rows that contain it |
| `per_train_row = Counter(i for g in test_grams for i in index.get(g, ()))` | for one test question, how many of its 13-grams each training row shares |
| `best / len(test_grams) >= THRESHOLD` | the share coming from the single most similar training row |
What we learned while building this (a real elite lesson): the first version flagged 13% of QA test questions as contaminated after exact de-duplication. Inspection showed false positives: exam boilerplate ("which of the following is the most appropriate next step in management", standard lab panels, "is brought to the emergency department because of a 2-day history of"). Counting a single shared 13-gram over-reports leakage in templated domains; requiring that half of a question comes from one training row fixed it, and a planted near-copy (one word changed) was still caught.
Watch out: this checks YOUR training data. The base model's pre-training data may already contain MedQA (it is public). That is why Med-PaLM 2, HealthBench and MedHELM also evaluate on new, private or expert-written items. Say this in interviews.
CHECKPOINT: 0 near-duplicates (or a short list you then remove from training and re-train).

## SECTION 7.3 - Prove a gain is real (paired bootstrap confidence interval)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: a 95% confidence interval for "DPO accuracy minus SFT accuracy".
Step 1. Run:
```
python 33_significance.py --a qa --b qa-dpo
```
Step 2. Also compare base vs SFT:
```
python 33_significance.py --a base --b qa
```
You should see: accuracy qa=... qa-dpo=... diff=+0.0xx, a 95% CI such as [-0.010, +0.052] and either SIGNIFICANT or NOT significant.
FILE scripts/33_significance.py:
@@include scripts/33_significance.py
| Line | What it does (plain English) |
| `def correct(row)` | 1 if the predicted letter equals the gold letter, else 0 |
| `a = [...]`, `b = [...]` | per-question correctness for the two models, on the SAME 500 questions |
| `idx = [rng.randrange(n) for _ in range(n)]` | a bootstrap sample: draw 500 question indices with replacement |
| `diffs.append(sum(b[i] - a[i] for i in idx) / n)` | the accuracy difference on that resampled test set; PAIRED because both models are scored on the same resampled questions |
| `rounds = 10000` | repeat 10,000 times to get the distribution of the difference |
| `low, high = diffs[2.5%], diffs[97.5%]` | the middle 95% of the differences = the confidence interval |
| `SIGNIFICANT if low > 0 or high < 0` | if the interval does not contain 0, the difference is unlikely to be noise |
Rule of thumb you should be able to say in an interview: on 500 multiple-choice questions, the 95% interval of a single accuracy is about +/- 4 points, so a 2-point gain is usually NOT significant; paired tests are tighter because both models share the hard questions.
CHECKPOINT: you can state each comparison as "diff X, 95% CI [L, H]".

## SECTION 7.4 - Prove the judge is trustworthy (agreement with your own labels)
WHERE: VS Code Terminal and WinSCP (or VS Code editor), folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: Cohen's kappa between the LLM judge and you on 30 pairwise comparisons.
Step 1. Create the label sheet:
```
python 35_judge_agreement.py --judge-file ../outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
```
You should see: fill the 'human' column in .../outputs/reports/human_labels.csv, then run this script again
Step 2. Show the two answers for item i (repeat for i = 0..29; change I_NUMBER):
```
python -c "import json;a=[json.loads(l) for l in open('../outputs/preds/qa/qa-dpo.jsonl')];b=[json.loads(l) for l in open('../outputs/preds/qa/qa.jsonl')];i=I_NUMBER;print(a[i]['prompt'],'\n\nREFERENCE:',a[i]['reference'],'\n\nqa-dpo:',a[i]['prediction'],'\n\nqa:',b[i]['prediction'])"
```
Step 3. Open outputs/reports/human_labels.csv in VS Code (Explorer panel) and type qa-dpo, qa or tie in the human column for each row you judged. Save with Ctrl+S.
Step 4. Run again:
```
python 35_judge_agreement.py --judge-file ../outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
```
You should see: items=30 agreement=0.xx Cohen's kappa=0.xx and the interpretation line.
FILE scripts/35_judge_agreement.py:
@@include scripts/35_judge_agreement.py
| Line | What it does (plain English) |
| `judge = json.loads(...)["per_item"][: a.n]` | the judge's verdict per item, saved by 32_judge.py |
| `if not labels.exists(): ... raise SystemExit` | first run only creates the sheet for you |
| `human = {int(r["i"]): r["human"].strip() ...}` | your labels; empty rows are ignored |
| `observed = sum(j == h ...) / n` | raw agreement: how often you and the judge said the same |
| `expected = sum(jc[k] * hc[k] ...) / (n * n)` | agreement you would get by chance given how often each of you uses each label |
| `kappa = (observed - expected) / (1 - expected)` | Cohen's kappa: agreement beyond chance. 1 = perfect, 0 = no better than chance |
A lesson from testing: with a judge that always answered "qa-dpo", raw agreement was 0.80 but kappa was 0.00. High agreement can be meaningless when one label dominates; that is exactly why kappa exists.
Watch out: you are not a physician. For a real medical product the labels must come from clinicians (HealthBench used 262 physicians). For this project, be transparent: "judge-human agreement measured on 30 items by the author".
CHECKPOINT: you can report kappa with its interpretation.

## SECTION 7.5 - Criterion-level (rubric) scoring for structured summaries
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: precision/recall over the individual fields of the clinical summary, the HealthBench rubric idea made verifiable.
Step 1. Run:
```
python 36_summary_field_scores.py --models base summary
```
You should see: one line per model with field precision, recall, F1 and invalid_json counts. The fine-tuned summary adapter should have far fewer invalid JSON outputs and higher recall than base.
FILE scripts/36_summary_field_scores.py:
@@include scripts/36_summary_field_scores.py
| Line | What it does (plain English) |
| `def paths(value, prefix="")` | flattens nested JSON into field paths such as "patient information.age"; list items are merged |
| `elif value not in (None, "", "None"): out.add(...)` | a field only counts if it has a real value |
| `def load(text)` | parses JSON, also inside a ```json fence; None if invalid |
| `if pred is None: invalid += 1; fn += len(ref)` | an unreadable answer misses every criterion |
| `tp += len(ref & got)` | fields present in both = criteria met |
| `fp += len(got - ref)` | fields the model invented that the reference does not have = hallucination signal |
| `fn += len(ref - got)` | facts the model missed |
| `precision`, `recall`, `f1` | standard definitions over all fields of all notes |
Watch out: this checks that the right FIELDS are filled, not that each VALUE is correct. For values, combine it with the judge (Section 5.7) or exact match on simple fields (age, sex). Saying this limitation out loud is part of being elite.
CHECKPOINT: you have field precision/recall for base and summary.

## SECTION 7.6 - Zero-downtime adapter rollout and rollback (canary)
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI, Compose stack from Section 6.7 running
GOAL: switch task qa from qa-dpo to a new adapter version without restarting vLLM, then roll back in seconds.
Step 1. Ask vLLM (from inside the api container, because vLLM's port is not published) which adapters it serves:
```
docker compose -f deploy/docker-compose.yml exec api python -c "import urllib.request,json; print([m['id'] for m in json.load(urllib.request.urlopen('http://vllm:8000/v1/models'))['data']])"
```
You should see: ['base', 'qa', 'notes', 'summary', 'qa-dpo'].
Step 2. Change the route for qa in deploy/.env (open the file and add this line, then save):
```
MODEL_QA=qa
```
Step 3. Recreate ONLY the api container (vLLM keeps running and keeps its GPU memory):
```
cd deploy && docker compose --env-file .env up -d api && cd ..
```
Step 4. Check the route:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: $(grep ^API_KEY deploy/.env | cut -d= -f2)" -d '{"task":"qa","input":"Which vitamin deficiency causes scurvy?\n\nOptions:\nA) A\nB) B12\nC) C\nD) D"}' | jq -r .model
```
You should see: qa (the SFT adapter now answers). Rollback = set MODEL_QA=qa-dpo and repeat Step 3.
| Concept | What it means |
| canary | send a small share of traffic to the new version and watch metrics before switching everyone (with two API deployments behind a load balancer, or a percentage router in the API) |
| rollback | returning to the previous version; here one variable and a 2-second container restart |
| why it is cheap | adapters are tiny and vLLM can host many at once; the expensive base model never reloads |
Watch out: the Compose file does not enable VLLM_ALLOW_RUNTIME_LORA_UPDATING (production safety). To add a brand-new adapter in Compose, add it to --lora-modules and recreate vllm, or enable runtime updates only on a private network.
CHECKPOINT: you switched and rolled back without vLLM restarting (docker compose ps shows vllm Up for the same duration).

## SECTION 7.7 - Report and model card
WHERE: VS Code Terminal, folder ~/-MEDSCRIPTAI/scripts, (.venv-eval) active
GOAL: outputs/reports/REPORT.md ready for your Hugging Face model card, README and interviews.
Step 1. Run:
```
python 34_report.py
```
Step 2. Download outputs/reports with WinSCP (drag the folder to your laptop).
Step 3. On your Hub adapter page click Edit model card and paste REPORT.md, then add: base model, data sources and licences, training settings (r=16, alpha=16, LR 2e-4, 1 epoch, DPO beta 0.1, LR 5e-6), contamination result, significance results, judge kappa, limitations, "Built with Llama", and "not for clinical use".
FILE scripts/34_report.py:
@@include scripts/34_report.py
| Line | What it does (plain English) |
| `metrics = json.loads((reports / "metrics.json").read_text())` | all automatic metrics from Section 5.6/6.4 |
| `cols = [...]` | the columns of the Markdown table |
| `lines.append(f"\| {name} \| ...")` | one table row per task/model; "-" where a metric does not apply |
| `for path in sorted(reports.glob("judge_*.json"))` | every judge result file |
| final lines | judge description, "Built with Llama", and the not-for-clinical-use statement |
CHECKPOINT: REPORT.md opens and shows your numbers.

## 7.8 What elite teams would do next (pick one as your stretch goal)
- **RL with verifiable rewards (GRPO).** For multiple-choice QA the reward is checkable (correct letter = 1). TRL's GRPOTrainer and Unsloth support it; this is the DeepSeek-R1 / Tulu 3 style of post-training (verify current TRL/Unsloth GRPO docs).
- **Other preference losses.** Compare DPO with SimPO (loss_type="sigmoid_norm" in TRL), IPO, KTO or ORPO on the same pairs; report the significance test for each.
- **Iterative on-policy DPO.** Regenerate pairs with the DPO model and train a second round; stop when the win rate stops moving.
- **Serving efficiency.** FP8 or AWQ-quantised base model for serving, prefix caching for the long shared system prompt, speculative decoding; measure tokens/second and cost per 1,000 requests.
- **Safety evaluation.** A red-team set (self-harm, dosage requests, emergencies) with expected refusals or escalation messages; track refusal rate next to helpfulness.
- **Privacy and regulation.** Real clinical notes contain PHI: de-identify, keep traces private (self-hosted LangSmith or redaction), and know that software making clinical decisions is regulated (for example FDA "software as a medical device").

## 7.9 Common mistake at this level
The mistake: running elite-looking analyses that are not tied to a decision: a kappa nobody reads, a CI computed on a different test set than the headline number, a contamination check run after the paper is written.
How to spot it: every analysis answers one question that would change what you ship or what you claim. Write the question first ("is DPO better than SFT on QA accuracy?"), then the analysis, then the decision ("ship qa-dpo" or "keep qa").
