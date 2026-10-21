# PART 7 - ELITE: prove your results are real (how top labs work)

## BEFORE YOU START: why this part exists
"Accuracy went from 61% to 64%" means nothing until you can answer four questions: (1) Is there test leakage? (2) Is 3 points on 500 questions real or luck? (3) Who graded it and can you trust the grader? (4) Can you roll a new version out and back safely? Top labs (Google's Med-PaLM 2, OpenAI's HealthBench, Stanford's MedHELM, AI2's Tulu 3) answer these before publishing. In this part you create five small scripts that answer them for YOUR project.
| Question | Tool you build | Section |
| leakage? | 04_contamination_check.py (13-gram near-duplicate check) | 7.2 |
| real or luck? | 33_significance.py (paired bootstrap confidence interval) | 7.3 |
| trust the judge? | 35_judge_agreement.py (Cohen's kappa vs your labels) | 7.4 |
| fields right? | 36_summary_field_scores.py (criterion-level scoring) | 7.5 |
| safe rollout? | one environment variable + Compose (no new file) | 7.6 |
| documentation | 34_report.py (model card) | 7.7 |

## 7.1 Who defines the top level today
| Who / what | Contribution you should know | Where |
| Tim Dettmers et al. (University of Washington), 2023 | QLoRA: 4-bit NF4 + LoRA, fine-tuning big models on one GPU | https://arxiv.org/abs/2305.14314 |
| Edward Hu et al. (Microsoft), 2021 | LoRA: low-rank adapters | https://arxiv.org/abs/2106.09685 |
| Rafael Rafailov, Archit Sharma, Eric Mitchell et al. (Stanford), 2023 | DPO | https://arxiv.org/abs/2305.18290 |
| Lewis Tunstall et al. (Hugging Face H4), 2023 | Zephyr: SFT + DPO on AI feedback; the recipe this project follows | https://arxiv.org/abs/2310.16944 |
| Nathan Lambert et al. (AI2), 2024 | Tulu 3: open post-training (SFT, on-policy DPO, RL with verifiable rewards) with decontamination | https://arxiv.org/abs/2411.15124 |
| Karan Singhal et al. (Google), Nature Medicine 2025 | Med-PaLM 2: 86.5% on MedQA and physician pairwise evaluation | https://doi.org/10.1038/s41591-024-03423-7 |
| Rahul K. Arora et al. (OpenAI), 2025 | HealthBench: 5,000 conversations graded with 48,562 physician-written rubric criteria | https://arxiv.org/abs/2505.08775 |
| Stanford CRFM / Stanford Medicine, 2025 | MedHELM: 121 real clinical tasks (notes, decision support, communication, admin) | https://arxiv.org/abs/2505.23802 |
| Lianmin Zheng et al. (LMSYS), 2023 | LLM-as-a-judge biases and the position-swap trick | https://arxiv.org/abs/2306.05685 |
| Seungone Kim et al., 2024 | Prometheus 2: open judge models | https://arxiv.org/abs/2405.01535 |
| Woosuk Kwon et al. (vLLM, UC Berkeley), 2023 | PagedAttention, the memory manager behind vLLM | https://arxiv.org/abs/2309.06180 |
| Ying Sheng et al. (S-LoRA), 2023 | serving thousands of LoRA adapters on one base model | https://arxiv.org/abs/2311.03285 |
| Daniel and Michael Han (Unsloth) | fast LoRA/QLoRA kernels | https://github.com/unslothai/unsloth |
| Brown et al. (OpenAI), 2020, GPT-3 paper | 13-gram overlap contamination analysis | https://arxiv.org/abs/2005.14165 |

## SECTION 7.2 - Leakage check (13-gram near-duplicates)
BEFORE YOU START: exact de-duplication (Section 4.4) misses near-copies: the same question with one word changed. We cut every question into all its 13-word windows ("13-grams") and flag a test question if at least half of its 13-grams come from ONE training question. A single shared 13-gram proves nothing: exam questions share stock phrases such as "which of the following is the most appropriate next step in management".
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: the number of near-duplicate test questions (should be 0 or very close).
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 04_contamination_check.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/04_contamination_check.py
| Line | What it does (plain English) |
| `N = 13` | n-gram length, as in the GPT-3 analysis |
| `THRESHOLD = 0.5` | flag if at least half of a test question's 13-grams come from one training question |
| `def grams(text)` | lower-case words, drop the "Options:" block, return every 13-word window |
| `index = defaultdict(set)` | 13-gram -> ids of training rows that contain it |
| `per_train_row = Counter(...)` | for one test question: how many 13-grams it shares with each training row |
| `best / len(test_grams) >= THRESHOLD` | the share from the single most similar training row |
Step 6. Run:
```
cd ~/medscriptai/scripts && source ../.venv-eval/bin/activate && python 04_contamination_check.py
```
You should see (real output on the full data):
```
qa       test rows= 500  near-duplicates of training rows=  1 (0.2%)
   e.g. Three days after admission to the intensive care unit for septic shock and bacteremia from a urinary tract inf
notes    test rows= 200  near-duplicates of training rows=  0 (0.0%)
summary  test rows= 200  near-duplicates of training rows=  0 (0.0%)
```
What we learned while building this (say it in interviews): (1) the first version counted ANY shared 13-gram and reported 13% "leakage" in QA, all false alarms from boilerplate; (2) the first full-data run then found REAL leakage: 14% (notes) and 18.5% (summary), because the clinical-notes dataset repeats 927 notes up to 5 times under different IDs. Removing repeated notes before splitting fixed it (0.0%). (3) One QA question (0.2%) remains: report it, or drop that test item and report results on 499.
Watch out: this only checks YOUR training data. Llama's pre-training data may already contain MedQA (it is public). That is why Med-PaLM 2, HealthBench and MedHELM also use new, private or expert-written test items.
CHECKPOINT: you can state the leakage numbers for all three tasks.

## SECTION 7.3 - Is the gain real? (paired bootstrap)
BEFORE YOU START: the bootstrap. Your 500 test questions are one sample of all possible questions. The bootstrap draws 500 questions WITH replacement from your 500, computes the accuracy difference, and repeats this 10,000 times. The middle 95% of those differences is the confidence interval (CI). If the CI does not contain 0, the difference is unlikely to be luck. "Paired" = both models are scored on the same resampled questions, which removes question difficulty from the noise.
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: a 95% CI for "model B accuracy minus model A accuracy".
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 33_significance.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/33_significance.py
| Line | What it does (plain English) |
| `def correct(row)` | 1 if the predicted letter equals the gold letter, else 0 |
| `a = [...]`, `b = [...]` | per-question correctness of the two models on the SAME questions |
| `idx = [rng.randrange(n) for _ in range(n)]` | one bootstrap sample: n random question indices, with replacement |
| `diffs.append(sum(b[i] - a[i] for i in idx) / n)` | the accuracy difference on that sample |
| `low, high = diffs[2.5%], diffs[97.5%]` | the 95% confidence interval |
| `SIGNIFICANT if low > 0 or high < 0` | the interval does not include 0 |
Step 6. Compare SFT with DPO, then base with SFT:
```
python 33_significance.py --a qa --b qa-dpo && python 33_significance.py --a base --b qa
```
You should see, for each comparison, three lines like:
```
accuracy qa=0.6xx  qa-dpo=0.6xx  diff=+0.0xx
95% CI of the difference: [-0.0xx, +0.0xx]
SIGNIFICANT          (or: NOT significant: the CI contains 0, do not claim an improvement)
```
Rule of thumb: on 500 questions a single accuracy has about +/- 4 points of uncertainty, so a 2-point gain is usually NOT significant; the paired test is tighter than comparing two separate intervals.
CHECKPOINT: you can write each comparison as "diff X, 95% CI [L, H], significant yes/no".

## SECTION 7.4 - Can you trust the judge? (Cohen's kappa)
BEFORE YOU START: kappa. You and the judge both label the same 30 duels (qa-dpo, qa or tie). Raw agreement can look high by chance: if the judge always says "qa-dpo" and you say "qa-dpo" 80% of the time, you "agree" 80% while the judge learned nothing. Cohen's kappa removes chance agreement: 1 = perfect, 0 = no better than chance. We saw exactly this in testing: 0.80 agreement but kappa 0.00.
WHERE: VS Code Explorer (create), VS Code editor, VS Code Terminal with (.venv-eval)
GOAL: kappa between the judge and you on 30 pairwise comparisons.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 35_judge_agreement.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/35_judge_agreement.py
| Line | What it does (plain English) |
| `judge = json.loads(...)["per_item"][: a.n]` | the judge's verdict per item (saved by 32_judge.py) |
| `if not labels.exists(): ... raise SystemExit` | the first run only creates an empty sheet for you |
| `human = {...}` | your labels; empty rows are ignored |
| `observed` | how often you and the judge agree |
| `expected` | agreement expected by chance from how often each of you uses each label |
| `kappa = (observed - expected) / (1 - expected)` | agreement beyond chance |
Step 6. Create the label sheet:
```
python 35_judge_agreement.py --judge-file ../outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
```
You should see: "fill the 'human' column in .../outputs/reports/human_labels.csv, then run this script again"
Step 7. Show item 0 (then repeat with 1, 2, ... 29 by changing the number after i=):
```
python -c "import json;a=[json.loads(l) for l in open('../outputs/preds/qa/qa-dpo.jsonl')];b=[json.loads(l) for l in open('../outputs/preds/qa/qa.jsonl')];i=0;print(a[i]['prompt'],'\n\nREFERENCE:',a[i]['reference'],'\n\nqa-dpo:',a[i]['prediction'],'\n\nqa:',b[i]['prediction'])"
```
Step 8. In the Explorer open outputs > reports > human_labels.csv.
Step 9. For each row type qa-dpo, qa or tie after the comma (example line: 0,qa-dpo) and press Ctrl+S when done.
Step 10. Run again:
```
python 35_judge_agreement.py --judge-file ../outputs/reports/judge_qa_pairwise_qa-dpo_vs_qa.json
```
You should see: items=30 agreement=0.xx Cohen's kappa=0.xx and "kappa > 0.6 = judge is usable; 0.4-0.6 = use with care; < 0.4 = do not trust this judge".
Watch out: you are not a physician. A real medical product needs clinician labels (HealthBench used 262 physicians). For this project say: "judge-human agreement measured on 30 items by the author".
CHECKPOINT: you can report kappa and what it means.

## SECTION 7.5 - Criterion-level scoring for the summary task
BEFORE YOU START: HealthBench grades every answer against a list of specific criteria. Our summary task allows a verifiable version: every field in the reference JSON (for example "patient information.age") is one criterion. Recall = how many reference fields the model filled (missed facts lower it). Precision = how many of the model's fields exist in the reference (invented fields lower it: a hallucination signal).
WHERE: VS Code Explorer (create), then VS Code Terminal with (.venv-eval)
GOAL: field precision/recall for base vs your summary adapter.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 36_summary_field_scores.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/36_summary_field_scores.py
| Line | What it does (plain English) |
| `def paths(value, prefix="")` | flattens nested JSON into field paths such as "patient information.age" |
| `elif value not in (None, "", "None")` | a field counts only if it has a real value |
| `def load(text)` | parse JSON (also inside a ```json block); None if invalid |
| `if pred is None: invalid += 1; fn += len(ref)` | an unreadable answer misses every criterion |
| `tp`, `fp`, `fn` | fields in both / invented by the model / missed by the model |
| `precision`, `recall`, `f1` | the standard definitions over all fields |
Step 6. Run:
```
python 36_summary_field_scores.py --models base summary
```
You should see: two lines like "summary    field precision=0.xxx recall=0.xxx F1=0.xxx invalid_json=x/200". Expect the fine-tuned adapter to have far fewer invalid JSON outputs and a higher recall than base.
Watch out: this checks that the right FIELDS are filled, not that each VALUE is correct. Combine it with the judge or with exact checks on simple fields (age, sex).
CHECKPOINT: you have field precision/recall for both models.

## SECTION 7.6 - Zero-downtime rollout and rollback
BEFORE YOU START: in production you never "just replace" a model. You switch traffic to the new version, watch, and switch back in seconds if it is worse. Your API picks the adapter from an environment variable (MODEL_QA), and vLLM keeps all adapters loaded, so a switch only restarts the small API container, never the model server.
WHERE: VS Code Terminal and editor, Compose stack from Section 6B.6 running
GOAL: switch task qa from qa-dpo to qa and back, without restarting vLLM.
Step 1. Ask vLLM (from inside the api container, because vLLM's port is not published) which adapters it serves:
```
cd ~/medscriptai && docker compose -f deploy/docker-compose.yml exec api python -c "import urllib.request,json; print([m['id'] for m in json.load(urllib.request.urlopen('http://vllm:8000/v1/models'))['data']])"
```
You should see: ['base', 'qa', 'notes', 'summary', 'qa-dpo']
Step 2. Open deploy/.env in the editor, add this line at the end, and press Ctrl+S:
```
MODEL_QA=qa
```
Step 3. Recreate ONLY the api container:
```
cd ~/medscriptai/deploy && docker compose --env-file .env up -d api && cd ..
```
Step 4. Check which adapter now answers qa:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: $(grep ^API_KEY deploy/.env | cut -d= -f2)" -d '{"task":"qa","input":"Which vitamin deficiency causes scurvy?\n\nOptions:\nA) A\nB) B12\nC) C\nD) D"}' | jq -r .model
```
You should see: qa
Step 5. Roll back: change the line in deploy/.env to MODEL_QA=qa-dpo, save, and repeat Step 3.
Step 6. Prove vLLM never restarted:
```
docker compose -f deploy/docker-compose.yml ps
```
You should see: vllm still "Up" for the same long time; only api was recreated.
| Concept | Meaning |
| canary | send a small share of traffic to the new version first and compare metrics |
| rollback | return to the previous version; here one variable and a 2-second restart |
CHECKPOINT: you switched and rolled back while vllm stayed up.

## SECTION 7.7 - Report and model card
WHERE: VS Code Explorer (create), VS Code Terminal with (.venv-eval), then WinSCP and the Hub website
GOAL: outputs/reports/REPORT.md for your model card, README and interviews.
Step 1. Right-click the folder scripts.
Step 2. Click New File...
Step 3. Type 34_report.py and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include scripts/34_report.py
| Line | What it does (plain English) |
| `metrics = json.loads((reports / "metrics.json").read_text())` | all automatic metrics |
| `cols = [...]` | the columns of the table |
| `lines.append(f"\| {name} \| ...")` | one table row per task/model ("-" where a metric does not apply) |
| `for path in sorted(reports.glob("judge_*.json"))` | every judge result |
| final lines | judge description, "Built with Llama", not-for-clinical-use statement |
Step 6. Run:
```
python 34_report.py
```
You should see: a Markdown table (# MedScriptAI evaluation report ...) printed and saved as outputs/reports/REPORT.md.
Step 7. In WinSCP, drag /home/ubuntu/medscriptai/outputs/reports from the right panel to your laptop.
Step 8. On your Hub adapter page click Edit model card, paste REPORT.md, and add: base model, data sources and licences, training settings (r=16, alpha=16, LR 2e-4, 1 epoch; DPO beta 0.1, LR 5e-6), leakage results (7.2), significance results (7.3), judge kappa (7.4), limitations, "Built with Llama", "not for clinical use". Click Commit changes.
CHECKPOINT: the model card shows your numbers.

## 7.8 What elite teams would try next
- **RL with verifiable rewards (GRPO).** For multiple-choice QA the reward is checkable (correct letter = 1). TRL's GRPOTrainer and Unsloth support it (verify current docs).
- **Other preference losses.** DPO vs SimPO (TRL loss_type="sigmoid_norm"), IPO, KTO, ORPO on the same pairs, each with a significance test.
- **Iterative on-policy DPO.** Regenerate pairs with the DPO model and train a second round; stop when the win rate stops moving.
- **Serving efficiency.** FP8 or AWQ base model, prefix caching for the shared system prompt, speculative decoding; measure tokens per second and cost per 1,000 requests.
- **Safety evaluation.** A red-team set (self-harm, dosing, emergencies) with expected refusals or escalation; track refusal rate next to helpfulness.
- **Privacy and regulation.** Real notes contain patient data: de-identify, keep traces private, and know that software making clinical decisions is regulated (for example FDA "software as a medical device").

## SECTION 7.9 - Commit
Step 1. Run:
```
cd ~/medscriptai && git add . && git commit -m "Part 7: leakage, significance, judge agreement, field scores, report"
```

## PART 7 RESULT CHECK
| Check | Where | Expected |
| leakage | 04_contamination_check.py | qa about 0.2%, notes 0.0%, summary 0.0% |
| significance | 33_significance.py | a CI for each comparison and a clear yes/no |
| judge reliability | 35_judge_agreement.py | kappa with interpretation |
| field scores | 36_summary_field_scores.py | precision/recall/invalid_json for base and summary |
| rollout | Section 7.6 | switch + rollback with vllm still up |
| model card | Hub page | REPORT.md + limitations + "Built with Llama" |

## 7.10 Common mistake at this level
The mistake: elite-looking analyses that are not tied to a decision: a kappa nobody reads, a CI computed on a different test set than the headline number, a leakage check run after the results are published.
How to spot it: write the question first ("is DPO better than SFT on QA accuracy?"), then the analysis, then the decision ("ship qa-dpo" or "keep qa").
