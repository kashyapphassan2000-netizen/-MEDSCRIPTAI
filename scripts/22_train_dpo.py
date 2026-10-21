"""Part 6: DPO preference alignment that continues from your SFT adapter.

python scripts/22_train_dpo.py --task qa     ->  outputs/adapters/dpo-qa
"""
from unsloth import FastLanguageModel, PatchDPOTrainer   # Unsloth first
PatchDPOTrainer()                                         # no-op in current Unsloth; kept for older versions

import argparse
import json
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from common import PREF, OUT

p = argparse.ArgumentParser()
p.add_argument("--task", default="qa", choices=["qa", "notes", "summary"])
p.add_argument("--beta", type=float, default=0.1)       # how strongly to stay close to the SFT model
p.add_argument("--lr", type=float, default=5e-6)        # DPO needs a MUCH smaller LR than SFT
p.add_argument("--epochs", type=float, default=1.0)
a = p.parse_args()
sft_dir = OUT / "adapters" / f"sft-{a.task}"
out_dir = OUT / "adapters" / f"dpo-{a.task}"

# 1) Load base + your SFT adapter as TRAINABLE. DPO will keep improving these same LoRA weights.
model, tokenizer = FastLanguageModel.from_pretrained(str(sft_dir), max_seq_length=2048, load_in_4bit=True)
tokenizer.padding_side = "left"                           # TRL's DPO collator expects left padding

# 2) Preference data: {"prompt": [...], "chosen": [...], "rejected": [...]} (conversational format).
pairs = load_dataset("json", data_files=str(PREF / f"{a.task}_pairs.jsonl"), split="train")
pairs = pairs.select_columns(["prompt", "chosen", "rejected"])
split = pairs.train_test_split(test_size=0.05, seed=42)  # 5% held out to watch reward accuracy

cfg = DPOConfig(
    output_dir=str(OUT / "checkpoints" / f"dpo-{a.task}"),
    beta=a.beta,
    loss_type=["sigmoid"],                # the original DPO loss
    max_length=2048,                      # prompt + answer are truncated to this many tokens
    per_device_train_batch_size=2,        # each example = 2 sequences (chosen + rejected)
    gradient_accumulation_steps=8,        # effective batch 16 pairs
    num_train_epochs=a.epochs,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    seed=3407,
)

# ref_model=None + a model that already has an adapter -> TRL copies it into a frozen "ref" adapter,
# so the reference policy is your SFT model (textbook DPO) without loading a second 8B model.
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, processing_class=tokenizer,
                     train_dataset=split["train"], eval_dataset=split["test"])
print("adapters inside the model:", list(trainer.model.peft_config.keys()))   # expect ['default', 'ref']

trainer.train()
metrics = trainer.evaluate()
print({k: round(v, 4) for k, v in metrics.items() if "rewards" in k})   # accuracies should be > 0.5

# Save ONLY the trained adapter ("default"), not the frozen "ref" copy.
model.save_pretrained(str(out_dir), selected_adapters=["default"])
tokenizer.save_pretrained(str(out_dir))
(out_dir / "dpo_eval.json").write_text(json.dumps(metrics, indent=1))
print(f"DONE adapter={out_dir}")
