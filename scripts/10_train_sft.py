"""Step 4: QLoRA supervised fine-tuning (SFT) of Llama-3.1-8B-Instruct with Unsloth + TRL + PEFT.

One run = one task = one LoRA adapter, e.g.  python scripts/10_train_sft.py --task qa
"""
from unsloth import FastLanguageModel                    # import Unsloth FIRST so it can patch TRL/transformers
from unsloth.chat_templates import train_on_responses_only

import argparse                                          # command-line options
import json                                              # save the training log
from datasets import load_dataset                        # read our .jsonl files
from trl import SFTConfig, SFTTrainer                    # Hugging Face TRL trainer
from common import SFT, OUT, BASE_MODEL_4BIT

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["qa", "notes", "summary"])
p.add_argument("--max-train", type=int, default=None, help="use only the first N training rows")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (-1 = use epochs)")
p.add_argument("--lr", type=float, default=2e-4)        # LoRA learns fast; 2e-4 is the usual QLoRA rate
p.add_argument("--rank", type=int, default=16)          # LoRA rank r: size of the adapter matrices
p.add_argument("--batch", type=int, default=8)          # examples per GPU step
p.add_argument("--grad-accum", type=int, default=2)     # effective batch = batch x grad-accum = 16
p.add_argument("--train-file", default=None, help="override the training .jsonl (Part 5 uses this)")
p.add_argument("--run-name", default=None)
a = p.parse_args()
run = a.run_name or f"sft-{a.task}"
out_dir = OUT / "adapters" / run

# 1) Load the base model already quantised to 4-bit (this is the "Q" in QLoRA).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_4BIT,     # 4-bit NF4 weights: ~5.7 GB on disk instead of ~16 GB
    max_seq_length=2048,            # longest example we keep is ~1,600 tokens
    load_in_4bit=True,
    dtype=None,                     # None = auto: bfloat16 on A100/L40/H100
)

# 2) Attach small trainable LoRA matrices to every attention and MLP projection.
model = FastLanguageModel.get_peft_model(
    model,
    r=a.rank,
    lora_alpha=a.rank,              # scaling = alpha / r = 1.0
    lora_dropout=0.0,               # 0 is fastest in Unsloth and works well
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # recompute activations to save VRAM
    random_state=3407,
)
model.print_trainable_parameters()  # expect ~42M trainable of ~8B (about 0.5%)


# 3) Turn each {"messages": [...]} row into one string using Llama-3.1's own chat template.
#    The template starts with <|begin_of_text|>; the trainer adds it again when tokenising,
#    so we remove it here to end up with exactly ONE begin token (double BOS hurts quality).
def to_text(batch):
    texts = [tokenizer.apply_chat_template(m, tokenize=False).removeprefix(tokenizer.bos_token)
             for m in batch["messages"]]
    return {"text": texts}


train_file = a.train_file or str(SFT / f"{a.task}_train.jsonl")
train = load_dataset("json", data_files=train_file, split="train")
if a.max_train:
    train = train.select(range(min(a.max_train, len(train))))
val = load_dataset("json", data_files=str(SFT / f"{a.task}_val.jsonl"), split="train").select(range(100))
train = train.map(to_text, batched=True, remove_columns=train.column_names)
val = val.map(to_text, batched=True, remove_columns=val.column_names)
print(train[0]["text"][:600])       # eyeball one formatted example before spending GPU time

# 4) Training settings.
cfg = SFTConfig(
    output_dir=str(OUT / "checkpoints" / run),
    dataset_text_field="text",
    max_length=2048,
    per_device_train_batch_size=a.batch,
    gradient_accumulation_steps=a.grad_accum,
    num_train_epochs=a.epochs,
    max_steps=a.max_steps,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",     # smooth decay to ~0 by the end
    warmup_steps=10,                # ramp the LR up for the first 10 steps
    optim="adamw_8bit",             # 8-bit optimiser states: big VRAM saving
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,             # keep only the 2 newest checkpoints on disk
    report_to="none",
    seed=3407,
)
trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train,
                     eval_dataset=val, args=cfg)

# 5) Only learn from the assistant's answer, not from the system/user text.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

stats = trainer.train()

# 6) Save ONLY the adapter (~80-170 MB at r=16), not the 16 GB base model.
model.save_pretrained(str(out_dir))
tokenizer.save_pretrained(str(out_dir))
(out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
print(f"DONE adapter={out_dir} train_loss={stats.training_loss:.4f}")
