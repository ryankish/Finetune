import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# Paths and configurations
model_name = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "Orion-zhen/dpo-codealpaca-emoji"
cache_dir = "/data2/ryan/finetune/cache"
output_dir = "/data2/ryan/finetune/output"

# Create directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load tokenizer and model with memory optimizations
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_cache=False,  # Disable KV cache
    torch_dtype=torch.float16,
    device_map="auto"  # Spread model across GPUs
)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, cache_dir=cache_dir)

def preprocess(example):
    prompt = example["prompt"].strip()
    chosen = tokenizer(
        prompt + "\n" + example["chosen"].strip(),
        truncation=True,
        max_length=512,
        padding="max_length"
    )["input_ids"]
    rejected = tokenizer(
        prompt + "\n" + example["rejected"].strip(),
        truncation=True,
        max_length=512,
        padding="max_length"
    )["input_ids"]
    return {"chosen": chosen, "rejected": rejected}

dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names, batched=True)

# Configure DPO training
dpo_config = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-7,
    max_grad_norm=0.1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=1,
    eval_strategy="no",
    save_total_limit=2,
    fp16=True,
    report_to="none",
    max_length=512,
    max_prompt_length=256,
    max_target_length=256,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    precompute_ref_log_probs=False,
)

# Train the model
trainer.train()

# Save final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
