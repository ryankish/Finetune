import os
from functools import cache
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# Load the base model and tokenizer
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir="/data2/ryan/finetune/cache")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir="/data2/ryan/finetune/cache")
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset("Orion-zhen/dpo-codealpaca-emoji", cache_dir="/data2/ryan/finetune/cache")

# Configure DPO training
dpo_config = DPOConfig(
    beta=0.1,  # Preference loss coefficient
    max_length=128,  # Maximum length of the sequences
    num_train_epochs=3,  # Number of training epochs
    learning_rate=5e-5,  # Learning rate
    output_dir="/data2/ryan/finetune/out",  # Output directory
)
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Batch size per GPU
    gradient_accumulation_steps=2,   # To simulate a larger batch size if needed
    learning_rate=5e-7,             # Learning rate
    # num_train_epochs=1,             # Number of training epochs
    fp16=True,                      # Mixed precision for faster training
    evaluation_strategy="steps",    # Evaluation strategy
    save_steps=2,                 # Save model checkpoint every 500 steps
    save_total_limit=2,             # Limit total checkpoints
    report_to="none",               # Disable reporting to third-party tools like W&B
    logging_dir="/data2/ryan/finetune/logs",  # Logging directory
    output_dir="/data2/ryan/finetune/out",  # Output directory
)
# Initialize the DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,   # Pass the custom training arguments here
    beta=dpo_config.beta,  # Pass the DPO configuration here
    max_length=dpo_config.max_length,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# Start the training
dpo_trainer.train()

dpo_trainer.save_model(os.path.join("/data2/ryan/finetune/out"))