import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig

# Configure base directory
BASE_DIR = "/data2/ryan/finetune"
os.makedirs(BASE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_cache_dir: Optional[str] = field(
        default="/data2/ryan/finetune/cache",
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    dataset_name: str = field(
        default="Orion-zhen/dpo-codealpaca-emoji",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_cache_dir: str = field(
        default="/data2/ryan/finetune/cache",
        metadata={"help": "Where to store the downloaded datasets"}
    )

def format_data(example):
    """Format the dataset examples for DPO training"""
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

def main():
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Update training args with more stable defaults
    training_args.output_dir = os.path.join(BASE_DIR, "outputs")
    training_args.remove_unused_columns = False
    
    # Very conservative training parameters
    training_args.gradient_accumulation_steps = 16
    training_args.per_device_train_batch_size = 1
    training_args.learning_rate = 1e-6
    training_args.max_grad_norm = 0.3
    training_args.logging_steps = 10
    training_args.num_train_epochs = 3
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = "epoch"
    training_args.fp16 = True
    training_args.warmup_ratio = 0.1
    training_args.gradient_checkpointing = True
    training_args.beta = 0.1
    
    # Optimizer parameters
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.999
    training_args.adam_epsilon = 1e-8
    training_args.weight_decay = 0.01

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        padding_side="left"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=not training_args.gradient_checkpointing,
    )

    # Load dataset
    dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.dataset_cache_dir
    )

    # Print dataset info
    logger.info("Dataset loaded. Checking first example:")
    logger.info(dataset["train"][0])

    # Format dataset if needed
    train_dataset = dataset["train"].map(
        format_data,
        remove_columns=dataset["train"].column_names
    )
    
    eval_dataset = dataset.get("validation")
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            format_data,
            remove_columns=eval_dataset.column_names
        )

    # Print formatted example
    logger.info("Formatted example:")
    logger.info(train_dataset[0])

    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
        max_target_length=256,
        loss_type="sigmoid",
        precompute_ref_log_probs=False,
    )

    # Train the model
    train_result = dpo_trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(BASE_DIR, "models", "final_model")
    dpo_trainer.save_model(final_model_path)
    
    # Save training metrics
    metrics = train_result.metrics
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info(f"Training completed successfully! Model saved to {final_model_path}")

if __name__ == "__main__":
    main()