import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DPOTrainer

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
from dataclasses import dataclass, field
from typing import Optional, List

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

@dataclass
class DPOConfig:
    """
    Arguments for DPO training specifics.
    """
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for DPO loss"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for training"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum length for prompt sequences"}
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "Maximum length for target sequences"}
    )

def main():
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig, TrainingArguments))
    model_args, data_args, dpo_args, training_args = parser.parse_args_into_dataclasses()

    # Set training arguments output directory
    training_args.output_dir = os.path.join(BASE_DIR, "outputs")

    # Load tokenizer and cache it
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir  # Updated from cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model and cache it
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=model_args.model_cache_dir  # Updated from cache_dir
    )

    # Load dataset with caching
    dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.dataset_cache_dir  # Updated from cache_dir
    )

    # training_args.remove_unused_columns = False
    # training_args.label_names = []
    # training_args.gradient_checkpointing = True
    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference model (copy of initial model)
        args=training_args,
        beta=dpo_args.beta,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        max_length=dpo_args.max_length,
        max_prompt_length=dpo_args.max_prompt_length,
        max_target_length=dpo_args.max_target_length,
        data_collator=None,  # Use default DPO collator
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