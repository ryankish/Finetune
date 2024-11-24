from email.policy import default
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
        # default="Orion-zhen/dpo-codealpaca-emoji",
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_cache_dir: str = field(
        default="/data2/ryan/finetune/cache",
        metadata={"help": "Where to store the downloaded datasets"}
    )

def main():
    # Create necessary directories
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Update training args with our defaults
    training_args.output_dir = os.path.join(BASE_DIR, "outputs")
    training_args.remove_unused_columns = False
    training_args.gradient_accumulation_steps = 4
    training_args.per_device_train_batch_size = 4
    training_args.learning_rate = 5e-5
    training_args.logging_steps = 10
    training_args.num_train_epochs = 3
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = "epoch"

    # Load tokenizer and cache it
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load dataset with caching
    dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.dataset_cache_dir
    )

    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
        max_target_length=256,
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