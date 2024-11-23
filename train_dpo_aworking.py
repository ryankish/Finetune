import sys
import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig

# Configure base directory and logging
BASE_DIR = "/data2/ryan/finetune"
os.makedirs(BASE_DIR, exist_ok=True)

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
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_cache_dir: Optional[str] = field(
        default="/data2/ryan/finetune/cache",
        metadata={"help": "Where to store the pretrained models"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="Orion-zhen/dpo-codealpaca-emoji",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_cache_dir: str = field(
        default="/data2/ryan/finetune/cache",
        metadata={"help": "Where to store the datasets"}
    )

def validate_and_format_example(example: Dict) -> Dict:
    """Validate and format a single example for DPO training."""
    logger.debug(f"Raw example: {example}")
    
    # Check required fields
    required_fields = ["prompt", "chosen", "rejected"]
    for field in required_fields:
        if field not in example:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(example[field], str):
            raise ValueError(f"Field {field} must be a string")
        if not example[field].strip():
            raise ValueError(f"Field {field} cannot be empty")
    
    # Format the example
    formatted = {
        "prompt": example["prompt"].strip(),
        "chosen": example["chosen"].strip(),
        "rejected": example["rejected"].strip()
    }
    
    logger.debug(f"Formatted example: {formatted}")
    return formatted

def prepare_dataset(dataset: Dataset) -> Dataset:
    """Prepare the dataset for DPO training."""
    # Print dataset info
    logger.info(f"Original dataset features: {dataset.features}")
    logger.info(f"Original dataset size: {len(dataset)}")
    logger.info("First example before formatting:")
    logger.info(dataset[0])
    
    try:
        # Try to format the first example to catch any issues early
        first_formatted = validate_and_format_example(dataset[0])
        logger.info("First example after formatting:")
        logger.info(first_formatted)
        
        # Format all examples
        formatted_dataset = dataset.map(
            validate_and_format_example,
            remove_columns=dataset.column_names,
            desc="Formatting dataset"
        )
        
        logger.info(f"Formatted dataset size: {len(formatted_dataset)}")
        logger.info(f"Formatted dataset features: {formatted_dataset.features}")
        
        return formatted_dataset
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set very conservative training parameters
    training_args.output_dir = os.path.join(BASE_DIR, "outputs")
    training_args.remove_unused_columns = False
    training_args.gradient_accumulation_steps = 1
    training_args.per_device_train_batch_size = 1
    training_args.learning_rate = 1e-5  # Even lower learning rate
    training_args.max_grad_norm = 0.9   # Stricter gradient clipping
    training_args.logging_steps = 1
    training_args.num_train_epochs = 3
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = "epoch"
    training_args.fp16 = False
    training_args.warmup_ratio = 0.1
    training_args.gradient_checkpointing = True
    training_args.beta = 0.1
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
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False  # Disable KV cache
    )

    # Load and prepare dataset
    raw_dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.dataset_cache_dir
    )
    for example in raw_dataset["train"]:
        try:
            validate_and_format_example(example)
        except Exception as e:
            logger.error(f"Dataset issue with example {example}: {str(e)}")
    
    # Prepare train dataset
    train_dataset = prepare_dataset(raw_dataset["train"])
    example = raw_dataset["train"][0]
    # print(example)
    # sys.exit()
    
    # Prepare eval dataset if it exists
    eval_dataset = None
    if "validation" in raw_dataset:
        eval_dataset = prepare_dataset(raw_dataset["validation"])

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
    
    # Save everything
    final_model_path = os.path.join(BASE_DIR, "models", "final_model")
    dpo_trainer.save_model(final_model_path)
    dpo_trainer.save_metrics("train", train_result.metrics)
    dpo_trainer.save_state()

    logger.info(f"Training completed. Model saved to {final_model_path}")

if __name__ == "__main__":
    main()