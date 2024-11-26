from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer
from trl import DPOTrainer, DPOConfig  # Fixed import
import logging
import os
import json
import time
from transformers import TrainerCallback

# Configure base directory and cache paths
BASE_DIR = "../resources/"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "outputs")

# Set up global logger
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """
    Arguments for DPO training
    """
    model_name: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "the model name"}
    )
    model_cache_dir: Optional[str] = field(
        default=CACHE_DIR,
        metadata={"help": "Where to store the pretrained models"}
    )
    dataset_name: Optional[str] = field(
        default="eborgnia/anthropic-hh-dpo",
        metadata={"help": "the dataset name"}
    )
    dataset_cache_dir: Optional[str] = field(
        default=CACHE_DIR,
        metadata={"help": "Where to store the datasets"}
    )
    experiment_name: Optional[str] = field(
        default="default_experiment",
        metadata={"help": "Name of the experiment for organizing outputs"}
    )
    log_with: Optional[str] = field(
        default="none",
        metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "the batch size"}
    )
    seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16,
        metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use PEFT or not"}
    )
    output_dir: Optional[str] = field(
        default=None,  # Will be set based on experiment_name
        metadata={"help": "the output directory"}
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "the number of training steps"}
    )
    logging_steps: Optional[int] = field(
        default=1,
        metadata={"help": "the number of logging steps"}
    )
    save_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of updates steps before checkpoint saves"}
    )
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "the beta parameter for DPO training"}
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={"help": "the maximum prompt length"}
    )
    max_completion_length: Optional[int] = field(
        default=768,
        metadata={"help": "the maximum completion length"}
    )

def setup_experiment_directory(experiment_name):
    """Set up the experiment directory structure"""
    experiment_dir = os.path.join(OUTPUT_BASE_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

class TimingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.total_steps = 0
        self.times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("Starting training...")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time if self.step_start_time else 0
        self.times.append(step_time)
        self.total_steps += 1
        return control

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        
        logger.info(
            f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
            f"Total steps: {self.total_steps}\n"
            f"Average time per step: {sum(self.times) / len(self.times):.2f}s"
        )
        return control

def validate_and_format_example(example):
    """Validate and format a single example for DPO training."""
    required_fields = ["prompt", "chosen", "rejected"]
    for field in required_fields:
        if field not in example:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(example[field], str):
            raise ValueError(f"Field {field} must be a string")
        if not example[field].strip():
            raise ValueError(f"Field {field} cannot be empty")
    
    return {
        "prompt": example["prompt"].strip(),
        "chosen": example["rejected"].strip(),
        "rejected": example["chosen"].strip()
    }

def prepare_dataset(dataset, sample_percentage=0.01):
    """Prepare the dataset for DPO training with optional sampling."""
    logger.info(f"Original dataset size: {len(dataset)}")
    
    sample_size = int(len(dataset) * sample_percentage)
    sampled_dataset = dataset.select(range(sample_size))
    logger.info(f"Sampled dataset size: {len(sampled_dataset)}")
    
    formatted_dataset = sampled_dataset.map(
        validate_and_format_example,
        remove_columns=sampled_dataset.column_names,
        desc="Formatting dataset"
    )
    
    logger.info(f"Formatted dataset size: {len(formatted_dataset)}")
    return formatted_dataset

def save_training_args_to_json(args, output_dir):
    """Save training arguments to JSON file"""
    args_dict = vars(args)
    serializable_args = {key: str(value) if not isinstance(value, (int, float, bool, str, type(None))) 
                        else value for key, value in args_dict.items()}
    
    output_path = os.path.join(output_dir, "training_args.json")
    with open(output_path, "w") as f:
        json.dump(serializable_args, f, indent=4)
    logger.info(f"Training arguments saved to {output_path}")

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize accelerator first
    accelerator = Accelerator()
    
    # Set up experiment directory
    args.output_dir = setup_experiment_directory(args.experiment_name)

    # Set up logging only on main process
    if accelerator.is_local_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.model_cache_dir,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Configure device mapping based on the distributed setup
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        device_map = None  # DeepSpeed will handle device mapping
    elif accelerator.distributed_type == DistributedType.MULTI_GPU:
        device_map = {"": accelerator.local_process_index}
    else:
        device_map = "auto"

    # Set up model loading configuration
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
        torch_dtype = torch.bfloat16
    else:
        quantization_config = None
        torch_dtype = torch.bfloat16  # Default to bfloat16

    # Load model with proper device mapping
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.model_cache_dir,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        use_cache=False,
        trust_remote_code=True
    )

    # Load and prepare dataset
    raw_dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.dataset_cache_dir
    )
    train_dataset = prepare_dataset(raw_dataset["train"], sample_percentage=0.01)
    eval_dataset = None

    # Set up DPO training arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_length=args.seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        report_to=args.log_with,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        # Add distributed training specific arguments
        local_rank=accelerator.local_process_index,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
    )

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        precompute_ref_log_probs=False,
        callbacks=[TimingCallback()],
        ref_model=None,
    )

    # Train the model
    trainer.train()

    # Save the model and training args only on the main process
    if accelerator.is_local_main_process:
        final_model_path = os.path.join(args.output_dir, "models", "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        trainer.save_model(final_model_path)
        save_training_args_to_json(args, args.output_dir)
        logger.info(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()