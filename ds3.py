import sys
import os
import torch
import torch.distributed as dist
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
from accelerate.state import PartialState
torch.cuda.empty_cache()

# Configure base directory and logging
BASE_DIR = "../resources/"

# Set up global logger
logger = logging.getLogger(__name__)

def init_distributed():
    """Initialize distributed environment with proper error handling"""
    try:
        # Get environment variables
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world_size > 1:
            # Set the device
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                
            # Initialize process group with timeout
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=world_size,
                    rank=rank,
                    timeout=datetime.timedelta(minutes=1)
                )
                
            # Synchronize processes
            if dist.is_initialized():
                dist.barrier()
                
        return local_rank, world_size, rank
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {str(e)}")
        raise

def setup_device():
    """Setup device and process group properly"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if torch.cuda.is_available() and local_rank != -1:
        # Set the device before initializing process group
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0))
            )
    
    return local_rank if local_rank != -1 else 0

# Function to set up directories dynamically
def setup_experiment_directory(experiment_name):
    experiment_dir = os.path.join(BASE_DIR, "outputs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

# Rest of your existing class definitions remain the same...
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="openai-community/gpt2",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_cache_dir: Optional[str] = field(
        default=os.path.join(BASE_DIR, "cache"),
        metadata={"help": "Where to store the pretrained models"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="Orion-zhen/dpo-codealpaca-emoji",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_cache_dir: str = field(
        default=os.path.join(BASE_DIR, "cache"),
        metadata={"help": "Where to store the datasets"}
    )


@dataclass
class TrainingArguments(DPOConfig):
    experiment_name: str = field(
        default="default_experiment",
        metadata={"help": "Name of the experiment. Used for organizing outputs."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where all outputs will be stored."}
    )
    loss_type: str = field(
        default="sigmoid"
    )
    prompt_length: int = field(
        default=512
    )
    chosen_length: int = field(
        default=256
    )
    rejected_length: int = field(
        default=256
    )



class TimingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.total_steps = 0
        self.times = []

    def on_init_end(self, args, state, control, **kwargs):
        """Called when trainer initialization is done."""
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training starts."""
        self.start_time = time.time()
        logger.info("Starting training...")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Called before each training step."""
        self.step_start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.times.append(step_time)
            self.total_steps += 1
            
            # Calculate statistics
            avg_time = sum(self.times) / len(self.times)
            if len(self.times) > 1:
                recent_avg = sum(self.times[-10:]) / min(10, len(self.times))
            else:
                recent_avg = avg_time
                
            logger.info(
                f"Step {self.total_steps} - "
                f"Time: {step_time:.2f}s - "
                f"Average: {avg_time:.2f}s - "
                f"Recent Average (10 steps): {recent_avg:.2f}s"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends."""
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

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        return control

    def on_save(self, args, state, control, **kwargs):
        """Called when saving the model."""
        return control

    def on_log(self, args, state, control, logs, **kwargs):
        """Called when logging occurs."""
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        """Called before each prediction step."""
        return control

def cleanup():
    """Cleanup function with proper error handling"""
    try:
        if dist.is_initialized():
            # Synchronize before destroying
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


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
    try:
        # Initialize distributed environment
        local_rank, world_size, rank = init_distributed()

        # Parse arguments
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Set up experiment directory
        training_args.output_dir = setup_experiment_directory(training_args.experiment_name)

        # Set up logging only on main process
        if rank == 0:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(os.path.join(training_args.output_dir, "training.log")),
                    logging.StreamHandler()
                ]
            )
            logger.info(f"Experiment Directory: {training_args.output_dir}")
            logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

        # Initialize partial state for accelerate
        state = PartialState()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.model_cache_dir,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Set training arguments for distributed training
        training_args.local_rank = local_rank
        training_args.n_gpu = 1
        training_args.remove_unused_columns = False
        training_args.deepspeed = "ds1.json"
        training_args.offload_optimizer = False
        
        # Load model with distributed settings
        with state.local_main_process_first():
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.model_cache_dir,
                torch_dtype=torch.float16,
                device_map={"": local_rank},
                use_cache=False,
                low_cpu_mem_usage=True
            )
            model.gradient_checkpointing_enable()

        # Load and prepare dataset
        with state.local_main_process_first():
            raw_dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=data_args.dataset_cache_dir
            )
            train_dataset = prepare_dataset(raw_dataset["train"])
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
            max_length=8,
            max_prompt_length=4,
            max_target_length=4,
            loss_type="sigmoid",
            precompute_ref_log_probs=False,
            callbacks=[TimingCallback()] if rank == 0 else [],
            ref_model=None,
        )

        # Train the model
        train_result = dpo_trainer.train()

        # Save everything only on main process
        if rank == 0:
            final_model_path = os.path.join(training_args.output_dir, "models", "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            dpo_trainer.save_model(final_model_path)
            dpo_trainer.save_metrics("train", train_result.metrics)
            dpo_trainer.save_state()
            logger.info(f"Training completed. Model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()