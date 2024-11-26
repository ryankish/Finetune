import sys
import os
import json
import torch
import logging
import time
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
torch.cuda.empty_cache()
os.environ["DEEPSPEED_LOG_LEVEL"] = "info"
os.environ["DEEPSPEED_LOG_LEVEL_STDOUT"] = "info"




# Configure base directory and logging
BASE_DIR = "../resources/"

# Set up global logger
logger = logging.getLogger(__name__)

# Function to set up directories dynamically
def setup_experiment_directory(experiment_name):
    experiment_dir = os.path.join(BASE_DIR, "outputs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_training_args_to_json(training_args, output_dir):
    args_dict = vars(training_args)
    def serialize(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    serializable_args = {key: serialize(value) for key, value in args_dict.items()}
    output_path = os.path.join(output_dir, "training_args.json")
    with open(output_path, "w") as f:
        json.dump(serializable_args, f, indent=4)
    logger.info("Training arguments saved as JSON.")
    print(f"Training arguments saved to {output_path}")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_cache_dir: Optional[str] = field(
        default=os.path.join(BASE_DIR, "cache"),
        metadata={"help": "Where to store the pretrained models"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        # default="Orion-zhen/dpo-codealpaca-emoji",
        default="eborgnia/anthropic-hh-dpo",
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
        batch_size_per_gpu = args.per_device_train_batch_size
        
        logger.info(
            f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
            f"Total steps: {self.total_steps}\n"
            f"Average time per step: {sum(self.times) / len(self.times):.2f}s"
            f"Batch size per GPU: {batch_size_per_gpu}"
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
        "chosen": example["rejected"].strip(),
        "rejected": example["chosen"].strip()
    }
    
    logger.debug(f"Formatted example: {formatted}")
    return formatted

def prepare_dataset(dataset: Dataset, sample_percentage: float = 0.005) -> Dataset:
    """Prepare the dataset for DPO training with optional sampling."""
    # Print dataset info
    logger.info(f"Original dataset features: {dataset.features}")
    logger.info(f"Original dataset size: {len(dataset)}")
    logger.info("First example before formatting:")
    logger.info(dataset[0])

    try:
        # Sample a fraction of the dataset
        sample_size = int(len(dataset) * sample_percentage)
        # sampled_indices = random.sample(range(len(dataset)), sample_size)
        sampled_indices = range(sample_size)
        sampled_dataset = dataset.select(sampled_indices)

        logger.info(f"Sampled dataset size: {len(sampled_dataset)}")
        
        # Try to format the first example to catch any issues early
        first_formatted = validate_and_format_example(sampled_dataset[0])
        logger.info("First example after formatting:")
        logger.info(first_formatted)

        # Format all examples
        formatted_dataset = sampled_dataset.map(
            validate_and_format_example,
            remove_columns=sampled_dataset.column_names,
            desc="Formatting dataset"
        )
        
        logger.info(f"Formatted dataset size: {len(formatted_dataset)}")
        logger.info(f"Formatted dataset features: {formatted_dataset.features}")
        
        return formatted_dataset

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def freeze_model_parameters(model, layers_to_freeze):
    """
    Freezes specific layers or parts of the model.

    Args:
        model: The LLaMA model.
        layers_to_freeze: A list of layer names or indices to freeze.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            logger.info(f"Froze parameter: {name}")
        else:
            param.requires_grad = True
            logger.info(f"Trainable parameter: {name}")

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up experiment directory
    training_args.output_dir = setup_experiment_directory(training_args.experiment_name)

    # Set up logging after defining output_dir
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(training_args.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    global logger  # Access global logger
    logger.info(f"Experiment Directory: {training_args.output_dir}")

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
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        use_cache=False,  # Disable KV cache
        low_cpu_mem_usage=True,  # Add this line
        trust_remote_code=True
    )
    layers_to_freeze = [[f"model.layers.{l}.self_attn.q_proj.weight",
                     f"model.layers.{l}.self_attn.k_proj.weight",
                     f"model.layers.{l}.self_attn.v_proj.weight",
                     f"model.layers.{l}.self_attn.o_proj.weight",
                     f"model.layers.{l}.mlp.gate_proj.weight",
                     f"model.layers.{l}.mlp.up_proj.weight",
                     f"model.layers.{l}.mlp.down_proj.weight",
                     f"model.layers.{l}.input_layernorm.weight",
                     f"model.layers.{l}.post_attention_layernorm.weight"
                     ] for l in range(0, 4)]

    # other_params_to_freeze = ['model.embed_tokens.weight']  # Convert to a list
    other_params_to_freeze = []
    params_to_freeze = [param for layer in layers_to_freeze for param in layer] + other_params_to_freeze
    freeze_model_parameters(model, params_to_freeze)
    model.gradient_checkpointing_enable() # https://huggingface.co/docs/transformers/main/deepspeed?zero-config=ZeRO-1

    # Load and prepare dataset
    raw_dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.dataset_cache_dir
    )
    
    # training_args.data_sample = 0.025
    training_args.data_sample = 0.075
    train_dataset = prepare_dataset(raw_dataset["train"], sample_percentage=training_args.data_sample)
    eval_dataset = None
    # if "validation" in raw_dataset:
    #     eval_dataset = prepare_dataset(raw_dataset["validation"])

    # ARGUEMENTS
    training_args.remove_unused_columns = False
    training_args.offload_optimizer = False
    training_args.loss_type = 'sigmoid'
    training_args.max_prompt_length = 256
    training_args.max_length = 1024
    training_args.max_completion_length = training_args.max_length - training_args.max_prompt_length 
    training_args.logging_steps = 1
    training_args.beta = 0.1
    training_args.num_train_epochs = 3

    # DEEPSPEED OVERWRITES
    training_args.learning_rate=5e-03
    training_args.per_device_train_batch_size = 20
    training_args.gradient_accumulation_steps = 16
    # training_args.train_batch_size = 8
    training_args.max_grad_norm = 1.5
    training_args.weight_decay=0.01
    # training_args.dataset_num_proc = None # try this
    
    # Initialize DPO Trainer with timing callback
    timing_callback = TimingCallback()
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        precompute_ref_log_probs=False,
        callbacks=[timing_callback],
        ref_model=None,
    )

    # Train the model
    train_result = dpo_trainer.train()

    if training_args.local_rank == 0:
        # Save the model
        final_model_path = os.path.join(training_args.output_dir, "models", "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        dpo_trainer.save_model(final_model_path)
        dpo_trainer.save_metrics("train", train_result.metrics)
        dpo_trainer.save_state()
        save_training_args_to_json(training_args, training_args.output_dir)

        logger.info(f"[{training_args.local_rank}] Training completed. Model saved to {final_model_path}")
    else:
        logger.info(f"[{training_args.local_rank}] Skipping saving as this process is not rank 0.")


if __name__ == "__main__":
    main()