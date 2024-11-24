from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="llama-3.2",
        metadata={"help": "Path to pretrained model or model identifier from Hugging Face hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (e.g., a Git branch or tag)"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow custom code from model repository"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype to use (e.g., 'float16', 'bfloat16', or 'auto')"}
    )
    attn_implementation: str = field(
        default="default",
        metadata={"help": "Type of attention implementation to use"}
    )
