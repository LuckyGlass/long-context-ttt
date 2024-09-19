from dataclasses import dataclass, field
from typing import Optional
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments
import logging


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@dataclass
class ModelArguments:
    model_name_or_path: str
    model_max_length: Optional[int] = field(default=None)
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    len_segment: int = field(
        default=2,
        metadata={
            "help": (
                "The number of blocks in a segment."
            )
        }
    )
    len_offset: int = field(
        default=1,
        metadata={
            "help": (
                "The offset from one segment to the next segment."
            )
        }
    )
    prepend_title: bool = field(
        default=False,
        metadata={"help": "If set to True, the first datapoint is \"<bos> Title: [title], Content: \" -> \"[first segment]\", and during evaluation the model is prompted to answer based on the [title] article."}
    )
    debug_size: Optional[int] = field(default=None)
    recite_prompt_type: Optional[str] = field(
        default="basic",
        metadata={"help": "The type of recite prompt, options: basic"}
    )
    sent_token: bool = field(default=False)
    num_generate_qa: int = field(default=0)
    generator_name_or_path: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments:
    use_lora: Optional[bool] = field(default=False)
    full_ft: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=8)
    load_in_4bit: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
    gather_batches: Optional[bool] = field(default=False)
    
    def __post_init__(self):
        if self.use_lora and self.full_ft:
            raise ValueError("--use_lora is in conflict with --full_ft")


@dataclass
class GlobalTestArguments:
    input_file: Optional[str] = field(default=None)
    compute_attention: bool = field(default=False)
    attention_output_dir: Optional[str] = field(default=None)
