from dataclasses import dataclass, field
from typing import Optional, Any, List
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
import logging


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=None)
    is_peft_model: bool = field(default=False)
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


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
    pad_to_max_length: bool = field(default=True)
    ttt_recite_first: bool = field(default=False)
    ttt_enable_ICL: bool = field(default=True)
    qa_loss_weight: float = field(default=1.0)
    enable_diverse_qa: bool = field(default=False)
    # involve_qa_epochs, shared with CustomTrainingArguments
    num_timeline_reorder: int = field(default=0)
    num_timeline_reorder_events: List[int] = field(default_factory=lambda: [5])
    append_question: bool = field(default=False)
    is_sentence_based: bool = field(default=False)
    
    def __post_init__(self):
        if len(self.num_timeline_reorder_events) == 1:
            self.num_timeline_reorder_events = self.num_timeline_reorder_events[0]
        elif len(self.num_timeline_reorder_events) == 2:
            self.num_timeline_reorder_events = tuple(self.num_timeline_reorder_events)
        else:
            raise ValueError("--num_timeline_reorder_events accepts an integer or a pair of integers.")


@dataclass
class CustomTrainingArguments:
    use_lora: bool = field(default=False)
    full_ft: bool = field(default=False)
    lora_rank: int = field(default=8)
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    gather_batches: bool = field(default=False)
    involve_qa_epochs: int = field(default=0)
    
    def __post_init__(self):
        if self.use_lora and self.full_ft:
            raise ValueError("--use_lora is in conflict with --full_ft")


@dataclass
class GlobalTestArguments:
    input_file: Optional[str] = field(default=None)
    compute_attention: bool = field(default=False)
    attention_output_dir: Optional[str] = field(default=None)
    eval_batch_size: int = field(default=1)
    overwrite: bool = field(default=True)
    enable_ICL: bool = field(default=True)
    recite_first: bool = field(default=False)
    
    def __post_init__(self):
        pass


def parse_args(class_clusters: tuple[Any|tuple[Any]], no_dict: tuple[Any], return_config: bool=False):
    class_set = set()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            class_set.update(set(cluster))
        else:
            class_set.add(cluster)
    class_tuple = tuple(class_set)
    parser = HfArgumentParser(class_tuple)
    arg_list = parser.parse_args_into_dataclasses()
    arg_dict = {c: a for c, a in zip(class_tuple, arg_list)}
    returns = ()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            temp = {}
            for item in cluster:
                temp.update(dict(vars(arg_dict[item])))
            returns += (temp,)
        else:
            if cluster in no_dict:
                returns += (arg_dict[cluster],)
            else:
                returns += (dict(vars(arg_dict[cluster])),)
    if return_config:
        config = {}
        for arg in arg_list:
            config.update({k: v for k, v in dict(vars(arg)).items() if isinstance(v, int|float|bool|str)})
        return returns, config
    return returns
