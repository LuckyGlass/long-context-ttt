import torch
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    PreTrainedTokenizer
)
from torch.utils.data import Dataset
from .context_dataset import LooGLEDataset
from .model import load_trainer


def train(dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, **kwargs):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        dataset (Dataset): the dataset to train on.
        tokenizer (PreTrainedTokenizer): a Llama tokenizer (or other tokenizers with chat template).
        training_args (TrainingArguments): transformers-style training arguments, used for the trainer.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    # load tokenzier
    torch.cuda.empty_cache()  # Manually release memory
    # Load and finetune the model
    trainer, model = load_trainer(dataset, tokenizer, training_args, **kwargs)
    trainer.train()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer


def LooGLEtrain(datapoint: dict, training_args: TrainingArguments, **kwargs):
    """Fine-tune the model and the corresponding tokenizer on a LooGLE task.
    Args:
        datapoint (dict): a LooGLE-style datapoint, containing `input`, `title`, `qa_pairs`.
        tokenizer (PreTrainedTokenizer): a Llama tokenizer (or other tokenizers with chat template).
        training_args (TrainingArguments): transformers-style training arguments, used for the trainer.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
        model_max_length (int): OPTIONAL; the texts will be clipped or padded to model_max_length tokens.
        block_size (int): OPTIONAL; the number of tokens in a block; a block is the unit of segments and offsets.
        len_segment (int): OPTIONAL; the number of units in a segment; the article is divided into segments.
        len_offset (int): OPTIONAL; the number of units per offset; it determines the offset from one segment to the next one.
        prepend_title (bool): OPTIONAL; whether to prompt the model with the title.
        sent_token (bool): OPTIONAL; whether to insert a `<|reserved_special_token_249|>` between each two sentences; if enabled, the model must be trained to recognize this token.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    tokenizer_kwargs = {
        "cache_dir": kwargs.get("cache_dir", None),
        "use_auth_token": kwargs.get("use_auth_token", False),
        "revision": kwargs.get("model_revision", "main"),
        "use_fast": True, 
    }
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name_or_path'], **tokenizer_kwargs)
    dataset = LooGLEDataset(datapoint, tokenizer, **kwargs)
    return train(dataset, tokenizer, training_args, **kwargs)
