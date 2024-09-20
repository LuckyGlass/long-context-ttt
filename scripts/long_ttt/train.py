import torch
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    PreTrainedTokenizer
)
from torch.utils.data import Dataset
from .model import load_trainer
from typing import Optional
from copy import deepcopy


def train(dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, enable_sequential_training: bool=False, sequential_training_epochs: Optional[int]=None, **kwargs):
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
    # Post-sequential training
    if enable_sequential_training:
        seq_training_args = deepcopy(training_args)
        seq_kwargs = deepcopy(kwargs)
        seq_training_args.num_train_epochs = sequential_training_epochs
        seq_training_args.per_device_train_batch_size = 1
        seq_training_args.gradient_accumulation_steps = 1
        seq_kwargs['gather_batches'] = False
        seq_trainer, model = load_trainer(dataset, tokenizer, seq_training_args, model=trainer.model, optimizer=trainer.optimizer, **seq_kwargs)  # Load the previous model and optimizer
        seq_trainer.train()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
