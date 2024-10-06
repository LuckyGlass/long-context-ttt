import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer
)
import math
from .model import load_model, load_optimizer
from .context_dataset import ContextDataset
from typing import Optional, Type
from copy import deepcopy


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        The major change is to support weighting the samples.
        The averaging strategy is a little different...
        """
        if self.label_smoother is not None:
            raise NotImplementedError("WeightedTrainer does not support label_smoother.")
        weights = inputs.pop('weights')
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = []
        for weight_case, logits_case, labels_case in zip(weights, shift_logits, shift_labels):
            losses.append(weight_case * loss_fn(logits_case, labels_case))
        loss = sum(losses) / len(losses)
        return (loss, outputs) if return_outputs else loss


def load_trainer(training_dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, eval_dataset: Optional[Dataset]=None, model_name_or_path: Optional[str]=None, gather_batches: bool=False, model: Optional[torch.nn.Module]=None, optimizer: Optional[torch.optim.Optimizer]=None, trainer_cls: Type=Trainer, **kwargs):
    """Load the training and the model (if the model is not instantiated).
    Args:
        training_dataset (Dataset):
        tokenizer (PreTrainedTokenizer):
        training_args (TrainingArguments):
        eval_dataset (Dataset): OPTIONAL, default to `None`.
        model_name_or_path (str): OPTIONAL, default to `None`; if the model is not instantiated, assign it to load the model.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        model (Module): OPTIONAL, default to `None`; an instantiated model.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
    Returns:
        trainer_model_pair (tuple[Trainer, Module]): the trainer and the model to train.
    """
    training_args = deepcopy(training_args)
    if gather_batches:
        training_args.gradient_accumulation_steps = len(training_dataset)
    # If the model is not given, then load the model
    if model is None:
        model = load_model(model_name_or_path, tokenizer, **kwargs)
    if optimizer is None:
        optimizer = load_optimizer(model, **vars(training_args))
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )
    return trainer, model


def train(dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, involve_qa_epochs: int=0, **kwargs):
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
    dataset.disable_qa()
    trainer, model = load_trainer(dataset, tokenizer, training_args, trainer_cls=WeightedTrainer, **kwargs)
    trainer.train()
    # Load the dataset with QA pairs and continue-finetune the model
    if involve_qa_epochs > 0:
        dataset.enable_qa()
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        trainer_syn, model = load_trainer(dataset, tokenizer, training_args_syn, model=model, optimizer=trainer.optimizer, trainer_cls=WeightedTrainer, **kwargs)
        trainer_syn.train()
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
