from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)
from .my_llama import MyLlamaForCausalLM
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from typing import Optional
from copy import deepcopy
import torch


def load_tokenizer(model_name_or_path: str):
    tokenizer_kwargs = {"use_fast": True, }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name_or_path: str, tokenizer: PreTrainedTokenizer, load_in_4bit: bool=False, load_in_8bit: bool=False, **kwargs):
    """Load the base model.
    Args:
        model_name_or_path (str):
        tokenizer (PreTrainedTokenizer): the model input will adapt to the width of the tokenizer.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
    Returns:
        model (PreTrainedModel): the base model.
    """
    model_cls = AutoModelForCausalLM
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        model_base = model_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype = torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:
        model_base = model_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype = torch.bfloat16,
            device_map="auto",
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model_base.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model_base.resize_token_embeddings(len(tokenizer))
    model_base.enable_input_require_grads()
    return model_base


def load_model(model_name_or_path: str, tokenizer: PreTrainedTokenizer, use_lora: bool=False, lora_rank: Optional[int]=None, full_ft: bool=False, **kwargs):
    """Load the trainable model.
    Args:
        model_name_or_path (str):
        tokenizer (PreTrainedTokenizer): the model input will adapt to the width of the tokenizer.
        use_lora (bool): OPTIONAL, default to `False`; whether to use LoRA.
        lora_rank (int): OPTIONAL, default to `None`; assign it when `use_lora=True`.
        full_ft (bool): OPTIONAL, default to `False`; whether to full-fine-tune the model.
        load_in_4bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        load_in_8bit (bool): OPTIONAL, default to `False`; it must be used with `use_lora=True`.
        cache_dir (str): OPTIONAL, default to `None`.
        model_revision (str): OPTIONAL, default to `"main"`.
        use_auth_token (bool): OPTIONAL, default to `False`.
    Returns:
        model (PreTrainedModel): the model to train.
    """
    model_base = load_base_model(model_name_or_path, tokenizer, **kwargs)
    # Load the model
    if use_lora:
        # Init LoRA model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_mode=False,
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=.0,
            init_lora_weights='gaussian',
        )
        model = get_peft_model(model_base, peft_config)
    elif full_ft:
        model = model_base
    else:
        raise ValueError("Fine-tuning only the last few layers is deprecated for poor performance. Please assign use_lora=True or full_ft=True.")
        """
        model = model_base
        param_learnable = []
        pattern_layer = r"layers.(\d+)."
        for name, param in model.named_parameters():
            layer = re.findall(pattern_layer, name)
            if len(layer) == 1 and int(layer[0]) >= 32 - training_args.train_layers:
                param_learnable.append(param)
        """
    torch.cuda.empty_cache()  # Manually release memory
    return model


def load_optimizer(model: torch.nn.Module, learning_rate: float=5e-5, force_return: bool=False, adam_beta1: float=0.9, adam_beta2: float=0.999, adam_epsilon: float=1e-8, weight_decay: float=.0, **kwargs):
    """Load the optimizer.
    Args:
        model (Module): the trainable model.
        force_return (bool): OPTIONAL, default to `False`; if `force_return=True`, it will instantiate the optimizer; otherwise it will return `None` and the optimizer will be instantiated in the trainer.
        learning_rate (float): OPTIONAL, default to `5e-5`.
        adam_beta1 (float): OPTIONAL, default to `0.9`.
        adam_beta2 (float): OPTIONAL, default to `0.999`.
        adam_epsilon (float): OPTIONAL, default to `1e-8`.
        weight_decay (float): OPTIONAL, default to `0.0`.
    """
    if force_return:
        param_learnable = {name: param for name, param in model.named_parameters()}
        torch.cuda.empty_cache()  # Manually release memory
        optimizer = torch.optim.AdamW(
            param_learnable,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )
    else:
        optimizer = None
    return optimizer
