from gated_mem_Llama import GMLlamaForCausalLM
from transformers import BitsAndBytesConfig
import torch

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=[f"layers.{i}.self_attn.mem_proj" for i in range(32)]+[f"layers.{i}.self_attn.gate_proj" for i in range(32)] + ["lm_head"]
)

model = GMLlamaForCausalLM.from_pretrained("./newmodel", device_map="auto", quantization_config=config)

for n, p in model.named_parameters():
    if "norm" in n:
        p.requires_grad_(False)
    if "embed_tokens" in n:
        p.requires_grad_(False)
    if "lm_head" in n:
        p.requires_grad_(False)
    print(n, p.dtype, p.requires_grad)