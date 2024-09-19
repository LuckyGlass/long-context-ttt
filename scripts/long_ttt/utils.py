import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer
from typing import Optional


def printGPU(info=""):
    if len(info) > 0:
        print("-" * 10 + " " + info + " " + "-" * 10)
    import GPUtil
    GPUs = GPUtil.getGPUs()
    print(f"{'ID':4s} {'Name':24s} {'MemoryUsed':16s} {'MemoryTotal':16s} {'Temperature':16s}")
    for GPU in GPUs:
        print(f"{GPU.id:<4d} {GPU.name:<24s} {GPU.memoryUsed:<16.3f} {GPU.memoryTotal:<16.3f} {GPU.temperature:<16f}")


def locate_tokens(pattern: Tensor, module: Tensor, threshold: float=0.5):
    pattern = pattern.flatten().to(module.device)
    module = module.flatten()
    assert pattern.numel() <= module.numel()
    for i in range(module.numel() - pattern.numel() + 1):
        if torch.mean((pattern == module[i:i+pattern.numel()]).float()) >= threshold:
            return i, i + pattern.numel()
    return None


def plot_attentions(attentions: Tensor, ranges: list, output_path: str='output_image.png'):
    plt.clf()
    fig, axs = plt.subplots(4, 8, figsize=(40, 20))
    for ax, attention in zip(axs.flat, attentions):
        # ax.plot(torch.arange(attention.numel()), attention.cpu(), linewidth=0.5)
        ax.set_ylim(top=0.5)
        for lr in ranges:
            if lr is not None:
                ax.plot(torch.arange(lr[0], lr[1]), attention[lr[0]:lr[1]].cpu(), linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def get_average_attention(tokenizer: PreTrainedTokenizer, attentions: Tensor, input_ids: Tensor, evidences: list[str], output_path: Optional[str]=None):
    """Compute the average attention on the evidences.
    Args:
        model (PreTrainedModel): the model used, currently only supporting LlamaModel.
        tokenizer (PreTrainedTokenizer): the tokenizer used to encode `input_ids`.
        hidden_states (tuple[Tensor]): the hidden states.
        evidences (list[str]): the evidences.
    Returns:
        avg_attns (list[float]): the average attention on each evidence.
    """
    # 0. locate the evidences
    ranges = []
    unfound_counter = 0
    input_ids = input_ids.flatten()
    for evidence in evidences:
        tokens = tokenizer(evidence, add_special_tokens=False, return_tensors='pt').input_ids.flatten().to(input_ids.device)
        ranges.append(locate_tokens(tokens, input_ids))
        if ranges[-1] is None:
            unfound_counter += 1
    if output_path is not None:
        plot_attentions(attentions, ranges, output_path=output_path)
    else:
        plot_attentions(attentions, ranges)
    results = [None if lr is None else torch.mean(attentions[:, lr[0]:lr[1]], dim=1).cpu() for lr in ranges]
    return results
