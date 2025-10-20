import math

import torch
from torch import Tensor, nn


# https://github.com/pytorch/torchtitan/blob/v0.2.0/torchtitan/models/utils.py#L363
def compute_model_tflop(
    dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_dim: int,
    vocab_size: int,
    num_layers: int,
    seqlen: int,
    training: bool,
):
    # linear layers
    attn_proj_params = 2 * dim * (num_qo_heads + num_kv_heads) * head_dim  # qkv and o projections
    mlp_params = 3 * dim * mlp_dim  # gate, up, and down projections
    perlayer_linear_flop = 2 * seqlen * (attn_proj_params + mlp_params)

    # divide by 2 due to causal attention
    perlayer_attn_flop = 4 * num_qo_heads * head_dim * seqlen * seqlen // 2

    lmhead_flop = 2 * seqlen * dim * vocab_size

    flop = (perlayer_linear_flop + perlayer_attn_flop) * num_layers + lmhead_flop
    if training:
        flop *= 3  # each matmul in forward requires 2 matmuls in backward
    return flop / 1e12


def get_gpu_tflops():
    name = torch.cuda.get_device_name()
    lookup = {
        "5090": 240,
    }
    for k, v in lookup.items():
        if k in name.lower():
            return v
    return 0


@torch.no_grad()
def get_grad_norm(model: nn.Module):
    grad_norm_sq = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)
    if hasattr(grad_norm_sq, "full_tensor"):
        grad_norm_sq = grad_norm_sq.full_tensor()
    return grad_norm_sq.item() ** 0.5


def get_optimizer(optim: str, model: nn.Module, lr: float, weight_decay: float, **kwargs):
    allowed = dict(torch=torch)
    optim_cls = eval(optim, allowed)
    if optim_cls in (torch.optim.AdamW, torch.optim.Adam):
        kwargs.update(fused=True)  # force fused impl
    return optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")


class LRSchedule:
    def __init__(
        self,
        lr: float,
        n_steps: int,
        warmup: float = 0.0,
        decay: float = 0.0,
        decay_type: str = "linear",
    ) -> None:
        self.lr = lr
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2
        assert decay_type in ("linear", "cosine")

    def get_lr(self, step: int) -> float:
        if step < self.t1:
            return self.lr * step / self.t1
        if step < self.t2:
            return self.lr
        if step < self.t3:
            progress = (step - self.t2) / (self.t3 - self.t2)
            if self.decay_type == "linear":
                return self.lr * (1 - progress)
            elif self.decay_type == "cosine":
                return 0.5 * self.lr * (1 + math.cos(progress * math.pi))
        return 0.0

    def set_lr(self, step: int, optim: torch.optim.Optimizer):
        lr = self.get_lr(step)
        for param_group in optim.param_groups:
            if isinstance(param_group["lr"], Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr
