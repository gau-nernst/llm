import torch
from torch import Tensor


def compute_rope(pos_ids: Tensor, rope_theta: float, dim: int) -> Tensor:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=pos_ids.device) / dim))
    freqs = pos_ids.unsqueeze(-1) * inv_freq
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [*pos_ids.shape, dim]


# NOTE: if x and pos_embeds are BF16, the computation is done in BF16
def apply_rope(x: Tensor, pos_embeds: Tensor) -> Tensor:
    # x: [*, L, num_heads, dim]
    # pos_embeds: [*, L, dim]
    # pos_embeds may have fewer leading dimensions than x's
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = pos_embeds.unsqueeze(-2).chunk(2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1).to(x.dtype)
