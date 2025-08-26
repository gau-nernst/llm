import torch.nn.functional as F
from torch import Tensor


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    # TODO: add varlen attention
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=True,
    ).transpose(1, 2)
