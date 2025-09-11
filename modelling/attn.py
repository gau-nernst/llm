from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class VarlenInfo(NamedTuple):
    cu_seqlens: Tensor
    max_seqlen: int  # to know how many blocks to launch


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    varlen_info: VarlenInfo | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    if varlen_info is not None:
        # switch to public API if it's merged + PyTorch 2.9 is released
        # https://github.com/pytorch/pytorch/pull/162326
        out, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            varlen_info.cu_seqlens,
            varlen_info.cu_seqlens,
            varlen_info.max_seqlen,
            varlen_info.max_seqlen,
            dropout_p,
            is_causal,
            return_debug_mask=False,
        )

    else:
        # F.sdpa() only dispatches FA/CuDNN kernels for 4D tensors
        if q.ndim == k.ndim == v.ndim == 3:
            q, k, v = q[None], k[None], v[None]
            to_squeeze = True
        else:
            to_squeeze = False
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=True,
        ).transpose(1, 2)
        if to_squeeze:
            out = out[0]

    return out
