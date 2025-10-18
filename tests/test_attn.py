import torch
from torch import Tensor
from utils import check_close

from modelling.attn import VarlenInfo, attention


def test_varlen_attn():
    # model specs
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    bs = 4
    q_list, k_list, v_list, grad_list = [], [], [], []
    cu_seqlens = [0]
    max_seqlen = 0
    for _ in range(bs):
        seqlen = torch.randint(16, 1024, (1,)).item()
        q_list.append(randn(seqlen, num_q_heads, head_dim) + 0.1)
        k_list.append(randn(seqlen, num_kv_heads, head_dim) + 0.1)
        v_list.append(randn(seqlen, num_kv_heads, head_dim) + 0.1)
        grad_list.append(randn(seqlen, num_q_heads, head_dim) + 0.1)
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
        max_seqlen = max(max_seqlen, seqlen)

    q, k, v, grad = map(torch.cat, (q_list, k_list, v_list, grad_list))
    q, k, v = map(Tensor.requires_grad_, (q, k, v))
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")

    out = attention(q, k, v, varlen_info=VarlenInfo(cu_seqlens, max_seqlen))
    out.backward(grad)

    out_ref, q_grad_ref, k_grad_ref, v_grad_ref = [], [], [], []
    for q_, k_, v_, grad_ in zip(q_list, k_list, v_list, grad_list):
        q_, k_, v_ = map(Tensor.requires_grad_, (q_, k_, v_))
        out_ = attention(q_, k_, v_)
        out_.backward(grad_)

        out_ref.append(out_)
        q_grad_ref.append(q_.grad)
        k_grad_ref.append(k_.grad)
        v_grad_ref.append(v_.grad)

    out_ref, q_grad_ref, k_grad_ref, v_grad_ref = map(torch.cat, (out_ref, q_grad_ref, k_grad_ref, v_grad_ref))

    check_close(out, out_ref, rtol=1.6e-2, atol=1e-3, pct=0.0)
    check_close(q.grad, q_grad_ref, rtol=1.6e-2, atol=1e-2, pct=0.0)
    check_close(k.grad, k_grad_ref, rtol=1.6e-2, atol=1e-2, pct=0.0)
    check_close(v.grad, v_grad_ref, rtol=1.6e-2, atol=1e-3, pct=0.0)
