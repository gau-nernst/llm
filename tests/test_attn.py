import torch

from modelling.attn import VarlenInfo, attention


def test_varlen_attn():
    # model specs
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128

    bs = 4
    q_list = []
    k_list = []
    v_list = []
    cu_seqlens = [0]
    max_seqlen = 0
    for _ in range(bs):
        seqlen = torch.randint(16, 1024, (1,)).item()
        q_list.append(torch.randn(seqlen, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"))
        k_list.append(torch.randn(seqlen, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"))
        v_list.append(torch.randn(seqlen, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") + 0.5)
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
        max_seqlen = max(max_seqlen, seqlen)

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")

    out = attention(q, k, v, varlen_info=VarlenInfo(cu_seqlens, max_seqlen))
    out_ref = torch.cat([attention(q_, k_, v_) for q_, k_, v_ in zip(q_list, k_list, v_list)], dim=0)
    torch.testing.assert_close(out, out_ref)
