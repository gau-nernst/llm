import torch
from transformers import AutoModelForCausalLM, Qwen3Config
from utils import check_close

from modelling.attn import VarlenInfo
from modelling.qwen3 import Qwen3ForCausalLM


def test_qwen3():
    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda"

    # test using FP32
    model = Qwen3ForCausalLM.from_pretrained(model_id).to(device).float()
    model_ref = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    max_id = model.cfg.vocab_size - 1000
    input_ids = torch.randint(0, max_id, size=(2, 16), device=device)
    with torch.no_grad():
        out = model(input_ids)
        out_ref = model_ref(input_ids).logits
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)

    input_embeds = torch.randn(2, 16, model.cfg.hidden_size, device=device)
    with torch.no_grad():
        out = model(input_embeds=input_embeds)
        out_ref = model_ref(inputs_embeds=input_embeds).logits
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)


def test_qwen3_varlen():
    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda"

    model = Qwen3ForCausalLM.from_pretrained(model_id).to(device)
    max_id = model.cfg.vocab_size - 1000

    # remove layers to avoid errors build up
    while len(model.model.layers) >= 5:
        model.model.layers.pop(-1)

    bs = 2
    tokens_list = []
    offsets = [0]
    max_seqlen = 0
    pos_ids_list = []
    for _ in range(bs):
        seqlen = torch.randint(16, 512, size=(1,)).item()
        tokens_list.append(torch.randint(0, max_id, (seqlen,), device=device))
        offsets.append(offsets[-1] + seqlen)
        max_seqlen = max(max_seqlen, seqlen)
        pos_ids_list.append(torch.arange(seqlen, device=device))

    tokens = torch.cat(tokens_list, dim=0)
    offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    pos_ids = torch.cat(pos_ids_list, dim=0)
    varlen_info = VarlenInfo(offsets, max_seqlen)

    with torch.no_grad():
        out = model(tokens, pos_ids=pos_ids, varlen_info=varlen_info)
        out_ref = torch.cat([model(tokens_) for tokens_ in tokens_list], dim=0)

    check_close(out, out_ref, rtol=1e-1, atol=1e-2, pct=1e-2)


def test_qwen3_tie_embeddings():
    model_id = "Qwen/Qwen3-0.6B"

    cfg = Qwen3Config.from_pretrained(model_id)
    model = Qwen3ForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    model = Qwen3ForCausalLM.from_pretrained(model_id)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_qwen3_autocast():
    model_id = "Qwen/Qwen3-0.6B"
    dtype = torch.bfloat16

    cfg = Qwen3Config.from_pretrained(model_id)
    model = Qwen3ForCausalLM(cfg)
    model.model.compute_dtype = dtype

    max_id = model.cfg.vocab_size - 1000
    input_ids = torch.randint(0, max_id, size=(2, 16))
    out = model(input_ids)

    assert model.lm_head.weight.dtype == torch.float32
    assert out.dtype == torch.bfloat16
