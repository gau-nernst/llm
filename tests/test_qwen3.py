import torch
from transformers import AutoModelForCausalLM

from qwen3 import Qwen3ForCausalLM


def test_qwen3():
    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda"

    # test using FP32
    model = Qwen3ForCausalLM.from_pretrained(model_id).to(device).float()
    model_ref = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    input_ids = torch.randint(0, model.cfg.vocab_size - 1000, size=(1, 16), device=device)
    with torch.no_grad():
        out = model(input_ids)
        out_ref = model_ref(input_ids).logits
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)

    input_embeds = torch.randn(1, 16, model.cfg.hidden_size, device=device)
    with torch.no_grad():
        out = model(input_embeds=input_embeds)
        out_ref = model_ref(inputs_embeds=input_embeds).logits
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)
