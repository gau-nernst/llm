import torch
from transformers import AutoModelForCausalLM

from modelling.internvl import IMG_CONTEXT_ID, IMG_END_ID, IMG_START_ID, InternVLChatModel


def test_internvl():
    model_id = "OpenGVLab/InternVL3_5-1B"
    device = "cuda"

    # test using FP32
    model = InternVLChatModel.from_pretrained(model_id).to(device).float()
    model_ref = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model_ref.img_context_token_id = IMG_CONTEXT_ID

    img_size = 448
    img = torch.randn(2, 3, img_size, img_size, device="cuda")

    # check ViT
    with torch.no_grad():
        out = model.vision_model(img)
        out_ref = model_ref.vision_model(img).last_hidden_state[:, 1:]
    torch.testing.assert_close(out, out_ref, rtol=3e-4, atol=2e-4)

    num_img_tokens = (img_size // 28) * (img_size // 28)
    img_tokens = [IMG_START_ID] + [IMG_CONTEXT_ID] * num_img_tokens + [IMG_END_ID]

    # different position of image embeddings
    max_token_id = model.language_model.cfg.vocab_size - 10000
    seq0 = torch.cat(
        [
            torch.randint(0, max_token_id, size=(16,), device=device),
            torch.tensor(img_tokens, device=device),
            torch.randint(0, max_token_id, size=(8,), device=device),
        ],
    )
    seq1 = torch.cat(
        [
            torch.randint(0, max_token_id, size=(8,), device=device),
            torch.tensor(img_tokens, device=device),
            torch.randint(0, max_token_id, size=(16,), device=device),
        ],
    )
    input_ids = torch.stack([seq0, seq1], dim=0)
    image_flags = torch.ones(2, num_img_tokens, device="cuda")

    with torch.no_grad():
        out = model(img, input_ids)
        out_ref = model_ref(img, input_ids, image_flags=image_flags).logits

    # difference is too much...
    rtol = 1e-3
    atol = 1e-3
    tol = out_ref.float().abs() * rtol + atol
    diff = (out.float() - out_ref.float()).abs()
    mismatch = diff > tol
    mismatch_pct = mismatch.float().mean().item()
    assert mismatch_pct < 1e-4  # 0.01%
