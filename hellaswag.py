import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

CACHE_DIR = Path(__file__).parent / "data_cache"


# https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.4/lm_eval/tasks/hellaswag/utils.py
def preprocess(text: str):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def get_hellaswag(tokenizer_id: str, split: str = "validation", no_cache: bool = False):
    cache_key = f"hellaswag_{split}_{tokenizer_id.replace('/', '_')}.bin"
    cache_path = CACHE_DIR / cache_key

    ds = load_dataset("Rowan/hellaswag", split=split)
    max_seqlen = 192 + 1  # forward pass will use 192 tokens = multiple of 64

    if cache_path.exists() and not no_cache:
        return torch.from_numpy(np.memmap(cache_path, dtype=np.int32, mode="r")).view(-1, 4, max_seqlen)

    # pad to 256 to avoid recompilation
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    data = torch.empty(len(ds), 4, max_seqlen, dtype=torch.int32)

    for row_idx, row in enumerate(ds):
        ctx = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        for ending_idx, ending in enumerate(row["endings"]):
            toks = tokenizer(preprocess(f"{ctx} {ending}"))["input_ids"]
            assert len(toks) <= data.shape[-1]
            data[row_idx, ending_idx, : len(toks)] = torch.tensor(toks)
            data[row_idx, ending_idx, len(toks) :] = -100  # F.cross_entropy() will ignore this

    if not no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fp = np.memmap(cache_path, dtype=np.int32, mode="w+", shape=data.numel())
        fp[:] = data.view(-1).numpy()
        fp.flush()

    return data


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer_id: str, split: str = "validation", pbar: bool = True) -> None:
    device = next(model.parameters()).device

    data = get_hellaswag(tokenizer_id, split).to(device)  # (10042, 4, 193)
    labels_raw = load_dataset("Rowan/hellaswag", split=split)["label"]
    labels = torch.tensor([int(x) for x in labels_raw], device=device)

    n_correct = 0
    bsize = 16
    model.eval()
    for start in tqdm(
        range(0, data.shape[0], bsize),
        desc=f"Evaluate hellaswag {split}",
        disable=not pbar,
        dynamic_ncols=True,
    ):
        end = min(start + bsize, data.shape[0])
        data_batch = data[start:end]

        # remove last token. clip to remove ignore_index=-100 for F.cross_entropy
        inputs = data_batch[..., :-1].flatten(0, 1).clip(0)  # (B * 4, 192)
        logits = model(inputs).float().flatten(0, 1)  # (B * 4 * 192, vocab_size)
        loss = F.cross_entropy(logits, data_batch[..., 1:].flatten().long(), reduction="none")  # (B * 4 * 192)
        loss = loss.view(end - start, 4, -1).sum(-1)  # (B, 4)
        preds = loss.argmin()  # (B,)

        labels_batch = labels[start:end]
        n_correct += (preds == labels_batch).sum().item()

    return n_correct / data.shape[0]
