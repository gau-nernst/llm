import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor, nn


# cast to activation dtype
class Linear(nn.Linear):
    def forward(self, x: Tensor):
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


# this also matches the behavior of single-GPU/DDP with FSDP
# i.e. norm weight is cast to BF16
class RMSNorm(nn.RMSNorm):
    def forward(self, x: Tensor):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


def make_merge_hook(old_keys: list[str], new_key: str):
    def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f"{prefix}{old_keys[0]}_proj.weight" not in state_dict:
            return
        w_list = [state_dict.pop(f"{prefix}{key}_proj.weight") for key in old_keys]
        state_dict[f"{prefix}{new_key}_proj.weight"] = torch.cat(w_list, dim=0)

    return hook


def load_hf_state_dict(model_id: str) -> dict[str, Tensor]:
    state_dict = dict()

    for filename in list_repo_files(model_id):
        if not filename.endswith(".safetensors"):
            continue

        local_path = hf_hub_download(model_id, filename)
        state_dict.update(safetensors.torch.load_file(local_path))

    return state_dict
