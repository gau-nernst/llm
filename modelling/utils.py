import safetensors.torch
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor


def load_hf_state_dict(model_id: str) -> dict[str, Tensor]:
    state_dict = dict()

    for filename in list_repo_files(model_id):
        if not filename.endswith(".safetensors"):
            continue

        local_path = hf_hub_download(model_id, filename)
        state_dict.update(safetensors.torch.load_file(local_path))

    return state_dict
