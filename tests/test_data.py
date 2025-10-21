from torch import Tensor

from pretrain import TokenDataset
from sft import PackedData, PackedDataset


def test_token_dataset():
    tokenizer_id = "Qwen/Qwen3-0.6B"
    repo_id = "allenai/c4"
    split = "train"
    name = "en"

    ds = TokenDataset(tokenizer_id, repo_id, split, name)

    # basic reading
    for _ in range(20):
        data = next(ds)
        assert isinstance(data, Tensor)
        assert data.shape == (ds.seqlen + 1,)

    # stateful-ness
    resumed_ds = TokenDataset(tokenizer_id, repo_id, split, name)
    resumed_ds.load_state_dict(ds.state_dict())

    for _ in range(5):
        assert (next(ds) == next(resumed_ds)).all()


def tets_packed_dataset():
    tokenizer_id = "Qwen/Qwen3-0.6B"
    repo_id = "allenai/tulu-3-sft-mixture"
    split = "train"
    maxlen = 2048

    ds = PackedDataset(tokenizer_id, repo_id, split, maxlen=maxlen)

    # basic reading
    for _ in range(20):
        data = next(ds)
        assert isinstance(data, PackedData)
        assert data.inputs.shape[0] <= maxlen

    # stateful-ness
    resumed_ds = PackedDataset(tokenizer_id, repo_id, split, maxlen=maxlen)
    resumed_ds.load_state_dict(ds.state_dict())

    for _ in range(5):
        for this, that in zip(next(ds), next(resumed_ds)):
            if isinstance(this, Tensor):
                assert (this == that).all()
            else:
                assert this == that
