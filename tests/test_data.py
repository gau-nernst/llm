from torch import Tensor

from pretrain import TokenDataset


def test_shard():
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
    state_dict = ds.state_dict()
    resumed_ds = TokenDataset(tokenizer_id, repo_id, split, name)
    resumed_ds.load_state_dict(state_dict)

    for _ in range(5):
        assert (next(ds) == next(resumed_ds)).all()
