import pytest

from data import JsonlShard, ParquetShard


@pytest.mark.parametrize(
    "shard_type,repo_id,path,col",
    [
        ("parquet", "allenai/tulu-3-sft-mixture", "data/train-00000-of-00006.parquet", "messages"),
        ("jsonl", "allenai/c4", "en/c4-train.00000-of-01024.json.gz", "text"),
    ],
)
def test_shard(shard_type: str, repo_id: str, path: str, col: str):
    shard_cls = dict(
        parquet=ParquetShard,
        jsonl=JsonlShard,
    )[shard_type]
    shard = shard_cls(repo_id, path, columns=[col])

    # basic reading
    # this particular file has 1000 rows in a rowgroup.
    # we intentionally read more than 1 rowgroup to test stateful feature later.
    for _ in range(1040):
        row = next(shard)
    assert isinstance(row, dict)
    assert len(row) == 1
    assert col in row.keys()

    # stateful
    state_dict = shard.state_dict()
    resumed_shard = shard_cls(repo_id, path, columns=[col])
    resumed_shard.load_state_dict(state_dict)

    for _ in range(5):
        assert next(shard) == next(resumed_shard)
