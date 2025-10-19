from data import ParquetShard


def test_parquet_shard():
    repo_id = "allenai/tulu-3-sft-mixture"
    path = "data/train-00000-of-00006.parquet"

    shard = ParquetShard(repo_id, path, columns=["messages"])

    # basic reading
    # this particular file has 1000 rows in a rowgroup.
    # we intentionally read more than 1 rowgroup to test stateful feature later.
    for _ in range(1040):
        row = next(shard)
    assert isinstance(row, dict)
    assert len(row) == 1
    assert "messages" in row.keys()

    # stateful
    state_dict = shard.state_dict()
    resumed_shard = ParquetShard(repo_id, path, columns=["messages"])
    resumed_shard.load_state_dict(state_dict)

    for _ in range(5):
        assert next(shard) == next(resumed_shard)
