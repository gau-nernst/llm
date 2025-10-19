import io
import json
from typing import Sequence

import fsspec
import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import hf_hub_download


def read_shard(path: str, columns: list[str] | None = None):
    if path.endswith(".parquet"):
        return read_parquet(path, columns)
    elif path.endswith((".jsonl", ".json.gz")):
        return read_jsonl(path, columns)
    else:
        raise ValueError(f"Unsupported {path=}")


def read_parquet(path: str, columns: list[str] | None = None):
    with fsspec.open(path) as f, pq.ParquetFile(f) as pq_f:
        for group_id in range(pq_f.num_row_groups):
            yield from pq_f.read_row_group(group_id, columns).to_pylist()


def read_jsonl(path: str, columns: list[str] | None = None):
    compression = "gzip" if path.endswith(".gz") else None
    with fsspec.open(path, compression=compression) as f:
        for line in f:
            row = json.loads(line)
            if columns is not None:
                row = {col: row[col] for col in columns}
            yield row


def get_hf_dataset_path(repo_id: str, split: str, name: str = "default"):
    path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
    content = open(path).read()
    meta = yaml.safe_load(io.StringIO(content.split("---\n")[1]))
    configs = meta["configs"]

    glob_pattern = None
    for config in configs:
        if config["config_name"] != name:
            continue
        for data_file in config["data_files"]:
            if data_file["split"] != split:
                continue
            glob_pattern = data_file["path"]
            break
        if glob_pattern is not None:
            break

    return glob_pattern


def infinite_stream(data: Sequence, seed: int):
    rng = torch.Generator("cpu").manual_seed(seed)
    while True:
        indices = torch.randperm(len(data), generator=rng).tolist()
        yield from (data[idx] for idx in indices)
