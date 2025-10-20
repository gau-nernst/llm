import io
import json
from typing import Iterable, Sequence

import fsspec
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
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


# yield a deterministic sequence
def shuffle_iter(n: int, seed: int):
    rng = torch.Generator("cpu").manual_seed(seed)
    while True:
        yield from torch.randperm(n, generator=rng).tolist()


def distribute_iter(iter_: Iterable):
    """With world size N, rank i will take elements whose index % world_size == rank"""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    counter = 0
    for x in iter_:
        if counter == rank:
            yield x
        counter = (counter + 1) % world_size
