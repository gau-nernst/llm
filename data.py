import copy
import io
from typing import Sequence

import fsspec
import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import hf_hub_download


def get_hf_dataset_configs(repo_id: str):
    path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
    content = open(path).read()
    meta = yaml.safe_load(io.StringIO(content.split("---\n")[1]))
    return meta["configs"]


# all operations are stateful
class ParquetShard:
    def __init__(self, repo_id: str, path: str, columns: list[str] | None = None):
        self.path = f"hf://datasets/{repo_id}/{path}"
        self.columns = list(columns)

        self.row_group_id = 0
        self.row_id = 0
        self.init()

    def init(self):
        self.fs_file = fsspec.open(self.path)
        self.pq_file = pq.ParquetFile(self.fs_file.__enter__())
        self.read_row_group()

    def read_row_group(self):
        # materialize in RAM
        self.row_group = self.pq_file.read_row_group(self.row_group_id, self.columns).to_pylist()

    def state_dict(self):
        return dict(row_group_id=self.row_group_id, row_id=self.row_id)

    def load_state_dict(self, state_dict):
        state_dict = copy.deepcopy(state_dict)
        self.row_group_id = state_dict.pop("row_group_id")
        self.row_id = state_dict.pop("row_id")
        assert len(state_dict) == 0  # make sure we have consumed everything
        self.init()

    def __iter__(self):
        return self

    def __next__(self):
        row = self.row_group[self.row_id]
        self.row_id += 1

        # finish this row group
        if self.row_id == len(self.row_group):
            self.row_id = 0
            self.row_group_id += 1

            # finish this file
            if self.row_group_id == self.pq_file.num_row_groups:
                raise StopIteration

            self.read_row_group()

        return row

    def __len__(self):
        return self.pq_file.metadata.num_rows


def infinite_stream(data: Sequence, seed: int):
    rng = torch.Generator("cpu").manual_seed(seed)
    while True:
        indices = torch.randperm(len(data), generator=rng)
        for idx in indices:
            yield data[idx]
