import copy
import io
import json
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


class HFShard:
    def __init__(self, repo_id: str, path: str, columns: list[str] | None = None):
        self.path = f"hf://datasets/{repo_id}/{path}"
        self.columns = list(columns)
        self.file = None

    def open_file(self, compression: str | None = None):
        self.file = fsspec.open(self.path, compression=compression)
        return self.file.__enter__()

    def close_file(self):
        if self.file is not None:
            self.file.__exit__(None, None, None)
            self.file = None


# all operations are stateful
class ParquetShard(HFShard):
    def __init__(self, repo_id: str, path: str, columns: list[str] | None = None) -> None:
        super().__init__(repo_id, path, columns)
        self.row_group_id = 0
        self.row_id = 0
        self.init()

    def init(self):
        self.close_file()
        self.pq_file = pq.ParquetFile(self.open_file())
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
                self.close_file()
                raise StopIteration

            self.read_row_group()

        return row

    def __len__(self):
        return self.pq_file.metadata.num_rows


class JsonlShard(HFShard):
    def __init__(self, repo_id: str, path: str, columns: list[str] | None = None) -> None:
        super().__init__(repo_id, path, columns)
        self.row_id = 0
        self.init()

    def init(self):
        self.close_file()
        self.row_iter = iter(self.open_file(compression="gzip"))

        # rewind
        for _ in range(self.row_id):
            next(self.row_iter)

    def state_dict(self):
        return dict(row_id=self.row_id)

    def load_state_dict(self, state_dict):
        state_dict = copy.deepcopy(state_dict)
        self.row_id = state_dict.pop("row_id")
        assert len(state_dict) == 0  # make sure we have consumed everything
        self.init()

    def __iter__(self):
        return self

    def __next__(self):
        # NOTE: we don't know when we will reach EOF ahead of time
        try:
            row = json.loads(next(self.row_iter))
            if self.columns is not None:
                row = {col: row[col] for col in self.columns}
            self.row_id += 1

        except StopIteration:
            self.close_file()
            raise

        return row


def infinite_stream(data: Sequence, seed: int):
    rng = torch.Generator("cpu").manual_seed(seed)
    while True:
        indices = torch.randperm(len(data), generator=rng)
        for idx in indices:
            yield data[idx]
