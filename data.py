import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer


# adapted from https://github.com/pytorch/torchtitan/blob/v0.2.0/torchtitan/datasets/hf_datasets.py
# must have "text" column e.g.
# - allenai/c4
# - HuggingFaceFW/fineweb-edu
class HFTextDataset(IterableDataset):
    def __init__(
        self,
        dataset: str,
        subset: str,
        split: str,
        tokenizer: str,
        seqlen: int,
        eval: bool,
    ) -> None:
        self.ds = load_dataset(dataset, name=subset, split=split, streaming=True).select_columns("text")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.seqlen = seqlen
        self.eval = eval

        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1

        if world_size > 1:
            self.ds = split_dataset_by_node(self.ds, rank, world_size)
        self._epoch = 0
        self._buffer: list[int] = []

    def __iter__(self):
        # does HF datasets split samples among data workers automatically?
        worker_info = get_worker_info()
        if worker_info is not None:
            assert worker_info.num_workers == 1

        SAMPLE_LEN = self.seqlen + 1
        while True:
            self.ds.set_epoch(self._epoch)
            for sample in self.ds:
                self._buffer.extend(self.tokenizer(sample["text"])["input_ids"])

                while len(self._buffer) >= SAMPLE_LEN:
                    sample = torch.tensor(self._buffer[:SAMPLE_LEN], dtype=torch.int32)
                    self._buffer = self._buffer[SAMPLE_LEN:]
                    yield sample[:-1], sample[1:]

            self._epoch += 1
            if self.eval:
                break

    def state_dict(self):
        return dict(
            ds=self.ds.state_dict(),
            _epoch=self._epoch,
            _buffer=list(self._buffer),  # make a copy
        )

    def load_state_dict(self, state_dict: dict):
        self.ds.load_state_dict(state_dict["ds"])
        self._epoch = state_dict["_epoch"]
        self._buffer = list(state_dict["_buffer"])  # make a copy
