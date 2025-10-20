# NOTE: due to sequence packing, there is no concept of "batch_size" in this script.
# thus, when num GPUs and/or grad_accum increase, num tokens processed per optimizer
# step also increases. this is DIFFERENT from pretrain.py, which maintains the same
# batch size (hence num tokens processed) per optimizer step.

import argparse
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import distribute_iter, shuffle_iter
from modelling import Qwen3ForCausalLM
from modelling.attn import VarlenInfo
from train_utils import (
    LRSchedule,
    compute_model_tflop,
    create_model,
    create_optimizer,
    get_gpu_tflops,
    get_grad_norm,
    print_model_stats,
)

logger = logging.getLogger(__file__)


class PackedData(NamedTuple):
    inputs: Tensor
    targets: Tensor
    pos_ids: Tensor
    offsets: Tensor
    max_seqlen: int

    def cuda(self):
        return PackedData(*[x.cuda() if isinstance(x, Tensor) else x for x in self])


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer_id: str,
        repo_id: str,
        split: str,
        name: str | None = None,
        maxlen: int = 2048,
        seed: int = 2025,
    ) -> None:
        self.ds = load_dataset(repo_id, name=name, split=split)["messages"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.maxlen = maxlen
        self.seed = seed
        self.load_state_dict()

    def state_dict(self):
        return dict(cnt=self.cnt, curr_tokens=self.curr_tokens)

    def load_state_dict(self, state_dict: dict | None = None):
        self.id_iter = shuffle_iter(len(self.ds), self.seed)
        self.id_iter = distribute_iter(self.id_iter)

        if state_dict is not None:
            self.cnt = state_dict["cnt"]
            self.curr_tokens = state_dict["curr_tokens"]

            for _ in range(self.cnt):  # rewind
                next(self.id_iter)

        else:
            self.cnt = 0
            self.load_next_row()

    def load_next_row(self):
        while True:
            row_id = next(self.id_iter)
            self.cnt += 1

            # guarantee that len(self.curr_tokens) <= self.maxlen
            self.curr_tokens = self.tokenizer.apply_chat_template(self.ds[row_id], add_generation_prompt=False)
            if len(self.curr_tokens) <= self.maxlen:
                break

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1
        return self

    def __next__(self):
        inputs = []
        targets = []
        pos_ids = []
        offsets = [0]
        max_seqlen = 0

        while True:
            # it's guaranteed that len(new_tokens) <= self.maxlen
            if len(inputs) + len(self.curr_tokens) - 1 > self.maxlen:
                break

            seqlen = len(self.curr_tokens) - 1
            inputs.extend(self.curr_tokens[:-1])
            targets.extend(self.curr_tokens[1:])
            pos_ids.extend(range(seqlen))
            offsets.append(offsets[-1] + seqlen)
            max_seqlen = max(max_seqlen, seqlen)

            # get the next row, since we have consumed the current row.
            self.load_next_row()

        return PackedData(
            inputs=torch.tensor(inputs, dtype=torch.int32),
            targets=torch.tensor(targets, dtype=torch.int32),
            pos_ids=torch.tensor(pos_ids, dtype=torch.int32),
            offsets=torch.tensor(offsets, dtype=torch.int32),
            max_seqlen=max_seqlen,
        )


def get_loss(model: Qwen3ForCausalLM, data: PackedData):
    varlen_info = VarlenInfo(data.offsets, data.max_seqlen)
    logits = model(data.inputs, pos_ids=data.pos_ids, varlen_info=varlen_info)
    return F.cross_entropy(logits.float(), data.targets.long())


def main(args: argparse.Namespace):
    rank = int(os.environ.get("RANK", 0))
    is_master = rank == 0
    world_size = 1
    is_dist = args.dist is not None
    is_ddp = args.dist == "ddp"
    is_fsdp = args.dist == "fsdp"

    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        if is_master:
            print(f"Using distributed training with {world_size=}")

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
    if args.profile:
        args.n_steps = 5
    args.torch_version = torch.__version__

    model = create_model(args.model, args.act_ckpt, args.dist)
    if is_master:
        print_model_stats(model)

    optim = create_optimizer(args.optim, model, args.lr, args.weight_decay, **args.optim_kwargs)
    if args.lr_schedule_kwargs is not None:
        lr_schedule = LRSchedule(args.lr, args.n_steps, **args.lr_schedule_kwargs)
    else:
        lr_schedule = None

    ds = PackedDataset(args.model, maxlen=args.maxlen, seed=args.seed, **args.ds)
    dloader = StatefulDataLoader(
        ds,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        snapshot_every_n_steps=args.ckpt_interval,
    )

    args.save_dir = Path("runs/sft") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if is_master:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        logger = wandb.init(
            dir="wandb_logs",
            config=args,
            project=args.project,
            name=args.run_name,
            tags=["sft"],
            mode="disabled" if args.profile else None,
        )

    step = 0
    if args.resume is not None:
        # TODO: test with DDP. make it work with FSDP
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        dloader.load_state_dict(ckpt["dloader"])
        step = ckpt["step"]

    def compute_tflop(seqlen_list: list[int]):
        cfg = model.cfg
        return sum(
            compute_model_tflop(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.intermediate_size,
                cfg.vocab_size,
                cfg.num_hidden_layers,
                seqlen,
                training=True,
            )
            for seqlen in seqlen_list
        )

    gpu_tflops = get_gpu_tflops()
    num_tokens_seen_millions = 0
    num_samples_seen = 0
    tokens_cnt = 0
    tflop_cnt = 0

    log_interval = 50
    pbar = tqdm(total=args.n_steps, initial=step, dynamic_ncols=True, disable=not is_master)
    model.train()
    loss_fn = get_loss if is_fsdp else torch.compile(get_loss, dynamic=True)

    # run forward+backward on the largest possible sequence first for PyTorch to allocate
    # enough memory + make sure we don't OOM on the largest sequence.
    # => no memory increase during training + less fragmentation.
    # in practice, there is still memory increase.
    batch = PackedData(
        inputs=torch.randint(model.cfg.vocab_size, size=(args.maxlen,), dtype=torch.int32, device="cuda"),
        targets=torch.randint(model.cfg.vocab_size, size=(args.maxlen,), dtype=torch.int32, device="cuda"),
        pos_ids=torch.arange(args.maxlen, dtype=torch.int32, device="cuda"),
        offsets=torch.tensor([0, args.maxlen], dtype=torch.int32, device="cuda"),
        max_seqlen=args.maxlen,
    )
    loss_fn(model, batch).backward()
    optim.zero_grad()  # clear gradient

    if args.profile and is_master:
        torch._inductor.config.triton.unique_kernel_names = True
        prof = torch.profiler.profile()

    dloader_iter = iter(dloader)
    time0 = time.perf_counter()
    while step < args.n_steps:
        # DDP: disable gradient all-reduce for non-last micro-steps
        for i in range(args.grad_accum):
            batch = next(dloader_iter)

            with model.no_sync() if (is_ddp and i < args.grad_accum - 1) else contextlib.nullcontext():
                loss = loss_fn(model, batch.cuda())
                loss.backward()

            num_samples_seen += batch.offsets.shape[0] - 1
            tokens_cnt += batch.inputs.shape[0]
            tflop_cnt += compute_tflop(batch.offsets.diff().tolist())

        if lr_schedule is not None:
            lr_schedule.set_lr(step, optim)

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if is_fsdp:
                grad_norm = grad_norm.full_tensor()
        else:
            grad_norm = None

        if step % log_interval == 0:
            if is_dist:
                dist.all_reduce(loss, dist.ReduceOp.AVG)
            log_dict = dict(
                loss=loss.item(),
                grad_norm=grad_norm if grad_norm is not None else get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
            )
            if is_master:
                logger.log(log_dict, step=step)
                pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()
        if args.profile and step == 1 and is_master:
            prof.start()

        if step % log_interval == 0 and is_master:
            time1 = time.perf_counter()
            num_tokens_seen_millions += tokens_cnt / 1e6
            log_dict = dict(
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                num_samples_seen=num_samples_seen,
                num_tokens_seen_millions=num_tokens_seen_millions,
                tokens_per_second=tokens_cnt / (time1 - time0),
                tflops=tflop_cnt / (time1 - time0),
            )
            if gpu_tflops != 0:
                log_dict["mfu"] = log_dict["tflops"] / gpu_tflops
            time0 = time1
            tokens_cnt = 0
            tflop_cnt = 0
            logger.log(log_dict, step=step)

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = dict(
                model=model.state_dict(),
                optim=optim.state_dict(),
                dloader=dloader.state_dict(),
                step=step,
            )
            if is_fsdp:  # FSDP saves on all ranks
                torch.save(ckpt, args.save_dir / f"last_{rank}.pth")
            elif is_master:  # single-device or DDP - only rank 0
                torch.save(ckpt, args.save_dir / "last.pth")

        # TODO: add validation

    if is_master:
        logger.finish()
        if args.profile:
            prof.stop()
            prof.export_chrome_trace("trace.json.gz")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--act_ckpt", action="store_true")
    parser.add_argument("--dist", choices=["ddp", "fsdp"])

    parser.add_argument("--ds", type=json.loads, default=dict(repo_id="allenai/tulu-3-sft-mixture", split="train"))
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--maxlen", type=int, default=8192)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--lr_schedule_kwargs", type=json.loads)
    parser.add_argument("--clip_grad_norm", type=int)

    parser.add_argument("--resume")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    main(args)
