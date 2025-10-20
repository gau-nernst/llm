# torchrun --standalone --nproc_per_node=2 pretrain.py

import argparse
import contextlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import fsspec
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3Config

from data import distribute_iter, get_hf_dataset_path, read_shard, shuffle_iter
from hellaswag import evaluate_hellaswag
from modelling import Qwen3ForCausalLM
from train_utils import LRSchedule, compute_model_tflop, get_gpu_tflops, get_grad_norm, get_optimizer, print_model_stats


class TokenDataset(IterableDataset):
    def __init__(
        self,
        tokenizer_id: str,
        repo_id: str,
        split: str,
        name: str = "default",
        seqlen: int = 2048,
        seed: int = 2025,
    ) -> None:
        glob_pattern = get_hf_dataset_path(repo_id, split, name)
        if glob_pattern is None:
            raise RuntimeError(f"{split=}, {name=} not found for {repo_id}")

        full_pattern = f"hf://datasets/{repo_id}/{glob_pattern}"
        shards = fsspec.filesystem("hf").glob(full_pattern)
        paths = [shard.removeprefix(f"datasets/{repo_id}/") for shard in shards]
        paths.sort()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.repo_id = repo_id
        self.shards = paths
        self.seqlen = seqlen
        self.seed = seed
        self.load_state_dict()

    def state_dict(self):
        return dict(shard_cnt=self.shard_cnt, row_cnt=self.row_cnt, buffer=list(self.buffer))

    def load_state_dict(self, state_dict: dict | None = None):
        if state_dict is not None:
            self.shard_cnt = state_dict["shard_cnt"]
            self.row_cnt = state_dict["row_cnt"]
            self.buffer = list(state_dict["buffer"])
        else:
            self.shard_cnt = 0
            self.row_cnt = 0
            self.buffer = []

        self.shard_id_iter = shuffle_iter(len(self.shards), self.seed)
        self.shard_id_iter = distribute_iter(self.shard_id_iter)
        for _ in range(self.shard_cnt):  # rewind
            next(self.shard_id_iter)

        path = self.shards[next(self.shard_id_iter)]
        self.row_iter = read_shard(f"hf://datasets/{self.repo_id}/{path}", ["text"])
        for _ in range(self.row_cnt):  # rewind
            next(self.row_iter)

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1
        return self

    def __next__(self):
        while len(self.buffer) < self.seqlen + 1:
            try:
                row = next(self.row_iter)
                self.row_cnt += 1

            except StopIteration:  # exhaust current shard
                path = self.shards[next(self.shard_id_iter)]
                self.row_iter = read_shard(f"hf://datasets/{self.repo_id}/{path}", ["text"])
                self.shard_cnt += 1
                self.row_cnt = 0
                continue

            # tokenize this row
            self.buffer.extend(self.tokenizer(row["text"])["input_ids"])

        data = torch.tensor(self.buffer[: self.seqlen + 1], dtype=torch.int32)
        self.buffer = self.buffer[self.seqlen + 1 :]
        return data


def get_loss(model: Qwen3ForCausalLM, data: Tensor):
    logits = model(data[..., :-1]).float().flatten(0, 1)  # [B * L, vocab_size]
    labels = data[..., 1:].long().flatten()  # [B * L]
    return F.cross_entropy(logits, labels)


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

    assert args.bsize % (args.gradient_accumulation * world_size) == 0
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
    if args.profile:
        args.n_steps = 5
    args.torch_version = torch.__version__

    cfg = Qwen3Config.from_pretrained(args.model)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    model.model.compute_dtype = torch.bfloat16
    model.model.act_ckpt = args.act_ckpt

    tflop_per_sample = compute_model_tflop(
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.num_hidden_layers,
        args.seqlen,
        training=True,
    )
    tflop_per_train_step = tflop_per_sample * args.bsize
    gpu_tflops = get_gpu_tflops()

    if is_ddp:
        # initialize model on rank 0, then DDP will broadcast to other ranks
        model.to_empty(device="cuda")
        if is_master:
            model.init_weights()
        model = DDP(model)

    elif is_fsdp:
        # init model after sharding
        for layer in model.model.layers:
            fully_shard(layer)
            layer.compile()  # FSDP is more performant when compiling this way
        fully_shard(model)
        model.to_empty(device="cuda")
        model.init_weights()

    else:
        # single-GPU case
        model.to_empty(device="cuda")
        model.init_weights()

    if is_master:
        print_model_stats(model)

    optim = get_optimizer(args.optim, model, args.lr, args.weight_decay, **args.optim_kwargs)
    if args.lr_schedule_kwargs is not None:
        lr_schedule = LRSchedule(args.lr, args.n_steps, **args.lr_schedule_kwargs)
    else:
        lr_schedule = None

    ds = TokenDataset(args.model, seqlen=args.seqlen, seed=args.seed, **args.ds)
    forward_bsize = args.bsize // (args.gradient_accumulation * world_size)
    dloader = StatefulDataLoader(
        ds,
        batch_size=forward_bsize,
        num_workers=1,
        pin_memory=True,
        snapshot_every_n_steps=args.ckpt_interval,
    )

    args.save_dir = Path("runs/pretrain") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if is_master:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        logger = wandb.init(
            dir="wandb_logs",
            config=args,
            project=args.project,
            name=args.run_name,
            tags=["pretrain"],
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

    log_interval = 50
    pbar = tqdm(total=args.n_steps, initial=step, dynamic_ncols=True, disable=not is_master)
    model.train()
    loss_fn = get_loss if is_fsdp else torch.compile(get_loss)
    time0 = time.perf_counter()
    if args.profile and is_master:
        torch._inductor.config.triton.unique_kernel_names = True
        prof = torch.profiler.profile()

    dloader_iter = iter(dloader)
    while step < args.n_steps:
        # DDP: disable gradient all-reduce for non-last micro-steps
        with model.no_sync() if is_ddp else contextlib.nullcontext():
            for _ in range(args.gradient_accumulation - 1):
                loss = loss_fn(model, next(dloader_iter).cuda())
                loss.backward()
        loss = loss_fn(model, next(dloader_iter).cuda())
        loss.backward()

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
            tokens_per_batch = args.bsize * args.seqlen
            time1 = time.perf_counter()
            log_dict = dict(
                max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                num_tokens_seen_millions=tokens_per_batch * step / 1e6,
                tokens_per_second=tokens_per_batch * log_interval / (time1 - time0),
                tflops=tflop_per_train_step * log_interval / (time1 - time0),
            )
            if gpu_tflops != 0:
                log_dict["mfu"] = log_dict["tflops"] / gpu_tflops
            time0 = time1
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

        if args.hellaswag_interval > 0 and step % args.hellaswag_interval == 0:
            # we must evaluate on all ranks for FSDP to work
            acc = evaluate_hellaswag(model, tokenizer_id=args.model)
            if is_master:
                logger.log(dict(hellaswag_acc=acc), step=step)

            if is_dist:
                dist.barrier()
            model.train()

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

    parser.add_argument("--ds", type=json.loads, default=dict(repo_id="allenai/c4", split="train", name="en"))
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--lr_schedule_kwargs", type=json.loads)
    parser.add_argument("--clip_grad_norm", type=int)

    parser.add_argument("--hellaswag_interval", type=int, default=0)

    parser.add_argument("--resume")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    main(args)
