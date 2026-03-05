"""
Stage 1: CrossDoc-JEPA Pretraining — v2

Fix: EMA update is called AFTER scaler.step() + scheduler.step(), not inside forward().
     This guarantees the target encoder only moves once valid gradients have been applied.

Fix: Sentence-level JEPA loss (sent_loss) is now part of the pretraining objective
     via CD-JEPA's hierarchical prediction, warming up JSE's shared sent_predictor.
"""

import os, sys, yaml, math, logging, argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import RobertaTokenizerFast, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import CrossDocJEPA
from data.dataset import WikipediaClusterDataset, pretrain_collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def cosine_ema(step: int, total: int, base: float = 0.996, final: float = 0.999) -> float:
    return final - (final - base) * (math.cos(math.pi * step / total) + 1) / 2


def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def save_ckpt(model, opt, sched, scaler, step, config, path):
    m = model.module if hasattr(model, "module") else model
    torch.save({"step": step, "model": m.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "config": config}, path)
    log.info(f"Saved checkpoint: {path}")


def pretrain(config: Dict):
    use_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world_size = setup_dist() if use_dist else (0, 1)
    is_main = rank == 0
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    tokenizer = RobertaTokenizerFast.from_pretrained(config["encoder_model"])
    dataset = WikipediaClusterDataset(
        cluster_file=config["train_cluster_file"],
        tokenizer=tokenizer,
        max_docs_per_cluster=config.get("max_docs", 8),
        max_paras=config.get("max_paras", 12),
        max_sents_per_para=config.get("max_sents_per_para", 6),
        max_sent_len=config.get("max_sent_len", 64),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_dist else None
    loader  = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                         shuffle=(sampler is None), num_workers=config.get("num_workers", 4),
                         collate_fn=pretrain_collate_fn, pin_memory=True, drop_last=True)

    model = CrossDocJEPA(config).to(device)
    if use_dist:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Separate LR groups: encoder gets lower LR to preserve pretrained representations
    inner = model.module if use_dist else model
    enc_params  = [p for n, p in inner.named_parameters()
                   if "target_encoder" not in n and ("context_encoder" in n or "hde" in n)]
    pred_params = [p for n, p in inner.named_parameters()
                   if "target_encoder" not in n and "context_encoder" not in n and "hde" not in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": config["lr"] * 0.1},
        {"params": pred_params, "lr": config["lr"]},
    ], weight_decay=config.get("weight_decay", 0.05))

    total_steps  = config["num_steps"]
    warmup_steps = config.get("warmup_steps", 1000)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler(enabled=config.get("fp16", True))

    step, epoch = 0, 0
    running = {}
    log_interval  = config.get("log_interval",  50)
    save_interval = config.get("save_interval", 5000)

    while step < total_steps:
        epoch += 1
        if use_dist: sampler.set_epoch(epoch)

        for batch in loader:
            if step >= total_steps: break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            with autocast(enabled=config.get("fp16", True)):
                m = model.module if use_dist else model
                losses = m.pretrain_forward(batch)
                loss   = losses["total_jepa_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── EMA update — AFTER optimizer step ─────────────────────────────
            tau = cosine_ema(step, total_steps,
                             config.get("ema_base", 0.996), config.get("ema_final", 0.999))
            # broadcast_dist=True syncs target_encoder across all GPUs after EMA
            (model.module if use_dist else model).cd_jepa.update_target_encoder(tau, broadcast_dist=use_dist)

            step += 1

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    running[k] = running.get(k, 0.0) + v.item()

            if is_main and step % log_interval == 0:
                lr  = scheduler.get_last_lr()[0]
                msg = f"Step {step:6d}/{total_steps} | LR {lr:.2e} | EMA τ {tau:.4f}"
                for k, v in running.items():
                    if k != "total_jepa_loss":
                        msg += f" | {k}: {v/log_interval:.4f}"
                log.info(msg)
                running = {}

            if is_main and step % save_interval == 0:
                save_ckpt(model, optimizer, scheduler, scaler, step, config,
                          Path(config["output_dir"]) / f"pretrain_step_{step}.pt")

    if is_main:
        save_ckpt(model, optimizer, scheduler, scaler, step, config,
                  Path(config["output_dir"]) / "pretrain_final.pt")
        log.info("Pretraining complete.")

    if use_dist: dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    pretrain(config)
