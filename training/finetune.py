"""
Stage 2: CrossDoc-JEPA Finetuning — v4
Fix: gate_logit isolated to dedicated low-LR parameter group (lr * 0.01)
to prevent gradient-magnitude-driven oscillation of the SRD convergence gate.
"""

import os, sys, yaml, logging, argparse
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import RobertaTokenizerFast, BartTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import CrossDocJEPA
from data.dataset import MultiNewsDataset, finetune_collate_fn
from evaluation.evaluate import evaluate_model, log_gate_alpha

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def build_optimizer(model, config):
    """
    Four parameter groups with carefully separated learning rates:

    1. gate_logit  (lr * 0.01) — SRD convergence gate scalar.
       A single global scalar receiving gradients from the entire (B, N, H) blob.
       Standard LR would cause violent oscillation as the model flickers between
       trusting and ignoring the JEPA predictor. Highly constrained LR ensures
       smooth, monotone convergence of gate_alpha over training.

    2. encoder params (lr * 0.20) — HDE + context encoder.
       Pretrained RoBERTa weights adapted to news domain at a conservative rate.

    3. new params (lr * 1.00) — FAD dual-attn blocks, SRD layers, JSE refiner,
       BART decoder layers above freeze threshold. Full learning rate.

    4. target_encoder — frozen (EMA-only). requires_grad = False throughout.
    """
    freeze_bart = config.get("freeze_bart_layers", 6)
    gate_params, enc_params, new_params = [], [], []

    for name, param in model.named_parameters():
        if "target_encoder" in name:
            param.requires_grad_(False)
            continue
        # Isolate the SRD convergence gate scalar into its own group
        if "srd.gate_logit" in name:
            gate_params.append(param)
        elif "context_encoder" in name or "hde" in name:
            enc_params.append(param)
        elif "bart.model.decoder.layers" in name:
            try:   layer_num = int(name.split("layers.")[1].split(".")[0])
            except: layer_num = freeze_bart
            if layer_num < freeze_bart:
                param.requires_grad_(False)
            else:
                new_params.append(param)
        else:
            new_params.append(param)

    lr = config["lr"]
    wd = config.get("weight_decay", 0.01)
    return torch.optim.AdamW([
        {"params": gate_params, "lr": lr * 0.10, "weight_decay": 0.0},  # gate: no wd
        {"params": enc_params,  "lr": lr * 0.20, "weight_decay": wd},
        {"params": new_params,  "lr": lr * 1.00, "weight_decay": wd},
    ])


def finetune(config):
    use_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1

    is_main = rank == 0
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_log_path = str(out_dir / "gate_alpha.jsonl")

    enc_tok = RobertaTokenizerFast.from_pretrained(config["encoder_model"])
    dec_tok = BartTokenizerFast.from_pretrained(config["decoder_model"])

    ds_kw = dict(max_docs=config.get("max_docs",5), max_paras=config.get("max_paras",12),
                 max_sents_per_para=config.get("max_sents_per_para",6),
                 max_sent_len=config.get("max_sent_len",64),
                 max_summary_len=config.get("max_summary_len",256))
    train_ds = MultiNewsDataset("train",      enc_tok, dec_tok, **ds_kw)
    val_ds   = MultiNewsDataset("validation", enc_tok, dec_tok, **ds_kw)

    tr_sampler   = DistributedSampler(train_ds) if use_dist else None
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=tr_sampler, shuffle=(tr_sampler is None),
                              num_workers=config.get("num_workers",2),
                              collate_fn=finetune_collate_fn, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=config.get("eval_batch_size",2),
                              shuffle=False, num_workers=config.get("num_workers", 8), collate_fn=finetune_collate_fn)

    model = CrossDocJEPA(config).to(device)
    ckpt  = config.get("pretrain_checkpoint")
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        if is_main: log.info("Loaded pretrain checkpoint.")

    if use_dist:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    inner     = model.module if use_dist else model
    optimizer = build_optimizer(inner, config)
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler   = get_linear_schedule_with_warmup(optimizer, config.get("warmup_steps",500), total_steps)
    scaler      = GradScaler(enabled=config.get("fp16", True))

    best_r2 = 0.0; global_step = 0
    log_int  = config.get("log_interval", 20)
    eval_int = config.get("eval_interval", 500)
    gate_log_int = config.get("gate_log_interval", 100)  # log gate_alpha every N steps
    running  = {}

    for epoch in range(config["num_epochs"]):
        model.train()
        if use_dist: tr_sampler.set_epoch(epoch)

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(enabled=config.get("fp16", True)):
                m   = model.module if use_dist else model
                out = m.finetune_forward(
                    batch=batch,
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                    # oracle_salience provides ROUGE-1 recall per sentence vs reference.
                    # Without this, jse.ranking_loss() is never called, score_refiner
                    # receives no direct gradients, and salience_loss = 0 throughout.
                    oracle_salience=batch.get("oracle_scores"),
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm",1.0))
            scaler.step(optimizer); scaler.update(); scheduler.step()

            (model.module if use_dist else model).cd_jepa.update_target_encoder(broadcast_dist=use_dist)

            for k in ("loss","ce_loss","jepa_loss","salience_loss","faithfulness_loss"):
                running[k] = running.get(k, 0.0) + out.get(k, torch.tensor(0.)).item()
            global_step += 1

            # Log gate_alpha for convergence analysis
            if is_main and global_step % gate_log_int == 0:
                log_gate_alpha(out.get("gate_alpha", 0.0), global_step, gate_log_path)

            if is_main and global_step % log_int == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(
                    f"Ep {epoch+1} | Step {global_step} | "
                    + " | ".join(f"{k}: {v/log_int:.4f}" for k, v in running.items())
                    + f" | LR {lr:.2e} | gate_α {out.get('gate_alpha',0.):.4f}"
                )
                running = {}

            if is_main and global_step % eval_int == 0:
                model.eval()
                m = model.module if use_dist else model
                metrics = evaluate_model(
                    m, val_loader, dec_tok, device,
                    max_samples=config.get("eval_samples", 200),
                    min_length=config.get("eval_min_length", 10),
                    length_penalty=config.get("eval_length_penalty", 2.0),
                    run_factcc=config.get("eval_factcc", False),
                )
                log.info(f"[Eval] R-1 {metrics['rouge1']:.4f} | R-2 {metrics['rouge2']:.4f} "
                         f"| R-L {metrics['rougeL']:.4f} | BertScore {metrics.get('bertscore',0.):.4f}")
                if metrics["rouge2"] > best_r2:
                    best_r2 = metrics["rouge2"]
                    torch.save({"model": m.state_dict(), "metrics": metrics},
                               out_dir / "best_model.pt")
                    log.info(f"↑ New best R-2: {best_r2:.4f}")
                model.train()

        if is_main:
            m = model.module if use_dist else model
            ckpt_path = out_dir / f"epoch_{epoch+1}.pt"
            torch.save({"model": m.state_dict(), "epoch": epoch + 1, "config": config}, ckpt_path)
            log.info(f"Saved epoch checkpoint: {ckpt_path}")

    if is_main:
        log.info(f"Finetuning complete. Best R-2: {best_r2:.4f}")
        log.info(f"Gate alpha log: {gate_log_path}")
        log.info("To plot convergence: python evaluation/evaluate.py --plot_gate "
                 f"--gate_log {gate_log_path} --config ... --checkpoint ...")
    if use_dist: dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    finetune(config)
