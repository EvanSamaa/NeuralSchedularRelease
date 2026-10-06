"""Training and evaluation machinery shared by the classifier entry points."""

import argparse
import json
import logging
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .constants import phase_labels
from .data.dataset import BackchannelJointDataset, make_collate_fn
from .metrics import evaluate, format_eval_table
from .models.factory import (
    ARCH_CLASS_NAMES,
    ARCH_JOINT,
    ARCH_MMF2F,
    ARCH_PAPER_FUSION,
    architecture_of,
    build_model_from_config,
    build_tokenizer,
    find_tokenizer_dir,
    load_tokenizer,
    load_weights,
    resolve_checkpoint,
    save_checkpoint,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

NO_DECAY_NAME_PARTS = ("bias", "LayerNorm.weight", "layer_norm.weight")


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse ``argv``; values from a ``--config`` YAML file replace the parser
    defaults and are themselves overridden by explicit command-line arguments.

    The file is a flat mapping of argument names (without dashes) to values.
    """
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file of argument values; command-line arguments take precedence.")
    known, _ = _peek_config(argv)
    if known is not None:
        import yaml

        with open(known, "r") as f:
            values = yaml.safe_load(f) or {}
        dests = {action.dest: action for action in parser._actions}
        unknown = sorted(set(values) - set(dests))
        if unknown:
            parser.error(f"unknown keys in {known}: {unknown}")
        for key in values:
            dests[key].required = False
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def _peek_config(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, rest = pre.parse_known_args(argv)
    return known.config, rest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_run(args, log_name: str, logger_name: str) -> logging.Logger:
    """Create the output directory, log to ``<output_dir>/<log_name>`` and the
    console, and record the arguments in ``args.json``."""
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_name)),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(logger_name)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Args: {vars(args)}")
    return logger


def make_tensorboard_writer(output_dir: str):
    """A ``SummaryWriter`` in ``<output_dir>/tensorboard``, or ``None`` without tensorboard."""
    if SummaryWriter is None:
        return None
    return SummaryWriter(os.path.join(output_dir, "tensorboard"))


def build_optimizer_and_scheduler(model, lr: float, weight_decay: float, warmup_ratio: float,
                                  total_steps: int, logger: Optional[logging.Logger] = None):
    """AdamW over the trainable parameters (one learning rate; no weight decay
    on biases and LayerNorm weights) with linear warm-up and linear decay."""
    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(part in name for part in NO_DECAY_NAME_PARTS):
            nodecay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    if logger:
        logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")
    return optimizer, scheduler


def log_model_size(model, arch: str, args, logger: logging.Logger) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ({ARCH_CLASS_NAMES[arch]}): {trainable:,} trainable / {total:,} total parameters")
    if args.text_only:
        logger.info("ABLATION: text_only (audio branch disabled)")
    if args.audio_only:
        logger.info("ABLATION: audio_only (text branch disabled)")


def _log_validation(val_metrics: Dict, epoch: int, avg_train_loss: float, label_names: List[str],
                    num_classes: int, logger, writer) -> None:
    if logger:
        logger.info(
            f"Epoch {epoch} — "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc']:.4f} | "
            f"val_macro_eer={val_metrics['macro_eer']:.4f} | "
            f"bc_f1={val_metrics['bc_f1']:.4f} "
            f"(P={val_metrics['bc_precision']:.3f} R={val_metrics['bc_recall']:.3f})"
        )
        logger.info("\n" + format_eval_table(val_metrics, label_names))
    if writer:
        for key in ("loss", "accuracy", "macro_f1", "weighted_f1", "macro_auc", "macro_eer",
                    "bc_f1", "bc_precision", "bc_recall"):
            writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
        for c, name in enumerate(label_names[:num_classes]):
            if not math.isnan(val_metrics["per_class_auc"][c]):
                writer.add_scalar(f"val/auc/{name}", val_metrics["per_class_auc"][c], epoch)
            if not math.isnan(val_metrics["per_class_eer"][c]):
                writer.add_scalar(f"val/eer/{name}", val_metrics["per_class_eer"][c], epoch)
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, criterion,
    device, num_classes, label_names, output_dir, model_cfg,
    epochs, patience, grad_clip, log_interval, logger, writer, tokenizer=None,
) -> float:
    """Train with early stopping on validation macro-F1.

    After every epoch the model is saved to ``<output_dir>/last_model``; the
    best one so far (by validation macro-F1) also goes to
    ``<output_dir>/best_model`` together with ``best_metrics.json``. Training
    stops after ``patience`` epochs without improvement. Returns the best
    macro-F1.
    """
    best_macro_f1 = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                audio=batch["audio"].to(device),
                audio_lens=batch["audio_lens"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            if (batch_idx + 1) % log_interval == 0 and logger:
                logger.info(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"loss={epoch_loss / num_batches:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / max(num_batches, 1)
        if logger:
            logger.info(f"Epoch {epoch} — evaluating...")
        val_metrics = evaluate(model, val_loader, device, num_classes, label_names)
        _log_validation(val_metrics, epoch, avg_train_loss, label_names, num_classes, logger, writer)

        save_checkpoint(model, os.path.join(output_dir, "last_model"), model_cfg, tokenizer)
        if logger:
            logger.info("  Saved last checkpoint")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            best_dir = os.path.join(output_dir, "best_model")
            save_checkpoint(model, best_dir, model_cfg, tokenizer)
            save_metrics = {k: v for k, v in val_metrics.items() if k != "classification_report"}
            save_metrics["epoch"] = epoch
            save_metrics["train_loss"] = avg_train_loss
            with open(os.path.join(best_dir, "best_metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=2)
            if logger:
                logger.info(f"  ** New best macro-F1: {best_macro_f1:.4f} (epoch {epoch}) **")
        else:
            patience_counter += 1
            if logger:
                logger.info(f"  No improvement ({patience_counter}/{patience}). Best: {best_macro_f1:.4f}")
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

    return best_macro_f1


def make_val_loader(val_ds, args, tokenizer) -> DataLoader:
    """Unshuffled loader with twice the training batch size."""
    return DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=make_collate_fn(pad_token_id=tokenizer.pad_token_id),
    )


def _fallback_architecture(args, default_arch: str) -> str:
    if getattr(args, "use_mmf2f", False):
        return ARCH_MMF2F
    if getattr(args, "use_paper_fusion", False):
        return ARCH_PAPER_FUSION
    return default_arch


# Command-line arguments that stand in for config.json keys when a checkpoint has no config.
_CONFIG_ARGS = (
    "roberta_name", "hubert_name", "gpt2_name", "fusion_layers", "fusion_heads", "fusion_ffn",
    "fusion_dropout", "head_dropout", "audio_stride", "text_only", "audio_only",
    "paper_fusion_option", "paper_audio_proj_dim", "mmf2f_hidden_dim", "mmf2f_rank",
)


def run_eval_only(args, logger: logging.Logger, default_arch: str = ARCH_JOINT) -> Dict:
    """Evaluate ``args.checkpoint`` on ``args.val_file``.

    Architecture, phase and label names come from the checkpoint's
    ``config.json``; without one they come from the command-line arguments
    (``default_arch`` when no architecture flag is set). The metrics table is
    logged and printed, and the metrics are written to ``args.eval_output`` or
    ``<output_dir>/eval_metrics.json``.
    """
    if args.checkpoint is None:
        raise ValueError("--eval_only requires --checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    sd_path, cfg = resolve_checkpoint(args.checkpoint)
    logger.info(f"Loading checkpoint: {sd_path}")
    if cfg is not None:
        logger.info("Found config.json — using saved architecture / phase.")
    else:
        logger.warning(
            "No config.json beside checkpoint — falling back to CLI args. "
            "Make sure --phase and the model arguments match training."
        )
        cfg = {"architecture": _fallback_architecture(args, default_arch)}

    phase = cfg.get("phase", args.phase)
    num_classes, label_names = phase_labels(phase)
    if "label_names" in cfg:
        label_names = cfg["label_names"]
        num_classes = cfg.get("num_classes", num_classes)
    logger.info(f"Phase: {phase} ({num_classes} classes)")

    arch = architecture_of(cfg)
    defaults = {key: getattr(args, key) for key in _CONFIG_ARGS if hasattr(args, key)}
    defaults["num_classes"] = num_classes
    text_model_name = defaults.get(
        "roberta_name" if arch in (ARCH_JOINT, ARCH_PAPER_FUSION) else "gpt2_name"
    )
    text_model_name = cfg.get(
        "roberta_name" if arch in (ARCH_JOINT, ARCH_PAPER_FUSION) else "gpt2_name", text_model_name
    )

    tok_dir = find_tokenizer_dir(args.checkpoint, extra=[os.path.join(args.output_dir, "tokenizer")])
    if tok_dir is not None:
        logger.info(f"Loading tokenizer from {tok_dir}")
        tokenizer = load_tokenizer(arch, tok_dir)
    else:
        logger.warning(
            f"No saved tokenizer found — re-creating from {text_model_name} with BC special tokens."
        )
        tokenizer = build_tokenizer(arch, text_model_name)

    logger.info(f"Loading val set: {args.val_file}")
    val_ds = BackchannelJointDataset(
        args.val_file, tokenizer, args.max_length,
        phase=phase, max_audio_seconds=args.max_audio_seconds,
        exclude_repetition=cfg.get("exclude_repetition", getattr(args, "exclude_repetition", False)),
        audio_root=getattr(args, "audio_root", None),
    )
    logger.info(f"Val: {len(val_ds)} examples")
    val_loader = make_val_loader(val_ds, args, tokenizer)

    logger.info(f"Building {ARCH_CLASS_NAMES[arch]} (num_classes={num_classes})")
    model = build_model_from_config(cfg, tokenizer_len=len(tokenizer), defaults=defaults).to(device)
    load_weights(model, sd_path, device, strict=getattr(args, "strict", False), warn=logger.warning)
    logger.info("Weights loaded.")

    logger.info("Running evaluation...")
    metrics = evaluate(model, val_loader, device, num_classes, label_names)
    table = format_eval_table(metrics, label_names)
    logger.info("\n" + table)
    print("\n" + table, flush=True)

    out_path = args.eval_output or os.path.join(args.output_dir, "eval_metrics.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    save_metrics["checkpoint"] = sd_path
    save_metrics["val_file"] = args.val_file
    save_metrics["phase"] = phase
    save_metrics["label_names"] = label_names
    with open(out_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    logger.info(f"Wrote eval metrics to {out_path}")
    return metrics
