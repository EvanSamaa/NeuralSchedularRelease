"""Train the text-only RoBERTa backchannel classifier.

    python -m neural_scheduler.backchannel.train_text_only \
        --train_file train.jsonl --val_file dev.jsonl \
        --output_dir runs/text_merged --phase merged

A ``RobertaForSequenceClassification`` over the same text format as the joint
classifier, without audio. ``--init_checkpoint`` copies the encoder of an
earlier run (for example a ``--phase binary`` warm-up) and re-initialises the
classification head.

Outputs in ``--output_dir``: ``best_model/`` and ``last_model/`` in Hugging
Face format (``best_metrics.json`` in ``best_model``), ``tokenizer/``,
``args.json``, ``train.log`` and ``tensorboard/``. Load the result with
:class:`neural_scheduler.backchannel.predictor.TextOnlyBackchannelClassifier`.
"""

import argparse
import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from .constants import PHASES, phase_labels
from .data.dataset import BackchannelTextDataset, compute_class_weights
from .metrics import evaluate_text_only
from .models.factory import add_bc_tokens
from .training import make_tensorboard_writer, set_seed, setup_run

NO_DECAY_NAME_PARTS = ("bias", "LayerNorm.weight")


def load_model_and_tokenizer(model_name: str, num_labels: int, init_checkpoint: Optional[str] = None):
    """Return ``(model, tokenizer)`` with the backchannel tokens added.

    With ``init_checkpoint`` the encoder weights (and vocabulary size) come
    from that checkpoint and the classification head is new.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    add_bc_tokens(tokenizer)

    if init_checkpoint and os.path.exists(init_checkpoint):
        logging.info(f"Loading encoder from checkpoint: {init_checkpoint}")
        checkpoint_model = RobertaForSequenceClassification.from_pretrained(init_checkpoint)
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
        config.vocab_size = checkpoint_model.config.vocab_size
        model = RobertaForSequenceClassification(config)
        model.roberta.load_state_dict(checkpoint_model.roberta.state_dict())
        logging.info(f"Loaded encoder weights, reinitialized classifier head for {num_labels} classes")
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def train(
    model, train_loader, val_loader, optimizer, scheduler, criterion, device,
    num_classes, label_names, output_dir,
    epochs=10, patience=3, grad_clip=1.0, log_interval=50, logger=None, writer=None,
) -> float:
    """Train with early stopping on validation macro-F1; returns the best macro-F1."""
    best_macro_f1 = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(outputs.logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                    f"loss={epoch_loss / num_batches:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / max(num_batches, 1)
        if logger:
            logger.info(f"Epoch {epoch} — evaluating...")
        val_metrics = evaluate_text_only(model, val_loader, device, num_classes, label_names)

        if logger:
            logger.info(
                f"Epoch {epoch} — "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f} | "
                f"bc_f1={val_metrics['bc_f1']:.4f} "
                f"(P={val_metrics['bc_precision']:.3f} R={val_metrics['bc_recall']:.3f})"
            )
            logger.info(f"\n{val_metrics['classification_report']}")
        if writer:
            for key in ("loss", "accuracy", "macro_f1", "weighted_f1", "bc_f1", "bc_precision", "bc_recall"):
                writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

        last_dir = os.path.join(output_dir, "last_model")
        model.save_pretrained(last_dir)
        if logger:
            logger.info(f"  Saved last checkpoint to {last_dir}")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            best_dir = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_dir)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the text-only RoBERTa backchannel classifier")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Checkpoint whose encoder initialises this run.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--phase", type=str, default="full", choices=list(PHASES),
                        help="binary: 2 classes; full: 7 classes; merged: 3 classes "
                             "(Repetition examples are dropped).")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--class_weight_method", type=str, default="sqrt_inv",
                        choices=["sqrt_inv", "inv", "uniform"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    logger = setup_run(args, "train.log", "train_bc")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    num_classes, label_names = phase_labels(args.phase)
    logger.info(f"Phase: {args.phase} ({num_classes} classes)")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name, num_labels=num_classes, init_checkpoint=args.init_checkpoint
    )
    model.to(device)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    logger.info("Loading datasets...")
    train_dataset = BackchannelTextDataset(args.train_file, tokenizer, args.max_length, phase=args.phase)
    val_dataset = BackchannelTextDataset(args.val_file, tokenizer, args.max_length, phase=args.phase)
    logger.info(f"Train: {len(train_dataset)} examples")
    logger.info(f"Val:   {len(val_dataset)} examples")

    train_label_counts = train_dataset.get_label_counts()
    logger.info(f"Train label distribution: {dict(train_label_counts)}")
    class_weights = compute_class_weights(train_label_counts, num_classes, method=args.class_weight_method)
    logger.info(f"Class weights ({args.class_weight_method}): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in named if not any(nd in n for nd in NO_DECAY_NAME_PARTS)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in named if any(nd in n for nd in NO_DECAY_NAME_PARTS)],
             "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    writer = make_tensorboard_writer(args.output_dir)
    logger.info("Starting training...")
    best_f1 = train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device,
        num_classes=num_classes, label_names=label_names, output_dir=args.output_dir,
        epochs=args.epochs, patience=args.patience, grad_clip=args.grad_clip,
        log_interval=args.log_interval, logger=logger, writer=writer,
    )
    if writer:
        writer.close()

    logger.info(f"Training complete. Best macro-F1: {best_f1:.4f}")
    logger.info(f"Best model: {os.path.join(args.output_dir, 'best_model')}")
    logger.info(f"Last model: {os.path.join(args.output_dir, 'last_model')}")


if __name__ == "__main__":
    main()
