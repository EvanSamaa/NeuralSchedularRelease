"""Train the backchannel classifier.

    python -m neural_scheduler.backchannel.train \
        --train_file train.jsonl --val_file dev.jsonl \
        --output_dir runs/joint_merged --phase merged

Architectures:

* default – :class:`~neural_scheduler.backchannel.models.JointBCClassifier`
  (RoBERTa + frozen HuBERT + fusion transformer); the one the pipeline uses.
* ``--use_paper_fusion`` – late-fusion baseline with RoBERTa. Recipe ``opt2``
  needs a stage-1 run (``--use_paper_fusion --text_only``) whose checkpoint is
  passed as ``--text_model_checkpoint``; ``opt1`` trains in one stage.
* ``--use_mmf2f`` – GPT-2 + HuBERT with low-rank multimodal fusion.

(The late-fusion baseline with GPT-2 has its own entry point,
:mod:`neural_scheduler.backchannel.train_gpt2_baseline`.)

Phases: ``merged`` (NoBackchannel / Acknowledgment / Reactive; drops
Repetition), ``binary`` (backchannel or not), ``full`` (seven classes).

Input files are the JSONL written by :mod:`neural_scheduler.backchannel.data.prep`.
Outputs in ``--output_dir``: ``best_model/`` and ``last_model/`` (``model.pt``,
``config.json``, ``tokenizer/``; ``best_metrics.json`` in ``best_model``),
``tokenizer/``, ``args.json``, ``train.log`` and ``tensorboard/``.

``--eval_only --checkpoint <dir>`` evaluates instead of training (same as
:mod:`neural_scheduler.backchannel.eval`). ``--config file.yaml`` supplies
argument values, e.g. ``configs/backchannel/joint_merged.yaml``.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import PHASES, phase_labels
from .data.dataset import (
    BackchannelJointDataset,
    compute_class_weights,
    make_balanced_sampler,
    make_collate_fn,
)
from .models import JointBCClassifier, MMF2FModel, PaperFusionClassifier
from .models.factory import (
    ARCH_JOINT,
    ARCH_MMF2F,
    ARCH_PAPER_FUSION,
    build_tokenizer,
    resolve_checkpoint,
)
from .training import (
    build_optimizer_and_scheduler,
    log_model_size,
    make_tensorboard_writer,
    make_val_loader,
    parse_args_with_config,
    run_eval_only,
    set_seed,
    setup_run,
    train_loop,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the joint text+audio backchannel classifier")

    p.add_argument("--train_file", type=str, default=None,
                   help="Required for training; ignored with --eval_only.")
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--audio_root", type=str, default=None,
                   help="Directory that relative audio_span paths are resolved against.")

    p.add_argument("--roberta_name", type=str, default="roberta-base")
    p.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    p.add_argument("--audio_stride", type=int, default=4,
                   help="Stride of the audio downsampling convolution (4: 50 Hz -> 12.5 Hz)")

    p.add_argument("--fusion_layers", type=int, default=2)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--fusion_ffn", type=int, default=3072)
    p.add_argument("--fusion_dropout", type=float, default=0.1)
    p.add_argument("--head_dropout", type=float, default=0.1)

    p.add_argument("--text_only", action="store_true", help="Disable the audio branch.")
    p.add_argument("--audio_only", action="store_true", help="Disable the text branch.")

    p.add_argument("--use_paper_fusion", action="store_true",
                   help="Late-fusion baseline (Wang et al., ICASSP 2024) with RoBERTa.")
    p.add_argument("--paper_fusion_option", type=str, default="opt2", choices=["opt1", "opt2"],
                   help="opt1: text encoder trained jointly with the fusion layer. "
                        "opt2: text encoder loaded from --text_model_checkpoint and frozen.")
    p.add_argument("--paper_audio_proj_dim", type=int, default=256)
    p.add_argument("--text_model_checkpoint", type=str, default=None,
                   help="Checkpoint of a --use_paper_fusion --text_only run; required for opt2.")

    p.add_argument("--use_mmf2f", action="store_true",
                   help="GPT-2 + HuBERT with low-rank multimodal fusion (Liu et al., 2018). "
                        "Uses --gpt2_name instead of --roberta_name.")
    p.add_argument("--gpt2_name", type=str, default="openai-community/gpt2")
    p.add_argument("--mmf2f_hidden_dim", type=int, default=256)
    p.add_argument("--mmf2f_rank", type=int, default=16)
    p.add_argument("--mmf2f_drop_rate", type=float, default=0.0,
                   help="Probability of dropping one modality for a training batch.")
    p.add_argument("--mmf2f_text_ckpt", type=str, default=None,
                   help="Unimodal text checkpoint to warm-start GPT-2 from.")
    p.add_argument("--mmf2f_audio_ckpt", type=str, default=None,
                   help="Unimodal audio checkpoint to warm-start HuBERT from.")

    p.add_argument("--phase", type=str, default="merged", choices=list(PHASES))
    p.add_argument("--exclude_repetition", action="store_true",
                   help="With --phase binary, drop Repetition examples.")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--class_weight_method", type=str, default="sqrt_inv",
                   choices=["sqrt_inv", "inv", "uniform"])
    p.add_argument("--balance_nobc", action="store_true",
                   help="Sample NoBackchannel as often as all backchannel classes together "
                        "(2 * N_bc draws per epoch, with replacement).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=50)

    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; evaluate --checkpoint on --val_file.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint directory (model.pt + config.json) or a model.pt file.")
    p.add_argument("--eval_output", type=str, default=None,
                   help="Where to write eval metrics (default <output_dir>/eval_metrics.json).")
    p.add_argument("--strict", action="store_true",
                   help="With --eval_only, fail on missing or unexpected checkpoint keys.")
    return p


def parse_args(argv=None):
    return parse_args_with_config(build_parser(), argv)


def architecture_from_args(args) -> str:
    if args.use_mmf2f and args.use_paper_fusion:
        raise ValueError("--use_mmf2f and --use_paper_fusion are mutually exclusive.")
    if args.use_mmf2f:
        return ARCH_MMF2F
    if args.use_paper_fusion:
        return ARCH_PAPER_FUSION
    return ARCH_JOINT


def build_model(args, arch: str, num_classes: int, tokenizer_len: int, logger):
    """Construct the model selected on the command line, including warm starts."""
    if arch == ARCH_MMF2F:
        model = MMF2FModel(
            gpt2_name=args.gpt2_name,
            hubert_name=args.hubert_name,
            num_classes=num_classes,
            hidden_dim=args.mmf2f_hidden_dim,
            lmf_rank=args.mmf2f_rank,
            tokenizer_len=tokenizer_len,
            text_only=args.text_only,
            audio_only=args.audio_only,
            drop_rate=args.mmf2f_drop_rate,
        )
        if args.mmf2f_text_ckpt or args.mmf2f_audio_ckpt:
            n_t, n_a = model.load_unimodal_checkpoints(
                text_ckpt=args.mmf2f_text_ckpt, audio_ckpt=args.mmf2f_audio_ckpt
            )
            logger.info(f"  Warm-started from unimodal checkpoints: {n_t} text tensors, "
                        f"{n_a} audio tensors copied.")
        return model

    if arch == ARCH_PAPER_FUSION:
        # A text-only run is stage 1 of opt2 and must train the text encoder.
        freeze_text = args.paper_fusion_option == "opt2" and not args.text_only
        if freeze_text and args.text_model_checkpoint is None:
            raise ValueError(
                "--paper_fusion_option opt2 requires --text_model_checkpoint (a stage-1 "
                "checkpoint from a run with --use_paper_fusion --text_only). Use "
                "--paper_fusion_option opt1 to train in one stage."
            )
        logger.info(
            f"Building PaperFusionClassifier (option={args.paper_fusion_option}, "
            f"freeze_text={freeze_text}, audio_proj_dim={args.paper_audio_proj_dim})"
        )
        model = PaperFusionClassifier(
            roberta_name=args.roberta_name,
            hubert_name=args.hubert_name,
            num_classes=num_classes,
            audio_proj_dim=args.paper_audio_proj_dim,
            tokenizer_len=tokenizer_len,
            fusion_option=args.paper_fusion_option,
            freeze_text=freeze_text,
            text_only=args.text_only,
            audio_only=args.audio_only,
        )
        if freeze_text:
            sd_path, _ = resolve_checkpoint(args.text_model_checkpoint)
            logger.info(f"Loading stage-1 text encoder from {sd_path}")
            n_copied, n_skipped = model.load_text_encoder_from_checkpoint(sd_path)
            logger.info(f"  Copied {n_copied} roberta.* tensors (skipped {n_skipped} non-matching keys)")
        return model

    return JointBCClassifier(
        roberta_name=args.roberta_name,
        hubert_name=args.hubert_name,
        num_classes=num_classes,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_ffn=args.fusion_ffn,
        fusion_dropout=args.fusion_dropout,
        head_dropout=args.head_dropout,
        audio_stride=args.audio_stride,
        tokenizer_len=tokenizer_len,
        text_only=args.text_only,
        audio_only=args.audio_only,
    )


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    logger = setup_run(args, "eval.log" if args.eval_only else "train.log", "train_joint")

    if args.eval_only:
        run_eval_only(args, logger)
        return
    if args.train_file is None:
        raise ValueError("--train_file is required for training (omit only with --eval_only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    num_classes, label_names = phase_labels(args.phase)
    logger.info(f"Phase: {args.phase} ({num_classes} classes)")
    arch = architecture_from_args(args)

    tokenizer = build_tokenizer(arch, args.gpt2_name if arch == ARCH_MMF2F else args.roberta_name)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    logger.info("Loading datasets...")
    dataset_kwargs = dict(
        phase=args.phase, max_audio_seconds=args.max_audio_seconds,
        exclude_repetition=args.exclude_repetition, audio_root=args.audio_root,
    )
    train_ds = BackchannelJointDataset(args.train_file, tokenizer, args.max_length, **dataset_kwargs)
    val_ds = BackchannelJointDataset(args.val_file, tokenizer, args.max_length, **dataset_kwargs)
    logger.info(f"Train: {len(train_ds)} examples")
    logger.info(f"Val:   {len(val_ds)} examples")

    train_counts = train_ds.get_label_counts()
    logger.info(f"Train label distribution: {dict(train_counts)}")
    class_weights = compute_class_weights(train_counts, num_classes, method=args.class_weight_method)
    logger.info(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_sampler = None
    if args.balance_nobc:
        train_sampler = make_balanced_sampler(train_ds, nobc_label=0)
        n_bc = sum(v for k, v in train_counts.items() if k != 0)
        logger.info(
            f"Balanced sampler enabled: {train_counts.get(0, 0)} NoBC → downsampled to match "
            f"{n_bc} BC samples ({train_sampler.num_samples} draws/epoch, with replacement)."
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=make_collate_fn(pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = make_val_loader(val_ds, args, tokenizer)

    model_cfg = {
        "roberta_name": args.roberta_name,
        "hubert_name": args.hubert_name,
        "num_classes": num_classes,
        "fusion_layers": args.fusion_layers,
        "fusion_heads": args.fusion_heads,
        "fusion_ffn": args.fusion_ffn,
        "fusion_dropout": args.fusion_dropout,
        "head_dropout": args.head_dropout,
        "audio_stride": args.audio_stride,
        "tokenizer_len": len(tokenizer),
        "text_only": args.text_only,
        "audio_only": args.audio_only,
        "phase": args.phase,
        "label_names": label_names,
        "use_paper_fusion": args.use_paper_fusion,
        "paper_fusion_option": args.paper_fusion_option,
        "paper_audio_proj_dim": args.paper_audio_proj_dim,
        "use_mmf2f": args.use_mmf2f,
        "gpt2_name": args.gpt2_name,
        "mmf2f_hidden_dim": args.mmf2f_hidden_dim,
        "mmf2f_rank": args.mmf2f_rank,
        "mmf2f_drop_rate": args.mmf2f_drop_rate,
        "exclude_repetition": args.exclude_repetition,
        "architecture": arch,
    }

    model = build_model(args, arch, num_classes, len(tokenizer), logger).to(device)
    log_model_size(model, arch, args, logger)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.lr, args.weight_decay, args.warmup_ratio,
        total_steps=len(train_loader) * args.epochs, logger=logger,
    )
    writer = make_tensorboard_writer(args.output_dir)

    logger.info("Starting training...")
    best_f1 = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, num_classes=num_classes, label_names=label_names,
        output_dir=args.output_dir, model_cfg=model_cfg,
        epochs=args.epochs, patience=args.patience, grad_clip=args.grad_clip,
        log_interval=args.log_interval, logger=logger, writer=writer, tokenizer=tokenizer,
    )
    if writer:
        writer.close()

    logger.info(f"Training complete. Best macro-F1: {best_f1:.4f}")
    logger.info(f"Best model: {os.path.join(args.output_dir, 'best_model')}")


if __name__ == "__main__":
    main()
