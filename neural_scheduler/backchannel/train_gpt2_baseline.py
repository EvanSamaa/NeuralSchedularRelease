"""Train the GPT-2 + HuBERT late-fusion baseline (Wang et al., ICASSP 2024).

    # stage 1: fine-tune GPT-2 on text alone
    python -m neural_scheduler.backchannel.train_gpt2_baseline \
        --train_file train.jsonl --val_file dev.jsonl \
        --output_dir runs/gpt2_stage1 --text_only

    # stage 2 (opt2): freeze that text encoder, train the fusion layer
    python -m neural_scheduler.backchannel.train_gpt2_baseline \
        --train_file train.jsonl --val_file dev.jsonl \
        --output_dir runs/gpt2_stage2 --paper_fusion_option opt2 \
        --text_model_checkpoint runs/gpt2_stage1/best_model

``--paper_fusion_option opt1`` trains GPT-2 and the fusion layer together in
one stage. The tokenizer gets a dedicated pad token and truncates on the left,
because the classifier reads the last non-pad token. Outputs follow
:mod:`neural_scheduler.backchannel.train`.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import PHASES, phase_labels
from .data.dataset import (
    BackchannelJointDataset,
    build_balanced_sampler,
    compute_class_weights,
    make_collate_fn,
)
from .models import GPT2PaperFusionClassifier
from .models.factory import ARCH_PAPER_FUSION_GPT2, build_tokenizer, resolve_checkpoint
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
    p = argparse.ArgumentParser(description="Train the GPT-2 + HuBERT late-fusion baseline")

    p.add_argument("--train_file", type=str, default=None,
                   help="Required for training; ignored with --eval_only.")
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--audio_root", type=str, default=None,
                   help="Directory that relative audio_span paths are resolved against.")

    p.add_argument("--gpt2_name", type=str, default="gpt2")
    p.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)

    p.add_argument("--text_only", action="store_true",
                   help="Disable the audio branch; stage 1 of the opt2 recipe.")
    p.add_argument("--audio_only", action="store_true", help="Disable the text branch.")

    p.add_argument("--paper_fusion_option", type=str, default="opt2", choices=["opt1", "opt2"],
                   help="opt1: text encoder trained jointly with the fusion layer. "
                        "opt2: text encoder loaded from --text_model_checkpoint and frozen.")
    p.add_argument("--paper_audio_proj_dim", type=int, default=256)
    p.add_argument("--text_model_checkpoint", type=str, default=None,
                   help="Checkpoint of a --text_only run; required for opt2.")

    p.add_argument("--phase", type=str, default="merged", choices=list(PHASES))
    p.add_argument("--binary", action="store_true", help="Shorthand for --phase binary.")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--class_weight_method", type=str, default=None,
                   choices=["sqrt_inv", "inv", "uniform"],
                   help="Loss class weighting. Default: uniform with --balanced_sampling, "
                        "sqrt_inv otherwise.")
    p.add_argument("--balanced_sampling", action="store_true",
                   help="Rebalance class frequencies of the training set with a weighted sampler.")
    p.add_argument("--balanced_mode", type=str, default="uniform", choices=["uniform", "sqrt"],
                   help="uniform: equal probability per class; sqrt: weight 1/sqrt(class count).")

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
    args = parse_args_with_config(build_parser(), argv)
    if args.binary:
        args.phase = "binary"
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    logger = setup_run(args, "eval.log" if args.eval_only else "train.log", "train_joint")

    if args.eval_only:
        run_eval_only(args, logger, default_arch=ARCH_PAPER_FUSION_GPT2)
        return
    if args.train_file is None:
        raise ValueError("--train_file is required for training (omit only with --eval_only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    num_classes, label_names = phase_labels(args.phase)
    logger.info(f"Phase: {args.phase} ({num_classes} classes)")

    tokenizer = build_tokenizer(ARCH_PAPER_FUSION_GPT2, args.gpt2_name)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    logger.info("Loading datasets...")
    dataset_kwargs = dict(
        phase=args.phase, max_audio_seconds=args.max_audio_seconds, audio_root=args.audio_root
    )
    train_ds = BackchannelJointDataset(args.train_file, tokenizer, args.max_length, **dataset_kwargs)
    val_ds = BackchannelJointDataset(args.val_file, tokenizer, args.max_length, **dataset_kwargs)
    logger.info(f"Train: {len(train_ds)} examples")
    logger.info(f"Val:   {len(val_ds)} examples")

    train_counts = train_ds.get_label_counts()
    logger.info(f"Train label distribution: {dict(train_counts)}")

    cw_method = args.class_weight_method
    if cw_method is None:
        cw_method = "uniform" if args.balanced_sampling else "sqrt_inv"
    if args.balanced_sampling and cw_method != "uniform":
        logger.warning(
            f"Balanced sampling is ON and class_weight_method='{cw_method}' — this stacks two "
            "corrections on top of each other. Pass --class_weight_method uniform to disable "
            "the loss reweighting."
        )
    class_weights = compute_class_weights(train_counts, num_classes, method=cw_method)
    logger.info(f"Class weights ({cw_method}): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    sampler = None
    if args.balanced_sampling:
        sampler = build_balanced_sampler(train_ds.get_labels(), num_classes, mode=args.balanced_mode)
        logger.info(f"Balanced sampling: ON (mode={args.balanced_mode}); validation set is left as-is.")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=make_collate_fn(pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = make_val_loader(val_ds, args, tokenizer)

    model_cfg = {
        "gpt2_name": args.gpt2_name,
        "hubert_name": args.hubert_name,
        "num_classes": num_classes,
        "tokenizer_len": len(tokenizer),
        "text_only": args.text_only,
        "audio_only": args.audio_only,
        "phase": args.phase,
        "label_names": label_names,
        "paper_fusion_option": args.paper_fusion_option,
        "paper_audio_proj_dim": args.paper_audio_proj_dim,
        "architecture": ARCH_PAPER_FUSION_GPT2,
    }

    # A text-only run is stage 1 of opt2 and must train the text encoder.
    freeze_text = args.paper_fusion_option == "opt2" and not args.text_only
    if freeze_text and args.text_model_checkpoint is None:
        raise ValueError(
            "--paper_fusion_option opt2 requires --text_model_checkpoint (a stage-1 checkpoint "
            "from a run with --text_only). Use --paper_fusion_option opt1 to train in one stage."
        )
    logger.info(
        f"Building GPT2PaperFusionClassifier (text=GPT-2 [{args.gpt2_name}], "
        f"option={args.paper_fusion_option}, freeze_text={freeze_text}, "
        f"audio_proj_dim={args.paper_audio_proj_dim})"
    )
    model = GPT2PaperFusionClassifier(
        gpt2_name=args.gpt2_name,
        hubert_name=args.hubert_name,
        num_classes=num_classes,
        audio_proj_dim=args.paper_audio_proj_dim,
        tokenizer_len=len(tokenizer),
        fusion_option=args.paper_fusion_option,
        freeze_text=freeze_text,
        text_only=args.text_only,
        audio_only=args.audio_only,
    ).to(device)
    if freeze_text:
        sd_path, _ = resolve_checkpoint(args.text_model_checkpoint)
        logger.info(f"Loading stage-1 GPT-2 text encoder from {sd_path}")
        n_copied, n_skipped = model.load_text_encoder_from_checkpoint(sd_path)
        logger.info(f"  Copied {n_copied} gpt2.* tensors (skipped {n_skipped} non-matching keys)")
    log_model_size(model, ARCH_PAPER_FUSION_GPT2, args, logger)

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
