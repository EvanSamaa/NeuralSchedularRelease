"""Inference-time ablations of a trained backchannel classifier.

    python -m neural_scheduler.backchannel.ablation \
        --checkpoint runs/joint_merged/best_model \
        --val_file test.jsonl \
        --output_json runs/joint_merged/ablation_results.json \
        --num_samples 11000

The model is loaded once and evaluated on perturbed copies of the same
examples, so differences between ablations are paired:

* ``full``            – unmodified input.
* ``no_audio``        – waveform replaced by a single silent sample.
* ``no_future_text``  – lines after the target line removed.
* ``no_current_text`` – target line reduced to ``<speaker>: <bc_slot>``.
* ``no_past_text``    – lines before the target line removed.
* ``no_past_bc``      – every ``[BC:*]`` tag removed.
* ``audio_only``      – whole text replaced by ``A: <bc_slot>``.

All seven run by default. Both modality branches of the model stay enabled
whatever the checkpoint's ``text_only`` / ``audio_only`` setting; ablation is
done on the inputs. Supported architectures: the joint classifier and the
late-fusion baselines.

The output JSON holds, per ablation, the metrics of
:func:`neural_scheduler.backchannel.metrics.evaluate_full`, and a ``summary``
block that is also printed as tables.
"""

import argparse
import json
import logging
import math
import os
import random
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader

from .constants import BC_SLOT_TOKEN, BC_TAG_TOKENS, LINE_SEP
from .data.dataset import BackchannelJointDataset, make_collate_fn
from .metrics import evaluate_full
from .models.factory import (
    ARCH_CLASS_NAMES,
    ARCH_MMF2F,
    architecture_of,
    build_model_from_config,
    build_tokenizer,
    find_tokenizer_dir,
    load_tokenizer,
    load_weights,
    resolve_checkpoint,
)
from .training import set_seed

ALL_ABLATIONS = [
    'full',
    'no_audio',
    'no_future_text',
    'no_current_text',
    'no_past_text',
    'no_past_bc',
    'audio_only',
]

DEFAULT_ABLATIONS = [
    'full',
    'audio_only',
    'no_audio',
    'no_future_text',
    'no_current_text',
    'no_past_text',
    'no_past_bc',
]


def _find_target_line_idx(lines: List[str]) -> int:
    """Index of the line containing ``<bc_slot>``, or -1."""
    for i, ln in enumerate(lines):
        if BC_SLOT_TOKEN in ln:
            return i
    return -1


def _strip_bc_tags(text: str) -> str:
    out = text
    for tag in BC_TAG_TOKENS:
        out = out.replace(' ' + tag, '').replace(tag, '')
    return out


def perturb_text(input_text: str, ablation: str) -> str:
    """Apply a text-side ablation. ``full`` and ``no_audio`` leave the text unchanged."""
    if ablation in ('full', 'no_audio'):
        return input_text

    lines = input_text.split(LINE_SEP)
    tgt = _find_target_line_idx(lines)

    if ablation == 'audio_only':
        return f"A: {BC_SLOT_TOKEN}"
    if ablation == 'no_past_bc':
        return LINE_SEP.join(_strip_bc_tags(ln) for ln in lines)
    if tgt < 0:
        return input_text
    if ablation == 'no_future_text':
        return LINE_SEP.join(lines[:tgt + 1])
    if ablation == 'no_past_text':
        return LINE_SEP.join(lines[tgt:])
    if ablation == 'no_current_text':
        target = lines[tgt]
        speaker = target.split(':', 1)[0].strip() if ':' in target else 'A'
        lines[tgt] = f"{speaker}: {BC_SLOT_TOKEN}"
        return LINE_SEP.join(lines)
    raise ValueError(f"Unknown ablation: {ablation}")


def make_ablation_collate(base_collate: Callable, ablation: str) -> Callable:
    """For ``no_audio``, zero the batched waveform and set every length to one
    sample (the shortest input HuBERT's length formula accepts)."""
    if ablation != 'no_audio':
        return base_collate

    def _collate(batch):
        out = base_collate(batch)
        out['audio'] = torch.zeros_like(out['audio'])
        out['audio_lens'] = torch.ones_like(out['audio_lens'])
        return out

    return _collate


class TextPerturbingDataset(torch.utils.data.Dataset):
    """View of a :class:`BackchannelJointDataset` with a text ablation applied.

    The perturbed texts are computed once; items are produced by the base
    dataset so tokenisation and audio loading stay in one place.
    """

    def __init__(self, base: BackchannelJointDataset, ablation: str):
        self.base = base
        self.ablation = ablation
        self._examples_orig = base.examples
        self._examples_perturbed = [
            {**ex, 'input_text': perturb_text(ex['input_text'], ablation)} for ex in base.examples
        ]

    def __len__(self):
        return len(self._examples_perturbed)

    def __getitem__(self, idx):
        self.base.examples = self._examples_perturbed
        try:
            return self.base[idx]
        finally:
            self.base.examples = self._examples_orig


def stratified_subset_indices(
    dataset: BackchannelJointDataset,
    n_samples: int,
    seed: int = 42,
) -> List[int]:
    """Pick `n_samples` indices, stratified by label where possible.

    If n_samples >= len(dataset), returns all indices unshuffled.
    """
    if n_samples is None or n_samples >= len(dataset):
        return list(range(len(dataset)))

    rng = random.Random(seed)
    by_label = {}
    for i, ex in enumerate(dataset.examples):
        lbl = dataset._label_for(ex)
        by_label.setdefault(lbl, []).append(i)

    # Shuffle each bucket
    for v in by_label.values():
        rng.shuffle(v)

    # Allocate proportionally, with at least 1 per non-empty class
    total = len(dataset)
    chosen: List[int] = []
    for lbl, idxs in by_label.items():
        share = max(1, round(n_samples * len(idxs) / total))
        chosen.extend(idxs[:share])

    # If we overshot (rounding), trim; if we undershot, top up randomly
    if len(chosen) > n_samples:
        rng.shuffle(chosen)
        chosen = chosen[:n_samples]
    elif len(chosen) < n_samples:
        leftover = list(set(range(total)) - set(chosen))
        rng.shuffle(leftover)
        chosen.extend(leftover[: n_samples - len(chosen)])

    rng.shuffle(chosen)
    return chosen


def build_model_and_load(checkpoint_path: str, tokenizer, device: torch.device, logger: logging.Logger):
    """Load the model of a checkpoint with both modality branches enabled.

    Returns ``(model, num_classes, label_names, phase, cfg)``.
    """
    sd_path, cfg = resolve_checkpoint(checkpoint_path)
    if cfg is None:
        raise RuntimeError(
            f"No config.json found beside checkpoint {checkpoint_path}. It must give at "
            "least phase, num_classes, label_names and the architecture."
        )
    arch = architecture_of(cfg)
    if arch == ARCH_MMF2F:
        raise ValueError("Ablations are not supported for the MMF2F baseline.")

    phase = cfg['phase']
    num_classes = cfg['num_classes']
    label_names = cfg['label_names']
    logger.info(f"Loading {ARCH_CLASS_NAMES[arch]} (phase={phase}, num_classes={num_classes})")

    model = build_model_from_config(
        cfg, tokenizer_len=len(tokenizer), text_only=False, audio_only=False
    ).to(device)
    load_weights(model, sd_path, device, warn=logger.warning)
    logger.info("Weights loaded.")
    return model, num_classes, label_names, phase, cfg


def build_tokenizer_for_checkpoint(checkpoint_path: str, fallback_name: str, logger):
    """Tokenizer saved with the checkpoint, or a fresh one built from ``fallback_name``."""
    _, cfg = resolve_checkpoint(checkpoint_path)
    arch = architecture_of(cfg or {})
    tok_dir = find_tokenizer_dir(checkpoint_path)
    if tok_dir is not None:
        logger.info(f"Loading tokenizer from {tok_dir}")
        return load_tokenizer(arch, tok_dir)
    logger.warning(f"No saved tokenizer found — re-creating from {fallback_name} with BC special tokens.")
    return build_tokenizer(arch, fallback_name)


def run_ablations(args, logger) -> Dict:
    """Run the requested ablations and write ``args.output_json``."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    ablations = args.ablations or DEFAULT_ABLATIONS
    for ab in ablations:
        if ab not in ALL_ABLATIONS:
            raise ValueError(f"Unknown ablation '{ab}'. Choose from: {ALL_ABLATIONS}")

    tokenizer = build_tokenizer_for_checkpoint(args.checkpoint, args.roberta_name, logger)
    model, num_classes, label_names, phase, cfg = build_model_and_load(
        args.checkpoint, tokenizer, device, logger,
    )
    model.eval()

    base_ds = BackchannelJointDataset(
        args.val_file, tokenizer, args.max_length,
        phase=phase, max_audio_seconds=cfg.get('max_audio_seconds', args.max_audio_seconds),
        audio_root=args.audio_root,
    )
    logger.info(f"Loaded validation set: {len(base_ds)} examples")

    if args.num_samples is not None and args.num_samples < len(base_ds):
        idxs = stratified_subset_indices(base_ds, args.num_samples, seed=args.seed)
        logger.info(f"Subsetting to {len(idxs)} examples (stratified by label, seed={args.seed}).")
        base_ds.examples = [base_ds.examples[i] for i in idxs]

    logger.info(f"Effective evaluation set size: {len(base_ds)}")
    logger.info(f"Label distribution: {dict(base_ds.get_label_counts())}")

    base_collate = make_collate_fn(pad_token_id=tokenizer.pad_token_id)
    results: Dict[str, Dict] = {}

    for ablation in ablations:
        logger.info('=' * 78)
        logger.info(f"Running ablation: {ablation}")
        logger.info('=' * 78)

        loader = DataLoader(
            TextPerturbingDataset(base_ds, ablation),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=make_ablation_collate(base_collate, ablation),
        )
        metrics = evaluate_full(model, loader, device, num_classes, label_names)

        metrics_to_save = {k: v for k, v in metrics.items() if k != 'classification_report'}
        metrics_to_save['classification_report_text'] = metrics['classification_report']

        bcb = metrics['bc_binary']
        logger.info(
            f"[{ablation}] "
            f"acc={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"macro_auc={metrics['macro_auc']:.4f}  "
            f"macro_eer={metrics['macro_eer']:.4f}  "
            f"|  bc_binary: f1={bcb['f1']:.4f} "
            f"auc={bcb['auc']:.4f} eer={bcb['eer']:.4f}"
        )
        results[ablation] = metrics_to_save

    out = {
        'checkpoint': args.checkpoint,
        'val_file': args.val_file,
        'phase': phase,
        'num_classes': num_classes,
        'label_names': label_names,
        'num_samples_evaluated': len(base_ds),
        'num_samples_requested': args.num_samples,
        'seed': args.seed,
        'ablations': results,
    }
    out['summary'] = _build_summary_table(results, label_names)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote ablation results to {args.output_json}")

    print('\n' + _format_summary_string(out['summary'], label_names), flush=True)
    return out


def _build_summary_table(results: Dict[str, Dict], label_names: List[str]) -> Dict:
    """Compact summary: one row per ablation with the headline metrics.

    Includes both the multiclass headline and the binary BC-vs-NoBC headline
    for each ablation, plus per-class accuracy / F1 / AUC / EER.
    """
    summary = {}
    for ab, m in results.items():
        bcb = m.get('bc_binary', {})
        per_class = {}
        for i, name in enumerate(label_names):
            per_class[name] = {
                'accuracy': _safe_get(m.get('per_class_accuracy'), i),
                'precision': _safe_get(m.get('per_class_precision'), i),
                'recall': _safe_get(m.get('per_class_recall'), i),
                'f1': _safe_get(m.get('per_class_f1'), i),
                'auc': _safe_get(m.get('per_class_auc'), i),
                'eer': _safe_get(m.get('per_class_eer'), i),
                'support': _safe_get(m.get('per_class_support'), i),
            }
        summary[ab] = {
            'multiclass': {
                'loss': m.get('loss'),
                'accuracy': m.get('accuracy'),
                'macro_f1': m.get('macro_f1'),
                'weighted_f1': m.get('weighted_f1'),
                'macro_auc': m.get('macro_auc'),
                'macro_eer': m.get('macro_eer'),
            },
            'bc_binary': {
                'accuracy': bcb.get('accuracy'),
                'precision': bcb.get('precision'),
                'recall': bcb.get('recall'),
                'f1': bcb.get('f1'),
                'auc': bcb.get('auc'),
                'eer': bcb.get('eer'),
            },
            'per_class': per_class,
        }
    return summary


def _safe_get(arr, i, default=None):
    if arr is None:
        return default
    try:
        return arr[i]
    except (IndexError, KeyError, TypeError):
        return default


def _format_summary_string(summary: Dict, label_names: List[str]) -> str:
    rows = []

    def _fmt(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return '   n/a'
        return f"{x:.4f}"

    full = summary.get('full')

    # ----- Multiclass headline -----
    rows.append('=' * 110)
    rows.append('  ABLATION SUMMARY — MULTICLASS')
    rows.append('=' * 110)
    header = (
        f"  {'Ablation':<18} "
        f"{'Acc':>8} {'MacroF1':>9} {'WtdF1':>8} "
        f"{'MacroAUC':>9} {'MacroEER':>9} {'Loss':>8}"
    )
    rows.append(header)
    rows.append('-' * 110)
    for ab, blk in summary.items():
        m = blk['multiclass']
        row = (
            f"  {ab:<18} "
            f"{_fmt(m['accuracy']):>8} "
            f"{_fmt(m['macro_f1']):>9} "
            f"{_fmt(m['weighted_f1']):>8} "
            f"{_fmt(m['macro_auc']):>9} "
            f"{_fmt(m['macro_eer']):>9} "
            f"{_fmt(m['loss']):>8}"
        )
        if full is not None and ab != 'full' \
                and m['macro_f1'] is not None \
                and full['multiclass']['macro_f1'] is not None:
            delta = m['macro_f1'] - full['multiclass']['macro_f1']
            row += f"   (ΔMacroF1={delta:+.4f})"
        rows.append(row)

    # ----- Binary BC-vs-NoBC headline -----
    rows.append('')
    rows.append('=' * 110)
    rows.append('  ABLATION SUMMARY — BINARY (BC vs NoBC)')
    rows.append('=' * 110)
    header = (
        f"  {'Ablation':<18} "
        f"{'Acc':>8} {'F1':>8} {'P':>8} {'R':>8} "
        f"{'AUC':>9} {'EER':>9}"
    )
    rows.append(header)
    rows.append('-' * 110)
    for ab, blk in summary.items():
        m = blk['bc_binary']
        row = (
            f"  {ab:<18} "
            f"{_fmt(m['accuracy']):>8} "
            f"{_fmt(m['f1']):>8} "
            f"{_fmt(m['precision']):>8} "
            f"{_fmt(m['recall']):>8} "
            f"{_fmt(m['auc']):>9} "
            f"{_fmt(m['eer']):>9}"
        )
        if full is not None and ab != 'full' \
                and m['f1'] is not None \
                and full['bc_binary']['f1'] is not None:
            delta = m['f1'] - full['bc_binary']['f1']
            row += f"   (ΔBC-F1={delta:+.4f})"
        rows.append(row)

    # ----- Per-class breakdown (one block per class) -----
    rows.append('')
    rows.append('=' * 110)
    rows.append('  PER-CLASS BREAKDOWN')
    rows.append('=' * 110)
    for cls_name in label_names:
        rows.append('')
        rows.append(f'  Class: {cls_name}')
        rows.append('  ' + '-' * 96)
        header = (
            f"  {'Ablation':<18} "
            f"{'Acc':>8} {'P':>8} {'R':>8} {'F1':>8} "
            f"{'AUC':>9} {'EER':>9} {'Support':>9}"
        )
        rows.append(header)
        for ab, blk in summary.items():
            m = blk['per_class'].get(cls_name, {})
            sup = m.get('support')
            sup_s = f"{sup:>9d}" if isinstance(sup, int) else f"{'n/a':>9}"
            rows.append(
                f"  {ab:<18} "
                f"{_fmt(m.get('accuracy')):>8} "
                f"{_fmt(m.get('precision')):>8} "
                f"{_fmt(m.get('recall')):>8} "
                f"{_fmt(m.get('f1')):>8} "
                f"{_fmt(m.get('auc')):>9} "
                f"{_fmt(m.get('eer')):>9} "
                f"{sup_s}"
            )

    rows.append('=' * 110)
    return '\n'.join(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Run inference-time ablations on a trained backchannel classifier.'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Checkpoint directory (model.pt + config.json) or a model.pt file.')
    p.add_argument('--val_file', type=str, required=True,
                   help='Evaluation JSONL written by neural_scheduler.backchannel.data.prep.')
    p.add_argument('--output_json', type=str, required=True,
                   help='Where to write the per-ablation metrics.')
    p.add_argument('--audio_root', type=str, default=None,
                   help='Directory that relative audio_span paths are resolved against.')
    p.add_argument('--num_samples', type=int, default=None,
                   help='Number of examples to evaluate (stratified by label). Default: all.')
    p.add_argument('--ablations', type=str, nargs='+', default=None, choices=ALL_ABLATIONS,
                   help='Ablations to run. Default: all.')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--max_audio_seconds', type=float, default=15.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--roberta_name', type=str, default='roberta-base',
                   help='Text model to build a tokenizer from if none was saved with the checkpoint.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('ablation')
    logger.info(f"Args: {vars(args)}")
    run_ablations(args, logger)


if __name__ == '__main__':
    main()
