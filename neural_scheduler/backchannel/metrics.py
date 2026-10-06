"""Evaluation metrics for the backchannel classifiers.

Reported for every model: loss, accuracy, macro / weighted F1, precision and
recall, backchannel-vs-none F1 / precision / recall (any non-zero class counts
as a backchannel), and one-vs-rest AUC and equal error rate per class.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the threshold at which it is reached
    for a one-vs-rest binary problem.

    EER is the operating point on the ROC curve where FPR == FNR (i.e.
    FPR == 1 - TPR). We compute this by finding the threshold that
    minimizes |FPR - FNR| on the ROC curve and linearly interpolating.

    Returns
    -------
    (eer, threshold)
        eer       : float in [0, 1]; NaN if the class is degenerate
                    (only one label present in y_true).
        threshold : float; NaN if degenerate.
    """
    # Need both positives and negatives present
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float('nan'), float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    # Find the index where FPR and FNR cross (or are closest)
    diff = fpr - fnr
    # If there's an exact crossing, interpolate between the two surrounding points
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        # Linear interp between i and i+1 to get the crossing point
        x0, x1 = diff[i], diff[i + 1]
        if x1 != x0:
            t = -x0 / (x1 - x0)
        else:
            t = 0.0
        eer = float(fpr[i] + t * (fpr[i + 1] - fpr[i]))
        thr = float(thresholds[i] + t * (thresholds[i + 1] - thresholds[i]))
    else:
        # No crossing — pick closest point
        idx = int(np.argmin(np.abs(diff)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        thr = float(thresholds[idx])
    return eer, thr


def compute_per_class_auc_eer(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> Dict:
    """
    Compute per-class one-vs-rest AUC and EER, plus macro-averaged values.

    Parameters
    ----------
    y_true   : [N] integer class labels
    y_proba  : [N, num_classes] softmax probabilities
    num_classes : number of classes

    Returns
    -------
    dict with keys:
        per_class_auc  : list of length num_classes (NaN if class absent)
        per_class_eer  : list of length num_classes (NaN if class absent)
        macro_auc      : mean of valid (non-NaN) per-class AUCs
        macro_eer      : mean of valid (non-NaN) per-class EERs
    """
    per_class_auc, per_class_eer, per_class_thr = [], [], []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        y_sc = y_proba[:, c]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            # Degenerate: class absent (or all positives) — undefined
            per_class_auc.append(float('nan'))
            per_class_eer.append(float('nan'))
            per_class_thr.append(float('nan'))
            continue
        try:
            auc = float(roc_auc_score(y_bin, y_sc))
        except ValueError:
            auc = float('nan')
        eer, thr = compute_eer(y_bin, y_sc)
        per_class_auc.append(auc)
        per_class_eer.append(eer)
        per_class_thr.append(thr)

    valid_aucs = [a for a in per_class_auc if not math.isnan(a)]
    valid_eers = [e for e in per_class_eer if not math.isnan(e)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
    macro_eer = float(np.mean(valid_eers)) if valid_eers else float('nan')

    return {
        'per_class_auc': per_class_auc,
        'per_class_eer': per_class_eer,
        'per_class_eer_threshold': per_class_thr,
        'macro_auc': macro_auc,
        'macro_eer': macro_eer,
    }


def format_eval_table(metrics: Dict, label_names: List[str]) -> str:
    """
    Render evaluation metrics as a nicely formatted text table.

    Sections:
      1. Overall summary (loss, accuracy, F1s, macro AUC, macro EER)
      2. Per-class AUC / EER / F1 / support
      3. Confusion matrix (counts)
      4. sklearn classification report (precision/recall/F1/support)
    """
    lines = []
    sep_thick = '=' * 78
    sep_thin = '-' * 78

    # -- Overall summary --
    lines.append(sep_thick)
    lines.append('  EVALUATION RESULTS')
    lines.append(sep_thick)
    lines.append(f"  Loss             : {metrics['loss']:.4f}")
    lines.append(f"  Accuracy         : {metrics['accuracy']:.4f}")
    lines.append(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    lines.append(f"  Weighted F1      : {metrics['weighted_f1']:.4f}")
    lines.append(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    lines.append(f"  Weighted Precision: {metrics['weighted_precision']:.4f}")
    lines.append(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    lines.append(f"  Weighted Recall  : {metrics['weighted_recall']:.4f}")
    lines.append(f"  Macro AUC (OvR)  : {metrics['macro_auc']:.4f}")
    lines.append(f"  Macro EER (OvR)  : {metrics['macro_eer']:.4f}")
    lines.append(
        f"  BC vs NoBC       : F1={metrics['bc_f1']:.4f}  "
        f"P={metrics['bc_precision']:.4f}  R={metrics['bc_recall']:.4f}"
    )

    # -- Per-class AUC/EER table --
    lines.append('')
    lines.append(sep_thin)
    lines.append('  PER-CLASS METRICS (one-vs-rest)')
    lines.append(sep_thin)
    header = f"  {'Class':<18} {'AUC':>8}  {'EER':>8}  {'EER thr':>10}  {'Support':>8}"
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))

    per_class_support = metrics.get('per_class_support', [0] * len(label_names))
    for i, name in enumerate(label_names):
        auc = metrics['per_class_auc'][i]
        eer = metrics['per_class_eer'][i]
        thr = metrics['per_class_eer_threshold'][i]
        sup = per_class_support[i] if i < len(per_class_support) else 0
        auc_s = f"{auc:.4f}" if not math.isnan(auc) else '   n/a'
        eer_s = f"{eer:.4f}" if not math.isnan(eer) else '   n/a'
        thr_s = f"{thr:.4f}" if not math.isnan(thr) else '     n/a'
        lines.append(
            f"  {name:<18} {auc_s:>8}  {eer_s:>8}  {thr_s:>10}  {sup:>8d}"
        )
    lines.append('  ' + '-' * (len(header) - 2))
    macro_auc_s = (
        f"{metrics['macro_auc']:.4f}" if not math.isnan(metrics['macro_auc']) else '   n/a'
    )
    macro_eer_s = (
        f"{metrics['macro_eer']:.4f}" if not math.isnan(metrics['macro_eer']) else '   n/a'
    )
    lines.append(
        f"  {'MACRO AVG':<18} {macro_auc_s:>8}  {macro_eer_s:>8}  "
        f"{'':>10}  {sum(per_class_support):>8d}"
    )

    # -- Confusion matrix --
    cm = np.array(metrics['confusion_matrix'])
    lines.append('')
    lines.append(sep_thin)
    lines.append('  CONFUSION MATRIX  (rows = true, cols = predicted)')
    lines.append(sep_thin)
    name_w = max(len(n) for n in label_names) + 2
    col_w = max(8, max(len(n) for n in label_names) + 2)
    header_cells = ' '.join(f"{n[:col_w-1]:>{col_w}}" for n in label_names)
    lines.append(f"  {'':<{name_w}}{header_cells}")
    for i, name in enumerate(label_names):
        row = ' '.join(f"{int(cm[i, j]):>{col_w}d}" for j in range(len(label_names)))
        lines.append(f"  {name:<{name_w}}{row}")

    # -- sklearn classification report --
    lines.append('')
    lines.append(sep_thin)
    lines.append('  CLASSIFICATION REPORT')
    lines.append(sep_thin)
    for rline in metrics['classification_report'].splitlines():
        lines.append('  ' + rline)
    lines.append(sep_thick)

    return '\n'.join(lines)



@torch.no_grad()
def collect_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device,
                        num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run a joint (text + audio) model over a loader.

    Returns ``(labels, predictions, probabilities, mean_loss)`` where the loss
    is unweighted cross-entropy averaged over batches.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, num_batches = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            audio=batch['audio'].to(device),
            audio_lens=batch['audio_lens'].to(device),
        )
        labels = batch['label'].to(device)
        total_loss += criterion(logits, labels).item()
        num_batches += 1

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes))
    return np.array(all_labels), np.array(all_preds), probs, total_loss / max(num_batches, 1)


def backchannel_detection_scores(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """``(f1, precision, recall)`` of backchannel vs. none, collapsing all non-zero classes."""
    binary_preds = (preds > 0).astype(int)
    binary_labels = (labels > 0).astype(int)
    f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)
    hits = (binary_preds & binary_labels).sum()
    return float(f1), float(hits / max(binary_preds.sum(), 1)), float(hits / max(binary_labels.sum(), 1))


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_names: List[str],
) -> Dict:
    """Evaluate a joint model; see the module docstring for the metrics."""
    all_labels, all_preds, all_probs, loss = collect_predictions(model, dataloader, device, num_classes)

    accuracy = (all_preds == all_labels).mean() if len(all_labels) else 0.0
    averaged = {}
    for average in ('macro', 'weighted'):
        averaged[f'{average}_f1'] = f1_score(all_labels, all_preds, average=average, zero_division=0)
        averaged[f'{average}_precision'] = precision_score(
            all_labels, all_preds, average=average, zero_division=0)
        averaged[f'{average}_recall'] = recall_score(
            all_labels, all_preds, average=average, zero_division=0)
    bc_f1, bc_precision, bc_recall = backchannel_detection_scores(all_labels, all_preds)
    auc_eer = compute_per_class_auc_eer(all_labels, all_probs, num_classes)

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=label_names[:num_classes],
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        'loss': loss,
        'accuracy': float(accuracy),
        'macro_f1': float(averaged['macro_f1']),
        'weighted_f1': float(averaged['weighted_f1']),
        'macro_precision': float(averaged['macro_precision']),
        'weighted_precision': float(averaged['weighted_precision']),
        'macro_recall': float(averaged['macro_recall']),
        'weighted_recall': float(averaged['weighted_recall']),
        'bc_f1': bc_f1,
        'bc_precision': bc_precision,
        'bc_recall': bc_recall,
        'per_class_auc': auc_eer['per_class_auc'],
        'per_class_eer': auc_eer['per_class_eer'],
        'per_class_eer_threshold': auc_eer['per_class_eer_threshold'],
        'macro_auc': auc_eer['macro_auc'],
        'macro_eer': auc_eer['macro_eer'],
        'per_class_support': [int((all_labels == c).sum()) for c in range(num_classes)],
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }


def evaluate_full(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_names: List[str],
) -> Dict:
    """Like :func:`evaluate`, with per-class precision / recall / F1 and a
    ``bc_binary`` block for backchannel vs. none.

    ``per_class_accuracy`` is per-class recall. The binary AUC and EER use
    ``1 - P(NoBackchannel)`` as the score.
    """
    y_true, y_pred, y_prob, loss = collect_predictions(model, dataloader, device, num_classes)
    class_ids = list(range(num_classes))

    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    per_class_recall = recall_score(
        y_true, y_pred, labels=class_ids, average=None, zero_division=0).tolist()
    auc_eer = compute_per_class_auc_eer(y_true, y_prob, num_classes)

    binary_pred = (y_pred > 0).astype(int)
    binary_true = (y_true > 0).astype(int)
    bc_f1 = float(f1_score(binary_true, binary_pred, average='binary', zero_division=0))
    bc_precision = float(precision_score(binary_true, binary_pred, zero_division=0))
    bc_recall = float(recall_score(binary_true, binary_pred, zero_division=0))
    bc_accuracy = float((binary_pred == binary_true).mean()) if len(binary_true) else 0.0

    bc_score = 1.0 - y_prob[:, 0] if y_prob.size else np.zeros(0)
    bc_auc = bc_eer = bc_eer_thr = float('nan')
    if 0 < binary_true.sum() < len(binary_true):
        try:
            bc_auc = float(roc_auc_score(binary_true, bc_score))
        except ValueError:
            bc_auc = float('nan')
        bc_eer, bc_eer_thr = compute_eer(binary_true, bc_score)

    report = classification_report(
        y_true, y_pred, labels=class_ids, target_names=label_names[:num_classes], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class_accuracy': list(per_class_recall),
        'per_class_precision': precision_score(
            y_true, y_pred, labels=class_ids, average=None, zero_division=0).tolist(),
        'per_class_recall': per_class_recall,
        'per_class_f1': f1_score(
            y_true, y_pred, labels=class_ids, average=None, zero_division=0).tolist(),
        'per_class_support': [int((y_true == c).sum()) for c in class_ids],
        'per_class_auc': auc_eer['per_class_auc'],
        'per_class_eer': auc_eer['per_class_eer'],
        'per_class_eer_threshold': auc_eer['per_class_eer_threshold'],
        'macro_auc': auc_eer['macro_auc'],
        'macro_eer': auc_eer['macro_eer'],
        'bc_binary': {
            'accuracy': bc_accuracy,
            'precision': bc_precision,
            'recall': bc_recall,
            'f1': bc_f1,
            'auc': bc_auc,
            'eer': bc_eer,
            'eer_threshold': bc_eer_thr,
            'positive_support': int(binary_true.sum()),
            'negative_support': int(len(binary_true) - binary_true.sum()),
        },
        'bc_f1': bc_f1,
        'bc_precision': bc_precision,
        'bc_recall': bc_recall,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }


@torch.no_grad()
def evaluate_text_only(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_names: List[str],
) -> Dict:
    """Evaluate a ``RobertaForSequenceClassification`` model on text-only batches."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, num_batches = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        labels = batch['label'].to(device)
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
        ).logits
        total_loss += criterion(logits, labels).item()
        num_batches += 1
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    bc_f1, bc_precision, bc_recall = backchannel_detection_scores(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=label_names[:num_classes],
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': float((all_preds == all_labels).mean()),
        'macro_f1': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'bc_f1': bc_f1,
        'bc_precision': bc_precision,
        'bc_recall': bc_recall,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }
