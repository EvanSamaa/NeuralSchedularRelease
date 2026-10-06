"""Datasets, collation and samplers for the backchannel classifiers."""

import json
import logging
import math
import os
from collections import Counter
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..constants import HUBERT_SR, ORIGINAL_TO_MERGED
from .audio import load_audio_span


def label_for_phase(example: Dict, phase: str) -> int:
    """Integer label of a prepared example under a training phase."""
    if phase == "binary":
        return 0 if example["label"] == "NoBackchannel" else 1
    if phase == "merged":
        return ORIGINAL_TO_MERGED[example["label"]]
    return example["label_id"]


def read_examples(filepath: str, phase: str, exclude_repetition: bool = False) -> List[Dict]:
    """Read a prepared JSONL file. The merged phase always drops ``Repetition``
    (it has no merged class); the binary phase drops it on request."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex["label"] == "Repetition" and (
                phase == "merged" or (phase == "binary" and exclude_repetition)
            ):
                continue
            examples.append(ex)
    return examples


class BackchannelJointDataset(Dataset):
    """Text + audio examples.

    Items are ``{input_ids, attention_mask, audio, audio_len, label}`` with
    unpadded text and the main speaker's audio span at 16 kHz. Truncation
    follows ``tokenizer.truncation_side``. ``audio_root`` is prepended to
    relative ``audio_span.wav`` paths.
    """

    def __init__(
        self,
        filepath: str,
        tokenizer,
        max_length: int = 512,
        phase: str = "merged",
        max_audio_seconds: float = 15.0,
        exclude_repetition: bool = False,
        audio_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        self.max_audio_seconds = max_audio_seconds
        self.audio_root = audio_root

        examples = read_examples(filepath, phase, exclude_repetition)
        self.examples = [ex for ex in examples if "audio_span" in ex]
        missing_audio = len(examples) - len(self.examples)
        if missing_audio > 0:
            logging.warning(f"{filepath}: skipped {missing_audio} examples missing audio_span")

    def __len__(self):
        return len(self.examples)

    def _label_for(self, ex: Dict) -> int:
        return label_for_phase(ex, self.phase)

    def _resolve_wav(self, wav_path: str) -> str:
        if self.audio_root is not None and not os.path.isabs(wav_path):
            return os.path.join(self.audio_root, wav_path)
        return wav_path

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["input_text"],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )
        span = ex["audio_span"]
        audio = load_audio_span(
            wav_path=self._resolve_wav(span["wav"]),
            start_sec=float(span["start"]),
            end_sec=float(span["end"]),
            target_sr=HUBERT_SR,
            max_seconds=self.max_audio_seconds,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "audio": torch.from_numpy(audio),
            "audio_len": torch.tensor(len(audio), dtype=torch.long),
            "label": torch.tensor(self._label_for(ex), dtype=torch.long),
        }

    def get_label_counts(self) -> Counter:
        return Counter(self._label_for(ex) for ex in self.examples)

    def get_labels(self) -> List[int]:
        """Integer labels in dataset order."""
        return [self._label_for(ex) for ex in self.examples]


class BackchannelTextDataset(Dataset):
    """Text-only examples padded to ``max_length``: ``{input_ids, attention_mask, label}``."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 512, phase: str = "full"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        self.examples = read_examples(filepath, phase)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_for_phase(ex, self.phase), dtype=torch.long),
        }

    def get_label_counts(self) -> Counter:
        return Counter(label_for_phase(ex, self.phase) for ex in self.examples)


def make_collate_fn(pad_token_id: int):
    """Collate :class:`BackchannelJointDataset` items, right-padding text with
    ``pad_token_id`` and audio with zeros. Returns
    ``{input_ids, attention_mask, audio, audio_lens, label}``."""

    def _collate(batch):
        batch_size = len(batch)
        max_text = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.full((batch_size, max_text), pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((batch_size, max_text), dtype=torch.long)
        for i, b in enumerate(batch):
            length = b["input_ids"].size(0)
            input_ids[i, :length] = b["input_ids"]
            attn_mask[i, :length] = b["attention_mask"]

        max_audio = max(b["audio"].size(0) for b in batch)
        audio = torch.zeros((batch_size, max_audio), dtype=torch.float32)
        audio_lens = torch.zeros((batch_size,), dtype=torch.long)
        for i, b in enumerate(batch):
            length = b["audio"].size(0)
            audio[i, :length] = b["audio"]
            audio_lens[i] = length

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "audio": audio,
            "audio_lens": audio_lens,
            "label": torch.stack([b["label"] for b in batch]),
        }

    return _collate


def compute_class_weights(label_counts: Counter, num_classes: int, method: str = "sqrt_inv") -> torch.Tensor:
    """Loss weights normalised to mean 1.

    ``sqrt_inv``: ``sqrt(N / count)``; ``inv``: ``N / count``; ``uniform``: ones.
    A class absent from ``label_counts`` is treated as having one example.
    """
    total = sum(label_counts.values())
    weights = torch.ones(num_classes)
    if method == "uniform":
        return weights
    for cls_id in range(num_classes):
        count = label_counts.get(cls_id, 1)
        if method == "sqrt_inv":
            weights[cls_id] = math.sqrt(total / count)
        elif method == "inv":
            weights[cls_id] = total / count
    return weights / weights.mean()


def make_balanced_sampler(dataset: BackchannelJointDataset, nobc_label: int = 0) -> WeightedRandomSampler:
    """Sampler that draws NoBackchannel as often as all backchannel classes together.

    NoBackchannel examples get weight ``N_bc / N_nobc`` and the rest 1; an
    epoch is ``2 * N_bc`` draws with replacement.
    """
    label_arr = torch.tensor(dataset.get_labels(), dtype=torch.long)
    n_nbc = int((label_arr == nobc_label).sum())
    n_bc = int((label_arr != nobc_label).sum())
    if n_nbc == 0 or n_bc == 0:
        raise ValueError(
            f"make_balanced_sampler: dataset has {n_nbc} NoBC and {n_bc} BC "
            "samples — cannot balance a degenerate distribution."
        )
    weights = torch.where(
        label_arr == nobc_label,
        torch.full_like(label_arr, n_bc / n_nbc, dtype=torch.float),
        torch.full_like(label_arr, 1.0, dtype=torch.float),
    )
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n_bc * 2, replacement=True)


def build_balanced_sampler(labels: List[int], num_classes: int, mode: str = "uniform") -> WeightedRandomSampler:
    """Sampler that rebalances all classes while keeping ``len(labels)`` draws per epoch.

    ``uniform``: per-example weight ``1 / count[class]`` (equal class
    probability); ``sqrt``: ``1 / sqrt(count[class])`` (partial rebalancing).
    """
    assert mode in ("uniform", "sqrt"), f"unknown balanced_mode: {mode!r}"
    counts = Counter(labels)
    cls_w = torch.zeros(num_classes, dtype=torch.double)
    for c in range(num_classes):
        n = counts.get(c, 0)
        if n == 0:
            continue
        cls_w[c] = 1.0 / n if mode == "uniform" else 1.0 / math.sqrt(n)
    sample_w = torch.tensor([cls_w[lbl].item() for lbl in labels], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(labels), replacement=True)
