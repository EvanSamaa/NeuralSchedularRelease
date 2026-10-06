"""Batch collation for scheduler training."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

LABEL_PAD_ID = -100


@dataclass
class DataCollatorForSeq2SeqWithChannels:
    """Right-pad a list of prepared features into a batch of long tensors.

    Each feature holds ``input_ids`` and optionally ``attention_mask``,
    ``labels``, ``generation_start_index`` and ``remaining_durations``.
    ``input_ids`` are padded with the tokenizer's pad id, the attention mask
    with 0, labels and durations with -100. ``generation_start_index`` is
    passed through as a ``[batch]`` tensor so the model can derive channel ids.
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        generation_start_indices = [f.pop("generation_start_index", None) for f in features]
        remaining_durations = [f.pop("remaining_durations", None) for f in features]

        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_durations = []

        for idx, f in enumerate(features):
            input_ids = f["input_ids"]
            attention_mask = f.get("attention_mask", [1] * len(input_ids))
            labels = f.get("labels", None)
            padding_length = max_length - len(input_ids)

            batch_input_ids.append(list(input_ids) + [pad_token_id] * padding_length)
            batch_attention_mask.append(list(attention_mask) + [0] * padding_length)
            if labels is not None:
                batch_labels.append(list(labels) + [LABEL_PAD_ID] * padding_length)
            if remaining_durations[idx] is not None:
                batch_durations.append(list(remaining_durations[idx]) + [LABEL_PAD_ID] * padding_length)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
        if batch_labels:
            batch["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        if generation_start_indices[0] is not None:
            batch["generation_start_index"] = torch.tensor(generation_start_indices, dtype=torch.long)
        if batch_durations:
            batch["remaining_durations"] = torch.tensor(batch_durations, dtype=torch.long)
        return batch
