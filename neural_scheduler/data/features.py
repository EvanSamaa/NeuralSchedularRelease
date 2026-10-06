"""Turn preprocessed JSONL records into training features, and set up the tokenizer."""

from typing import Any, Dict, List

from transformers import AutoTokenizer

from ..constants import GLM4_VOICE_MODEL_ID, MAX_TURNS
from ..vocab import add_scheduler_tokens


def prepare_features(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Build labels, attention masks and full-length duration labels for a batch of records.

    Input columns (one list entry per record):

    * ``input_ids``: the full sequence as vocab ids;
    * ``generation_start_index``: index of the first dialogue token;
    * ``generation_end_index`` (optional): index of ``<|end-of-dialogue|>``;
      defaults to the sequence length;
    * ``remaining_durations`` (optional): duration labels for the dialogue
      region only, of length ``generation_end_index - generation_start_index``.

    Output columns: ``input_ids``; ``labels`` (script region set to -100, so
    the loss covers the dialogue only); ``attention_mask`` (all ones);
    ``generation_start_index``; and ``remaining_durations`` padded with -100 to
    the sequence length when the input had them.
    """
    input_ids_batch = examples["input_ids"]
    start_indices = examples["generation_start_index"]

    has_gen_end = "generation_end_index" in examples
    if has_gen_end:
        end_indices = examples["generation_end_index"]

    has_durations = "remaining_durations" in examples
    if has_durations:
        durations_batch = examples["remaining_durations"]

    labels_batch = []
    attention_mask_batch = []
    processed_durations_batch = []

    for i, (ids, start_idx) in enumerate(zip(input_ids_batch, start_indices)):
        end_idx = end_indices[i] if has_gen_end else len(ids)

        label = list(ids)
        for j in range(len(label)):
            if j < start_idx:
                label[j] = -100
        labels_batch.append(label)
        attention_mask_batch.append([1] * len(ids))

        if has_durations:
            dialogue_durations = list(durations_batch[i])
            expected_dialogue_len = end_idx - start_idx
            if len(dialogue_durations) != expected_dialogue_len:
                raise ValueError(
                    f"Sample {i}: Dialogue duration mismatch! "
                    f"Expected {expected_dialogue_len} durations (gen_start={start_idx}, gen_end={end_idx}), "
                    f"but got {len(dialogue_durations)} durations"
                )

            full_durations = [-100] * start_idx + dialogue_durations + [-100] * (len(ids) - end_idx)
            if len(full_durations) != len(ids):
                raise ValueError(
                    f"Sample {i}: Duration length mismatch after padding! "
                    f"input_ids={len(ids)}, durations={len(full_durations)} "
                    f"(script_len={start_idx}, dialogue_len={len(dialogue_durations)}, "
                    f"trailing_len={len(ids) - end_idx})"
                )
            processed_durations_batch.append(full_durations)

    result = {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "attention_mask": attention_mask_batch,
        "generation_start_index": start_indices,
    }
    if has_durations:
        result["remaining_durations"] = processed_durations_batch
    return result


def setup_tokenizer(model_id: str = GLM4_VOICE_MODEL_ID, max_turns: int = MAX_TURNS):
    """Load a GLM-4-Voice tokenizer and append the scheduler tokens.

    ``model_id`` may be the stock model or a scheduler checkpoint directory
    (whose tokenizer already has the tokens). Padding is on the right.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    add_scheduler_tokens(tokenizer, max_turns)
    tokenizer.padding_side = "right"
    return tokenizer
