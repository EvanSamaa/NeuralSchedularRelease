"""Loading, input construction, scoring and sampling shared by the evaluations.

Sequences follow the scheduler layout::

    <|start-of-script-A|> script A <|end-of-script-A|>
    <|start-of-script-B|> script B <|end-of-script-B|>
    <|start-of-dialogue|> A0 B0 A1 B1 ... <|end-of-dialogue|>

"Context" below means everything up to and including ``<|start-of-dialogue|>``.
Token lists are either *codec* ids (0..16383) or *vocab* ids; each function
says which it expects.
"""

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..constants import CHANNEL_DIALOGUE_A, CHANNEL_SCRIPT, SILENCE_TOKEN_ID
from ..vocab import (
    END_DIALOGUE,
    END_SCRIPT_A,
    END_SCRIPT_B,
    START_DIALOGUE,
    START_SCRIPT_A,
    START_SCRIPT_B,
    audio_token,
)

# Largest codec id kept when a script is given as codec ids. Script regions of
# the released datasets also hold turn markers; expressed as "codec ids" (vocab
# id minus the audio offset) those are >= 16389 and are dropped by this bound.
MAX_SCRIPT_CODE = 16388


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl_examples(file_path: str, num_examples: Optional[int] = None, seed: Optional[int] = None) -> List[Dict]:
    """Read a JSONL dataset and draw ``num_examples`` records without replacement.

    With ``seed`` the draw is reproducible. With ``seed=None`` the global
    ``random`` state is used, which is how the research scripts sampled (they
    never seeded it). ``num_examples=None`` returns every record in file order.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    if num_examples is None or not examples:
        return examples
    sample_size = min(num_examples, len(examples))
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(examples, sample_size)


def extract_script_contents(tokenizer, input_ids: List[int], return_dialogue: bool = False):
    """Split a full sequence into its script A, script B (and dialogue) vocab ids.

    Markers are excluded. A region whose markers are missing is returned empty.
    """
    marker_tokens = [START_SCRIPT_A, END_SCRIPT_A, START_SCRIPT_B, END_SCRIPT_B, START_DIALOGUE, END_DIALOGUE]
    markers = {token: tokenizer.convert_tokens_to_ids(token) for token in marker_tokens}

    def get_slice(start_tok, end_tok):
        try:
            start_idx = input_ids.index(markers[start_tok]) + 1
            end_idx = input_ids.index(markers[end_tok])
            return input_ids[start_idx:end_idx]
        except (ValueError, KeyError):
            return []

    script_a_ids = get_slice(START_SCRIPT_A, END_SCRIPT_A)
    script_b_ids = get_slice(START_SCRIPT_B, END_SCRIPT_B)
    if return_dialogue:
        dialogue_ids = get_slice(START_DIALOGUE, END_DIALOGUE)
        return script_a_ids, script_b_ids, dialogue_ids
    return script_a_ids, script_b_ids


def prepare_inference_input(
    tokens_channel_a: List[int],
    tokens_channel_b: List[int],
    tokenizer,
    device: Union[str, torch.device] = "cuda",
    already_tokenized: bool = False,
    audio_token_max_value: int = MAX_SCRIPT_CODE,
) -> Tuple[torch.Tensor, int]:
    """Build the context ``[1, L]`` from two scripts.

    ``already_tokenized=True``: scripts are vocab ids and are used verbatim.
    ``already_tokenized=False``: scripts are codec ids; values above
    ``audio_token_max_value`` are dropped before conversion to ``<|audio_k|>``.

    Returns the context tensor and its length, which is the index where the
    dialogue starts.
    """
    if not already_tokenized:
        tokens_channel_a = [t for t in tokens_channel_a if t <= audio_token_max_value]
        tokens_channel_b = [t for t in tokens_channel_b if t <= audio_token_max_value]
        all_tokens = [START_SCRIPT_A]
        all_tokens += [audio_token(t) for t in tokens_channel_a]
        all_tokens += [END_SCRIPT_A, START_SCRIPT_B]
        all_tokens += [audio_token(t) for t in tokens_channel_b]
        all_tokens += [END_SCRIPT_B, START_DIALOGUE]
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    else:
        input_ids = tokenizer.convert_tokens_to_ids([START_SCRIPT_A]) + list(tokens_channel_a)
        input_ids += tokenizer.convert_tokens_to_ids([END_SCRIPT_A])
        input_ids += tokenizer.convert_tokens_to_ids([START_SCRIPT_B]) + list(tokens_channel_b)
        input_ids += tokenizer.convert_tokens_to_ids([END_SCRIPT_B])
        input_ids += tokenizer.convert_tokens_to_ids([START_DIALOGUE])
    generation_start_index = len(input_ids)
    return torch.tensor([input_ids], dtype=torch.long, device=device), generation_start_index


def remove_duplicated_silence_tokens(seq: List[int], target_val: int = SILENCE_TOKEN_ID, split_lengths=(2, 4)) -> List[int]:
    """Collapse the silence runs that delimit script segments to one token each.

    Runs of ``target_val`` whose length is in ``split_lengths`` are treated as
    delimiters; the pieces between them are re-joined with a single
    ``target_val`` and the result starts and ends with one.
    """
    blocks = []
    i = 0
    while i < len(seq):
        if seq[i] == target_val:
            start = i
            while i < len(seq) and seq[i] == target_val:
                i += 1
            blocks.append((start, i))
        else:
            i += 1

    subsequences = []
    current_start = 0
    for start, end in blocks:
        if (end - start) not in split_lengths:
            continue
        if start > current_start:
            subsequences.append(seq[current_start:start])
        current_start = end
    if current_start < len(seq):
        subsequences.append(seq[current_start:])

    out_sequence = [target_val]
    for subseq in subsequences:
        out_sequence.extend(subseq)
        out_sequence.append(target_val)
    return out_sequence


def separate_channels(generated_tokens: Sequence[int], tokenizer) -> Tuple[List[int], List[int]]:
    """De-interleave generated vocab ids into two codec-id lists.

    Non-audio tokens are skipped; parity is taken from the position in the
    generated sequence (even = A, odd = B).
    """
    channel_a_tokens, channel_b_tokens = [], []
    for i, token_id in enumerate(generated_tokens):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if token_str.startswith("<|audio_") and token_str.endswith("|>"):
            code = int(token_str[8:-2])
            if i % 2 == 0:
                channel_a_tokens.append(code)
            else:
                channel_b_tokens.append(code)
    return channel_a_tokens, channel_b_tokens


# ---------------------------------------------------------------------------
# Channel ids
# ---------------------------------------------------------------------------

def compute_channel_ids(seq_length: int, generation_start_index: int, device, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """Channel ids ``[1, seq_length]``: 0 for the context, then 1 (A), 2 (B) alternating."""
    positions = torch.arange(seq_length, device=device)
    relative = positions - generation_start_index
    channel_ids = torch.where(
        relative >= 0,
        relative % 2 + CHANNEL_DIALOGUE_A,
        torch.full_like(positions, CHANNEL_SCRIPT),
    )
    return channel_ids.to(dtype).unsqueeze(0)


def compute_batch_channel_ids(batch_size: int, seq_length: int, generation_start_index: Sequence[int], device) -> torch.Tensor:
    """Channel ids ``[B, seq_length]`` with one dialogue start per row."""
    rows = [compute_channel_ids(seq_length, int(generation_start_index[b]), device) for b in range(batch_size)]
    return torch.cat(rows, dim=0)


def compute_alignment_channel_ids(seq_length: int, context_length: int, device) -> torch.Tensor:
    """Channel ids as the silence-perturbation experiment built them: 0 for the
    context and for dialogue A, 1 for dialogue B.

    This is not the layout the model was trained with (0 / 1 / 2). It is kept
    because the published silence-perturbation curves were scored with it; see
    ``docs/evaluation.md``.
    """
    positions = torch.arange(seq_length, device=device)
    relative = positions - context_length
    channel_ids = torch.where(relative >= 0, relative % 2, torch.zeros_like(positions))
    return channel_ids.to(torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def interleave(dialogue_a: Sequence[int], dialogue_b: Sequence[int]) -> List[int]:
    """``[a0, b0, a1, b1, ...]``, truncated to the shorter channel."""
    out = []
    for a, b in zip(dialogue_a, dialogue_b):
        out.extend([a, b])
    return out


@torch.no_grad()
def compute_dialogue_nll(
    model,
    context_ids: torch.Tensor,
    context_length: int,
    dialogue_ids: Sequence[int],
    device: Union[str, torch.device] = "cuda",
    legacy_channel_ids: bool = False,
) -> Tuple[float, float]:
    """Perplexity and summed NLL of an interleaved dialogue given its context.

    ``dialogue_ids`` are vocab ids. One forward pass over context + dialogue;
    only dialogue tokens are scored. ``legacy_channel_ids`` selects
    :func:`compute_alignment_channel_ids` instead of the trained layout.

    Returns ``(perplexity, total_nll)``.
    """
    dialogue_tensor = torch.tensor([list(dialogue_ids)], dtype=torch.long, device=device)
    full_input_ids = torch.cat([context_ids, dialogue_tensor], dim=1)
    seq_length = full_input_ids.shape[1]
    if legacy_channel_ids:
        channel_ids = compute_alignment_channel_ids(seq_length, context_length, device)
    else:
        channel_ids = compute_channel_ids(seq_length, context_length, device)
    attention_mask = torch.ones_like(full_input_ids)

    outputs = model(
        input_ids=full_input_ids,
        attention_mask=attention_mask,
        channel_ids=channel_ids,
        use_cache=False,
        return_dict=True,
    )
    shift_logits = outputs.logits[:, context_length - 1:-1, :]
    shift_labels = full_input_ids[:, context_length:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    total_nll = -token_log_probs.sum().item()
    num_tokens = shift_labels.shape[1]
    perplexity = torch.exp(torch.tensor(total_nll / num_tokens)).item() if num_tokens > 0 else 1.0
    return perplexity, total_nll


@torch.no_grad()
def compute_full_nll(
    model,
    context_tokens: torch.Tensor,
    context_length: int,
    dialogue_tokens_a: torch.Tensor,
    dialogue_tokens_b: torch.Tensor,
    device: Union[str, torch.device] = "cuda",
    legacy_channel_ids: bool = True,
) -> Tuple[float, float]:
    """Perplexity and summed NLL for two aligned ``[1, T]`` channel tensors of vocab ids.

    Defaults to the legacy channel-id layout, which is what the
    silence-perturbation experiment used. Returns ``(perplexity, total_nll)``.
    """
    dialogue_ids = interleave(dialogue_tokens_a[0].tolist(), dialogue_tokens_b[0].tolist())
    return compute_dialogue_nll(
        model, context_tokens, context_length, dialogue_ids, device=device, legacy_channel_ids=legacy_channel_ids
    )


@torch.no_grad()
def compute_per_token_nll(model, context_ids: List[int], dialogue_ids: List[int], device) -> np.ndarray:
    """NLL of every dialogue token (vocab ids), shape ``(len(dialogue_ids),)``."""
    full_ids = list(context_ids) + list(dialogue_ids)
    prefix_len = len(context_ids)
    dialogue_len = len(dialogue_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    channel_ids = compute_channel_ids(len(full_ids), prefix_len, device)
    attention_mask = torch.ones(1, len(full_ids), dtype=torch.long, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        channel_ids=channel_ids,
        use_cache=False,
        return_dict=True,
    )
    dialogue_logits = outputs.logits[0, prefix_len - 1: prefix_len + dialogue_len - 1, :]
    dialogue_labels = input_ids[0, prefix_len:]
    token_nlls = torch.nn.CrossEntropyLoss(reduction="none")(dialogue_logits, dialogue_labels)
    return token_nlls.cpu().numpy()


def build_scoring_inputs(input_ids: torch.Tensor, generation_start_index: Sequence[int], device):
    """Inputs for :func:`score_sequences` from a ``[B, L]`` batch of full sequences.

    Returns ``({"input_ids", "labels"}, attention_mask, channel_ids)``; the
    attention mask is all ones, so rows must not be padded.
    """
    batch_size, seq_length = input_ids.shape
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    channel_ids = compute_batch_channel_ids(batch_size, seq_length, generation_start_index, device)
    return {"input_ids": input_ids, "labels": input_ids}, attention_mask, channel_ids


def score_sequences(model, inputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor, channel_ids: torch.Tensor):
    """Perplexity and mean NLL per row over *all* tokens after the first
    (script and dialogue alike). Returns ``(perplexities, mean_loss)``."""
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            channel_ids=channel_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    token_losses = torch.nn.CrossEntropyLoss(reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.size())

    token_losses = token_losses * attention_mask[..., 1:]
    mean_loss = token_losses.sum(dim=1) / attention_mask[..., 1:].sum(dim=1)
    return torch.exp(mean_loss), mean_loss


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_dialogue(
    model,
    tokenizer,
    tokens_channel_a: List[int],
    tokens_channel_b: List[int],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    device: Union[str, torch.device] = "cuda",
    return_channel_separated: bool = True,
    verbose: bool = False,
    already_tokenized: bool = False,
) -> Union[List[int], Tuple[List[int], List[int]]]:
    """Sample a dialogue token by token from the two scripts.

    Top-k is applied before nucleus filtering. Generation stops at
    ``<|end-of-dialogue|>`` or the tokenizer's EOS, or after
    ``max_new_tokens``. Returns two codec-id lists (A, B) or, with
    ``return_channel_separated=False``, the generated vocab ids.
    """
    model.eval()
    end_of_dialogue_token = tokenizer.convert_tokens_to_ids(END_DIALOGUE)
    eos_token_id = tokenizer.eos_token_id

    input_ids, generation_start_index = prepare_inference_input(
        tokens_channel_a, tokens_channel_b, tokenizer, device, already_tokenized=already_tokenized
    )
    if verbose:
        print(f"Context length: {generation_start_index}")

    generated_tokens = []
    current_input_ids = input_ids

    for step in range(max_new_tokens):
        channel_ids = compute_channel_ids(current_input_ids.shape[1], generation_start_index, current_input_ids.device)
        attention_mask = torch.ones_like(current_input_ids)

        # Every step re-encodes the full sequence; no key/value cache is carried over.
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            channel_ids=channel_ids,
            use_cache=True,
            return_dict=True,
        )

        next_token_logits = outputs.logits[:, -1, :]
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if do_sample:
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token_id = next_token.item()
        generated_tokens.append(next_token_id)
        if verbose and (step + 1) % 50 == 0:
            print(f"Generated {step + 1} tokens...")
        if next_token_id == end_of_dialogue_token or next_token_id == eos_token_id:
            break
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

    if generated_tokens and generated_tokens[-1] == end_of_dialogue_token:
        generated_tokens = generated_tokens[:-1]
    if return_channel_separated:
        return separate_channels(generated_tokens, tokenizer)
    return generated_tokens


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def to_serializable(obj):
    """Recursively convert numpy scalars/arrays (and tuples) for ``json.dump``."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_model_and_tokenizer(checkpoint: str, device: str):
    """Load the scheduler for an evaluation run (imported lazily so the
    library functions above work with any object exposing the same forward)."""
    from ..model import load_scheduler

    model, tokenizer = load_scheduler(checkpoint, device=device)
    model.eval()
    return model, tokenizer
