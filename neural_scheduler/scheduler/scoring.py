"""Scoring a candidate schedule: negative log-likelihood under the scheduler LM."""

from typing import Dict, List, Optional, Tuple

import torch

from ..constants import CHANNEL_DIALOGUE_A, CHANNEL_SCRIPT


def compute_nll_for_placement(
    model,
    script_ids: List[int],
    dialogue_ids: List[int],
    generation_start_index: int,
    special_tokens: Dict[str, int],
    device: str = "cuda",
    scoring_window: Optional[Tuple[int, int]] = None,
) -> Tuple[float, List[float]]:
    """NLL of ``dialogue_ids`` given the script, with one full forward pass.

    The model input is ``script_ids[:generation_start_index] + dialogue_ids +
    [<|end-of-dialogue|>]``. ``scoring_window`` is a ``(start, end)`` range of
    indices into the interleaved dialogue; only the tokens in it are summed.
    Without a window the whole dialogue (and the end marker) is scored.

    Returns ``(total_nll, per_token_nll)`` for the scored tokens.
    """
    full_ids = script_ids[:generation_start_index] + dialogue_ids + [special_tokens["end_dialogue"]]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    seq_len = input_ids.shape[1]

    positions = torch.arange(seq_len, device=device)
    channel_ids = torch.where(
        positions < generation_start_index,
        torch.full_like(positions, CHANNEL_SCRIPT),
        (positions - generation_start_index) % 2 + CHANNEL_DIALOGUE_A,
    ).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel_ids=channel_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    per_token_nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())[0].cpu().tolist()

    # per_token_nll[i] is the loss of token i + 1, hence the - 1 below.
    if scoring_window is not None:
        win_start = max(0, generation_start_index + scoring_window[0] - 1)
        win_end = min(len(per_token_nll), generation_start_index + scoring_window[1] - 1)
        scored = per_token_nll[win_start:win_end]
    else:
        scored = per_token_nll[max(0, generation_start_index - 1):]
    return sum(scored), scored
