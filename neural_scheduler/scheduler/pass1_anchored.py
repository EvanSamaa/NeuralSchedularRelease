"""Pass 1 for backchannel-only scheduling: primary turns are not searched.

Used when one speaker's audio already exists as a recording (the benchmark
setting): channel-A turns are committed at their known positions and only the
other speaker's backchannels are scheduled afterwards, by pass 2.
"""

import logging
from typing import Dict, List, Tuple

from .subsegments import assemble_turn_from_sub_segments, proportional_token_counts
from .types import PlacedTurn, TurnInfo

logger = logging.getLogger(__name__)


def _place_turn_at_known_timing(turn: TurnInfo, timing: Dict, silence_token_id: int) -> PlacedTurn:
    """Lay ``turn`` out as ``timing`` says and return it placed at ``timing['start_token']``.

    ``timing`` is ``{'turn_id', 'start_token', 'end_token', 'sub_segments':
    [{'start_token', 'end_token'}, ...]}`` in dialogue steps. The turn's speech
    tokens are divided among the sub-segments in proportion to their spans, the
    gaps between them are filled with silence, and tokens that do not fit in
    the spans are dropped. ``turn`` is modified in place.
    """
    blocks = timing.get("sub_segments") or [
        {"start_token": timing["start_token"], "end_token": timing["end_token"]}
    ]
    blocks = sorted(blocks, key=lambda s: s["start_token"])

    speech_tokens = list(turn.speech_token_ids)
    spans = [max(0, s["end_token"] - s["start_token"]) for s in blocks]
    total_span = sum(spans)

    if total_span <= 0 or not speech_tokens:
        block_tokens: List[List[int]] = [[] for _ in blocks]
    else:
        if len(blocks) == 1:
            counts = [len(speech_tokens)]
        else:
            counts = proportional_token_counts(spans, len(speech_tokens))

        # A block never holds more tokens than its span; the excess moves on to the next block.
        carry = 0
        clamped: List[int] = []
        for count, span in zip(counts, spans):
            allotted = min(span, count + carry)
            clamped.append(allotted)
            carry = (count + carry) - allotted
        if carry > 0:
            logger.warning(
                f"  Turn {turn.turn_id}: dropped {carry} speech token(s) that do not fit the "
                f"known timing (script has {len(speech_tokens)}, timing has room for {total_span})."
            )

        block_tokens = []
        offset = 0
        for count in clamped:
            block_tokens.append(speech_tokens[offset:offset + count])
            offset += count

    block_onsets = [s["start_token"] for s in blocks]
    tokens, sub_segments_meta, _, _ = assemble_turn_from_sub_segments(block_tokens, block_onsets, silence_token_id)

    turn_onset = timing["start_token"]
    front_pad = block_onsets[0] - turn_onset
    if front_pad > 0:
        tokens = [silence_token_id] * front_pad + tokens
    elif front_pad < 0:
        logger.warning(
            f"  Turn {turn.turn_id}: start_token={timing['start_token']} is after the first "
            f"sub-segment start {block_onsets[0]}; using the sub-segment start."
        )
        turn_onset = block_onsets[0]

    turn.start_token = turn_onset
    turn.end_token = turn_onset + len(tokens)
    turn.duration = len(tokens)
    turn.speech_token_ids = tokens
    turn.sub_segments = sub_segments_meta

    return PlacedTurn(turn_info=turn, onset_step=turn_onset)


def pass_1_anchor_primary_turns(
    primary_turns: List[TurnInfo],
    special_tokens: Dict[str, int],
    s1_turn_timing: List[Dict],
    verbose: bool = False,
) -> Tuple[List[PlacedTurn], Dict]:
    """Place primary turns without any search; returns ``(placed_turns, diagnostics)``.

    A channel-A turn with an entry in ``s1_turn_timing`` is laid out as that
    entry says. Any other primary turn is placed at its ``start_token``, which
    is a position inside the script rather than in the dialogue; the benchmark
    inputs contain no such turns.
    """
    silence_token_id = special_tokens["silence"]
    timing_by_turn = {entry["turn_id"]: entry for entry in (s1_turn_timing or [])}

    placed_turns: List[PlacedTurn] = []
    diagnostics = {
        "pass": 1,
        "mode": "bc_only_anchor_s1_search_s2",
        "num_turns": len(primary_turns),
        "num_s1_anchored": 0,
        "num_s2_searched": 0,
        "per_turn": [],
    }

    for turn in sorted(primary_turns, key=lambda t: t.turn_id):
        if turn.channel == "A" and turn.turn_id in timing_by_turn:
            placed = _place_turn_at_known_timing(turn, timing_by_turn[turn.turn_id], silence_token_id)
            mode = "gt_anchored"
            diagnostics["num_s1_anchored"] += 1
        else:
            if verbose:
                logger.warning(
                    f"  Turn {turn.turn_id} ch={turn.channel} has no known timing; "
                    f"placing it at its script position {turn.start_token}."
                )
            placed = PlacedTurn(turn_info=turn, onset_step=int(turn.start_token))
            mode = "fallback_no_gt"
            diagnostics["num_s2_searched"] += 1

        placed_turns.append(placed)
        diagnostics["per_turn"].append({
            "turn_id": turn.turn_id,
            "channel": turn.channel,
            "mode": mode,
            "script_onset": placed.onset_step,
            "best_onset": placed.onset_step,
            "shift": 0,
            "best_nll": 0.0,
            "nll_scores": [0.0],
            "candidates": [placed.onset_step],
            "duration": placed.turn_info.duration,
            "search_bounds": (placed.onset_step, placed.onset_step),
            "gap_constraint_applied": False,
        })
        if verbose and mode == "gt_anchored":
            logger.info(
                f"  Turn {turn.turn_id} ch=A anchored at {placed.onset_step}, "
                f"duration {placed.turn_info.duration}, {len(placed.turn_info.sub_segments)} sub-segment(s)"
            )

    placed_turns.sort(key=lambda p: p.onset_step)
    return placed_turns, diagnostics
