"""Splitting turns into speech blocks ("sub-segments") and putting them back together."""

import copy
from typing import Dict, List, Sequence, Tuple

from ..constants import TOKEN_RATE_HZ
from .types import PlacedTurn, TurnInfo


def proportional_token_counts(weights: Sequence[float], num_tokens: int) -> List[int]:
    """Split ``num_tokens`` in proportion to ``weights``.

    Counts are rounded independently; the rounding error goes to the block with
    the largest weight (the first one on ties). ``sum(weights)`` must be positive.
    """
    total = sum(weights)
    counts = [int(round(w / total * num_tokens)) for w in weights]
    diff = num_tokens - sum(counts)
    if diff != 0:
        largest_idx = max(range(len(counts)), key=lambda i: weights[i])
        counts[largest_idx] += diff
    return counts


def split_turn_tokens_by_sub_segments(turn: TurnInfo) -> List[List[int]]:
    """The turn's tokens divided among its sub-segments in proportion to their durations.

    Returns a single list when the turn has fewer than two sub-segments.
    """
    sub_segs = turn.sub_segments
    speech_tokens = turn.speech_token_ids

    if not sub_segs or len(sub_segs) <= 1:
        return [list(speech_tokens)]

    durations = [ss["end"] - ss["start"] for ss in sub_segs]
    if sum(durations) <= 0 or not speech_tokens:
        return [list(speech_tokens)]

    slices = []
    offset = 0
    for count in proportional_token_counts(durations, len(speech_tokens)):
        slices.append(speech_tokens[offset:offset + count])
        offset += count
    return slices


def assemble_turn_from_sub_segments(
    sub_seg_allocations: List[List[int]],
    sub_seg_onsets: List[int],
    silence_token_id: int,
) -> Tuple[List[int], List[Dict], int, int]:
    """Join sub-segments placed at ``sub_seg_onsets`` into one token list.

    Gaps between consecutive sub-segments are filled with silence.

    Returns ``(tokens, sub_segments_meta, turn_onset, turn_end)`` where the
    metadata is a list of ``{'start': sec, 'end': sec}`` in dialogue time.
    """
    assert len(sub_seg_allocations) == len(sub_seg_onsets)
    assert len(sub_seg_allocations) > 0

    turn_onset = sub_seg_onsets[0]
    tokens: List[int] = []
    meta: List[Dict] = []
    current_pos = turn_onset

    for seg_tokens, onset in zip(sub_seg_allocations, sub_seg_onsets):
        gap = onset - current_pos
        if gap > 0:
            tokens.extend([silence_token_id] * gap)
        tokens.extend(seg_tokens)
        meta.append({
            "start": onset / TOKEN_RATE_HZ,
            "end": (onset + len(seg_tokens)) / TOKEN_RATE_HZ,
        })
        current_pos = onset + len(seg_tokens)

    return tokens, meta, turn_onset, turn_onset + len(tokens)


def with_sub_segment_layout(turn: TurnInfo, tokens: List[int], sub_segments_meta: List[Dict]) -> TurnInfo:
    """Copy of ``turn`` carrying an assembled token list and its sub-segment metadata."""
    new_turn = copy.deepcopy(turn)
    new_turn.speech_token_ids = tokens
    new_turn.duration = len(tokens)
    new_turn.sub_segments = sub_segments_meta
    return new_turn


def build_host_turn_with_pause(
    host: PlacedTurn,
    bc_onset: int,
    pause_duration: int,
    silence_token_id: int,
) -> PlacedTurn:
    """Copy of ``host`` with ``pause_duration`` steps of silence inserted at ``bc_onset``.

    The copy's sub-segments become the part before and the part after the pause.
    """
    host_start = host.onset_step
    original_tokens = host.turn_info.speech_token_ids

    split_offset = max(0, min(bc_onset - host_start, len(original_tokens)))
    tokens_before = original_tokens[:split_offset]
    tokens_after = original_tokens[split_offset:]

    split_sec = (host_start + split_offset) / TOKEN_RATE_HZ
    pause_end_sec = split_sec + pause_duration / TOKEN_RATE_HZ

    new_sub_segments = []
    if tokens_before:
        new_sub_segments.append({"start": host_start / TOKEN_RATE_HZ, "end": split_sec})
    if tokens_after:
        new_sub_segments.append({
            "start": pause_end_sec,
            "end": pause_end_sec + len(tokens_after) / TOKEN_RATE_HZ,
        })

    return PlacedTurn(
        turn_info=with_sub_segment_layout(
            host.turn_info,
            tokens_before + [silence_token_id] * pause_duration + tokens_after,
            new_sub_segments,
        ),
        onset_step=host.onset_step,
        gap_to_previous=host.gap_to_previous,
        bc_offset_from_host_start=host.bc_offset_from_host_start,
    )


def split_turn_into_sub_segments(turn: TurnInfo) -> List[Dict]:
    """Sub-segments of a turn as dicts, tokens allocated in proportion to block durations.

    Each dict has ``index``, ``token_ids``, ``original_offset`` (token offset
    from the turn start), ``duration`` (tokens), ``start_sec`` and ``end_sec``.
    """
    sub_segs = turn.sub_segments
    speech_tokens = turn.speech_token_ids

    def whole_turn(start_sec: float, end_sec: float) -> List[Dict]:
        return [{
            "index": 0,
            "token_ids": list(speech_tokens),
            "original_offset": 0,
            "duration": len(speech_tokens),
            "start_sec": start_sec,
            "end_sec": end_sec,
        }]

    if not sub_segs:
        return whole_turn(0.0, len(speech_tokens) / TOKEN_RATE_HZ)
    if len(sub_segs) == 1:
        return whole_turn(sub_segs[0]["start"], sub_segs[0]["end"])

    durations_sec = [ss["end"] - ss["start"] for ss in sub_segs]
    if sum(durations_sec) <= 0:
        return whole_turn(sub_segs[0]["start"], sub_segs[-1]["end"])

    result = []
    token_offset = 0
    counts = proportional_token_counts(durations_sec, len(speech_tokens))
    for i, (count, ss) in enumerate(zip(counts, sub_segs)):
        seg_tokens = speech_tokens[token_offset:token_offset + count]
        result.append({
            "index": i,
            "token_ids": list(seg_tokens),
            "original_offset": token_offset,
            "duration": len(seg_tokens),
            "start_sec": ss["start"],
            "end_sec": ss["end"],
        })
        token_offset += count
    return result


def compute_original_sub_segment_onsets(turn_onset: int, sub_segments: List[Dict]) -> List[int]:
    """Onset step of each sub-segment, keeping the gaps recorded in their metadata."""
    if len(sub_segments) <= 1:
        return [turn_onset]

    onsets = [turn_onset]
    current_pos = turn_onset + sub_segments[0]["duration"]
    for i in range(1, len(sub_segments)):
        gap_sec = sub_segments[i]["start_sec"] - sub_segments[i - 1]["end_sec"]
        onset = current_pos + max(0, int(round(gap_sec * TOKEN_RATE_HZ)))
        onsets.append(onset)
        current_pos = onset + sub_segments[i]["duration"]
    return onsets
