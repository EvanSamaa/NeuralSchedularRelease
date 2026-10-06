"""Turn-ID assignment from the VAD segments of both channels.

A *turn* is a run of consecutive same-speaker VAD segments (in start-time order
over both channels). A turn that is at least half covered by an earlier turn
inherits that turn's id; this is how a backchannel ends up sharing the id of the
turn it overlaps. Ids are shared across both speakers and numbered contiguously
from 0 in order of first appearance.
"""

from typing import Dict, List


def merge_adjacent_same_speaker_segments(segments: List[Dict]) -> List[Dict]:
    """Merge neighbouring segments of the same channel in a start-sorted list.

    Each merged segment keeps the originals under ``sub_segments``.
    """
    if not segments:
        return []

    merged = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "channel": segments[0]["channel"],
        "sub_segments": [{"start": segments[0]["start"], "end": segments[0]["end"]}],
    }

    for seg in segments[1:]:
        if seg["channel"] == current["channel"]:
            current["end"] = max(current["end"], seg["end"])
            current["sub_segments"].append({"start": seg["start"], "end": seg["end"]})
        else:
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "channel": seg["channel"],
                "sub_segments": [{"start": seg["start"], "end": seg["end"]}],
            }

    merged.append(current)
    return merged


def compute_overlap_ratio(seg_a: Dict, seg_b: Dict) -> float:
    """Fraction of ``seg_b`` that is covered by ``seg_a``."""
    overlap_start = max(seg_a["start"], seg_b["start"])
    overlap_end = min(seg_a["end"], seg_b["end"])
    overlap_duration = max(0, overlap_end - overlap_start)
    seg_b_duration = seg_b["end"] - seg_b["start"]
    if seg_b_duration <= 0:
        return 0.0
    return overlap_duration / seg_b_duration


def assign_turn_ids(
    vad_timestamps_left_tokens: List[Dict[str, int]],
    vad_timestamps_right_tokens: List[Dict[str, int]],
    overlap_threshold: float = 0.5,
) -> List[Dict]:
    """Assign turn ids to the VAD segments (token units) of channels A (left) and B (right).

    1. Pool both channels and sort by ``(start, channel)``.
    2. Merge adjacent same-speaker segments into turns.
    3. A turn covered by at least ``overlap_threshold`` of its length by an
       earlier turn takes that turn's id (first match wins); otherwise a new id.
    4. Renumber ids contiguously.

    Returns the turns as dicts with ``start``, ``end``, ``channel``,
    ``sub_segments`` and ``turn_id``.
    """
    all_segments = []
    for ts in vad_timestamps_left_tokens:
        all_segments.append({"start": ts["start"], "end": ts["end"], "channel": "A"})
    for ts in vad_timestamps_right_tokens:
        all_segments.append({"start": ts["start"], "end": ts["end"], "channel": "B"})

    all_segments.sort(key=lambda x: (x["start"], x["channel"]))

    merged_segments = merge_adjacent_same_speaker_segments(all_segments)

    next_turn_id = 0
    assigned = []

    for seg in merged_segments:
        assigned_turn_id = None

        for prev_seg in assigned:
            if compute_overlap_ratio(prev_seg, seg) >= overlap_threshold:
                assigned_turn_id = prev_seg["turn_id"]
                break

        if assigned_turn_id is None:
            assigned_turn_id = next_turn_id
            next_turn_id += 1

        seg["turn_id"] = assigned_turn_id
        assigned.append(seg)

    unique_ids = sorted(set(s["turn_id"] for s in assigned))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    for seg in assigned:
        seg["turn_id"] = id_map[seg["turn_id"]]

    return assigned
