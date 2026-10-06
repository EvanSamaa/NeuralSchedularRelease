"""Turn-ID assignment for labelled sentences.

Consecutive same-speaker sentences form one segment. A segment that is covered
for at least half of its duration by an earlier segment inherits that
segment's turn id, which is how a backchannel ends up sharing a turn id with
the speech it overlaps.
"""

from typing import Dict, List, Tuple

from ..constants import SPEECH_LABEL


def _sub_segment(seg: Dict) -> Dict:
    return {
        "start": seg["start"],
        "end": seg["end"],
        "speaker_id": seg["speaker_id"],
        "text": seg["text"],
    }


def merge_adjacent_same_speaker(segments: List[Dict]) -> List[Dict]:
    """Merge runs of consecutive same-speaker segments, keeping the originals
    under ``sub_segments``."""
    if not segments:
        return []

    def make_merged(seg):
        return {
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": seg["speaker_id"],
            "sub_segments": [_sub_segment(seg)],
        }

    merged = []
    current = make_merged(segments[0])
    for seg in segments[1:]:
        if seg["speaker_id"] == current["speaker_id"]:
            current["end"] = max(current["end"], seg["end"])
            current["sub_segments"].append(_sub_segment(seg))
        else:
            merged.append(current)
            current = make_merged(seg)
    merged.append(current)
    return merged


def compute_overlap_ratio(seg_a: Dict, seg_b: Dict) -> float:
    """Fraction of ``seg_b`` covered by ``seg_a``."""
    overlap = max(0.0, min(seg_a["end"], seg_b["end"]) - max(seg_a["start"], seg_b["start"]))
    duration_b = seg_b["end"] - seg_b["start"]
    return overlap / duration_b if duration_b > 0 else 0.0


def assign_turn_ids(merged_segments: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
    """Give each segment a ``turn_id`` (first earlier segment that covers at
    least ``overlap_threshold`` of it, else a new id), then renumber
    contiguously. Modifies and returns the segments."""
    next_turn_id = 0
    assigned = []
    for seg in merged_segments:
        assigned_id = None
        for prev in assigned:
            if compute_overlap_ratio(prev, seg) >= overlap_threshold:
                assigned_id = prev["turn_id"]
                break
        if assigned_id is None:
            assigned_id = next_turn_id
            next_turn_id += 1
        seg["turn_id"] = assigned_id
        assigned.append(seg)

    unique_ids = sorted(set(s["turn_id"] for s in assigned))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    for seg in assigned:
        seg["turn_id"] = id_map[seg["turn_id"]]
    return assigned


def build_turn_records(
    sentences_0: List[Dict],
    sentences_1: List[Dict],
    backchannel_lookup: Dict[Tuple, str],
) -> List[Dict]:
    """Label every sentence and attach a turn id.

    ``backchannel_lookup`` maps ``(speaker_id, start, end)`` to
    ``"backchannel-<Class>"``; sentences not in it are labelled ``"Speech"``.
    Returns ``[{text, start, end, speaker_id, turn_id, label}]`` sorted by start.
    """
    all_sentences = []
    for speaker_id, sentences in ((0, sentences_0), (1, sentences_1)):
        for sent in sentences:
            all_sentences.append({
                "start": sent["start"],
                "end": sent["end"],
                "speaker_id": speaker_id,
                "text": sent["transcript"],
            })

    all_sentences.sort(key=lambda x: (x["start"], x["speaker_id"]))
    merged_with_turns = assign_turn_ids(merge_adjacent_same_speaker(all_sentences))

    records = []
    for seg in merged_with_turns:
        for sub in seg["sub_segments"]:
            key = (sub["speaker_id"], sub["start"], sub["end"])
            records.append({
                "text": sub["text"],
                "start": sub["start"],
                "end": sub["end"],
                "speaker_id": sub["speaker_id"],
                "turn_id": seg["turn_id"],
                "label": backchannel_lookup.get(key, SPEECH_LABEL),
            })

    records.sort(key=lambda x: x["start"])
    return records
