"""Reading turns out of the script region of a scheduler input sequence.

The input is::

    <|start-of-script-A|> SIL <|start-turn-N|> speech [SIL speech ...] <|end-turn-N|> SIL ... <|end-of-script-A|>
    <|start-of-script-B|> ...                                                                  <|end-of-script-B|>
    <|start-of-dialogue|> ...

A turn id that appears in one script is a primary turn. A turn id that appears
in both scripts is a primary turn on one channel and a backchannel on the other.
"""

from typing import Dict, List, Optional, Tuple

from ..constants import MAX_TURNS, TOKEN_RATE_HZ
from .types import TurnInfo

_SCRIPT_REGIONS = (
    ("A", "start_script_a", "end_script_a"),
    ("B", "start_script_b", "end_script_b"),
)


def _turn_marker_lookup(special_tokens: Dict[str, int]) -> Dict[int, Tuple[str, int]]:
    """Map a turn-marker vocab id to ``('start' | 'end', turn_id)``."""
    lookup = {}
    for i in range(MAX_TURNS):
        start_id = special_tokens.get(f"start_turn_{i}")
        end_id = special_tokens.get(f"end_turn_{i}")
        if start_id is not None:
            lookup.setdefault(start_id, ("start", i))
        if end_id is not None:
            lookup.setdefault(end_id, ("end", i))
    return lookup


def _script_region(input_ids: List[int], special_tokens: Dict[str, int], start_key: str, end_key: str) -> Optional[List[int]]:
    try:
        start_idx = input_ids.index(special_tokens[start_key]) + 1
        end_idx = input_ids.index(special_tokens[end_key])
    except ValueError:
        return None
    return input_ids[start_idx:end_idx]


def generation_start_index(input_ids: List[int], special_tokens: Dict[str, int]) -> int:
    """Index of the first dialogue token (just after ``<|start-of-dialogue|>``)."""
    try:
        return input_ids.index(special_tokens["start_dialogue"]) + 1
    except ValueError:
        return len(input_ids)


def compute_turn_boundaries_from_script_only(
    input_ids: List[int],
    special_tokens: Dict[str, int],
) -> List[Dict]:
    """One ``{turn_id, channel, start_sec, end_sec}`` entry per speech block of each turn.

    Times are positions inside the script, not dialogue time: each channel has
    its own running counter that advances over the speech and silence tokens
    inside turns, so the gaps between a turn's speech blocks are preserved.
    """
    silence_id = special_tokens["silence"]
    markers = _turn_marker_lookup(special_tokens)
    turn_boundaries = []

    for channel_label, start_key, end_key in _SCRIPT_REGIONS:
        script_region = _script_region(input_ids, special_tokens, start_key, end_key)
        if script_region is None:
            continue

        current_turn_id = None
        turn_position = 0  # position inside the current turn, speech and silence
        in_speech = False
        sub_seg_start_pos = 0
        sub_segments = []
        has_speech_in_seg = False
        channel_offset = 0  # tokens consumed by earlier turns of this channel

        def close_sub_segment():
            sub_segments.append({
                "start_sec": (channel_offset + sub_seg_start_pos) / TOKEN_RATE_HZ,
                "end_sec": (channel_offset + turn_position) / TOKEN_RATE_HZ,
            })

        def emit_turn():
            for ss in sub_segments:
                turn_boundaries.append({
                    "turn_id": current_turn_id,
                    "channel": channel_label,
                    "start_sec": ss["start_sec"],
                    "end_sec": ss["end_sec"],
                })

        for tok_id in script_region:
            marker = markers.get(tok_id)
            if marker is not None:
                kind, marker_turn = marker
                if kind == "start":
                    if current_turn_id is not None:
                        if has_speech_in_seg and in_speech:
                            close_sub_segment()
                        emit_turn()
                        channel_offset += turn_position
                    current_turn_id = marker_turn
                    turn_position = 0
                    sub_seg_start_pos = 0
                    sub_segments = []
                    in_speech = False
                    has_speech_in_seg = False
                elif current_turn_id == marker_turn:
                    if has_speech_in_seg and in_speech:
                        close_sub_segment()
                    emit_turn()
                    channel_offset += turn_position
                    current_turn_id = None
                    sub_segments = []
                    turn_position = 0
                    has_speech_in_seg = False
                    in_speech = False
                continue

            if current_turn_id is None:
                continue
            if tok_id == silence_id:
                if in_speech and has_speech_in_seg:
                    close_sub_segment()
                    in_speech = False
                    has_speech_in_seg = False
            else:
                if not in_speech:
                    sub_seg_start_pos = turn_position
                    in_speech = True
                has_speech_in_seg = True
            turn_position += 1

    turn_boundaries.sort(key=lambda x: (x["turn_id"], x["start_sec"]))
    return turn_boundaries


def extract_script_turn_content(
    input_ids: List[int],
    special_tokens: Dict[str, int],
) -> Dict[Tuple[str, int], List[int]]:
    """Speech tokens of each turn, keyed by ``(channel, turn_id)``. Silence is dropped."""
    result = {}
    silence_id = special_tokens["silence"]
    markers = _turn_marker_lookup(special_tokens)

    for channel_label, start_key, end_key in _SCRIPT_REGIONS:
        script_region = _script_region(input_ids, special_tokens, start_key, end_key)
        if script_region is None:
            continue

        current_turn_id = None
        current_tokens = []
        for token_id in script_region:
            marker = markers.get(token_id)
            if marker is not None:
                kind, marker_turn = marker
                if kind == "start":
                    if current_turn_id is not None:
                        result[(channel_label, current_turn_id)] = current_tokens
                    current_turn_id = marker_turn
                    current_tokens = []
                elif current_turn_id == marker_turn:
                    result[(channel_label, current_turn_id)] = current_tokens
                    current_turn_id = None
                    current_tokens = []
                continue
            if current_turn_id is not None and token_id != silence_id:
                current_tokens.append(token_id)

    return result


def parse_turn_structure(
    input_ids: List[int],
    special_tokens: Dict[str, int],
    turn_boundaries: List[Dict],
) -> Tuple[List[TurnInfo], List[TurnInfo]]:
    """Split the script's turns into ``(primary_turns, backchannel_turns)``.

    When a turn id is present on both channels, the channel with the larger
    total block duration is the primary turn and the other is a backchannel
    hosted by it.
    """
    turns_by_id = {}
    for tb in turn_boundaries:
        turns_by_id.setdefault(tb["turn_id"], []).append(tb)

    script_content = extract_script_turn_content(input_ids, special_tokens)

    def make_turn(tid: int, channel: str, entries: List[Dict]) -> TurnInfo:
        speech_tokens = script_content.get((channel, tid), [])
        return TurnInfo(
            turn_id=tid,
            channel=channel,
            start_token=int(entries[0]["start_sec"] * TOKEN_RATE_HZ),
            end_token=int(entries[-1]["end_sec"] * TOKEN_RATE_HZ),
            duration=len(speech_tokens),
            sub_segments=[{"start": e["start_sec"], "end": e["end_sec"]} for e in entries],
            speech_token_ids=speech_tokens,
        )

    primary_turns = []
    backchannel_turns = []

    for tid in sorted(turns_by_id.keys()):
        entries = turns_by_id[tid]
        channels_in_turn = set(e["channel"] for e in entries)

        if len(channels_in_turn) == 1:
            primary_turns.append(make_turn(tid, entries[0]["channel"], entries))
            continue

        channel_durations = {}
        for e in entries:
            ch = e["channel"]
            channel_durations[ch] = channel_durations.get(ch, 0) + (e["end_sec"] - e["start_sec"])
        primary_channel = max(channel_durations, key=channel_durations.get)

        for ch in channels_in_turn:
            turn = make_turn(tid, ch, [e for e in entries if e["channel"] == ch])
            if ch == primary_channel:
                primary_turns.append(turn)
            else:
                turn.is_backchannel = True
                turn.host_turn_id = tid
                backchannel_turns.append(turn)

    primary_turns.sort(key=lambda t: t.start_token)
    backchannel_turns.sort(key=lambda t: t.start_token)
    return primary_turns, backchannel_turns


def _is_in_script_region(input_ids: List[int], position: int, channel: str, special_tokens: Dict[str, int]) -> bool:
    suffix = "a" if channel == "A" else "b"
    try:
        start_idx = input_ids.index(special_tokens[f"start_script_{suffix}"])
        end_idx = input_ids.index(special_tokens[f"end_script_{suffix}"])
    except ValueError:
        return False
    return start_idx < position < end_idx


def strip_backchannels_from_script(
    input_ids: List[int],
    backchannel_turns: List[TurnInfo],
    special_tokens: Dict[str, int],
) -> Tuple[List[int], int]:
    """Remove the backchannel turns from the script so pass 1 sees primary turns only.

    Each removed span runs from the turn's start marker through its end marker
    and one trailing silence token. Returns ``(new_input_ids, generation_start_index)``.
    """
    bc_turn_ids_by_channel = {}
    for bc in backchannel_turns:
        bc_turn_ids_by_channel.setdefault(bc.channel, set()).add(bc.turn_id)

    silence_id = special_tokens["silence"]
    all_bc_turn_ids = set()
    for tid_set in bc_turn_ids_by_channel.values():
        all_bc_turn_ids.update(tid_set)

    new_ids = []
    i = 0
    while i < len(input_ids):
        token_id = input_ids[i]
        skip = False
        for tid in all_bc_turn_ids:
            if token_id != special_tokens.get(f"start_turn_{tid}"):
                continue
            end_turn_marker = special_tokens.get(f"end_turn_{tid}")
            in_script = any(
                tid in bc_turn_ids_by_channel.get(ch_label, set())
                and _is_in_script_region(input_ids, i, ch_label, special_tokens)
                for ch_label in ("A", "B")
            )
            if in_script:
                j = i + 1
                while j < len(input_ids) and input_ids[j] != end_turn_marker:
                    j += 1
                if j < len(input_ids):
                    j += 1
                    if j < len(input_ids) and input_ids[j] == silence_id:
                        j += 1
                i = j
                skip = True
                break

        if not skip:
            new_ids.append(input_ids[i])
            i += 1

    return new_ids, generation_start_index(new_ids, special_tokens)
