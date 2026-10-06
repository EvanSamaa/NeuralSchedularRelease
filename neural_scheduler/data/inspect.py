"""Inspect processed JSONL records.

``show`` prints the script regions, the interleaved dialogue with its duration
labels, a duration check and the turn structure::

    python -m neural_scheduler.data.inspect show --input_file train.jsonl --num_samples 2

``diagnose`` re-derives the turn assignment of a record from its stored VAD
timestamps, step by step, and compares it with the stored turn boundaries::

    python -m neural_scheduler.data.inspect diagnose --input_file train.jsonl --sample_idx 0
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

from ..constants import BASE_VOCAB_SIZE, MAX_TURNS, SILENCE_TOKEN_ID, TOKEN_RATE_HZ
from ..vocab import STRUCTURAL_TOKENS, end_turn_token, start_turn_token
from .durations import validate_remaining_durations
from .turns import compute_overlap_ratio, merge_adjacent_same_speaker_segments

_STRUCTURAL_KEYS = [
    "start_script_a", "end_script_a", "start_script_b", "end_script_b", "start_dialogue", "end_dialogue",
]
_DISPLAY = dict(zip(_STRUCTURAL_KEYS, STRUCTURAL_TOKENS))
_DISPLAY["silence"] = "SIL"


def default_special_token_ids(max_turns: int = MAX_TURNS) -> Dict[str, int]:
    """Special-token ids of the released vocabulary, without loading a tokenizer."""
    ids = {key: BASE_VOCAB_SIZE + i for i, key in enumerate(_STRUCTURAL_KEYS)}
    ids["silence"] = SILENCE_TOKEN_ID
    first_turn = BASE_VOCAB_SIZE + len(_STRUCTURAL_KEYS)
    for i in range(max_turns):
        ids[f"start_turn_{i}"] = first_turn + 2 * i
        ids[f"end_turn_{i}"] = first_turn + 2 * i + 1
    return ids


def load_special_token_ids(tokenizer_path: Optional[str] = None) -> Dict[str, int]:
    """Special-token ids, from ``tokenizer_path`` if given, else the released vocabulary."""
    if tokenizer_path is None:
        return default_special_token_ids()
    from ..vocab import get_special_token_ids
    from .sequence import load_text_tokenizer

    return get_special_token_ids(load_text_tokenizer(tokenizer_path))


def build_id_to_name(special_tokens: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in special_tokens.items() if isinstance(v, int)}


def format_token(token_id: int, id_to_name: Dict[int, str]) -> str:
    """Human-readable form of a vocab id."""
    name = id_to_name.get(token_id)
    if name is None:
        return f"tok_{token_id}"
    if name.startswith("start_turn_"):
        return start_turn_token(name[len("start_turn_"):])
    if name.startswith("end_turn_"):
        return end_turn_token(name[len("end_turn_"):])
    return _DISPLAY.get(name, f"<{name}>")


def _rule(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def visualize_full_statistics(sample: Dict) -> None:
    input_ids = sample["input_ids"]
    gen_start = sample["generation_start_index"]
    gen_end = sample["generation_end_index"]
    turn_boundaries = sample.get("turn_boundaries", [])

    _rule("OVERALL STATISTICS")
    print(f"  Total sequence length:  {len(input_ids)}")
    print(f"  Script region:          {gen_start} tokens (pos 0 to {gen_start - 1})")
    print(f"  Dialogue region:        {gen_end - gen_start} tokens (pos {gen_start} to {gen_end - 1})")
    if "window_start_sec" in sample and "window_end_sec" in sample:
        print(f"  Window:                 {sample['window_start_sec']:.2f}s - {sample['window_end_sec']:.2f}s")
    print(f"  Source:                 example {sample.get('source_file', '?')}, window {sample.get('window_idx', '?')}")
    if turn_boundaries:
        print(f"  Number of turns:        {len(set(tb['turn_id'] for tb in turn_boundaries))}")


def visualize_script_region(
    sample: Dict, special_tokens: Dict[str, int], id_to_name: Dict[int, str], max_tokens_per_region: int = 200
) -> None:
    """Print script A and script B token by token, with the turn each token belongs to."""
    input_ids = sample["input_ids"]
    script_tokens = input_ids[:sample["generation_start_index"]]
    silence_id = special_tokens["silence"]

    try:
        regions = [
            ("SCRIPT A", script_tokens.index(special_tokens["start_script_a"]),
             script_tokens.index(special_tokens["end_script_a"]) + 1),
            ("SCRIPT B", script_tokens.index(special_tokens["start_script_b"]),
             script_tokens.index(special_tokens["end_script_b"]) + 1),
        ]
    except ValueError:
        print("  Error: Could not find script boundaries")
        return

    for region_name, region_start, region_end in regions:
        region = input_ids[region_start:region_end]
        turn_count = sum(1 for t in region if id_to_name.get(t, "").startswith("start_turn_"))
        _rule(f"{region_name} REGION ({len(region)} tokens, {turn_count} turns)")

        display_len = min(len(region), max_tokens_per_region)
        current_turn = None
        print(f"{'Pos':<6} {'Token ID':<10} {'Token':<30} {'Type':<20}")
        print("-" * 80)

        for i in range(display_len):
            token_id = region[i]
            name = id_to_name.get(token_id)
            if token_id == silence_id:
                type_str = "SILENCE"
            elif name is None:
                type_str = f"SPEECH (turn {current_turn})" if current_turn else "SPEECH"
            elif name.startswith("start_turn_"):
                current_turn = name[len("start_turn_"):]
                type_str = f"TURN START ({current_turn})"
            elif name.startswith("end_turn_"):
                type_str = f"TURN END ({current_turn})"
                current_turn = None
            elif "script" in name:
                type_str = "SCRIPT MARKER"
            else:
                type_str = "SPECIAL"
            print(f"{region_start + i:<6} {token_id:<10} {format_token(token_id, id_to_name):<30} {type_str:<20}")

        if len(region) > display_len:
            print(f"... ({len(region) - display_len} more tokens)")


def visualize_dialogue_region(
    sample: Dict, special_tokens: Dict[str, int], id_to_name: Dict[int, str], max_display: int = 80
) -> None:
    """Print the head and tail of the interleaved dialogue with duration labels."""
    dialogue_tokens = sample["input_ids"][sample["generation_start_index"]:sample["generation_end_index"]]
    remaining_durations = sample["remaining_durations"]
    silence_id = special_tokens["silence"]

    display_len = min(len(dialogue_tokens), max_display)
    display_len -= display_len % 2

    _rule(f"DIALOGUE REGION ({len(dialogue_tokens)} tokens, showing first {display_len})")
    print(f"\n{'Pos':<6} {'Chan':<5} {'Token ID':<10} {'Token':<15} {'Dur':<6} {'Description':<30}")
    print("-" * 80)

    def row(i: int, describe: bool) -> str:
        token_id = dialogue_tokens[i]
        dur = remaining_durations[i] if i < len(remaining_durations) else "?"
        line = f"{i:<6} {'A' if i % 2 == 0 else 'B':<5} {token_id:<10} {format_token(token_id, id_to_name):<15} {dur:<6}"
        if describe:
            kind = "Silence" if token_id == silence_id else "Speech"
            line += f" {kind} (repeats {dur} more)"
        return line

    for i in range(display_len):
        print(row(i, describe=True))

    if len(dialogue_tokens) > display_len:
        print(f"... ({len(dialogue_tokens) - display_len} more tokens)")
        tail = 20
        print(f"\n... last {tail} tokens:")
        for i in range(max(0, len(dialogue_tokens) - tail), len(dialogue_tokens)):
            print(row(i, describe=False))


def visualize_duration_verification(sample: Dict, special_tokens: Dict[str, int]) -> int:
    """Check the duration labels against the dialogue tokens; returns the number of errors."""
    dialogue_tokens = sample["input_ids"][sample["generation_start_index"]:sample["generation_end_index"]]
    silence_id = special_tokens["silence"]

    _rule("DURATION VERIFICATION")
    errors = validate_remaining_durations(dialogue_tokens, sample["remaining_durations"])
    for message in errors[:10]:
        print(f"  {message}")
    print("  PASSED: all duration labels are consistent" if not errors else f"  FAILED: {len(errors)} runs")

    def longest_run(tokens: List[int]) -> int:
        longest = current = 0
        for t in tokens:
            current = current + 1 if t == silence_id else 0
            longest = max(longest, current)
        return longest

    for name, offset in (("A", 0), ("B", 1)):
        channel = dialogue_tokens[offset::2]
        silence = sum(1 for t in channel if t == silence_id)
        print(f"  Channel {name}: {len(channel)} tokens, {silence} silence "
              f"({100 * silence / max(1, len(channel)):.1f}%), longest silence run {longest_run(channel)}")
    return len(errors)


def visualize_turn_summary(sample: Dict) -> None:
    """Print one line per (turn, channel) and the raw VAD segments."""
    turn_boundaries = sample.get("turn_boundaries", [])
    _rule("TURN STRUCTURE")
    if not turn_boundaries:
        print("  No turns found")
        return

    turns_by_id: Dict[int, Dict[str, List[Dict]]] = {}
    for tb in turn_boundaries:
        turns_by_id.setdefault(tb["turn_id"], {}).setdefault(tb["channel"], []).append(tb)

    print(f"  Total unique turn IDs: {len(turns_by_id)}\n")
    print(f"  {'Turn':<6} {'Chan':<5} {'Segments':<45} {'Bar'}")
    print("  " + "-" * 75)
    for tid in sorted(turns_by_id):
        by_channel = turns_by_id[tid]
        for ch in sorted(by_channel):
            entries = by_channel[ch]
            segs_str = ", ".join(f"{e['start_sec']:.2f}-{e['end_sec']:.2f}s" for e in entries)
            total_dur = sum(e["end_sec"] - e["start_sec"] for e in entries)
            bar = "#" * min(30, max(1, int(total_dur * 5)))
            shared = " [backchannel]" if len(by_channel) > 1 else ""
            print(f"  {tid:<6} {ch:<5} {segs_str:<45} {bar} ({total_dur:.2f}s){shared}")

    for name, key in (("A", "vad_timestamps_left"), ("B", "vad_timestamps_right")):
        vad = sample.get(key, [])
        print(f"\n  Raw VAD, channel {name} ({len(vad)} segments):")
        for i, v in enumerate(vad):
            print(f"    [{i}] {v['start']:.2f}s - {v['end']:.2f}s ({v['end'] - v['start']:.2f}s)")


def visualize_sample_full(
    sample: Dict,
    special_tokens: Dict[str, int],
    sample_idx: int,
    script_only: bool = False,
    dialogue_only: bool = False,
    max_script_tokens: int = 200,
    max_dialogue_tokens: int = 80,
) -> None:
    """Print the full breakdown of one record."""
    id_to_name = build_id_to_name(special_tokens)
    print("\n" + "#" * 80)
    print(f"  SAMPLE {sample_idx}")
    print("#" * 80)

    visualize_full_statistics(sample)
    if not dialogue_only:
        visualize_script_region(sample, special_tokens, id_to_name, max_script_tokens)
    if not script_only:
        visualize_dialogue_region(sample, special_tokens, id_to_name, max_dialogue_tokens)
        visualize_duration_verification(sample, special_tokens)
    visualize_turn_summary(sample)


def vad_seconds_to_token_indices(vad_seconds: List[Dict[str, float]]) -> List[Dict[str, int]]:
    """Stored VAD seconds back to token indices (floor), as preprocessing did from samples."""
    return [
        {"start": int(ts["start"] * TOKEN_RATE_HZ), "end": int(ts["end"] * TOKEN_RATE_HZ)}
        for ts in vad_seconds
    ]


def diagnose_sample(sample: Dict, sample_idx: int, overlap_threshold: float = 0.5) -> None:
    """Trace the turn assignment of one record from its stored VAD timestamps."""
    _rule(f"TURN DIAGNOSTIC FOR SAMPLE {sample_idx}")

    channels = {}
    for name, key in (("A", "vad_timestamps_left"), ("B", "vad_timestamps_right")):
        seconds = sample.get(key, [])
        tokens = vad_seconds_to_token_indices(seconds)
        channels[name] = tokens
        print(f"\nChannel {name}: {len(seconds)} VAD segments (seconds -> token span)")
        for i, (sec, tok) in enumerate(zip(seconds, tokens)):
            print(f"  [{i}] {sec['start']:.6f}s - {sec['end']:.6f}s -> {tok['start']}-{tok['end']} "
                  f"({tok['end'] - tok['start']} tokens)")

    all_segments = [
        {"start": tok["start"], "end": tok["end"], "channel": name}
        for name in ("A", "B") for tok in channels[name]
    ]
    all_segments.sort(key=lambda x: (x["start"], x["channel"]))
    print(f"\nSorted by (start, channel): {len(all_segments)} segments")
    for i, seg in enumerate(all_segments):
        print(f"  [{i}] Ch {seg['channel']}: tokens {seg['start']}-{seg['end']}")
    if not all_segments:
        print("  No segments")
        return

    merged = merge_adjacent_same_speaker_segments(all_segments)
    print(f"\nAfter merging adjacent same-speaker segments: {len(merged)} turns")
    for i, m in enumerate(merged):
        subs = m["sub_segments"]
        print(f"  [{i}] Ch {m['channel']}: tokens {m['start']}-{m['end']}, sub-segments "
              + ", ".join(f"{s['start']}-{s['end']}" for s in subs))
        for j in range(len(subs)):
            for k in range(j + 1, len(subs)):
                if max(subs[j]["start"], subs[k]["start"]) < min(subs[j]["end"], subs[k]["end"]):
                    print(f"       WARNING: sub-segments [{j}] and [{k}] overlap")

    print(f"\nOverlap rule (threshold {overlap_threshold:.0%}):")
    next_turn_id = 0
    assigned = []
    for seg in merged:
        turn_id = None
        for prev_seg in assigned:
            ratio = compute_overlap_ratio(prev_seg, seg)
            if ratio >= overlap_threshold:
                turn_id = prev_seg["turn_id"]
                print(f"  Ch {seg['channel']} {seg['start']}-{seg['end']}: {ratio:.1%} covered by "
                      f"Ch {prev_seg['channel']} turn {turn_id} -> shares turn {turn_id}")
                break
        if turn_id is None:
            turn_id = next_turn_id
            next_turn_id += 1
            print(f"  Ch {seg['channel']} {seg['start']}-{seg['end']}: new turn {turn_id}")
        seg["turn_id"] = turn_id
        assigned.append(seg)

    id_map = {old: new for new, old in enumerate(sorted(set(s["turn_id"] for s in assigned)))}
    for seg in assigned:
        seg["turn_id"] = id_map[seg["turn_id"]]

    print("\nScript content per channel:")
    for name in ("A", "B"):
        print(f"  Channel {name}: SIL")
        for seg in (s for s in assigned if s["channel"] == name):
            subs = " SIL ".join(f"[{s['start']}-{s['end']}]" for s in seg["sub_segments"])
            dropped = "  (no marker tokens: left out of the script)" if seg["turn_id"] >= MAX_TURNS else ""
            print(f"    <|start-turn-{seg['turn_id']}|> {subs} <|end-turn-{seg['turn_id']}|> SIL{dropped}")

    reconstructed = sorted(
        (
            {"turn_id": seg["turn_id"], "channel": seg["channel"],
             "start_sec": sub["start"] / TOKEN_RATE_HZ, "end_sec": sub["end"] / TOKEN_RATE_HZ}
            for seg in assigned for sub in seg["sub_segments"]
        ),
        key=lambda x: (x["turn_id"], x["start_sec"]),
    )
    stored = sample.get("turn_boundaries", [])
    print(f"\nStored turn_boundaries: {len(stored)} entries; reconstructed: {len(reconstructed)} entries")
    print("  MATCH" if stored == reconstructed else "  MISMATCH between stored and reconstructed boundaries")

    channels_by_turn: Dict[int, set] = {}
    for seg in assigned:
        channels_by_turn.setdefault(seg["turn_id"], set()).add(seg["channel"])
    shared = sorted(t for t, chans in channels_by_turn.items() if len(chans) > 1)
    print(f"Turn ids present on both channels (backchannels): {shared if shared else 'none'}")


def _iter_selected(input_file: str, indices: Optional[List[int]], start: int, count: int):
    wanted = set(indices) if indices is not None else None
    last = max(wanted) if wanted else start + count - 1
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f):
            if line_num > last:
                break
            if (wanted is not None and line_num not in wanted) or (wanted is None and line_num < start):
                continue
            try:
                yield line_num, json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: could not parse line {line_num}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect processed scheduler JSONL records.")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="Print script, dialogue, duration check and turn structure")
    show.add_argument("--input_file", required=True)
    show.add_argument("--num_samples", type=int, default=1, help="Number of records from the start of the file")
    show.add_argument("--sample_indices", type=int, nargs="+", default=None, help="Specific 0-based record indices")
    show.add_argument("--script_only", action="store_true")
    show.add_argument("--dialogue_only", action="store_true")
    show.add_argument("--max_script_tokens", type=int, default=200)
    show.add_argument("--max_dialogue_tokens", type=int, default=80)
    show.add_argument("--tokenizer_path", default=None,
                      help="Resolve special-token ids from this tokenizer instead of the released vocabulary")

    diagnose = sub.add_parser("diagnose", help="Trace the turn assignment of a record")
    diagnose.add_argument("--input_file", required=True)
    diagnose.add_argument("--sample_idx", type=int, default=0)
    diagnose.add_argument("--num_samples", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.input_file):
        sys.exit(f"Error: file not found: {args.input_file}")

    shown = 0
    if args.command == "show":
        special_tokens = load_special_token_ids(args.tokenizer_path)
        for line_num, sample in _iter_selected(args.input_file, args.sample_indices, 0, args.num_samples):
            try:
                visualize_sample_full(
                    sample, special_tokens, line_num,
                    script_only=args.script_only, dialogue_only=args.dialogue_only,
                    max_script_tokens=args.max_script_tokens, max_dialogue_tokens=args.max_dialogue_tokens,
                )
                shown += 1
            except Exception:
                print(f"Error visualizing sample {line_num}:\n{traceback.format_exc()}")
    else:
        for line_num, sample in _iter_selected(args.input_file, None, args.sample_idx, args.num_samples):
            diagnose_sample(sample, line_num)
            shown += 1
    print(f"\nInspected {shown} records from {args.input_file}")


if __name__ == "__main__":
    main()
