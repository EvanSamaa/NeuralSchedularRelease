"""Prepare the Full-Duplex-Bench ICC backchannel subset for the backchannel-only pipeline.

Each example of the benchmark is a directory ``<id>/`` with ``input.wav`` (the
speaker) and ``transcription.json`` (word-level timestamps; listener
backchannels appear as ``[TAG]`` words). For every example this script

* splits the words into sentences at pauses, then merges sentences that are
  too short (few words, or brief) into a neighbour;
* removes the tags, recording each as a ground-truth backchannel event on the
  sentence it belongs to;
* cuts ``input.wav`` into one padded clip per sentence;
* writes ``<id>/split_audio/NNN.wav`` and ``<id>/meta.json``, and one record
  per example to a master JSONL consumed by pipeline step 1.

Outputs go to ``--out_dir`` and the benchmark directory is left untouched;
``--in_place`` writes next to ``input.wav`` instead. Paths in the outputs are
relative (to the JSONL's directory in the JSONL, to the example directory in
``meta.json``) unless ``--absolute_paths`` is given.

The benchmark data is not part of this repository; see ``docs/evaluation.md``.

    python -m neural_scheduler.evaluation.fdb.prepare_icc \\
        --dataset_dir FDB/v1_v1.5/dataset/icc_backchannel --out_dir OUT/icc_prepared
"""

import argparse
import json
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_PAUSE_THRESHOLD = 0.6  # seconds between words that starts a new sentence
DEFAULT_PADDING_SEC = 0.15  # silence kept before and after each sentence clip
DEFAULT_MIN_WORDS = 3  # sentences with this many words or fewer are merged
DEFAULT_MIN_DURATION_SEC = 2.0  # sentences shorter than this are merged
MIN_SEGMENT_SAMPLES = 100  # clips shorter than this are dropped


# ── Helpers ───────────────────────────────────────────────────────────

def is_tag(word: str) -> bool:
    return word.startswith("[") and word.endswith("]")


def words_to_sentences(words, pause_threshold=DEFAULT_PAUSE_THRESHOLD):
    """
    Split word-level entries into sentences based on inter-word pauses.
    Tags are filtered out before splitting, but their timestamps and
    positions relative to the surrounding words are recorded.

    Returns:
        sentences  : list of lists of word dicts (tags excluded)
        tag_events : list of dicts with keys:
                       tag        - the tag string e.g. "[BACKCHANNEL]"
                       timestamp  - [start, end] from the original word entry
                       sentence_idx - which sentence (0-indexed) this tag
                                      falls *before* (i.e. at the boundary
                                      between sentence_idx-1 and sentence_idx,
                                      or at the start if sentence_idx==0)
                                      More precisely: the sentence whose first
                                      word comes AFTER this tag.
    """
    if not words:
        return [], []

    tag_events = []

    sentences = []
    current_sentence = []
    pending_tags = []

    def flush_sentence():
        nonlocal current_sentence, pending_tags
        if current_sentence:
            sentences.append(current_sentence)
            current_sentence = []

    for i, word in enumerate(words):
        if is_tag(word["text"]):
            pending_tags.append(word)
            continue

        if current_sentence:
            prev_end   = current_sentence[-1]["timestamp"][1]
            curr_start = word["timestamp"][0]
            if curr_start - prev_end >= pause_threshold:
                for tag in pending_tags:
                    tag_events.append({
                        "tag":          tag["text"],
                        "timestamp":    tag["timestamp"],
                        "sentence_idx": len(sentences) + 1,
                    })
                pending_tags = []
                flush_sentence()
            else:
                for tag in pending_tags:
                    tag_events.append({
                        "tag":          tag["text"],
                        "timestamp":    tag["timestamp"],
                        "sentence_idx": len(sentences),
                    })
                pending_tags = []
        else:
            for tag in pending_tags:
                tag_events.append({
                    "tag":          tag["text"],
                    "timestamp":    tag["timestamp"],
                    "sentence_idx": len(sentences),
                })
            pending_tags = []

        current_sentence.append(word)

    flush_sentence()
    for tag in pending_tags:
        tag_events.append({
            "tag":          tag["text"],
            "timestamp":    tag["timestamp"],
            "sentence_idx": max(0, len(sentences) - 1),
        })

    return sentences, tag_events


def seg_duration(seg):
    """Word-span duration of a segment in seconds."""
    return seg[-1]["timestamp"][1] - seg[0]["timestamp"][0]


def merge_short_sentences(sentences, tag_events,
                          min_words=DEFAULT_MIN_WORDS,
                          min_duration_sec=DEFAULT_MIN_DURATION_SEC,
                          dialogue_id="?"):
    """
    Merge segments that are too short into an adjacent segment.

    A segment is "too short" if EITHER:
      - it has <= min_words words, OR
      - its word-span duration < min_duration_sec

    Merge rules (applied repeatedly until no segment qualifies):
      1. Merge into the NEXT segment if one exists.
      2. Otherwise merge into the PREVIOUS segment.
      3. A lone single-segment dialogue cannot be merged (logged as warning).

    tag_events are remapped so sentence_idx values stay correct after merges.

    Returns:
        merged_sentences : list of lists of word dicts
        remapped_events  : tag_events with updated sentence_idx values
    """
    if not sentences:
        return sentences, tag_events

    segs = [list(s) for s in sentences]

    # old_to_new[old_idx] = current new index; starts as identity
    old_to_new = list(range(len(segs)))

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segs):
            n_words  = len(segs[i])
            duration = seg_duration(segs[i])
            too_short_words    = n_words <= min_words
            too_short_duration = duration < min_duration_sec

            if too_short_words or too_short_duration:
                text_preview = " ".join(w["text"] for w in segs[i])
                reason = []
                if too_short_words:
                    reason.append(f"{n_words}w <= {min_words}")
                if too_short_duration:
                    reason.append(f"{duration:.2f}s < {min_duration_sec}s")
                reason_str = ", ".join(reason)

                if i + 1 < len(segs):
                    # Rule 1: merge forward into next segment
                    merged_into = i + 1
                    segs[merged_into] = segs[i] + segs[merged_into]
                    direction = "forward"
                elif i - 1 >= 0:
                    # Rule 2: merge backward into previous segment
                    merged_into = i - 1
                    segs[merged_into] = segs[merged_into] + segs[i]
                    direction = "backward"
                else:
                    # Rule 3: single segment, cannot merge
                    print(f"  [MERGE] WARNING: dialogue={dialogue_id}: "
                          f"single-segment, cannot merge ({reason_str}): '{text_preview}'")
                    i += 1
                    continue

                print(f"  [MERGE] dialogue={dialogue_id}: "
                      f"seg[{i}] ({reason_str}) '{text_preview}' "
                      f"→ merged {direction} into seg[{merged_into}]")

                segs.pop(i)

                # Rebuild old_to_new to account for the pop at index i.
                # Segments that mapped to i now map to merged_into (adjusted for pop).
                # Segments that mapped above i shift down by 1.
                post_pop_target = merged_into if merged_into < i else merged_into - 1
                new_old_to_new = []
                for cur_new in old_to_new:
                    if cur_new == i:
                        new_old_to_new.append(post_pop_target)
                    elif cur_new > i:
                        new_old_to_new.append(cur_new - 1)
                    else:
                        new_old_to_new.append(cur_new)
                old_to_new = new_old_to_new

                changed = True
                # Recheck position i (now holds what was i+1, or is gone)
            else:
                i += 1

    # Remap tag_events
    remapped_events = []
    for ev in tag_events:
        old_idx = min(ev["sentence_idx"], len(old_to_new) - 1)  # clamp trailing-tag edge case
        remapped_events.append({**ev, "sentence_idx": old_to_new[old_idx]})

    return segs, remapped_events


def save_wav(path, audio_np, sr, n_channels=1, sampwidth=2):
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(audio_np.tobytes())


# ── Per-example processing ────────────────────────────────────────────

def _path_for_output(path: Path, relative_to: Optional[Path]) -> str:
    return str(path) if relative_to is None else os.path.relpath(path, relative_to)


def process_example(
    ex_dir: Path,
    out_ex_dir: Optional[Path] = None,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
    padding_sec: float = DEFAULT_PADDING_SEC,
    min_words: int = DEFAULT_MIN_WORDS,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    jsonl_dir: Optional[Path] = None,
    absolute_paths: bool = False,
):
    """Process one example directory and return its master-JSONL record.

    Clips and ``meta.json`` are written to ``out_ex_dir`` (default: ``ex_dir``).
    Unless ``absolute_paths``, paths in the returned record are relative to
    ``jsonl_dir`` and paths in ``meta.json`` to ``out_ex_dir``.
    """
    out_ex_dir = Path(out_ex_dir) if out_ex_dir is not None else ex_dir
    wav_path = ex_dir / "input.wav"
    json_path = ex_dir / "transcription.json"

    with open(json_path) as f:
        words = json.load(f)

    with wave.open(str(wav_path)) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = np.int16 if sampwidth == 2 else np.int32
    audio = np.frombuffer(raw, dtype=dtype)

    tags_removed = sum(1 for w in words if is_tag(w["text"]))
    sentences, tag_events = words_to_sentences(words, pause_threshold)
    print(f"  Pause-split: {len(sentences)} segments")

    sentences, tag_events = merge_short_sentences(
        sentences, tag_events,
        min_words=min_words,
        min_duration_sec=min_duration_sec,
        dialogue_id=ex_dir.name,
    )
    print(f"  After merge: {len(sentences)} segments")

    split_dir = out_ex_dir / "split_audio"
    split_dir.mkdir(parents=True, exist_ok=True)

    turns = []
    audio_paths = []
    skipped = 0
    total_samples = len(audio)
    pad_samples = int(padding_sec * sr)

    for i, sent_words in enumerate(sentences):
        text = " ".join(w["text"] for w in sent_words)
        start_sec = sent_words[0]["timestamp"][0]
        end_sec = sent_words[-1]["timestamp"][1]

        slice_start_samp = max(0, int(start_sec * sr) - pad_samples)
        slice_end_samp = min(total_samples, int(end_sec * sr) + pad_samples)
        segment = audio[slice_start_samp:slice_end_samp]

        if len(segment) < MIN_SEGMENT_SAMPLES:
            print(f"  WARNING: skipping sentence {i} (only {len(segment)} samples): '{text}'")
            skipped += 1
            continue

        audio_path = split_dir / f"{len(turns):03d}.wav"
        save_wav(audio_path, segment, sr, n_channels, sampwidth)
        audio_paths.append(audio_path)

        turns.append({
            "turn_id": len(turns),
            "speaker": 1,
            "text": text,
            "audio_path": str(audio_path),
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "slice_start_sec": round(slice_start_samp / sr, 6),
            "slice_end_sec": round(slice_end_samp / sr, 6),
        })

    if skipped:
        print(f"  Skipped {skipped} empty/too-short segments")

    bc_by_sentence = {}
    for ev in tag_events:
        bc_by_sentence.setdefault(ev["sentence_idx"], []).append({
            "tag": ev["tag"],
            "timestamp": ev["timestamp"],
        })

    gt_bc_per_turn = []
    for i in range(len(sentences)):
        events = bc_by_sentence.get(i, [])
        gt_bc_per_turn.append({
            "turn_id": i,
            "bc_count": len(events),
            "bc_events": events,
        })

    def turns_with_paths(relative_to: Optional[Path]):
        return [
            {**turn, "audio_path": _path_for_output(path, relative_to)}
            for turn, path in zip(turns, audio_paths)
        ]

    meta_path = out_ex_dir / "meta.json"
    meta = {
        "dialogue_id": ex_dir.name,
        "turns": turns_with_paths(None if absolute_paths else out_ex_dir),
        "tags_removed": tags_removed,
        "gt_bc_per_turn": gt_bc_per_turn,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    text_field = " \n ".join(f"[S1] {t['text']}" for t in turns)
    print(f"  {len(sentences)} sentences, {tags_removed} tags removed, "
          f"{len(tag_events)} BC events assigned")

    record_base = None if absolute_paths else jsonl_dir
    return {
        "dialogue_id": ex_dir.name,
        "turns": turns_with_paths(record_base),
        "text": text_field,
        "tags_removed": tags_removed,
        "gt_bc_per_turn": gt_bc_per_turn,
        "meta_path": _path_for_output(meta_path, record_base),
    }


# ── Main ──────────────────────────────────────────────────────────────

def prepare_dataset(
    dataset_dir: str,
    output_jsonl: str,
    out_dir: Optional[str] = None,
    pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
    padding_sec: float = DEFAULT_PADDING_SEC,
    min_words: int = DEFAULT_MIN_WORDS,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    absolute_paths: bool = False,
):
    """Process every example directory of ``dataset_dir`` (numeric names first,
    in numeric order) and write the master JSONL. ``out_dir=None`` writes the
    per-example outputs into the dataset itself."""
    dataset_dir = Path(dataset_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    examples = sorted(
        [x for x in dataset_dir.iterdir() if x.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else float("inf"),
    )

    records = []
    for ex_dir in examples:
        print(f"\n{'='*40}\nExample {ex_dir.name}")
        try:
            records.append(process_example(
                ex_dir,
                out_ex_dir=(Path(out_dir) / ex_dir.name) if out_dir is not None else None,
                pause_threshold=pause_threshold,
                padding_sec=padding_sec,
                min_words=min_words,
                min_duration_sec=min_duration_sec,
                jsonl_dir=output_jsonl.parent,
                absolute_paths=absolute_paths,
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    with open(output_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_tags = sum(r["tags_removed"] for r in records)
    print(f"\nWrote {len(records)} records to {output_jsonl}")
    print(f"Total tags removed across dataset: {total_tags}")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Benchmark directory with one <id>/ folder per example.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where per-example clips and meta.json go (required unless --in_place).")
    parser.add_argument("--output_jsonl", type=str, default=None,
                        help="Master JSONL (default: <out_dir>/icc_backchannel_bc_gt.jsonl).")
    parser.add_argument("--in_place", action="store_true",
                        help="Write clips and meta.json into the benchmark directory itself.")
    parser.add_argument("--absolute_paths", action="store_true", help="Store absolute paths in the outputs.")
    parser.add_argument("--pause_threshold", type=float, default=DEFAULT_PAUSE_THRESHOLD)
    parser.add_argument("--padding_sec", type=float, default=DEFAULT_PADDING_SEC)
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--min_duration", type=float, default=DEFAULT_MIN_DURATION_SEC)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.in_place == (args.out_dir is not None):
        parser.error("give exactly one of --out_dir and --in_place")
    out_dir = None if args.in_place else args.out_dir
    if args.output_jsonl is None:
        if args.in_place:
            parser.error("--output_jsonl is required with --in_place")
        args.output_jsonl = str(Path(out_dir) / "icc_backchannel_bc_gt.jsonl")
    prepare_dataset(
        args.dataset_dir, args.output_jsonl, out_dir=out_dir,
        pause_threshold=args.pause_threshold, padding_sec=args.padding_sec,
        min_words=args.min_words, min_duration_sec=args.min_duration,
        absolute_paths=args.absolute_paths,
    )


if __name__ == "__main__":
    main()
