"""Locating pipeline outputs and baseline files for the analysis tools."""

import os
import re
import sys
from typing import List, Optional, Tuple

SEARCH_SUBDIRS = ["segment_search_hp", "segment_search_v2"]


def detect_search_subdir(sample_dir: str) -> str:
    """Return the first of :data:`SEARCH_SUBDIRS` that exists under ``sample_dir``."""
    for sub in SEARCH_SUBDIRS:
        if os.path.isdir(os.path.join(sample_dir, sub)):
            return sub
    raise FileNotFoundError(f"Neither of {SEARCH_SUBDIRS} found under: {sample_dir}")


def extract_sample_id(sample_dir: str) -> str:
    """Numeric id of a directory named ``sample_<N>``."""
    name = os.path.basename(sample_dir.rstrip("/"))
    m = re.search(r"sample_(\d+)$", name)
    if not m:
        raise ValueError(f"Could not extract sample ID from folder name: {name!r}")
    return m.group(1)


def read_lines(path: str) -> List[str]:
    """Non-empty, stripped lines of a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_ids(path: str) -> List[int]:
    """One integer example id per line."""
    return [int(line) for line in read_lines(path)]


def read_pairs(path: str) -> List[Tuple[str, str]]:
    """Lines of the form ``<wav_a>, <wav_b>``; malformed lines are reported and skipped."""
    pairs = []
    for lineno, line in enumerate(read_lines(path), start=1):
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            print(f"WARNING line {lineno}: expected 'wav_a, wav_b', got: {line!r}", file=sys.stderr)
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def report_missing(candidates: List[Tuple[str, str]]) -> bool:
    """Warn about every ``(path, label)`` that does not exist. True if any is missing."""
    missing = [(p, lbl) for p, lbl in candidates if not os.path.exists(p)]
    for p, lbl in missing:
        print(f"  WARNING: missing {lbl}: {p}", file=sys.stderr)
    if missing:
        print("  Skipping.", file=sys.stderr)
    return bool(missing)


def backchannel_sample_paths(sample_dir: str, personaplex_dir: str) -> Optional[dict]:
    """Resolve the three tracks of a backchannel comparison sample.

    Returns ``sample_id``, ``search_subdir``, ``channel_a`` (the speaker),
    ``model_b`` (the scheduler's listener channel) and ``personaplex``
    (``<personaplex_dir>/<sample_id>/output.wav``), or ``None`` after printing
    a warning if the directory layout is not recognised or a file is missing.
    """
    try:
        search_sub = detect_search_subdir(sample_dir)
        sample_id = extract_sample_id(sample_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"  WARNING: {e}  ->  skipping.", file=sys.stderr)
        return None
    search_dir = os.path.join(sample_dir, search_sub)
    paths = {
        "sample_id": sample_id,
        "search_subdir": search_sub,
        "channel_a": os.path.join(search_dir, "channel_a.wav"),
        "model_b": os.path.join(search_dir, "channel_b.wav"),
        "personaplex": os.path.join(personaplex_dir, sample_id, "output.wav"),
    }
    if report_missing([
        (paths["channel_a"], f"{search_sub}/channel_a.wav"),
        (paths["model_b"], f"{search_sub}/channel_b.wav"),
        (paths["personaplex"], f"personaplex/{sample_id}/output.wav"),
    ]):
        return None
    return paths


def turn_take_example_paths(ex_id: int, naive_dir: str, model_dir: str, model_subdir: str) -> Optional[dict]:
    """Resolve the naive and scheduled audio of a turn-taking example.

    Returns ``naive_dual`` (``<naive_dir>/example_XXXX/combined_dual.wav``,
    stereo) and ``model_ch_a`` / ``model_ch_b``
    (``<model_dir>/sample_N/<model_subdir>/channel_{a,b}.wav``), or ``None``
    after printing a warning if a file is missing.
    """
    paths = {
        "naive_dual": os.path.join(naive_dir, f"example_{ex_id:04d}", "combined_dual.wav"),
        "model_ch_a": os.path.join(model_dir, f"sample_{ex_id}", model_subdir, "channel_a.wav"),
        "model_ch_b": os.path.join(model_dir, f"sample_{ex_id}", model_subdir, "channel_b.wav"),
    }
    if report_missing([
        (paths["naive_dual"], "naive combined_dual.wav"),
        (paths["model_ch_a"], "model channel_a.wav"),
        (paths["model_ch_b"], "model channel_b.wav"),
    ]):
        return None
    return paths
