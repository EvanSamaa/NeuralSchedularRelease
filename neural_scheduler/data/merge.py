"""Merge shard files into the JSONL files the trainer reads.

Fisher: all shards are one pool; conversations (``source_file``) are assigned to
train / val / test (98 / 1 / 1 by default, seed 42) so that no conversation
is split across files::

    python -m neural_scheduler.data.merge --dataset fisher --input_dir SHARDS

Seamless Interaction: shards carry their split in the file name; improvised and
naturalistic shards of the same split are concatenated::

    python -m neural_scheduler.data.merge --dataset seamless --input_dir SHARDS --trainer_names
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

FISHER_SHARD_PATTERN = re.compile(r"^fisher_audio_tokens_v2_shard_(\d+)_of_(\d+)\.jsonl$")
SEAMLESS_SHARD_PATTERN = re.compile(
    r"^(seamless_interaction_(\w+?)_(train|dev|test))_audio_tokens_shard_(\d+)_of_(\d+)\.jsonl$"
)

FISHER_OUTPUT_TEMPLATE = "fisher_audio_tokens_v2_{split}.jsonl"
SEAMLESS_OUTPUT_TEMPLATE = "seamless_interaction_{split}_audio_tokens.jsonl"
# File names the released training launchers read. Seamless has no separate
# validation set in that layout: its test split is used for validation.
SEAMLESS_TRAINER_TEMPLATE = "audio_tokens_{split}_with_durations_and_turnID_as_tokens.jsonl"
SEAMLESS_TRAINER_SPLIT_NAMES = {"train": "train", "test": "val", "dev": "dev"}


# --- Fisher -------------------------------------------------------------------

def find_fisher_shards(input_dir: str) -> List[Dict]:
    """Fisher shard files sorted by shard id; warns about gaps in the shard set."""
    shards = []
    total_expected = None

    for filename in os.listdir(input_dir):
        match = FISHER_SHARD_PATTERN.match(filename)
        if not match:
            continue
        shard_id = int(match.group(1))
        total_shards = int(match.group(2))
        if total_expected is None:
            total_expected = total_shards
        elif total_expected != total_shards:
            raise ValueError(f"Inconsistent total_shards: saw both {total_expected} and {total_shards}")
        shards.append({
            "filename": filename,
            "filepath": os.path.join(input_dir, filename),
            "shard_id": shard_id,
            "total_shards": total_shards,
        })

    shards.sort(key=lambda x: x["shard_id"])

    if total_expected is not None:
        missing = set(range(total_expected)) - {s["shard_id"] for s in shards}
        if missing:
            print(f"WARNING: missing shards: {sorted(missing)}")

    return shards


def scan_source_files(shards: List[Dict]) -> Tuple[Dict, int]:
    """Count records per ``source_file`` over all shards.

    Returns ``(counts, total_lines)`` and stores each shard's ``line_count``.
    """
    source_counts = defaultdict(int)
    total_lines = 0
    for sf in shards:
        shard_lines = 0
        with open(sf["filepath"], "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON in {sf['filepath']}: {e}")
                if "source_file" not in obj:
                    raise KeyError(
                        f"Line in {sf['filepath']} missing 'source_file' field; "
                        "cannot do conversation-level splitting."
                    )
                source_counts[obj["source_file"]] += 1
                shard_lines += 1
        sf["line_count"] = shard_lines
        total_lines += shard_lines
    return dict(source_counts), total_lines


def assign_splits(
    source_counts: Dict, train_frac: float, val_frac: float, test_frac: float, seed: int
) -> Dict:
    """Assign each ``source_file`` to ``train``, ``val`` or ``test``.

    Ids are sorted, shuffled with ``random.Random(seed)`` and cut at the rounded
    fractions; the remainder goes to test. Fractions are of conversations, so
    record-level proportions differ slightly.
    """
    source_ids = sorted(source_counts.keys())
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    n = len(source_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    assignment = {}
    for i, sid in enumerate(source_ids):
        if i < n_train:
            assignment[sid] = "train"
        elif i < n_train + n_val:
            assignment[sid] = "val"
        else:
            assignment[sid] = "test"
    return assignment


def write_fisher_splits(shards: List[Dict], assignment: Dict, output_dir: str, dry_run: bool) -> Dict[str, int]:
    """Stream every record to the file of its conversation's split; returns record counts."""
    split_counts = {"train": 0, "val": 0, "test": 0}
    handles = {}
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        handles = {
            split: open(os.path.join(output_dir, FISHER_OUTPUT_TEMPLATE.format(split=split)), "w")
            for split in split_counts
        }
    try:
        for sf in shards:
            with open(sf["filepath"], "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    split = assignment[json.loads(line)["source_file"]]
                    if not dry_run:
                        handles[split].write(line + "\n")
                    split_counts[split] += 1
    finally:
        for h in handles.values():
            h.close()

    for h in handles.values():
        print(f"    Written to: {h.name}")
    return split_counts


def merge_fisher(args) -> None:
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_frac} "
            f"({args.train_frac} + {args.val_frac} + {args.test_frac})"
        )
    output_dir = args.output_dir or args.input_dir

    shards = find_fisher_shards(args.input_dir)
    if not shards:
        print("No shard files found.")
        return
    print(f"Found {len(shards)} shard file(s).")

    source_counts, total_lines = scan_source_files(shards)
    print(f"Total records: {total_lines}, conversations: {len(source_counts)}")
    for sf in shards:
        print(f"  Shard {sf['shard_id']:3d}: {sf['line_count']:8d} records")

    assignment = assign_splits(source_counts, args.train_frac, args.val_frac, args.test_frac, args.seed)
    per_split = defaultdict(int)
    for split in assignment.values():
        per_split[split] += 1
    print(f"Conversations: train={per_split['train']}, val={per_split['val']}, test={per_split['test']}")

    split_line_counts = write_fisher_splits(shards, assignment, output_dir, args.dry_run)
    for split in ("train", "val", "test"):
        n = split_line_counts[split]
        pct = 100.0 * n / total_lines if total_lines else 0.0
        print(f"  {split:5s}: {n:8d} records ({pct:.2f}%)")


# --- Seamless -----------------------------------------------------------------

def find_seamless_shards(input_dir: str) -> Dict[str, List[Dict]]:
    """Seamless shard files grouped by split, ordered by (data type, shard id).

    Data types sort alphabetically, so improvised shards precede naturalistic ones.
    """
    groups = defaultdict(list)

    for filename in os.listdir(input_dir):
        match = SEAMLESS_SHARD_PATTERN.match(filename)
        if not match:
            continue
        groups[match.group(3)].append({
            "filename": filename,
            "filepath": os.path.join(input_dir, filename),
            "dataset_name": match.group(1),
            "data_type": match.group(2),
            "split": match.group(3),
            "shard_id": int(match.group(4)),
            "total_shards": int(match.group(5)),
        })

    for key in groups:
        groups[key].sort(key=lambda x: (x["data_type"], x["shard_id"]))

    return dict(groups)


def concatenate_shards(shard_files: List[Dict], output_path: str, dry_run: bool = False) -> Dict:
    """Concatenate shard files (blank lines removed); returns per-shard and total record counts."""
    stats = {"shards": [], "total_lines": 0, "total_shards": len(shard_files)}

    if dry_run:
        for sf in shard_files:
            with open(sf["filepath"], "r") as f_in:
                line_count = sum(1 for _ in f_in)
            stats["shards"].append({"shard_id": sf["shard_id"], "lines": line_count})
            stats["total_lines"] += line_count
        return stats

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f_out:
        for sf in shard_files:
            shard_lines = 0
            with open(sf["filepath"], "r") as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(line + "\n")
                        shard_lines += 1
            stats["shards"].append({"shard_id": sf["shard_id"], "lines": shard_lines})
            stats["total_lines"] += shard_lines

    return stats


def seamless_output_name(split: str, trainer_names: bool) -> str:
    if trainer_names:
        return SEAMLESS_TRAINER_TEMPLATE.format(split=SEAMLESS_TRAINER_SPLIT_NAMES[split])
    return SEAMLESS_OUTPUT_TEMPLATE.format(split=split)


def merge_seamless(args) -> None:
    output_dir = args.output_dir or args.input_dir
    groups = find_seamless_shards(args.input_dir)
    if not groups:
        print("No shard files found.")
        return

    for split, shards in sorted(groups.items()):
        if args.splits and split not in args.splits:
            print(f"Skipping split '{split}' (not in {args.splits})")
            continue

        by_type = defaultdict(list)
        for s in shards:
            by_type[s["data_type"]].append(s)

        print(f"Split: {split} ({len(shards)} shard files)")
        for data_type, type_shards in sorted(by_type.items()):
            expected = type_shards[0]["total_shards"]
            missing_ids = set(range(expected)) - {s["shard_id"] for s in type_shards}
            print(f"  {data_type}: {len(type_shards)}/{expected} shards")
            if missing_ids:
                print(f"    WARNING: missing shards: {sorted(missing_ids)}")

        output_path = os.path.join(output_dir, seamless_output_name(split, args.trainer_names))
        stats = concatenate_shards(shards, output_path, dry_run=args.dry_run)
        verb = "Would write" if args.dry_run else "Wrote"
        print(f"  {verb} {stats['total_lines']} records to {output_path}")


# --- CLI ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge shard JSONL files into training files.")
    parser.add_argument("--dataset", required=True, choices=["fisher", "seamless"])
    parser.add_argument("--input_dir", required=True, help="Directory containing the shard JSONL files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: the input directory)")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be written without writing")

    fisher = parser.add_argument_group("fisher")
    fisher.add_argument("--train_frac", type=float, default=0.98)
    fisher.add_argument("--val_frac", type=float, default=0.01)
    fisher.add_argument("--test_frac", type=float, default=0.01)
    fisher.add_argument("--seed", type=int, default=42)

    seamless = parser.add_argument_group("seamless")
    seamless.add_argument("--splits", nargs="+", default=None, help="Only merge these splits (e.g. train test)")
    seamless.add_argument("--trainer_names", action="store_true",
                          help="Name the outputs as the released training scripts expect: "
                               "audio_tokens_{train,val}_with_durations_and_turnID_as_tokens.jsonl, "
                               "where 'val' is the Seamless test split")
    return parser


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    if args.dataset == "fisher":
        merge_fisher(args)
    else:
        merge_seamless(args)
    print("Dry run complete." if args.dry_run else "Done.")


if __name__ == "__main__":
    main()
