"""Shard arithmetic, per-shard logging and resume bookkeeping."""

import json
import logging
import os
from typing import Dict, List, Set


def get_shard_indices(total_size: int, num_shards: int, shard_id: int) -> List[int]:
    """Indices of shard ``shard_id`` when ``total_size`` items are cut into
    ``num_shards`` contiguous, balanced ranges (the first ``total_size % num_shards``
    shards get one extra item)."""
    base_size = total_size // num_shards
    remainder = total_size % num_shards

    if shard_id < remainder:
        start_idx = shard_id * (base_size + 1)
        end_idx = start_idx + base_size + 1
    else:
        start_idx = remainder * (base_size + 1) + (shard_id - remainder) * base_size
        end_idx = start_idx + base_size

    return list(range(start_idx, end_idx))


def setup_logging(shard_id: int, num_shards: int, log_dir: str, tag: str) -> logging.Logger:
    """Logger writing to the console and to ``processing_{tag}_shard_{id}_of_{N}.log``.

    ``tag`` names the dataset (and, for Seamless, data type and split) so that
    jobs sharing an output directory do not write to the same log file.
    """
    log_file = os.path.join(log_dir, f"processing_{tag}_shard_{shard_id}_of_{num_shards}.log")

    logger = logging.getLogger(f"{tag}_shard_{shard_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(f"%(asctime)s - Shard {shard_id}/{num_shards} - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(log_file), logging.StreamHandler()):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_processed_example_ids(output_file_path: str) -> Set:
    """``source_file`` values that already have at least one record in the output."""
    return set(get_processed_windows(output_file_path))


def get_processed_windows(output_file_path: str) -> Dict[object, Set[int]]:
    """Map each ``source_file`` in the output to the window indices already written.

    Lines that are not valid JSON (a write cut short by a killed job) are ignored.
    """
    processed: Dict[object, Set[int]] = {}
    if not os.path.exists(output_file_path):
        return processed
    with open(output_file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            processed.setdefault(data.get("source_file"), set()).add(data.get("window_idx"))
    return processed


def completed_marker_path(output_file_path: str) -> str:
    """Side file listing conversations that were processed to the end."""
    return output_file_path + ".completed"


def read_completed(output_file_path: str) -> Set:
    path = completed_marker_path(output_file_path)
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return {json.loads(line) for line in f if line.strip()}


def mark_completed(output_file_path: str, example_id) -> None:
    with open(completed_marker_path(output_file_path), "a") as f:
        f.write(json.dumps(example_id) + "\n")
