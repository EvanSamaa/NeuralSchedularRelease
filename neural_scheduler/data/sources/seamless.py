"""Seamless Interaction: one mono WAV per participant.

Files are named ``V{vendor}_S{session}_I{interaction}_P{participant}.wav`` under
``{data_root}/{improvised,naturalistic}/{train,dev,test}/``. The two participants
of an interaction become the left (A) and right (B) channels in sorted order of
the participant suffix.

Processing is driven by shard manifests: JSON files listing the conversations
of each shard, created once per (data_type, split) and then consumed by the
parallel tokenisation jobs.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from ...constants import SPEECH_TOKENIZER_SR

DATA_TYPES = ("naturalistic", "improvised")
SPLITS = ("train", "dev", "test")

_FILENAME_PATTERN = re.compile(r"V(\d+)_S(\d+)_I(\d+)_P(\d+[A-Z]?)\.wav$")


def parse_filename_to_dict(full_path: str) -> Dict:
    """Parse the vendor / session / interaction / participant ids out of a file name.

    The id fields are ``None`` when the name does not match the convention.
    """
    match = _FILENAME_PATTERN.search(os.path.basename(full_path))
    if match:
        vendor_id, session_id, interaction_id, participant_id = match.groups()
        return {
            "Full Path": full_path,
            "Vendor ID": int(vendor_id),
            "Session ID": int(session_id),
            "Interaction ID": int(interaction_id),
            "Participant ID": participant_id,
        }
    return {
        "Full Path": full_path,
        "Vendor ID": None, "Session ID": None,
        "Interaction ID": None, "Participant ID": None,
    }


def find_wav_files_recursive(root_dir: str):
    """Find all WAV files under ``root_dir``; returns a DataFrame of parsed metadata."""
    import pandas as pd

    file_metadata = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                file_metadata.append(parse_filename_to_dict(os.path.join(dirpath, filename)))
    return pd.DataFrame(file_metadata)


def group_files_by_context(df) -> List[List[str]]:
    """Group file paths by (vendor, session, interaction), in sorted key order."""
    if df.empty:
        return []
    df_valid = df.dropna(subset=["Vendor ID", "Session ID", "Interaction ID"])
    if df_valid.empty:
        return []
    grouped_paths = df_valid.groupby(["Vendor ID", "Session ID", "Interaction ID"])["Full Path"].apply(list)
    return grouped_paths.tolist()


def sort_participants(audio_paths: List[str]) -> List[str]:
    """Order an interaction's files by participant suffix; the first two become A and B."""
    return sorted(audio_paths, key=lambda x: x.split("_")[-1])


def validate_sample_files(audio_paths: List[str]) -> Tuple[bool, str, Dict[str, str]]:
    """Check that an interaction has two existing audio files."""
    if len(audio_paths) < 2:
        return False, f"Less than 2 audio files: {len(audio_paths)}", {}

    file_paths = {"audio_left": audio_paths[0], "audio_right": audio_paths[1]}
    missing = [f"{n}: {p}" for n, p in file_paths.items() if not os.path.exists(p)]

    if missing:
        return False, f"Missing files: {', '.join(missing)}", file_paths
    return True, "", file_paths


def discover_interactions(data_root: str, data_type: str, split: str) -> List[List[str]]:
    """All interaction groups (lists of file paths) of one (data_type, split)."""
    return group_files_by_context(find_wav_files_recursive(os.path.join(data_root, data_type, split)))


def interaction_name(audio_path: str, data_type: str, split: str) -> str:
    """Globally unique conversation name, e.g. ``improvised_train_V01_S0223_I00000127``."""
    stem = os.path.basename(audio_path)
    return f"{data_type}_{split}_" + stem.rsplit("_P", 1)[0]


# --- shard manifests ---------------------------------------------------------

def get_shard_manifest_dir(output_dir: str, data_type: str, split: str) -> str:
    return os.path.join(output_dir, f"shard_manifests_{data_type}_{split}")


def get_shard_manifest_path(output_dir: str, data_type: str, split: str, shard_id: int) -> str:
    return os.path.join(get_shard_manifest_dir(output_dir, data_type, split), f"shard_{shard_id:05d}.json")


def get_shard_metadata_path(output_dir: str, data_type: str, split: str) -> str:
    return os.path.join(get_shard_manifest_dir(output_dir, data_type, split), "metadata.json")


def create_shard_manifests(
    multi_participant_interactions: List[List[str]],
    output_dir: str, data_type: str, split: str, shard_count: int,
) -> None:
    """Split the interactions into ``shard_count`` manifests.

    Interactions without two existing files are dropped; the rest get
    consecutive ``example_id`` values and are divided into contiguous shards of
    ``ceil(total / shard_count)`` examples.
    """
    manifest_dir = get_shard_manifest_dir(output_dir, data_type, split)
    os.makedirs(manifest_dir, exist_ok=True)

    valid_samples = []
    skipped_count = 0

    print(f"Validating {len(multi_participant_interactions)} samples...")

    for idx, audio_paths in enumerate(multi_participant_interactions):
        audio_paths_sorted = sort_participants(audio_paths)
        is_valid, error_msg, _ = validate_sample_files(audio_paths_sorted)

        if is_valid:
            valid_samples.append({"original_idx": idx, "audio_paths": audio_paths_sorted})
        else:
            skipped_count += 1
            if skipped_count <= 10:
                print(f"  Skipping sample {idx}: {error_msg}")

    print(f"Valid samples: {len(valid_samples)}, Skipped: {skipped_count}")

    total_examples = len(valid_samples)
    examples_per_shard = (total_examples + shard_count - 1) // shard_count

    print(f"\nCreating {shard_count} shard manifests ({total_examples} examples, ~{examples_per_shard}/shard)")

    shard_info = []
    for shard_id in range(shard_count):
        start_idx = shard_id * examples_per_shard
        end_idx = min((shard_id + 1) * examples_per_shard, total_examples)
        if start_idx >= total_examples:
            break

        examples_data = [
            {"example_id": idx, "original_idx": valid_samples[idx]["original_idx"],
             "audio_paths": valid_samples[idx]["audio_paths"]}
            for idx in range(start_idx, end_idx)
        ]

        manifest = {
            "shard_id": shard_id, "shard_count": shard_count,
            "data_type": data_type, "split": split,
            "start_idx": start_idx, "end_idx": end_idx,
            "num_examples": len(examples_data), "examples": examples_data,
        }

        with open(get_shard_manifest_path(output_dir, data_type, split, shard_id), "w") as f:
            json.dump(manifest, f, indent=2)

        shard_info.append({"shard_id": shard_id, "num_examples": len(examples_data),
                           "start_idx": start_idx, "end_idx": end_idx})
        print(f"  Shard {shard_id}: examples {start_idx}-{end_idx-1} ({len(examples_data)} examples)")

    metadata = {
        "shard_count": len(shard_info), "total_examples": total_examples,
        "total_skipped": skipped_count, "data_type": data_type,
        "split": split, "shards": shard_info,
    }
    with open(get_shard_metadata_path(output_dir, data_type, split), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nShard manifests saved to: {manifest_dir}")


def load_shard_metadata(output_dir: str, data_type: str, split: str) -> Optional[Dict]:
    metadata_path = get_shard_metadata_path(output_dir, data_type, split)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_shard_manifest(output_dir: str, data_type: str, split: str, shard_id: int) -> Dict:
    """Load one shard manifest; raises if the manifests have not been created."""
    metadata = load_shard_metadata(output_dir, data_type, split)
    if metadata is None:
        raise FileNotFoundError(
            f"Shard metadata not found at {get_shard_metadata_path(output_dir, data_type, split)}. "
            "Create the manifests first (--create_manifests)."
        )
    if shard_id < 0 or shard_id >= metadata["shard_count"]:
        raise ValueError(f"Invalid shard_id {shard_id}. Valid range: 0 to {metadata['shard_count'] - 1}")

    with open(get_shard_manifest_path(output_dir, data_type, split, shard_id), "r") as f:
        return json.load(f)


def build_debug_manifest(multi_participant_interactions: List[List[str]]) -> Dict:
    """A single in-memory manifest covering every valid interaction (sequential runs)."""
    valid_samples = []
    for idx, audio_paths in enumerate(multi_participant_interactions):
        audio_paths_sorted = sort_participants(audio_paths)
        is_valid, _, _ = validate_sample_files(audio_paths_sorted)
        if is_valid:
            valid_samples.append({"example_id": idx, "original_idx": idx, "audio_paths": audio_paths_sorted})
    return {
        "shard_id": -1, "shard_count": 1,
        "num_examples": len(valid_samples), "examples": valid_samples,
    }


# --- audio -------------------------------------------------------------------

def load_and_resample_audio(audio_path: str, target_sr: int = SPEECH_TOKENIZER_SR) -> Tuple[torch.Tensor, int]:
    """Load a file as mono float32 and resample it to ``target_sr`` with librosa."""
    import librosa

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return torch.from_numpy(waveform).float(), sr
