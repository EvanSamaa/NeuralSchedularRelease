"""Find two-party interactions in a Seamless Interaction split."""

import logging
import os
import re
from typing import Dict, List

_FILENAME_PATTERN = re.compile(r"V(\d+)_S(\d+)_I(\d+)_P(\d+[A-Z]?)\.wav$")


def parse_filename_to_dict(full_path: str) -> Dict:
    """Parse a ``V{vendor}_S{session}_I{interaction}_P{participant}.wav`` path."""
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
        "Vendor ID": None,
        "Session ID": None,
        "Interaction ID": None,
        "Participant ID": None,
    }


def find_wav_files_recursive(root_dir: str) -> List[Dict]:
    """All WAV files under ``root_dir`` with their parsed identifiers, in walk order."""
    file_metadata = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                file_metadata.append(parse_filename_to_dict(os.path.join(dirpath, filename)))
    return file_metadata


def group_files_by_context(file_metadata: List[Dict]) -> List[List[str]]:
    """Group WAV paths by (vendor, session, interaction).

    Groups are ordered by that key; paths keep their input order within a
    group. Files whose name did not parse are left out.
    """
    groups: Dict[tuple, List[str]] = {}
    for meta in file_metadata:
        key = (meta["Vendor ID"], meta["Session ID"], meta["Interaction ID"])
        if None in key:
            continue
        groups.setdefault(key, []).append(meta["Full Path"])
    return [groups[key] for key in sorted(groups)]


def discover_interactions(
    dataset_root: str,
    split: str,
    dataset_type: str,
    logger: logging.Logger,
) -> List[Dict]:
    """List the interactions of ``dataset_root/dataset_type/split`` that have two
    WAV files with a transcript JSON beside each.

    The first two files of a group, sorted by file name, become the left and
    right speaker. ``interaction_id`` is the running index within this
    (split, dataset_type) and is what shards are assigned by.
    """
    data_split_path = os.path.join(dataset_root, dataset_type, split)
    logger.info(f"Scanning for WAV files in: {data_split_path}")

    grouped = group_files_by_context(find_wav_files_recursive(data_split_path))
    logger.info(f"Found {len(grouped)} interaction groups")

    interactions = []
    skipped = 0
    for group in grouped:
        if len(group) < 2:
            skipped += 1
            continue

        group_sorted = sorted(group, key=os.path.basename)
        wav_left, wav_right = group_sorted[0], group_sorted[1]
        transcript_left = os.path.splitext(wav_left)[0] + ".json"
        transcript_right = os.path.splitext(wav_right)[0] + ".json"

        missing = [p for p in (transcript_left, transcript_right) if not os.path.exists(p)]
        if missing:
            logger.debug(f"Missing transcript: {missing[0]}")
            skipped += 1
            continue

        interactions.append({
            "interaction_id": len(interactions),
            "identifier_left": os.path.splitext(os.path.basename(wav_left))[0],
            "identifier_right": os.path.splitext(os.path.basename(wav_right))[0],
            "wav_left": wav_left,
            "wav_right": wav_right,
            "transcript_left": transcript_left,
            "transcript_right": transcript_right,
        })

    logger.info(
        f"Valid interactions: {len(interactions)} | "
        f"Skipped (< 2 files or missing transcripts): {skipped}"
    )
    return interactions
