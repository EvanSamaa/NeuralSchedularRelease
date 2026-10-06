"""Label backchannels in Seamless Interaction and write one record per interaction.

    python -m neural_scheduler.backchannel.labeling.extract \
        --dataset_root /path/to/seamless-interaction \
        --split train --dataset_type naturalistic \
        --output_dir /path/to/backchannel_v2 \
        --qwen_model_path /path/to/Qwen3-Omni-30B-A3B-Instruct \
        --shard_id 0 --num_shards 72

For every interaction: split transcript sentences at long word gaps, pick
backchannel candidates with the rule pre-filter, classify each candidate with
Qwen3-Omni, keep the ones assigned a backchannel class, then give every
sentence a label and a turn id.

Files written to ``{output_dir}/{split}_{dataset_type}/``:

* ``shard{N}of{M}_interactions.jsonl`` – one record per interaction::

      {interaction_id, identifier_left, identifier_right, wav_left, wav_right,
       transcript_left, transcript_right, split, dataset_type, sr,
       num_turns, num_sentences, num_backchannels,
       turns: [{text, start, end, speaker_id (0 = left, 1 = right), turn_id, label}]}

  ``label`` is ``"Speech"`` or ``"backchannel-<Class>"``.
* ``shard{N}of{M}_failures.jsonl`` – interactions that raised, with traceback.
* ``summary_shard{N}of{M}.json`` and ``process_{split}_{dataset_type}_shard{N}of{M}.log``.
"""

import argparse
import json
import logging
import os
import random
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ..constants import BC_CLASSES, BC_LABEL_PREFIX, SEAMLESS_SR, SPEECH_LABEL
from .candidates import (
    create_backchannel_example,
    generate_prompt,
    get_conversation_bounds,
    identify_backchannel_candidates,
    preprocess_sentences,
)
from .discover import discover_interactions
from .prompt import SYSTEM_PROMPT
from .turns import build_turn_records
from .validator import BackchannelValidator

BACKCHANNEL_CLASSES = set(BC_CLASSES)
TRANSCRIPT_KEY = "metadata:transcript"
PATH_FIELDS = ("wav_left", "wav_right", "transcript_left", "transcript_right")


def load_transcript_sentences(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)[TRANSCRIPT_KEY]


def label_candidates(candidates, sentences_0, sentences_1, labeler, validator,
                     half_window_size: float, batch_size: int) -> Dict:
    """Classify candidates and return ``{(speaker_id, start, end): "backchannel-<Class>"}``
    for those the labeller assigned a backchannel class."""
    examples = [
        create_backchannel_example(cand, sentences_0, sentences_1, half_window_size)
        for cand in candidates
    ]
    prompts = []
    for example in examples:
        prompt_context, prompt_bc = generate_prompt(example)
        prompts.append([{"transcript": prompt_context}, {"transcript": prompt_bc}])

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        responses.extend(labeler.run_batch_labeling(batch, batch_size=len(batch)))

    validation = validator.validate_batch(responses)
    invalid_indices = set(e["index"] for e in validation["errors"])
    valid_data = iter(validation["valid_data"])

    lookup = {}
    for i, example in enumerate(examples):
        if i in invalid_indices:
            continue
        classification = next(valid_data).get("classification", "")
        if classification in BACKCHANNEL_CLASSES:
            bc = example["backchannel"]
            lookup[(bc["speaker_id"], bc["start"], bc["end"])] = f"{BC_LABEL_PREFIX}{classification}"
    return lookup


def process_single_interaction(
    interaction: Dict,
    qwen_wrapper,
    validator: BackchannelValidator,
    split: str,
    dataset_type: str,
    threshold_for_splitting: float = 0.8,
    threshold_for_bc: float = 0.3,
    half_window_size: float = 10.0,
    batch_size: int = 16,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Build the output record for one interaction."""
    sentences_0 = preprocess_sentences(
        load_transcript_sentences(interaction["transcript_left"]), gap_threshold=threshold_for_splitting
    )
    sentences_1 = preprocess_sentences(
        load_transcript_sentences(interaction["transcript_right"]), gap_threshold=threshold_for_splitting
    )

    min_time, max_time = get_conversation_bounds(sentences_0, sentences_1)
    candidates = identify_backchannel_candidates(
        sentences_0, sentences_1, min_time, max_time, temporal_threshold=threshold_for_bc
    )
    if logger:
        logger.info(f"  {len(candidates)} BC candidates found")

    backchannel_lookup = {}
    if candidates:
        backchannel_lookup = label_candidates(
            candidates, sentences_0, sentences_1, qwen_wrapper, validator,
            half_window_size, batch_size,
        )
        if logger:
            logger.info(f"  {len(backchannel_lookup)} confirmed BCs after validation")

    turns = build_turn_records(sentences_0, sentences_1, backchannel_lookup)
    return {
        "interaction_id": interaction["interaction_id"],
        "identifier_left": interaction["identifier_left"],
        "identifier_right": interaction["identifier_right"],
        "wav_left": interaction["wav_left"],
        "wav_right": interaction["wav_right"],
        "transcript_left": interaction["transcript_left"],
        "transcript_right": interaction["transcript_right"],
        "split": split,
        "dataset_type": dataset_type,
        "sr": SEAMLESS_SR,
        "num_turns": len(set(t["turn_id"] for t in turns)),
        "num_sentences": len(turns),
        "num_backchannels": sum(1 for t in turns if t["label"] != SPEECH_LABEL),
        "turns": turns,
    }


def read_processed_ids(path: str) -> set:
    """Interaction ids already present in an output file (unparsable lines are ignored)."""
    processed = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["interaction_id"])
                except (ValueError, KeyError):
                    pass
    return processed


def process_shard(
    shard_interactions: List[Dict],
    qwen_wrapper,
    validator: BackchannelValidator,
    split: str,
    dataset_type: str,
    output_interactions_path: str,
    output_failures_path: str,
    threshold_for_splitting: float,
    threshold_for_bc: float,
    half_window_size: float,
    batch_size: int,
    logger: logging.Logger,
    resume: bool = False,
    max_interactions: Optional[int] = None,
    relative_to: Optional[str] = None,
):
    """Process a shard, appending one line per interaction as it finishes.

    With ``resume`` the output files are appended to and interactions already
    in them are skipped. ``relative_to`` rewrites the audio and transcript
    paths in the records relative to that directory. Returns
    ``(ok, skipped, failed)`` counts.
    """
    processed_ids = set()
    if resume:
        processed_ids = read_processed_ids(output_interactions_path)
        if os.path.exists(output_interactions_path):
            logger.info(f"Resuming: {len(processed_ids)} interactions already processed")

    interactions_to_process = shard_interactions
    if max_interactions is not None:
        interactions_to_process = shard_interactions[:max_interactions]

    total = len(interactions_to_process)
    ok_count = skip_count = fail_count = 0
    mode = "a" if resume else "w"
    with open(output_interactions_path, mode) as f_ok, open(output_failures_path, mode) as f_fail:
        for i, interaction in enumerate(interactions_to_process):
            iid = interaction["interaction_id"]
            if iid in processed_ids:
                skip_count += 1
                continue

            logger.info(
                f"[{i + 1}/{total}] {interaction['identifier_left']} / {interaction['identifier_right']}"
            )
            try:
                record = process_single_interaction(
                    interaction=interaction,
                    qwen_wrapper=qwen_wrapper,
                    validator=validator,
                    split=split,
                    dataset_type=dataset_type,
                    threshold_for_splitting=threshold_for_splitting,
                    threshold_for_bc=threshold_for_bc,
                    half_window_size=half_window_size,
                    batch_size=batch_size,
                    logger=logger,
                )
                if relative_to is not None:
                    for field in PATH_FIELDS:
                        record[field] = os.path.relpath(record[field], relative_to)
                f_ok.write(json.dumps(record) + "\n")
                f_ok.flush()
                ok_count += 1
                logger.info(
                    f"  -> {record['num_sentences']} sentences | "
                    f"{record['num_turns']} turns | {record['num_backchannels']} BCs"
                )
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                failure = {
                    "interaction_id": iid,
                    "identifier_left": interaction["identifier_left"],
                    "identifier_right": interaction["identifier_right"],
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                f_fail.write(json.dumps(failure) + "\n")
                f_fail.flush()
                fail_count += 1

    logger.info(f"Shard done: {ok_count} ok | {skip_count} skipped | {fail_count} failed")
    return ok_count, skip_count, fail_count


def select_shard(interactions: List[Dict], shard_id: int, num_shards: int) -> List[Dict]:
    return [i for i in interactions if i["interaction_id"] % num_shards == shard_id]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Label backchannels in Seamless Interaction with Qwen3-Omni."
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Seamless Interaction root (contains improvised/ and naturalistic/).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--dataset_type", type=str, default="improvised",
                        choices=["improvised", "naturalistic"])
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Records go to <output_dir>/<split>_<dataset_type>/.")
    parser.add_argument("--qwen_model_path", type=str, required=True,
                        help="Local path or hub id of Qwen3-Omni-30B-A3B-Instruct.")

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)

    parser.add_argument("--threshold_splitting", type=float, default=0.8,
                        help="Split a sentence at word gaps of at least this many seconds.")
    parser.add_argument("--threshold_bc", type=float, default=0.3,
                        help="A candidate must be this far from the same speaker's other sentences.")
    parser.add_argument("--half_window", type=float, default=10.0,
                        help="Seconds of context shown to the labeller on each side.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_interactions", type=int, default=None,
                        help="Cap interactions per shard (for testing).")
    parser.add_argument("--relative_paths", action="store_true",
                        help="Store audio and transcript paths relative to --dataset_root "
                             "instead of as given.")
    return parser.parse_args(argv)


def setup_logging(output_dir: str, shard_tag: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"bc_labeling_{shard_tag}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(os.path.join(output_dir, f"process_{shard_tag}.log")),
        logging.StreamHandler(),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    shard_name = f"shard{args.shard_id}of{args.num_shards}"
    output_dir = os.path.join(args.output_dir, f"{args.split}_{args.dataset_type}")
    logger = setup_logging(output_dir, f"{args.split}_{args.dataset_type}_{shard_name}")
    logger.info(f"Args: {vars(args)}")

    all_interactions = discover_interactions(
        dataset_root=args.dataset_root, split=args.split, dataset_type=args.dataset_type, logger=logger
    )
    logger.info(f"Total interactions discovered: {len(all_interactions)}")
    shard_interactions = select_shard(all_interactions, args.shard_id, args.num_shards)
    logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_interactions)} interactions")

    output_interactions_path = os.path.join(output_dir, f"{shard_name}_interactions.jsonl")
    output_failures_path = os.path.join(output_dir, f"{shard_name}_failures.jsonl")

    logger.info("Loading Qwen model...")
    from .qwen_labeler import QwenOmniBatchLabeler

    qwen_wrapper = QwenOmniBatchLabeler(args.qwen_model_path, system_prompt=SYSTEM_PROMPT)
    validator = BackchannelValidator()
    logger.info("Models loaded.")

    ok, skipped, failed = process_shard(
        shard_interactions=shard_interactions,
        qwen_wrapper=qwen_wrapper,
        validator=validator,
        split=args.split,
        dataset_type=args.dataset_type,
        output_interactions_path=output_interactions_path,
        output_failures_path=output_failures_path,
        threshold_for_splitting=args.threshold_splitting,
        threshold_for_bc=args.threshold_bc,
        half_window_size=args.half_window,
        batch_size=args.batch_size,
        logger=logger,
        resume=args.resume,
        max_interactions=args.max_interactions,
        relative_to=args.dataset_root if args.relative_paths else None,
    )

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "split": args.split,
        "dataset_type": args.dataset_type,
        "interactions_ok": ok,
        "interactions_skipped": skipped,
        "interactions_failed": failed,
        "output_interactions": output_interactions_path,
        "output_failures": output_failures_path,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    summary_path = os.path.join(output_dir, f"summary_{shard_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Interactions: {output_interactions_path}")
    logger.info(f"Failures:     {output_failures_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
