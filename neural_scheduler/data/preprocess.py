"""Tokenise two-channel conversations into scheduler training records.

Fisher (one shard of an LMDB or of a directory of stereo files)::

    python -m neural_scheduler.data.preprocess --dataset fisher \
        --lmdb_path fisher_audio.lmdb --keys_path fisher_audio_keys.txt \
        --output_dir OUT --num_shards 21 --shard_id 0

Seamless Interaction (manifests once per data type and split, then one job per shard)::

    python -m neural_scheduler.data.preprocess --dataset seamless --create_manifests \
        --data_root /path/to/seamless-interaction --data_type improvised --split train \
        --output_dir OUT --num_shards 25
    python -m neural_scheduler.data.preprocess --dataset seamless \
        --data_type improvised --split train --output_dir OUT --shard_id 0

Each job writes one JSONL file with one record per 20.48 s window; merge the
shards with ``python -m neural_scheduler.data.merge``.
"""

import argparse
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..constants import GLM4_VOICE_MODEL_ID, GLM4_VOICE_TOKENIZER_ID, SPEECH_TOKENIZER_SR
from ..vocab import get_special_token_ids
from . import sharding
from .sequence import load_text_tokenizer, process_conversation
from .speech_tokenizer import load_speech_tokenizer
from .vad import load_silero_vad


@dataclass
class ConversationJob:
    """One conversation to process: its id, how to load it, and fields to add to its records."""

    example_id: object
    load: Callable[[], Tuple[torch.Tensor, torch.Tensor, int]]
    extra_fields: Dict = field(default_factory=dict)


@dataclass
class Models:
    whisper_model: object
    feature_extractor: object
    tokenizer: object
    special_tokens: Dict[str, int]
    vad_model: object
    get_speech_timestamps: Callable


@dataclass
class ShardSummary:
    processed: int = 0  # conversations processed in this run
    skipped: int = 0  # conversations already done (resume)
    errors: int = 0  # conversations that could not be loaded
    windows_written: int = 0
    windows_failed: int = 0
    windows_dropped_unknown_code: int = 0


def load_models(args) -> Models:
    print("Loading WhisperVQ model...")
    whisper_model, feature_extractor = load_speech_tokenizer(
        args.tokenizer_path, device=args.device, glm4voice_repo=args.glm4voice_repo
    )
    print("Loading tokenizer...")
    tokenizer = load_text_tokenizer(args.model_id, add_audio_16383=args.add_audio_16383_token)
    print("Loading VAD model...")
    vad_model, get_speech_timestamps = load_silero_vad(args.vad_repo)
    return Models(
        whisper_model, feature_extractor, tokenizer,
        get_special_token_ids(tokenizer), vad_model, get_speech_timestamps,
    )


def process_shard(
    jobs: List[ConversationJob],
    models: Models,
    output_file_path: str,
    logger,
    resume: bool = False,
    resume_per_window: bool = False,
    visualize_count: int = 0,
    device: str = "cuda",
) -> ShardSummary:
    """Process every conversation of a shard, appending records to ``output_file_path``.

    ``resume`` skips conversations that already have a record in the output.
    ``resume_per_window`` (implies ``resume``) instead skips conversations that
    were processed to the end and, for the others, only the windows already
    written, so a conversation interrupted half-way is completed.
    """
    resume = resume or resume_per_window
    summary = ShardSummary()

    done_windows: Dict = {}
    if resume_per_window:
        done_windows = sharding.get_processed_windows(output_file_path)
        processed_ids = sharding.read_completed(output_file_path)
        logger.info(f"Found {len(processed_ids)} completed and {len(done_windows)} started examples")
    elif resume:
        processed_ids = sharding.get_processed_example_ids(output_file_path)
        logger.info(f"Found {len(processed_ids)} already processed examples")
    else:
        processed_ids = set()
        marker = sharding.completed_marker_path(output_file_path)
        if os.path.exists(marker):
            os.remove(marker)

    visualized = 0

    def maybe_visualize(record: Dict) -> None:
        nonlocal visualized
        if visualized < visualize_count:
            from .inspect import visualize_sample_full

            visualize_sample_full(record, models.special_tokens, visualized)
            visualized += 1

    total = len(jobs)
    with open(output_file_path, "a" if resume else "w") as f_out:
        for position, job in enumerate(jobs):
            if job.example_id in processed_ids:
                summary.skipped += 1
                continue

            logger.info(f"[{position + 1}/{total}] Processing example {job.example_id}...")
            start_time = time.time()
            try:
                audio_left, audio_right, sr = job.load()
            except Exception as e:
                logger.error(f"Error loading example {job.example_id}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                summary.errors += 1
                continue

            stats = process_conversation(
                example_id=job.example_id,
                audio_left=audio_left,
                audio_right=audio_right,
                sr=sr,
                whisper_model=models.whisper_model,
                feature_extractor=models.feature_extractor,
                tokenizer=models.tokenizer,
                special_tokens=models.special_tokens,
                vad_model=models.vad_model,
                get_speech_timestamps=models.get_speech_timestamps,
                logger=logger,
                output_file=f_out,
                extra_fields=job.extra_fields,
                done_windows=done_windows.get(job.example_id),
                on_record=maybe_visualize,
                device=device,
            )
            sharding.mark_completed(output_file_path, job.example_id)

            summary.processed += 1
            summary.windows_written += stats.written
            summary.windows_failed += stats.failed
            summary.windows_dropped_unknown_code += stats.dropped_unknown_code
            logger.info(
                f"  Processed {stats.written} windows ({stats.failed} failed, of which "
                f"{stats.dropped_unknown_code} unknown audio code) in {time.time() - start_time:.2f}s"
            )

            if (position + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(
        f"Total windows written: {summary.windows_written}, failed: {summary.windows_failed} "
        f"(unknown audio code: {summary.windows_dropped_unknown_code})"
    )
    return summary


# --- dataset-specific job lists ---------------------------------------------

def fisher_jobs(source, shard_indices: List[int], source_id_style: str) -> List[ConversationJob]:
    def make_loader(idx: int):
        def load():
            sample = source.load(idx)
            return sample["left"], sample["right"], sample["sr"]
        return load

    return [
        ConversationJob(idx if source_id_style == "index" else source.name(idx), make_loader(idx))
        for idx in shard_indices
    ]


def seamless_jobs(
    manifest: Dict,
    data_type: str,
    split: str,
    source_id_style: str,
    relative_to: Optional[str] = None,
) -> List[ConversationJob]:
    from .sources import seamless

    def make_loader(paths: List[str]):
        def load():
            audio_left, sr = seamless.load_and_resample_audio(paths[0], SPEECH_TOKENIZER_SR)
            audio_right, _ = seamless.load_and_resample_audio(paths[1], SPEECH_TOKENIZER_SR)
            return audio_left, audio_right, sr
        return load

    jobs = []
    for example in manifest["examples"]:
        paths = example["audio_paths"]
        stored = [os.path.relpath(p, relative_to) for p in paths] if relative_to else paths
        example_id = (
            example["example_id"] if source_id_style == "index"
            else seamless.interaction_name(paths[0], data_type, split)
        )
        jobs.append(ConversationJob(
            example_id, make_loader(paths),
            {"audio_path_left": stored[0], "audio_path_right": stored[1]},
        ))
    return jobs


# --- CLI ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenise Fisher or Seamless Interaction audio into scheduler training records.",
    )
    parser.add_argument("--dataset", required=True, choices=["fisher", "seamless"])
    parser.add_argument("--output_dir", required=True, help="Directory for shard JSONL files, manifests and logs")
    parser.add_argument("--num_shards", "--shard_count", dest="num_shards", type=int, default=None,
                        help="Number of shards (Fisher: required; Seamless: used by --create_manifests, default 16)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard to process (Seamless: -1 processes everything sequentially without manifests)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to the shard file, skipping conversations that already have a record")
    parser.add_argument("--resume_per_window", action="store_true",
                        help="Like --resume, but also completes conversations that were interrupted half-way")
    parser.add_argument("--visualize", type=int, default=0, help="Print a breakdown of the first N records")
    parser.add_argument("--device", default="cuda")

    models = parser.add_argument_group("models")
    models.add_argument("--tokenizer_path", default=GLM4_VOICE_TOKENIZER_ID, help="WhisperVQ speech tokenizer")
    models.add_argument("--model_id", default=GLM4_VOICE_MODEL_ID, help="Source of the text tokenizer")
    models.add_argument("--glm4voice_repo", default=None,
                        help="GLM-4-Voice checkout (default: the GLM4VOICE_REPO environment variable)")

    models.add_argument("--vad_repo", default=None,
                        help="Silero VAD torch.hub spec or local checkout (default: the SILERO_VAD_REPO "
                             "environment variable, else the pinned release the published data was made with)")

    fisher = parser.add_argument_group("fisher")
    fisher.add_argument("--lmdb_path", default=None, help="Fisher LMDB")
    fisher.add_argument("--keys_path", default=None, help="Key list of the Fisher LMDB")
    fisher.add_argument("--audio_dir", default=None,
                        help="Directory of stereo Fisher files, as an alternative to the LMDB")

    seamless = parser.add_argument_group("seamless")
    seamless.add_argument("--data_type", default="naturalistic", choices=["naturalistic", "improvised"])
    seamless.add_argument("--split", default="test", choices=["train", "dev", "test"])
    seamless.add_argument("--data_root", default=None, help="Root of the Seamless Interaction audio")
    seamless.add_argument("--create_manifests", "--split_dataset", dest="create_manifests", action="store_true",
                          help="Only write the shard manifests (CPU only)")
    seamless.add_argument("--relative_audio_paths", action="store_true",
                          help="Store audio paths relative to --data_root in the records")

    variants = parser.add_argument_group("options that change the produced data (off for the released datasets)")
    variants.add_argument("--add_audio_16383_token", action="store_true",
                          help="Append <|audio_16383|> to the vocabulary (id 168842) instead of dropping windows "
                               "that contain codec id 16383; the model must be trained with the same vocabulary")
    variants.add_argument("--source_id_style", default="index", choices=["index", "name"],
                          help="'index' stores the integer position as source_file; 'name' stores a string that "
                               "is unique across data types and splits")
    return parser


def run_fisher(args) -> Optional[ShardSummary]:
    from .sources.fisher import FisherDirectorySource, FisherLMDBSource

    if args.num_shards is None:
        raise SystemExit("--num_shards is required for --dataset fisher")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"shard_id must be in range [0, {args.num_shards - 1}], got {args.shard_id}")
    if args.audio_dir:
        source = FisherDirectorySource(args.audio_dir, output_sr=SPEECH_TOKENIZER_SR)
    elif args.lmdb_path and args.keys_path:
        source = FisherLMDBSource(args.lmdb_path, args.keys_path, output_sr=SPEECH_TOKENIZER_SR)
    else:
        raise SystemExit("--dataset fisher needs either --lmdb_path and --keys_path, or --audio_dir")

    logger = sharding.setup_logging(args.shard_id, args.num_shards, args.output_dir, "fisher")
    shard_indices = sharding.get_shard_indices(len(source), args.num_shards, args.shard_id)
    logger.info(f"Shard {args.shard_id}: {len(shard_indices)} of {len(source)} conversations")

    output_file_path = os.path.join(
        args.output_dir, f"fisher_audio_tokens_v2_shard_{args.shard_id}_of_{args.num_shards}.jsonl"
    )
    logger.info(f"Output file: {output_file_path}")

    summary = process_shard(
        fisher_jobs(source, shard_indices, args.source_id_style), load_models(args),
        output_file_path, logger, resume=args.resume, resume_per_window=args.resume_per_window,
        visualize_count=args.visualize, device=args.device,
    )
    print(f"Output: {output_file_path}")
    return summary


def run_seamless(args) -> Optional[ShardSummary]:
    from .sources import seamless

    dataset_name = f"seamless_interaction_{args.data_type}_{args.split}"
    interactions = None
    if args.create_manifests or args.shard_id < 0:
        if not args.data_root:
            raise SystemExit("--data_root is required to scan for audio files")
        data_split_path = os.path.join(args.data_root, args.data_type, args.split)
        print(f"Scanning for WAV files in: {data_split_path}")
        interactions = seamless.discover_interactions(args.data_root, args.data_type, args.split)
        print(f"Found {len(interactions)} interaction groups")

    if args.create_manifests:
        seamless.create_shard_manifests(
            interactions, args.output_dir, args.data_type, args.split, args.num_shards or 16
        )
        return None

    if args.relative_audio_paths and not args.data_root:
        raise SystemExit("--relative_audio_paths needs --data_root")
    relative_to = args.data_root if args.relative_audio_paths else None

    if args.shard_id >= 0:
        manifest = seamless.load_shard_manifest(args.output_dir, args.data_type, args.split, args.shard_id)
        num_shards = seamless.load_shard_metadata(args.output_dir, args.data_type, args.split)["shard_count"]
        output_file_path = os.path.join(
            args.output_dir, f"{dataset_name}_audio_tokens_shard_{args.shard_id:05d}_of_{num_shards}.jsonl"
        )
    else:
        manifest = seamless.build_debug_manifest(interactions)
        num_shards = 1
        output_file_path = os.path.join(args.output_dir, f"{dataset_name}_audio_tokens_debug.jsonl")

    logger = sharding.setup_logging(args.shard_id, num_shards, args.output_dir, dataset_name)
    logger.info(f"Shard {args.shard_id}: {manifest['num_examples']} examples")

    summary = process_shard(
        seamless_jobs(manifest, args.data_type, args.split, args.source_id_style, relative_to),
        load_models(args), output_file_path, logger,
        resume=args.resume, resume_per_window=args.resume_per_window,
        visualize_count=args.visualize, device=args.device,
    )
    print(f"Output: {output_file_path}")
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = run_fisher(args) if args.dataset == "fisher" else run_seamless(args)
    if summary is not None:
        print(
            f"Shard {args.shard_id} summary: processed={summary.processed}, skipped={summary.skipped}, "
            f"errors={summary.errors}, windows={summary.windows_written}, "
            f"failed_windows={summary.windows_failed} "
            f"(unknown audio code: {summary.windows_dropped_unknown_code})"
        )


if __name__ == "__main__":
    main()
