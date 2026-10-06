"""Step 3: schedule tokenised dialogues and reconstruct their audio.

Reads the JSONL written by step 2 (one dialogue per line), runs the NLL search
of :mod:`neural_scheduler.scheduler` for each, and writes per dialogue::

    <output_root>/<checkpoint parent>_<checkpoint name>/sample_<id>/
        input_script/{channel_a,channel_b,channel_sum}.wav     the unscheduled voice lines
        <output_sub_dir>/
            channel_a.wav  channel_b.wav  channel_sum.wav      the scheduled dialogue
            timeline.json                                      the schedule and its scores
            *.png                                              per-pass plots
            dual_channel_waveform.mp4                          with --save_video

``<id>`` is the sample's index in the JSONL in ``three_pass`` mode and the name
of its cache directory in ``bc_only`` mode.

Usage::

    python -m neural_scheduler.pipeline.schedule \
        --tokenized_jsonl out/tokenized_result.jsonl --cache_base_dir out/segment_cache \
        --output_root out/scheduled --checkpoint /path/to/checkpoint
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

from .. import vocab
from ..scheduler import config, generate_dialogue_bc_only, generate_dialogue_three_pass
from .reconstruct import reconstruct_dialogue_from_cache

logger = logging.getLogger("neural_scheduler.pipeline.schedule")

MODE_THREE_PASS = "three_pass"
MODE_BC_ONLY = "bc_only"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schedule tokenised dialogues with the scheduler LM and reconstruct the audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = parser.add_argument_group("input / output")
    io.add_argument("--tokenized_jsonl", type=str, required=True,
                    help="tokenized_result.jsonl written by the tokenisation step.")
    io.add_argument("--cache_base_dir", type=str, default=None,
                    help="Directory holding example_XXXX/ cache folders; only used for samples "
                         "that carry no 'source_cache' field.")
    io.add_argument("--output_root", type=str, required=True, help="Root directory for results.")
    io.add_argument("--output_sub_dir", type=str, default="segment_search_v2",
                    help="Name of the per-sample folder that receives the scheduled dialogue.")
    io.add_argument("--checkpoint", type=str, required=True, help="Scheduler checkpoint directory.")
    io.add_argument("--device", type=str, default="cuda:0")
    io.add_argument("--num_examples", type=int, default=None, help="Process only this many samples.")
    io.add_argument("--randomize", action="store_true",
                    help="With --num_examples, pick the samples at random instead of taking the first ones.")
    io.add_argument("--output_sr", type=int, default=22050, help="Sample rate of the written WAV files.")
    io.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=None,
                    help="Render a dual-channel waveform video (default: on for three_pass, off for bc_only).")
    io.add_argument("--save_input_scripts", action=argparse.BooleanOptionalAction, default=True,
                    help="Also write the unscheduled voice lines of each speaker.")

    parser.add_argument("--mode", choices=[MODE_THREE_PASS, MODE_BC_ONLY], default=MODE_THREE_PASS,
                        help="three_pass schedules everything; bc_only keeps speaker 1 at the positions in "
                             "the sample's 's1_turn_timing' and only places speaker 2's backchannels.")

    p1 = parser.add_argument_group("pass 1: primary turn onsets (three_pass mode)")
    d1 = config.PASS1
    p1.add_argument("--pass1_search_window", type=int, default=d1.search_window,
                    help="Search +/- this many steps around the anchor.")
    p1.add_argument("--pass1_search_stride", type=int, default=d1.search_stride)
    p1.add_argument("--pass1_scoring_context", type=int, default=d1.scoring_context,
                    help="Steps scored on each side of the candidate turn.")
    p1.add_argument("--pass1_inter_turn_gap", type=int, default=d1.inter_turn_gap,
                    help="The anchor is the end of the previous turn plus this many steps.")
    p1.add_argument("--pass_1_schedule_subturns", action="store_true",
                    help="Search the pauses between the speech blocks of a turn.")
    p1.add_argument("--pass_1_speaker_1_subturn_min_pause", type=int, default=d1.speaker_1_subturn_min_pause)
    p1.add_argument("--pass_1_speaker_1_subturn_max_pause", type=int, default=d1.speaker_1_subturn_max_pause)
    p1.add_argument("--pass_1_speaker_2_subturn_min_pause", type=int, default=d1.speaker_2_subturn_min_pause)
    p1.add_argument("--pass_1_speaker_2_subturn_max_pause", type=int, default=d1.speaker_2_subturn_max_pause)
    p1.add_argument("--pass_1_subturn_search_stride", type=int, default=d1.subturn_search_stride)
    p1.add_argument("--pass_1_use_speaker_gap_constraints", action="store_true",
                    help="Restrict each onset to [min_gap, max_gap] after the previous turn.")
    p1.add_argument("--pass_1_speaker_1_min_gap", type=int, default=d1.speaker_1_min_gap,
                    help="Smallest gap when speaker 1 takes the floor; negative allows overlap.")
    p1.add_argument("--pass_1_speaker_1_max_gap", type=int, default=d1.speaker_1_max_gap)
    p1.add_argument("--pass_1_speaker_2_min_gap", type=int, default=d1.speaker_2_min_gap)
    p1.add_argument("--pass_1_speaker_2_max_gap", type=int, default=d1.speaker_2_max_gap)
    p1.add_argument("--pass_1_use_strong_overlap_constraints", action="store_true",
                    help="Force every speaker switch to overlap the previous turn.")
    p1.add_argument("--pass_1_strong_overlap_min", type=float, default=d1.strong_overlap_min,
                    help="Smallest overlap, as a fraction of the incoming turn.")
    p1.add_argument("--pass_1_strong_overlap_max", type=float, default=d1.strong_overlap_max)

    p2 = parser.add_argument_group("pass 2: backchannel onsets")
    d2 = config.PASS2
    cli = config.CLI_DEFAULT_OVERRIDES
    p2.add_argument("--pass2_search_stride", type=int, default=d2.search_stride)
    p2.add_argument("--pass2_scoring_context", type=int, default=cli["pass2_scoring_context"])
    p2.add_argument("--pass2_bc_position_limit", type=float, default=d2.bc_position_limit,
                    help="Fraction of the host's tail in which a backchannel may start.")
    p2.add_argument("--pass2_bc_effect", type=str, default=cli["pass2_bc_effect"],
                    help="Allowed host reactions, joined by '+': np (no pause), fp (full pause), hp (half pause).")
    p2.add_argument("--pass2_max_bc_overlap_ratio", type=float, default=d2.max_bc_overlap_ratio,
                    help="Fraction of a backchannel that must overlap its host.")
    p2.add_argument("--pass2_score_only_pauses", action="store_true",
                    help="When --pass2_bc_effect excludes np, use pauses to score onsets but do not insert them.")

    p3 = parser.add_argument_group("pass 3: re-timing after backchannels (three_pass mode)")
    d3 = config.PASS3
    p3.add_argument("--pass3_scoring_context", type=int, default=d3.scoring_context)
    p3.add_argument("--pass3_max_extra_gap", type=int, default=d3.max_extra_gap,
                    help="Extra silence allowed after the backchannels that precede a breakpoint.")
    return parser


def load_samples(path: str, num_examples: Optional[int] = None, randomize: bool = False) -> List[Dict]:
    """Read a JSONL file, optionally keeping the first or a random ``num_examples`` lines."""
    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if num_examples is not None:
        if randomize:
            samples = random.sample(samples, min(num_examples, len(samples)))
        else:
            samples = samples[:num_examples]
    return samples


def resolve_cache_dir(sample: Dict, sample_index: int, jsonl_path: str, cache_base_dir: Optional[str]) -> str:
    """Cache directory of a sample.

    ``source_cache`` may be absolute or relative to the JSONL file. Samples
    without it are looked up as ``<cache_base_dir>/example_XXXX``.
    """
    source_cache = sample.get("source_cache")
    if source_cache:
        if os.path.isabs(source_cache):
            return source_cache
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(jsonl_path)), source_cache))
    if cache_base_dir is None:
        raise ValueError(f"Sample {sample_index} has no 'source_cache'; pass --cache_base_dir.")
    return os.path.join(cache_base_dir, f"example_{sample_index:04d}")


def script_codes(tokenizer, input_ids: List[int]) -> Tuple[List[int], List[int]]:
    """Codec ids of the audio tokens in script A and script B (turn markers dropped)."""

    def region(start_token: str, end_token: str) -> List[int]:
        try:
            start = input_ids.index(tokenizer.convert_tokens_to_ids(start_token)) + 1
            end = input_ids.index(tokenizer.convert_tokens_to_ids(end_token))
        except ValueError:
            return []
        codes = []
        for token_id in input_ids[start:end]:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("<|audio_") and token.endswith("|>"):
                codes.append(int(token[8:-2]))
        return codes

    return (
        region(vocab.START_SCRIPT_A, vocab.END_SCRIPT_A),
        region(vocab.START_SCRIPT_B, vocab.END_SCRIPT_B),
    )


def run_search(args: argparse.Namespace, model, tokenizer, special_tokens: Dict[str, int], sample: Dict):
    """Run the search selected by ``args.mode`` on one sample."""
    pass2 = dict(
        pass2_search_stride=args.pass2_search_stride,
        pass2_scoring_context=args.pass2_scoring_context,
        pass2_bc_position_limit=args.pass2_bc_position_limit,
        pass2_bc_effect=args.pass2_bc_effect,
        pass2_max_bc_overlap_ratio=args.pass2_max_bc_overlap_ratio,
        pass2_score_only_pauses=args.pass2_score_only_pauses,
    )
    common = dict(
        model=model,
        tokenizer=tokenizer,
        input_ids=sample["input_ids"],
        special_tokens=special_tokens,
        device=args.device,
        verbose=True,
    )

    if args.mode == MODE_BC_ONLY:
        s1_turn_timing = sample.get("s1_turn_timing", [])
        if not s1_turn_timing:
            logger.warning(
                "  The sample has no 's1_turn_timing'; speaker 1's turns will be placed at their script "
                "positions, which is almost certainly wrong. Tokenise with neural_scheduler.pipeline.tokenize_icc."
            )
        return generate_dialogue_bc_only(s1_turn_timing=s1_turn_timing, **common, **pass2)

    return generate_dialogue_three_pass(
        pass1_search_window=args.pass1_search_window,
        pass1_search_stride=args.pass1_search_stride,
        pass1_scoring_context=args.pass1_scoring_context,
        pass1_inter_turn_gap=args.pass1_inter_turn_gap,
        pass_1_schedule_subturns=args.pass_1_schedule_subturns,
        pass_1_speaker_1_subturn_min_pause=args.pass_1_speaker_1_subturn_min_pause,
        pass_1_speaker_1_subturn_max_pause=args.pass_1_speaker_1_subturn_max_pause,
        pass_1_speaker_2_subturn_min_pause=args.pass_1_speaker_2_subturn_min_pause,
        pass_1_speaker_2_subturn_max_pause=args.pass_1_speaker_2_subturn_max_pause,
        pass_1_subturn_search_stride=args.pass_1_subturn_search_stride,
        pass_1_use_speaker_gap_constraints=args.pass_1_use_speaker_gap_constraints,
        pass_1_speaker_1_min_gap=args.pass_1_speaker_1_min_gap,
        pass_1_speaker_1_max_gap=args.pass_1_speaker_1_max_gap,
        pass_1_speaker_2_min_gap=args.pass_1_speaker_2_min_gap,
        pass_1_speaker_2_max_gap=args.pass_1_speaker_2_max_gap,
        pass_1_strong_overlap_mode=args.pass_1_use_strong_overlap_constraints,
        pass_1_strong_overlap_min=args.pass_1_strong_overlap_min,
        pass_1_strong_overlap_max=args.pass_1_strong_overlap_max,
        pass3_scoring_context=args.pass3_scoring_context,
        pass3_max_extra_gap=args.pass3_max_extra_gap,
        **common,
        **pass2,
    )


def process_sample(
    args: argparse.Namespace,
    model,
    tokenizer,
    special_tokens: Dict[str, int],
    sample: Dict,
    sample_index: int,
    model_name: str,
) -> Optional[str]:
    """Schedule one sample and write its outputs; returns the output folder, or None if skipped."""
    cache_dir = resolve_cache_dir(sample, sample_index, args.tokenized_jsonl, args.cache_base_dir)
    if not os.path.isdir(os.path.join(cache_dir, "source_map")):
        logger.warning(f"  Skipping sample {sample_index}: no source_map/ in {cache_dir}")
        return None

    if args.mode == MODE_BC_ONLY and sample.get("source_cache"):
        sample_name = f"sample_{os.path.basename(os.path.normpath(cache_dir))}"
    else:
        sample_name = f"sample_{sample_index}"
    out_folder = os.path.join(args.output_root, model_name, sample_name)
    schedule_folder = os.path.join(out_folder, args.output_sub_dir)
    os.makedirs(schedule_folder, exist_ok=True)

    if args.save_input_scripts:
        input_folder = os.path.join(out_folder, "input_script")
        os.makedirs(input_folder, exist_ok=True)
        script_a, script_b = script_codes(tokenizer, sample["input_ids"])
        try:
            reconstruct_dialogue_from_cache(
                channel_a_scheduled=script_a,
                channel_b_scheduled=script_b,
                cache_dir=cache_dir,
                output_dir=input_folder,
                output_sr=args.output_sr,
                logger=logger,
            )
        except Exception as error:
            if args.mode != MODE_BC_ONLY:
                raise
            logger.warning(f"  Skipping sample {sample_index}: could not write its input scripts ({error})")
            return None

    channel_a, channel_b, state = run_search(args, model, tokenizer, special_tokens, sample)

    # Imported here so that scheduling and reconstruction work without the plotting stack.
    from ..viz.three_pass import visualize_scheduling
    from ..viz.timeline import export_scheduling_state_json

    export_scheduling_state_json(state, os.path.join(schedule_folder, "timeline.json"))
    visualize_scheduling(state, save_dir=schedule_folder)

    wav_a, wav_b = reconstruct_dialogue_from_cache(
        channel_a_scheduled=channel_a,
        channel_b_scheduled=channel_b,
        cache_dir=cache_dir,
        output_dir=schedule_folder,
        output_sr=args.output_sr,
        logger=logger,
    )
    logger.info(f"  Scheduled dialogue: {len(wav_a)} and {len(wav_b)} samples at 16 kHz")

    if args.save_video:
        from ..viz.waveform_video import create_dual_waveform_video

        video_path = os.path.join(schedule_folder, "dual_channel_waveform.mp4")
        try:
            create_dual_waveform_video(
                os.path.join(schedule_folder, "channel_a.wav"),
                os.path.join(schedule_folder, "channel_b.wav"),
                video_path,
            )
        except Exception as error:
            logger.warning(f"  Video generation failed: {error}")

    return schedule_folder


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.save_video is None:
        args.save_video = args.mode == MODE_THREE_PASS

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ..model import load_scheduler

    logger.info(f"Loading scheduler from {args.checkpoint}")
    model, tokenizer = load_scheduler(args.checkpoint, device=args.device)
    model.eval()
    special_tokens = vocab.get_special_token_ids(tokenizer)

    checkpoint_parts = args.checkpoint.rstrip("/").split("/")
    model_name = "_".join(checkpoint_parts[-2:])

    samples = load_samples(args.tokenized_jsonl, args.num_examples, args.randomize)
    logger.info(f"Loaded {len(samples)} sample(s) from {args.tokenized_jsonl}")

    for sample_index, sample in enumerate(samples):
        logger.info(f"Sample {sample_index + 1}/{len(samples)} ({args.mode})")
        out = process_sample(args, model, tokenizer, special_tokens, sample, sample_index, model_name)
        if out is not None:
            logger.info(f"  Output in {out}")


if __name__ == "__main__":
    main()
