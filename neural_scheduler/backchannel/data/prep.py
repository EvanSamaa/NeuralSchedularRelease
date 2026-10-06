"""Turn labelled interactions into classifier examples.

    python -m neural_scheduler.backchannel.data.prep \
        --input_dir /path/to/backchannel_v2/train_naturalistic \
        --output_dir /path/to/bc_classification_data/train_naturalistic_with_audio

Segments that share a turn id form a *turn group*: the main speaker's speech
plus whatever the listener said during it. Each turn group with at most one
backchannel becomes one example whose text is the surrounding turn groups,
one line per speaker, joined by ``" </s> "``::

    A: My skin is very sensitive to it [BC:Neutral] </s> A: I'm having little red blotches <bc_slot> </s> B: oh no

* the target's main speaker is ``A``, the other speaker ``B``;
* the target line ends with ``<bc_slot>``; the backchannel itself is hidden;
* a context turn group with exactly one backchannel carries ``[BC:<Class>]``;
* context covers every turn group whose midpoint lies within
  ``--context_window`` seconds of the target's midpoint, before and after.

Where in the turn the backchannel falls is deliberately not encoded; timing is
the scheduler's job.

Label schemes (``--label_scheme``):

* ``semantic`` – the target label is the labeller's class.
* ``duration`` – a backchannel of at most ``--bc_duration_threshold`` seconds
  is labelled ``Neutral``, a longer one ``Emotional``; records gain a
  ``bc_duration`` field. Context tags keep the labeller's class.

One ``{split}.jsonl`` per value of the interactions' ``split`` field is
written, with negatives down-sampled to ``--neg_ratio`` times the positives,
plus ``data_summary.json`` and ``bc_duration_distribution.png``. Records::

    {input_text, label, label_id, interaction_id, turn_id, dataset_type, split,
     bc_start, bc_end, [bc_duration,] main_speech_duration,
     audio_span: {wav, start, end}, sr}

``audio_span`` is the main speaker's own speech in the target turn group on
that speaker's mono recording; audio is loaded at training time.
"""

import argparse
import glob
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ..constants import BC_CLASSES, BC_LABEL_PREFIX, BC_SLOT_TOKEN, FULL_LABEL_MAP, LINE_SEP, SPEECH_LABEL

LABEL_MAP = FULL_LABEL_MAP
LABEL_SCHEMES = ("semantic", "duration")
DEFAULT_BC_DURATION_THRESHOLD = 0.32
SHORT_BC_LABEL = "Neutral"
LONG_BC_LABEL = "Emotional"


def load_all_interactions(input_dir: str) -> List[Dict]:
    """Read every ``*_interactions.jsonl`` under ``input_dir`` (recursively)."""
    interactions = []
    seen_files = set()
    for pattern in [
        os.path.join(input_dir, "**", "*_interactions.jsonl"),
        os.path.join(input_dir, "*_interactions.jsonl"),
    ]:
        for path in glob.glob(pattern, recursive=True):
            if path in seen_files:
                continue
            seen_files.add(path)
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        interactions.append(json.loads(line))
    return interactions


def _bc_class(segment: Dict) -> str:
    return segment["label"].split("-", 1)[1]


def extract_turn_groups(
    interaction: Dict,
    label_scheme: str = "semantic",
    bc_duration_threshold: float = DEFAULT_BC_DURATION_THRESHOLD,
) -> List[Dict]:
    """Group an interaction's segments by turn id.

    Each group is ``{turn_id, segments, bc_segments, speech_segments, bc_count,
    label, main_speaker_id, main_speech_duration, turn_start, turn_end}``. The
    main speaker is the one with the most ``Speech`` duration. ``label`` is
    ``NoBackchannel`` without a backchannel, the class (per ``label_scheme``)
    with exactly one, and ``None`` with several.
    """
    groups_by_id = defaultdict(list)
    for seg in interaction["turns"]:
        groups_by_id[seg["turn_id"]].append(seg)

    turn_groups = []
    for turn_id in sorted(groups_by_id.keys()):
        segments = groups_by_id[turn_id]
        bc_segs = [s for s in segments if s["label"].startswith(BC_LABEL_PREFIX)]
        speech_segs = [s for s in segments if s["label"] == SPEECH_LABEL]

        speech_dur_by_speaker = defaultdict(float)
        for s in speech_segs:
            speech_dur_by_speaker[s["speaker_id"]] += s["end"] - s["start"]
        if speech_dur_by_speaker:
            main_speaker = max(speech_dur_by_speaker, key=speech_dur_by_speaker.get)
        else:
            main_speaker = segments[0]["speaker_id"]

        if len(bc_segs) == 1:
            if label_scheme == "duration":
                bc_dur = bc_segs[0]["end"] - bc_segs[0]["start"]
                label = SHORT_BC_LABEL if bc_dur <= bc_duration_threshold else LONG_BC_LABEL
            else:
                label = _bc_class(bc_segs[0])
        elif len(bc_segs) == 0:
            label = "NoBackchannel"
        else:
            label = None

        turn_groups.append({
            "turn_id": turn_id,
            "segments": segments,
            "bc_segments": bc_segs,
            "speech_segments": speech_segs,
            "bc_count": len(bc_segs),
            "label": label,
            "main_speaker_id": main_speaker,
            "main_speech_duration": sum(
                s["end"] - s["start"] for s in speech_segs if s["speaker_id"] == main_speaker
            ),
            "turn_start": min(s["start"] for s in segments),
            "turn_end": max(s["end"] for s in segments),
        })
    return turn_groups


def render_turn_group(turn_group: Dict, speaker_tag_fn, is_target: bool = False) -> List[str]:
    """Text lines of one turn group: the main speaker's speech as one line
    (ending in ``<bc_slot>`` for the target, or ``[BC:<Class>]`` for a context
    group with exactly one backchannel), then one line per listener speech
    segment."""
    main_speaker = turn_group["main_speaker_id"]
    lines = []

    speech_by_speaker = defaultdict(list)
    for seg in sorted(turn_group["speech_segments"], key=lambda s: s["start"]):
        speech_by_speaker[seg["speaker_id"]].append(seg)

    main_texts = [s["text"] for s in speech_by_speaker.get(main_speaker, [])]
    if main_texts:
        main_line = f"{speaker_tag_fn(main_speaker)}: {' '.join(main_texts)}"
        if is_target:
            main_line += f" {BC_SLOT_TOKEN}"
        elif turn_group["bc_count"] == 1:
            main_line += f" [BC:{_bc_class(turn_group['bc_segments'][0])}]"
        lines.append(main_line)

    for spk_id, segs in speech_by_speaker.items():
        if spk_id == main_speaker:
            continue
        for seg in segs:
            lines.append(f"{speaker_tag_fn(spk_id)}: {seg['text']}")
    return lines


def build_input_text(
    target_group: Dict,
    all_turn_groups: List[Dict],
    context_window: float = 10.0,
) -> Tuple[str, Optional[float], Optional[float]]:
    """Return ``(input_text, bc_start, bc_end)`` for a target turn group.

    Context turn groups with several backchannels are rendered without a tag.
    """
    main_speaker = target_group["main_speaker_id"]

    def speaker_tag_fn(speaker_id):
        return "A" if speaker_id == main_speaker else "B"

    target_mid = (target_group["turn_start"] + target_group["turn_end"]) / 2.0
    window_start = target_mid - context_window
    window_end = target_mid + context_window

    bc_start_time = None
    bc_end_time = None
    if target_group["bc_count"] == 1:
        bc_seg = target_group["bc_segments"][0]
        bc_start_time = bc_seg["start"]
        bc_end_time = bc_seg["end"]

    all_lines = []
    for tg in all_turn_groups:
        tg_mid = (tg["turn_start"] + tg["turn_end"]) / 2.0
        if tg_mid < window_start or tg_mid > window_end:
            continue
        is_target = tg["turn_id"] == target_group["turn_id"]
        if not is_target and tg["bc_count"] > 1:
            untagged = dict(tg)
            untagged["bc_count"] = 0
            lines = render_turn_group(untagged, speaker_tag_fn, is_target=False)
        else:
            lines = render_turn_group(tg, speaker_tag_fn, is_target=is_target)
        all_lines.extend(lines)

    return LINE_SEP.join(all_lines), bc_start_time, bc_end_time


def compute_audio_span(turn_group: Dict, interaction: Dict) -> Optional[Dict]:
    """``{wav, start, end}`` covering the main speaker's speech segments in the
    turn group, on ``wav_left`` for speaker 0 and ``wav_right`` for speaker 1.
    ``None`` if the main speaker has no speech segment."""
    main_speaker = turn_group["main_speaker_id"]
    main_segs = [s for s in turn_group["speech_segments"] if s["speaker_id"] == main_speaker]
    if not main_segs:
        return None

    if main_speaker == 0:
        wav_path = interaction["wav_left"]
    elif main_speaker == 1:
        wav_path = interaction["wav_right"]
    else:
        return None

    return {
        "wav": wav_path,
        "start": float(min(s["start"] for s in main_segs)),
        "end": float(max(s["end"] for s in main_segs)),
    }


def generate_examples(
    interactions: List[Dict],
    context_window: float = 10.0,
    min_speech_duration: float = 0.5,
    logger: Optional[logging.Logger] = None,
    label_scheme: str = "semantic",
    bc_duration_threshold: float = DEFAULT_BC_DURATION_THRESHOLD,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Build all examples. Returns ``(positives, negatives, stats)``.

    Turn groups with several backchannels, with less than
    ``min_speech_duration`` seconds of main speech, or without main-speaker
    audio are not used as targets.
    """
    if label_scheme not in LABEL_SCHEMES:
        raise ValueError(f"label_scheme must be one of {LABEL_SCHEMES}, got {label_scheme!r}")

    positive_examples = []
    negative_examples = []
    excluded_multi_bc = 0
    excluded_short = 0
    excluded_no_audio = 0

    for interaction in interactions:
        turn_groups = extract_turn_groups(interaction, label_scheme, bc_duration_threshold)
        for tg in turn_groups:
            if tg["label"] is None:
                excluded_multi_bc += 1
                continue
            if tg["main_speech_duration"] < min_speech_duration:
                excluded_short += 1
                continue

            input_text, bc_start, bc_end = build_input_text(
                target_group=tg, all_turn_groups=turn_groups, context_window=context_window
            )
            audio_span = compute_audio_span(tg, interaction)
            if audio_span is None:
                excluded_no_audio += 1
                continue

            example = {
                "input_text": input_text,
                "label": tg["label"],
                "label_id": LABEL_MAP[tg["label"]],
                "interaction_id": interaction["interaction_id"],
                "turn_id": tg["turn_id"],
                "dataset_type": interaction.get("dataset_type", ""),
                "split": interaction.get("split", ""),
                "bc_start": bc_start,
                "bc_end": bc_end,
            }
            if label_scheme == "duration":
                has_bc = bc_start is not None and bc_end is not None
                example["bc_duration"] = round(bc_end - bc_start, 3) if has_bc else None
            example["main_speech_duration"] = round(tg["main_speech_duration"], 3)
            example["audio_span"] = audio_span
            example["sr"] = interaction.get("sr", 16000)

            if tg["label"] == "NoBackchannel":
                negative_examples.append(example)
            else:
                positive_examples.append(example)

    stats = {
        "total_positive": len(positive_examples),
        "total_negative": len(negative_examples),
        "excluded_multi_bc": excluded_multi_bc,
        "excluded_short": excluded_short,
        "excluded_no_audio": excluded_no_audio,
        "positive_by_class": dict(Counter(e["label"] for e in positive_examples)),
    }

    if logger:
        logger.info(f"Positive examples: {len(positive_examples)}")
        logger.info(f"Negative examples: {len(negative_examples)}")
        logger.info(f"Excluded (multi-BC): {excluded_multi_bc}")
        logger.info(f"Excluded (short): {excluded_short}")
        logger.info(f"Excluded (no main-speaker audio): {excluded_no_audio}")
        for cls, count in sorted(stats["positive_by_class"].items()):
            logger.info(f"  {cls}: {count}")

    return positive_examples, negative_examples, stats


def balance_and_write(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    output_dir: str,
    neg_ratio: float = 4.0,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Per ``split`` value: keep at most ``neg_ratio`` negatives per positive,
    shuffle, and write ``{split}.jsonl``. Also writes ``data_summary.json``.

    The down-sampling applies to every split, test included.
    """
    rng = random.Random(seed)

    pos_by_split = defaultdict(list)
    neg_by_split = defaultdict(list)
    for ex in positive_examples:
        pos_by_split[ex["split"]].append(ex)
    for ex in negative_examples:
        neg_by_split[ex["split"]].append(ex)

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for split in sorted(set(list(pos_by_split.keys()) + list(neg_by_split.keys()))):
        pos = pos_by_split[split]
        neg = neg_by_split[split]

        target_neg_count = int(len(pos) * neg_ratio)
        if len(neg) > target_neg_count:
            rng.shuffle(neg)
            neg = neg[:target_neg_count]

        all_examples = pos + neg
        rng.shuffle(all_examples)

        output_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        class_counts = Counter(ex["label"] for ex in all_examples)
        summary[split] = {
            "total": len(all_examples),
            "positive": len(pos),
            "negative": len(neg),
            "class_counts": dict(class_counts),
            "output_path": output_path,
        }

        if logger:
            logger.info(f"\n{split}: {len(all_examples)} examples ({len(pos)} pos, {len(neg)} neg)")
            for cls, count in sorted(class_counts.items()):
                logger.info(f"  {cls}: {count}")

    summary_path = os.path.join(output_dir, "data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if logger:
        logger.info(f"\nSummary: {summary_path}")
    return summary


def plot_bc_duration_distribution(
    positive_examples: List[Dict],
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Histogram and density of backchannel durations, saved as
    ``bc_duration_distribution.png``; per-class statistics go to the logger.
    Skipped when matplotlib or scipy is missing."""
    try:
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError as e:
        if logger:
            logger.warning(f"Skipping duration plot — missing dependency: {e}")
        return

    durations_by_class = defaultdict(list)
    for ex in positive_examples:
        if ex["bc_start"] is not None and ex["bc_end"] is not None:
            dur = ex["bc_end"] - ex["bc_start"]
            if dur > 0:
                durations_by_class[ex["label"]].append(dur)

    present_classes = [c for c in BC_CLASSES if c in durations_by_class]
    if not present_classes:
        if logger:
            logger.warning("No BC duration data found — skipping plot.")
        return

    color = "#4C72B0"
    all_durs = np.array([d for durs in durations_by_class.values() for d in durs])
    x_min = 0.0
    x_max = min(np.percentile(all_durs, 99) * 1.1, all_durs.max())
    x_grid = np.linspace(x_min, x_max, 500)
    kde = gaussian_kde(all_durs, bw_method="scott")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(all_durs, bins=50, range=(x_min, x_max), density=True, color=color, alpha=0.35,
            label="_nolegend_")
    ax.fill_between(x_grid, kde(x_grid), alpha=0.2, color=color)
    ax.plot(x_grid, kde(x_grid), lw=2.2, color=color)

    median_val = np.median(all_durs)
    mean_val = np.mean(all_durs)
    ax.axvline(median_val, color="#C44E52", lw=1.5, ls="--", label=f"Median: {median_val:.2f}s")
    ax.axvline(mean_val, color="#DD8452", lw=1.5, ls=":", label=f"Mean: {mean_val:.2f}s")
    ax.plot(all_durs, np.full_like(all_durs, -0.01), "|", color=color, alpha=0.15, markersize=4)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Backchannel duration (s)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Backchannel Duration Distribution  "
        f"(n={len(all_durs)}, μ={mean_val:.2f}s, σ={np.std(all_durs):.2f}s)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    out_path = os.path.join(output_dir, "bc_duration_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"Duration distribution plot saved: {out_path}")
        logger.info("\nBC duration stats (seconds):")
        for cls in present_classes:
            durs = np.array(durations_by_class[cls])
            logger.info(
                f"  {cls:15s}  n={len(durs):5d}  mean={np.mean(durs):.3f}  "
                f"med={np.median(durs):.3f}  std={np.std(durs):.3f}  "
                f"p5={np.percentile(durs, 5):.3f}  p95={np.percentile(durs, 95):.3f}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare backchannel classification data from labelled interactions."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with *_interactions.jsonl shard files (searched recursively)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the per-split JSONL files")
    parser.add_argument("--context_window", type=float, default=10.0,
                        help="Context window in seconds on each side")
    parser.add_argument("--min_speech_duration", type=float, default=0.5,
                        help="Minimum main-speaker speech duration of a target turn group")
    parser.add_argument("--neg_ratio", type=float, default=4.0,
                        help="Negatives kept per positive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_scheme", type=str, default="semantic", choices=LABEL_SCHEMES,
                        help="semantic: labeller's class; duration: Neutral for short, "
                             "Emotional for long backchannels")
    parser.add_argument("--bc_duration_threshold", type=float, default=DEFAULT_BC_DURATION_THRESHOLD,
                        help="Longest backchannel (seconds) labelled Neutral under --label_scheme duration")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("prepare_bc_data")
    logger.info(f"Args: {vars(args)}")

    interactions = load_all_interactions(args.input_dir)
    logger.info(f"Loaded {len(interactions)} interactions")

    pos, neg, _ = generate_examples(
        interactions,
        context_window=args.context_window,
        min_speech_duration=args.min_speech_duration,
        logger=logger,
        label_scheme=args.label_scheme,
        bc_duration_threshold=args.bc_duration_threshold,
    )
    balance_and_write(
        positive_examples=pos,
        negative_examples=neg,
        output_dir=args.output_dir,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        logger=logger,
    )
    plot_bc_duration_distribution(positive_examples=pos, output_dir=args.output_dir, logger=logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
