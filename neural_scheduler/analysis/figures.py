"""Voice-activity figures comparing the scheduler with its baselines.

Two sub-commands, each writing one single-column PNG per sample:

``bc``
    Speaker turn, scheduler backchannels and PersonaPlex backchannels for the
    samples listed in ``--sample_list`` (one ``.../sample_<N>`` directory per
    line)::

        python -m neural_scheduler.analysis.figures bc \\
            --sample_list good_examples.txt \\
            --personaplex_dir personaplex/icc_output --out_dir vad_plots

``turn-take``
    Naive concatenation (both channels) against the scheduled dialogue (both
    channels) for the example ids listed in ``--selections``::

        python -m neural_scheduler.analysis.figures turn-take \\
            --selections selections.txt --naive_dir <run>/segment_cache \\
            --model_dir <run>/<checkpoint> --out_dir vad_plots

Voice activity is detected with Silero VAD.
"""

import argparse
import os

from ..viz.vad_plot import get_silero_model, load_mono_channel, load_wav_mono, plot_activity_bars, vad_silero
from ._inputs import backchannel_sample_paths, read_ids, read_lines, turn_take_example_paths

FIG_WIDTH = 3.5  # single column of a two-column paper

BC_COLORS = {
    "channel_a": "#4C9BE8",
    "model_b": "#E8824C",
    "personaplex": "#5DC46A",
}
BC_BAR_Y = {"channel_a": 2, "model_b": 0, "personaplex": 1}
BC_ROW_LABELS = {
    "channel_a": "Speaker Turn",
    "model_b": "BC (Neural Scheduler)",
    "personaplex": "BC (Personaplex)",
}
BC_FIG_HEIGHT = 1.0

TT_COLORS = {
    "naive_a": "#4C9BE8",
    "naive_b": "#1A5FA8",
    "proposed_a": "#E8824C",
    "proposed_b": "#A84C1A",
}
TT_BAR_Y = {"naive_a": 3, "naive_b": 2, "proposed_a": 1, "proposed_b": 0}
TT_FIG_HEIGHT = 1.6


def plot_vad_bc(segs_a, segs_model, segs_pp, duration, sample_id, out_path) -> None:
    """Three-row figure: speaker turn / scheduler backchannel / PersonaPlex backchannel."""
    plot_activity_bars(
        {"channel_a": segs_a, "model_b": segs_model, "personaplex": segs_pp},
        bar_y=BC_BAR_Y,
        colors=BC_COLORS,
        duration=duration,
        title=f"Voice Activity  —  sample {sample_id}",
        out_path=out_path,
        fig_size=(FIG_WIDTH, BC_FIG_HEIGHT),
        ylim=(-0.7, 2.7),
        row_labels=BC_ROW_LABELS,
    )


def plot_vad_turn_take(segs, duration, title, out_path, save_pdf=False) -> None:
    """Four-row figure: naive A/B above a divider, scheduled A/B below.

    ``segs`` maps ``naive_a``, ``naive_b``, ``proposed_a`` and ``proposed_b``
    to speech segments. The y ticks show row positions (3..0), not names.
    """
    plot_activity_bars(
        segs,
        bar_y=TT_BAR_Y,
        colors=TT_COLORS,
        duration=duration,
        title=title,
        out_path=out_path,
        fig_size=(FIG_WIDTH, TT_FIG_HEIGHT),
        ylim=(-0.6, 3.6),
        divider_y=(TT_BAR_Y["naive_b"] + TT_BAR_Y["proposed_a"]) / 2,
        integer_time_ticks=True,
        save_pdf=save_pdf,
    )


def run_bc(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    sample_dirs = read_lines(args.sample_list)
    print(f"Loaded {len(sample_dirs)} sample(s).")
    get_silero_model()

    vad_kwargs = dict(
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
    )
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"\n{'─'*60}")
        print(f"  Sample {idx:02d}  ->  {sample_dir}")
        paths = backchannel_sample_paths(sample_dir, args.personaplex_dir)
        if paths is None:
            continue

        print(f"  Loading audio ({paths['search_subdir']})...")
        wav_a, sr_a = load_wav_mono(paths["channel_a"])
        wav_model, sr_model = load_wav_mono(paths["model_b"])
        wav_pp, sr_pp = load_wav_mono(paths["personaplex"])

        # The baseline's first seconds are muted, as in the listening-test stimuli.
        wav_pp = wav_pp.clone()
        wav_pp[:int(args.silence_first_n * sr_pp)] = 0.0

        duration = max(len(wav_a) / sr_a, len(wav_model) / sr_model, len(wav_pp) / sr_pp)

        print("  Running Silero VAD...")
        segs_a = vad_silero(wav_a, sr_a, **vad_kwargs)
        segs_model = vad_silero(wav_model, sr_model, **vad_kwargs)
        segs_pp = vad_silero(wav_pp, sr_pp, **vad_kwargs)
        print(f"    speaker turn: {len(segs_a)} seg(s), "
              f"neural scheduler: {len(segs_model)}, "
              f"personaplex: {len(segs_pp)}")

        out_path = os.path.join(args.out_dir, f"vad_sample_{idx:02d}.png")
        plot_vad_bc(segs_a, segs_model, segs_pp, duration, paths["sample_id"], out_path)

    print(f"\nAll done. Plots in {args.out_dir}/")


def run_turn_take(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    ids = read_ids(args.selections)
    print(f"Loaded {len(ids)} example ID(s): {ids}")
    get_silero_model()

    vad_kwargs = dict(
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
    )
    for plot_idx, ex_id in enumerate(ids, start=1):
        print(f"\n{'─'*60}")
        print(f"  Example {ex_id}  (plot {plot_idx:02d})")
        paths = turn_take_example_paths(ex_id, args.naive_dir, args.model_dir, args.model_subdir)
        if paths is None:
            continue

        print("  Loading audio...")
        wav_naive_a, sr_na = load_mono_channel(paths["naive_dual"], channel=0)
        wav_naive_b, sr_nb = load_mono_channel(paths["naive_dual"], channel=1)
        wav_prop_a, sr_pa = load_wav_mono(paths["model_ch_a"])
        wav_prop_b, sr_pb = load_wav_mono(paths["model_ch_b"])

        duration = max(
            len(wav_naive_a) / sr_na,
            len(wav_naive_b) / sr_nb,
            len(wav_prop_a) / sr_pa,
            len(wav_prop_b) / sr_pb,
        )

        print(f"  Running Silero VAD (duration={duration:.1f}s)...")
        segs = {
            "naive_a": vad_silero(wav_naive_a, sr_na, **vad_kwargs),
            "naive_b": vad_silero(wav_naive_b, sr_nb, **vad_kwargs),
            "proposed_a": vad_silero(wav_prop_a, sr_pa, **vad_kwargs),
            "proposed_b": vad_silero(wav_prop_b, sr_pb, **vad_kwargs),
        }
        for k, s in segs.items():
            print(f"    {k}: {len(s)} segment(s)")

        out_path = os.path.join(args.out_dir, f"vad_example_{plot_idx:02d}.png")
        plot_vad_turn_take(
            segs, duration,
            title=f"Voice Activity — example {ex_id}",
            out_path=out_path,
            save_pdf=args.save_pdf,
        )

    print(f"\nAll done. Plots in {args.out_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice-activity comparison figures.")
    sub = parser.add_subparsers(dest="command", required=True)

    bc = sub.add_parser("bc", help="Speaker turn / scheduler backchannel / PersonaPlex backchannel.")
    bc.add_argument("--sample_list", required=True,
                    help="Text file with one sample_<N> directory per line.")
    bc.add_argument("--personaplex_dir", required=True,
                    help="Root of the PersonaPlex outputs (contains <N>/output.wav).")
    bc.add_argument("--out_dir", required=True)
    bc.add_argument("--threshold", type=float, default=0.4,
                    help="Silero speech probability threshold (default: 0.4). "
                         "Lower is more sensitive and catches quieter backchannels.")
    bc.add_argument("--min_speech_ms", type=int, default=100,
                    help="Minimum speech segment duration in ms (default: 100).")
    bc.add_argument("--min_silence_ms", type=int, default=80,
                    help="Minimum silence gap that splits segments, in ms (default: 80).")
    bc.add_argument("--silence_first_n", type=float, default=4.0,
                    help="Seconds muted at the start of the PersonaPlex output (default: 4.0).")
    bc.set_defaults(func=run_bc)

    tt = sub.add_parser("turn-take", help="Naive A/B against scheduled A/B.")
    tt.add_argument("--selections", required=True,
                    help="Text file with one integer example ID per line.")
    tt.add_argument("--naive_dir", required=True,
                    help="Root of the naive segment_cache (example_XXXX/ folders).")
    tt.add_argument("--model_dir", required=True,
                    help="Root of the scheduler output (sample_N/ folders).")
    tt.add_argument("--model_subdir", default="segment_search_v2",
                    help="Subdirectory of sample_N/ holding channel_a.wav and channel_b.wav.")
    tt.add_argument("--out_dir", required=True)
    tt.add_argument("--threshold", type=float, default=0.2)
    tt.add_argument("--min_speech_ms", type=int, default=100)
    tt.add_argument("--min_silence_ms", type=int, default=80)
    tt.add_argument("--save_pdf", action="store_true",
                    help="Also save a vector PDF alongside each PNG.")
    tt.set_defaults(func=run_turn_take)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
