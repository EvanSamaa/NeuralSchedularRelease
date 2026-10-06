"""Listening-test stimuli: A/B comparison videos and mixed audio.

Video sub-commands render, per item, a clip labelled "A", a silent pause and
a clip labelled "B", with the two systems assigned to A and B at random, and
write ``ab_mapping.json`` (the key needed by
:mod:`neural_scheduler.analysis.user_study`) next to the videos:

``videos-bc``
    Scheduler backchannels against PersonaPlex. Both mixes share the
    speaker's ``channel_a.wav``; the listener is ``channel_b.wav`` or
    ``<personaplex_dir>/<sample_id>/output.wav``.
``videos-turn-take``
    Naive concatenation (``combined_dual.wav``) against the scheduled
    dialogue (``channel_a.wav`` + ``channel_b.wav``).
``videos-paired``
    Arbitrary pairs of already-mixed wavs, one ``<wav_a>, <wav_b>`` per line.

Audio sub-commands export the same comparisons as plain wavs (for a demo
page or supplementary material): ``audio-bc``, ``audio-turn-take`` and
``audio-paired``.

Video audio is loudness-matched per channel to -23 LUFS with ffmpeg;
``audio-bc`` and ``audio-turn-take`` average the two channels without
loudness matching. ``ffmpeg`` and ``ffprobe`` must be on ``PATH`` for
everything except ``audio-bc`` and ``audio-turn-take``.

Pass ``--seed`` to make the A/B assignment reproducible; without it the
assignment differs on every run and ``ab_mapping.json`` is the only record.
"""

import argparse
import json
import os
import random
import tempfile

from ..viz import ffmpeg
from ._inputs import backchannel_sample_paths, read_ids, read_lines, read_pairs, report_missing, turn_take_example_paths

PERSONAPLEX_SILENCE_SEC = 4.0


def extract_label(wav_path: str) -> str:
    """Human-readable system label derived from a wav path.

    A path containing ``TTS_scenes/<name>/`` gives ``<name>``; a path
    containing a ``segment_search*`` directory gives the directory two levels
    above it; otherwise the parent directory name is used.
    """
    parts = wav_path.replace("\\", "/").split("/")

    if "TTS_scenes" in parts:
        idx = parts.index("TTS_scenes")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    for i, p in enumerate(parts):
        if p.startswith("segment_search"):
            if i >= 2:
                return parts[i - 2]
            elif i >= 1:
                return parts[i - 1]

    return os.path.basename(os.path.dirname(wav_path)) or "system"


def _write_mapping(mapping: list, out_dir: str) -> None:
    mapping_path = os.path.join(out_dir, "ab_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping written → {mapping_path}")
    print(f"All done. {len(mapping)} video(s) in {out_dir}/")


def _render(first_wav, first_label, second_wav, second_label, out_video, pause_sec, tmp) -> None:
    print(f"  A = {first_label}  |  B = {second_label}")
    print(f"  Rendering A, pause ({pause_sec}s), B → {out_video}")
    ffmpeg.render_ab_video(first_wav, second_wav, out_video, pause_sec, tmp)


# ---------------------------------------------------------------------------
# A/B videos
# ---------------------------------------------------------------------------

def make_bc_videos(sample_list: str, personaplex_dir: str, out_dir: str, pause_sec: float = 2.0) -> list:
    """Scheduler vs PersonaPlex backchannel videos; returns the A/B mapping."""
    os.makedirs(out_dir, exist_ok=True)
    sample_dirs = read_lines(sample_list)
    print(f"Loaded {len(sample_dirs)} sample(s).")

    mapping = []
    for video_idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"\n{'─'*60}")
        print(f"  Video {video_idx:02d}  →  {sample_dir}")
        print(f"{'─'*60}")
        paths = backchannel_sample_paths(sample_dir, personaplex_dir)
        if paths is None:
            continue
        ch_a, model_ch_b, personaplex_wav = paths["channel_a"], paths["model_b"], paths["personaplex"]
        video_file = f"bc_video_{video_idx:02d}.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            model_mixed = os.path.join(tmp, "model_mixed.wav")
            personaplex_mixed = os.path.join(tmp, "personaplex_mixed.wav")

            print("  Mixing model (channel_a + channel_b)…")
            ffmpeg.mix_two_mono_to_mono(
                ch_a, model_ch_b, model_mixed,
                label_a="channel_a", label_b="channel_b (model)",
            )

            print("  Mixing personaplex (channel_a + output.wav)…")
            personaplex_silenced = os.path.join(tmp, "personaplex_silenced.wav")
            ffmpeg.silence_first_n_seconds(personaplex_wav, personaplex_silenced, n=PERSONAPLEX_SILENCE_SEC)
            ffmpeg.mix_two_mono_to_mono(
                ch_a, personaplex_silenced, personaplex_mixed,
                label_a="channel_a", label_b="output.wav (personaplex)",
            )

            model_is_A = random.random() < 0.5
            if model_is_A:
                first_wav, first_label = model_mixed, "model"
                second_wav, second_label = personaplex_mixed, "personaplex"
            else:
                first_wav, first_label = personaplex_mixed, "personaplex"
                second_wav, second_label = model_mixed, "model"

            _render(first_wav, first_label, second_wav, second_label,
                    os.path.join(out_dir, video_file), pause_sec, tmp)

        mapping.append({
            "video_file": video_file,
            "sample_dir": sample_dir,
            "sample_id": paths["sample_id"],
            "search_subdir": paths["search_subdir"],
            "A": first_label,
            "B": second_label,
            "model_is_A": model_is_A,
            "channel_a_wav": ch_a,
            "model_ch_b": model_ch_b,
            "personaplex_wav": personaplex_wav,
        })

    _write_mapping(mapping, out_dir)
    return mapping


def make_turn_take_videos(
    selections: str, naive_dir: str, model_dir: str, out_dir: str,
    model_subdir: str = "segment_search_v2", pause_sec: float = 2.0,
    video_prefix: str = "turntake_video_",
) -> list:
    """Naive vs scheduled turn-taking videos; returns the A/B mapping."""
    os.makedirs(out_dir, exist_ok=True)
    ids = read_ids(selections)
    print(f"Loaded {len(ids)} example IDs: {ids}")

    mapping = []
    for video_idx, ex_id in enumerate(ids, start=1):
        print(f"\n{'─'*60}")
        print(f"  Video {video_idx:02d}  (example ID {ex_id})")
        print(f"{'─'*60}")
        paths = turn_take_example_paths(ex_id, naive_dir, model_dir, model_subdir)
        if paths is None:
            continue
        video_file = f"{video_prefix}{video_idx:02d}.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            naive_mixed = os.path.join(tmp, "naive_mixed.wav")
            model_mixed = os.path.join(tmp, "model_mixed.wav")

            print("  Mixing naive dual → mono…")
            ffmpeg.mix_dual_wav_to_mono(paths["naive_dual"], naive_mixed, label="naive")

            print("  Mixing model channels → mono…")
            ffmpeg.mix_two_mono_to_mono(
                paths["model_ch_a"], paths["model_ch_b"], model_mixed,
                label_a="model ch_a", label_b="model ch_b",
            )

            naive_is_A = random.random() < 0.5
            if naive_is_A:
                first_wav, first_label = naive_mixed, "naive"
                second_wav, second_label = model_mixed, "model"
            else:
                first_wav, first_label = model_mixed, "model"
                second_wav, second_label = naive_mixed, "naive"

            _render(first_wav, first_label, second_wav, second_label,
                    os.path.join(out_dir, video_file), pause_sec, tmp)

        mapping.append({
            "video_file": video_file,
            "example_id": ex_id,
            "A": first_label,
            "B": second_label,
            "naive_is_A": naive_is_A,
            "naive_dual": paths["naive_dual"],
            "model_ch_a": paths["model_ch_a"],
            "model_ch_b": paths["model_ch_b"],
        })

    _write_mapping(mapping, out_dir)
    return mapping


def _mono_input(raw_wav: str, tmp_path: str, name: str, target_sr: int) -> str:
    """Return a mono version of ``raw_wav`` (the file itself if already mono)."""
    channels = ffmpeg.get_channels(raw_wav)
    if channels == 1:
        return raw_wav
    print(f"  {name} has {channels} channels → downmixing to mono")
    ffmpeg.to_mono(raw_wav, tmp_path, target_sr)
    return tmp_path


def make_paired_videos(sample_list: str, out_dir: str, pause_sec: float = 2.0) -> list:
    """A/B videos from ``<wav_a>, <wav_b>`` pairs; returns the A/B mapping.

    Each wav is downmixed to mono if needed and normalised to -23 LUFS on its
    own. ``first_is_wav_a`` in the mapping tells which file was shown as A.
    """
    os.makedirs(out_dir, exist_ok=True)
    pairs = read_pairs(sample_list)
    print(f"Loaded {len(pairs)} pair(s).")

    mapping = []
    for video_idx, (raw_wav_a, raw_wav_b) in enumerate(pairs, start=1):
        print(f"\n{'─'*60}")
        print(f"  Video {video_idx:02d}")
        print(f"    wav_a: {raw_wav_a}")
        print(f"    wav_b: {raw_wav_b}")
        print(f"{'─'*60}")
        if report_missing([(raw_wav_a, "wav_a"), (raw_wav_b, "wav_b")]):
            continue

        label_a = extract_label(raw_wav_a)
        label_b = extract_label(raw_wav_b)
        print(f"  Labels: [{label_a}] vs [{label_b}]")
        video_file = f"video_{video_idx:02d}.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            mono_a = _mono_input(raw_wav_a, os.path.join(tmp, "mono_a.wav"), "wav_a", ffmpeg.STIMULUS_SR)
            mono_b = _mono_input(raw_wav_b, os.path.join(tmp, "mono_b.wav"), "wav_b", ffmpeg.STIMULUS_SR)

            norm_a = os.path.join(tmp, "norm_a.wav")
            norm_b = os.path.join(tmp, "norm_b.wav")
            ffmpeg.normalise(mono_a, norm_a, label=label_a)
            ffmpeg.normalise(mono_b, norm_b, label=label_b)

            first_is_a = random.random() < 0.5
            if first_is_a:
                first_wav, first_label = norm_a, label_a
                second_wav, second_label = norm_b, label_b
            else:
                first_wav, first_label = norm_b, label_b
                second_wav, second_label = norm_a, label_a

            _render(first_wav, first_label, second_wav, second_label,
                    os.path.join(out_dir, video_file), pause_sec, tmp)

        mapping.append({
            "video_file": video_file,
            "wav_a": raw_wav_a,
            "wav_b": raw_wav_b,
            "label_a": label_a,
            "label_b": label_b,
            "A": first_label,
            "B": second_label,
            "first_is_wav_a": first_is_a,
        })

    _write_mapping(mapping, out_dir)
    return mapping


# ---------------------------------------------------------------------------
# Mixed audio
# ---------------------------------------------------------------------------

def _resample_if_needed(waveform, src_sr: int, tgt_sr: int):
    import torchaudio

    if src_sr == tgt_sr:
        return waveform
    return torchaudio.transforms.Resample(src_sr, tgt_sr)(waveform)


def mix_average(a, b):
    """Zero-pad two 1-D tensors to equal length and average them."""
    import torch

    if len(a) < len(b):
        a = torch.nn.functional.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a):
        b = torch.nn.functional.pad(b, (0, len(a) - len(b)))
    return 0.5 * a + 0.5 * b


def _save_wav(path: str, waveform, sr: int) -> None:
    import torchaudio

    torchaudio.save(path, waveform.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16)
    print(f"    Saved -> {path}")


def export_bc_audio(
    sample_list: str, personaplex_dir: str, out_dir: str,
    silence_first_n: float = PERSONAPLEX_SILENCE_SEC, target_sr: int = 22050,
) -> int:
    """Write ``sample_<id>_proposed.wav`` and ``sample_<id>_personaplex.wav`` per sample.

    Returns the number of wavs written.
    """
    from ..viz.vad_plot import load_wav_mono

    os.makedirs(out_dir, exist_ok=True)
    sample_dirs = read_lines(sample_list)
    print(f"Loaded {len(sample_dirs)} sample(s). Expecting {len(sample_dirs) * 2} output files.")

    saved = 0
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"\n{'─'*60}")
        print(f"  Sample {idx:02d}  ->  {sample_dir}")
        paths = backchannel_sample_paths(sample_dir, personaplex_dir)
        if paths is None:
            continue

        wav_a, sr_a = load_wav_mono(paths["channel_a"])
        wav_model, sr_model = load_wav_mono(paths["model_b"])
        wav_pp, sr_pp = load_wav_mono(paths["personaplex"])

        wav_pp = wav_pp.clone()
        wav_pp[:int(silence_first_n * sr_pp)] = 0.0

        wav_a = _resample_if_needed(wav_a, sr_a, target_sr)
        wav_model = _resample_if_needed(wav_model, sr_model, target_sr)
        wav_pp = _resample_if_needed(wav_pp, sr_pp, target_sr)

        sample_id = paths["sample_id"]
        _save_wav(os.path.join(out_dir, f"sample_{sample_id}_proposed.wav"), mix_average(wav_a, wav_model), target_sr)
        _save_wav(os.path.join(out_dir, f"sample_{sample_id}_personaplex.wav"), mix_average(wav_a, wav_pp), target_sr)
        saved += 2

    print(f"\nDone. {saved} WAV files written to {out_dir}/")
    return saved


def export_turn_take_audio(
    selections: str, naive_dir: str, model_dir: str, out_dir: str,
    model_subdir: str = "segment_search_v2", target_sr: int = 22050, max_samples: int = 5,
) -> int:
    """Write ``example_XXXX_naive.wav`` and ``example_XXXX_proposed.wav`` for the
    first ``max_samples`` ids. Returns the number of wavs written."""
    from ..viz.vad_plot import load_mono_channel, load_wav_mono

    os.makedirs(out_dir, exist_ok=True)
    ids = read_ids(selections)[:max_samples]
    print(f"Processing first {len(ids)} example(s). Expecting {len(ids) * 2} output files.")

    saved = 0
    for ex_id in ids:
        print(f"\n{'─'*60}")
        print(f"  Example {ex_id}")
        paths = turn_take_example_paths(ex_id, naive_dir, model_dir, model_subdir)
        if paths is None:
            continue

        wav_naive_a, sr_na = load_mono_channel(paths["naive_dual"], channel=0)
        wav_naive_b, sr_nb = load_mono_channel(paths["naive_dual"], channel=1)
        wav_prop_a, sr_pa = load_wav_mono(paths["model_ch_a"])
        wav_prop_b, sr_pb = load_wav_mono(paths["model_ch_b"])

        wav_naive_a = _resample_if_needed(wav_naive_a, sr_na, target_sr)
        wav_naive_b = _resample_if_needed(wav_naive_b, sr_nb, target_sr)
        wav_prop_a = _resample_if_needed(wav_prop_a, sr_pa, target_sr)
        wav_prop_b = _resample_if_needed(wav_prop_b, sr_pb, target_sr)

        _save_wav(os.path.join(out_dir, f"example_{ex_id:04d}_naive.wav"), mix_average(wav_naive_a, wav_naive_b), target_sr)
        _save_wav(os.path.join(out_dir, f"example_{ex_id:04d}_proposed.wav"), mix_average(wav_prop_a, wav_prop_b), target_sr)
        saved += 2

    print(f"\nDone. {saved} WAV files written to {out_dir}/")
    return saved


def export_paired_audio(sample_list: str, out_dir: str, target_sr: int = 22050, number_offset: int = 5) -> int:
    """Write loudness-normalised ``sample_NN_proposed.wav`` (from ``wav_a``) and
    ``sample_NN_naive.wav`` (from ``wav_b``) per pair, numbered from
    ``number_offset``. Returns the number of wavs written."""
    os.makedirs(out_dir, exist_ok=True)
    pairs = read_pairs(sample_list)
    print(f"Loaded {len(pairs)} pair(s). Expecting {len(pairs) * 2} output files.")

    saved = 0
    for idx, (raw_wav_a, raw_wav_b) in enumerate(pairs):
        file_num = idx + number_offset
        print(f"\n{'─'*60}")
        print(f"  Pair {idx}  (file number {file_num:02d})")
        print(f"    wav_a: {raw_wav_a}")
        print(f"    wav_b: {raw_wav_b}")
        if report_missing([(raw_wav_a, "wav_a"), (raw_wav_b, "wav_b")]):
            continue

        out_a = os.path.join(out_dir, f"sample_{file_num:02d}_proposed.wav")
        out_b = os.path.join(out_dir, f"sample_{file_num:02d}_naive.wav")
        with tempfile.TemporaryDirectory() as tmp:
            mono_a = _mono_input(raw_wav_a, os.path.join(tmp, "mono_a.wav"), "wav_a", target_sr)
            mono_b = _mono_input(raw_wav_b, os.path.join(tmp, "mono_b.wav"), "wav_b", target_sr)
            ffmpeg.normalise(mono_a, out_a, target_sr, label=f"sample_{file_num:02d}_proposed")
            ffmpeg.normalise(mono_b, out_b, target_sr, label=f"sample_{file_num:02d}_naive")
        saved += 2

    print(f"\nDone. {saved} WAV files written to {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_video_args(p) -> None:
    p.add_argument("--out_dir", required=True, help="Output directory for videos and ab_mapping.json.")
    p.add_argument("--pause_sec", type=float, default=2.0,
                   help="Silent pause between the A and B clips (default: 2.0 s).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for a reproducible A/B assignment (default: unseeded).")


def _add_bc_inputs(p) -> None:
    p.add_argument("--sample_list", required=True,
                   help="Text file with one sample_<N> directory per line.")
    p.add_argument("--personaplex_dir", required=True,
                   help="Root of the PersonaPlex outputs (contains <N>/output.wav).")


def _add_turn_take_inputs(p) -> None:
    p.add_argument("--selections", required=True,
                   help="Text file with one integer example ID per line.")
    p.add_argument("--naive_dir", required=True,
                   help="Root of the naive segment_cache (example_XXXX/ folders).")
    p.add_argument("--model_dir", required=True,
                   help="Root of the scheduler output (sample_N/ folders).")
    p.add_argument("--model_subdir", default="segment_search_v2",
                   help="Subdirectory of sample_N/ holding channel_a.wav and channel_b.wav.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listening-test stimuli: A/B videos and mixed audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("videos-bc", help="A/B videos: scheduler vs PersonaPlex backchannels.")
    _add_bc_inputs(p)
    _add_video_args(p)
    p.set_defaults(func=lambda a: make_bc_videos(a.sample_list, a.personaplex_dir, a.out_dir, a.pause_sec))

    p = sub.add_parser("videos-turn-take", help="A/B videos: naive concatenation vs scheduler.")
    _add_turn_take_inputs(p)
    _add_video_args(p)
    p.add_argument("--video_prefix", default="turntake_video_",
                   help="File name prefix of the videos (default: turntake_video_; "
                        "use video_ for the plain naming).")
    p.set_defaults(func=lambda a: make_turn_take_videos(
        a.selections, a.naive_dir, a.model_dir, a.out_dir, a.model_subdir, a.pause_sec, a.video_prefix))

    p = sub.add_parser("videos-paired", help="A/B videos from '<wav_a>, <wav_b>' pairs.")
    p.add_argument("--sample_list", required=True, help="Text file; each line: <wav_a>, <wav_b>")
    _add_video_args(p)
    p.set_defaults(func=lambda a: make_paired_videos(a.sample_list, a.out_dir, a.pause_sec))

    p = sub.add_parser("audio-bc", help="Mixed wavs: scheduler vs PersonaPlex backchannels.")
    _add_bc_inputs(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--silence_first_n", type=float, default=PERSONAPLEX_SILENCE_SEC,
                   help="Seconds muted at the start of the PersonaPlex output (default: 4.0).")
    p.add_argument("--target_sr", type=int, default=22050,
                   help="Sample rate of the output files (default: 22050).")
    p.set_defaults(func=lambda a: export_bc_audio(
        a.sample_list, a.personaplex_dir, a.out_dir, a.silence_first_n, a.target_sr))

    p = sub.add_parser("audio-turn-take", help="Mixed wavs: naive vs scheduler.")
    _add_turn_take_inputs(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_sr", type=int, default=22050,
                   help="Sample rate of the output files (default: 22050).")
    p.add_argument("--max_samples", type=int, default=5,
                   help="Only the first N ids of --selections are exported (default: 5).")
    p.set_defaults(func=lambda a: export_turn_take_audio(
        a.selections, a.naive_dir, a.model_dir, a.out_dir, a.model_subdir, a.target_sr, a.max_samples))

    p = sub.add_parser("audio-paired", help="Normalised wavs from '<wav_a>, <wav_b>' pairs.")
    p.add_argument("--sample_list", required=True, help="Text file; each line: <wav_a>, <wav_b>")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_sr", type=int, default=22050)
    p.add_argument("--number_offset", type=int, default=5,
                   help="Number given to the first pair's files (default: 5).")
    p.set_defaults(func=lambda a: export_paired_audio(a.sample_list, a.out_dir, a.target_sr, a.number_offset))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
