"""Backchannel metrics on the Full-Duplex-Bench ICC backchannel subset.

Scores the listener channel produced by the scheduler's backchannel-only
pipeline: Jensen-Shannon divergence between the predicted and the ground-truth
backchannel timing distributions (JSD), turn-over rate (TOR) and backchannel
frequency.

The metric definitions (0.2 s windows, linear resampling of the ground-truth
distribution, JSD via ``scipy``) follow ``evaluation/eval_backchannel.py`` of
Full-Duplex-Bench (https://github.com/DanielLin94144/Full-Duplex-Bench, commit
d7336bd, CC BY-NC 4.0). This script is an independent adapter and differs from
the upstream protocol in ways that change the scores:

* no ASR transcript is used. Upstream decides turn-over from the words inside
  each speech segment; here a segment longer than 3 s counts as taking the turn
  (TOR = 1) and every shorter segment counts as one backchannel;
* a missing prediction is skipped (``--missing skip``) and leaves the mean;
  ``--missing silent`` scores it as a silent listener (JSD 1, TOR 0, frequency 0);
* the VAD is run without telling it the sampling rate, so it assumes 16 kHz
  whatever the file's rate is. ``--resample_for_vad`` resamples to 16 kHz first.

See ``docs/evaluation.md`` for the end-to-end flow.

    python -m neural_scheduler.evaluation.fdb.eval_backchannel \
        --pred_root OUT/inference_output/<model> --gt_distribution icc_gt_distribution.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

WINDOW_SIZE = 0.2  # seconds per bin of the timing distribution
EPSILON = 1e-10
TIME_THRESHOLD = 3  # seconds; a longer speech segment is a turn, not a backchannel

LAYOUT_RELPATHS = {
    "scheduler": ("sample_{id}", "segment_search_v2/channel_b.wav"),
    "fdb": ("{id}", "output.wav"),
}


def split_backchannels(segments: Sequence[Dict[str, float]], time_threshold: float = TIME_THRESHOLD) -> Tuple[List[List[float]], int]:
    """Separate VAD segments into backchannels and turn-taking.

    Returns ``(backchannels, tor)`` where ``backchannels`` holds ``[start, end]``
    of every segment no longer than ``time_threshold`` and ``tor`` is 1 if any
    segment is longer.
    """
    backchannels = []
    tor = 0
    for segment in segments:
        start_time, end_time = segment["start"], segment["end"]
        if end_time - start_time > time_threshold:
            tor = 1
        else:
            backchannels.append([start_time, end_time])
    return backchannels, tor


def timing_jsd(backchannels: Sequence[Sequence[float]], duration_sec: float, gt_dist: Optional[Sequence[float]],
               window_size: float = WINDOW_SIZE) -> float:
    """Jensen-Shannon distance between predicted and ground-truth backchannel timing.

    The prediction is a histogram over ``window_size`` bins counting the
    backchannels active in each bin; the ground truth is resampled to the same
    number of bins. With no predicted backchannel the result is 1 and the
    ground truth is not consulted.
    """
    if len(backchannels) == 0:
        return 1
    if gt_dist is None:
        raise KeyError("no ground-truth distribution for a sample with predicted backchannels")
    time_intervals = [0 for _ in range(int(duration_sec / window_size) + 1)]
    for interval in backchannels:
        start = int(interval[0] / window_size)
        end = int(interval[1] / window_size)
        for i in range(start, end + 1):
            if i < len(time_intervals):
                time_intervals[i] += 1

    time_intervals = np.array(time_intervals) + EPSILON
    time_intervals = time_intervals / np.sum(time_intervals)

    x_gt = np.linspace(0, 1, len(gt_dist))
    x_pred = np.linspace(0, 1, len(time_intervals))
    gt_dist_resized = interp1d(x_gt, gt_dist, kind="linear", fill_value="extrapolate")(x_pred)
    return jensenshannon(time_intervals, gt_dist_resized)


def score_sample(segments: Sequence[Dict[str, float]], duration_sec: float, gt_dist: Optional[Sequence[float]]) -> Dict[str, float]:
    """JSD, TOR, backchannel frequency (per second) and count for one sample."""
    backchannels, tor = split_backchannels(segments)
    return {
        "jsd": timing_jsd(backchannels, duration_sec, gt_dist),
        "tor": tor,
        "frequency": len(backchannels) / duration_sec,
        "bc_count": len(backchannels),
    }


def summarize(per_sample: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    jsd = [r["jsd"] for r in per_sample.values()]
    tor = [r["tor"] for r in per_sample.values()]
    freq = [r["frequency"] for r in per_sample.values()]
    return {
        "jsd_mean": float(np.mean(jsd)), "jsd_std": float(np.std(jsd)),
        "tor_mean": float(np.mean(tor)), "tor_std": float(np.std(tor)),
        "frequency_mean": float(np.mean(freq)), "frequency_std": float(np.std(freq)),
        "num_samples": len(jsd),
    }


def eval_backchannel(
    pred_root: str,
    gt_distribution_path: str,
    layout: str = "scheduler",
    channel_relpath: Optional[str] = None,
    missing: str = "skip",
    resample_for_vad: bool = False,
    verbose: bool = True,
) -> Dict:
    """Score every sample directory under ``pred_root``.

    ``layout`` is ``"scheduler"`` (``sample_<id>/<channel_relpath>``) or
    ``"fdb"`` (``<id>/output.wav``, the upstream layout, for baselines). Sample
    ids are the trailing integer of each directory name and index the
    ground-truth distribution JSON.
    """
    import torchaudio
    from silero_vad import get_speech_timestamps, load_silero_vad

    dir_template, default_relpath = LAYOUT_RELPATHS[layout]
    relpath = channel_relpath or default_relpath

    with open(gt_distribution_path, "r") as f:
        gt_distribution = json.load(f)
    vad_model = load_silero_vad()

    per_sample: Dict[str, Dict[str, float]] = {}
    skipped = []
    for entry in os.listdir(pred_root):
        sample_id = entry.split("_")[-1]
        if not sample_id.isdigit():
            continue
        wav_path = os.path.join(pred_root, dir_template.format(id=sample_id), relpath)
        if not os.path.exists(wav_path):
            if missing == "skip":
                skipped.append(sample_id)
                if verbose:
                    print(f"Skipping {sample_id}: {relpath} not found.")
                continue
            result = {"jsd": 1, "tor": 0, "frequency": 0.0, "bc_count": 0}
        else:
            wav, sr = torchaudio.load(wav_path)
            duration_sec = wav.shape[-1] / sr
            if resample_for_vad and sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            segments = get_speech_timestamps(wav, vad_model, return_seconds=True)
            result = score_sample(segments, duration_sec, gt_distribution.get(sample_id))
        per_sample[sample_id] = result
        if verbose:
            print(f"Sample {sample_id} - JSD: {result['jsd']:.4f}, BC Count: {result['bc_count']}")

    summary = summarize(per_sample) if per_sample else {"num_samples": 0}
    summary["skipped"] = skipped
    if verbose and per_sample:
        print("\n" + "-" * 50)
        print("[Result]")
        print(f"JSD - Mean: {summary['jsd_mean']:.4f} ± {summary['jsd_std']:.4f}")
        print(f"TOR - Mean: {summary['tor_mean']:.4f} ± {summary['tor_std']:.4f}")
        print(f"Frequency - Mean: {summary['frequency_mean']:.4f} ± {summary['frequency_std']:.4f}")
        print(f"Number of samples: {summary['num_samples']} (skipped: {len(skipped)})")
        print("-" * 50)
    return {"summary": summary, "per_sample": per_sample}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_root", type=str, required=True,
                        help="Directory holding one sub-directory per sample.")
    parser.add_argument("--gt_distribution", type=str, required=True,
                        help="Ground-truth distribution JSON of the benchmark (icc_gt_distribution.json).")
    parser.add_argument("--layout", choices=sorted(LAYOUT_RELPATHS), default="scheduler",
                        help="'scheduler': sample_<id>/<channel_relpath>; 'fdb': <id>/output.wav.")
    parser.add_argument("--channel_relpath", type=str, default=None,
                        help="Listener wav inside each sample directory "
                             "(default: segment_search_v2/channel_b.wav, or output.wav for --layout fdb).")
    parser.add_argument("--missing", choices=["skip", "silent"], default="skip",
                        help="What to do when a sample has no listener wav.")
    parser.add_argument("--resample_for_vad", action="store_true",
                        help="Resample to 16 kHz before VAD instead of letting the VAD assume 16 kHz.")
    parser.add_argument("--output_json", type=str, default=None, help="Write summary and per-sample scores here.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    results = eval_backchannel(
        args.pred_root, args.gt_distribution,
        layout=args.layout, channel_relpath=args.channel_relpath,
        missing=args.missing, resample_for_vad=args.resample_for_vad,
    )
    if args.output_json:
        from ..common import to_serializable

        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(to_serializable(results), f, indent=2)


if __name__ == "__main__":
    main()
