"""Silero VAD and voice-activity bar plots shared by the paper figures."""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio

SILERO_SR = 16000

FONTSIZE = 7  # 7-8 pt, the label size of a two-column paper
DPI = 300
BAR_HEIGHT = 0.8

_silero_model = None
_silero_utils = None


def get_silero_model():
    """Load Silero VAD through ``torch.hub`` once and cache it.

    Returns ``(model, utils)`` where ``utils[0]`` is ``get_speech_timestamps``.
    """
    global _silero_model, _silero_utils
    if _silero_model is None:
        print("  Loading Silero VAD model (first call)...")
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _silero_model.eval()
    return _silero_model, _silero_utils


def load_wav_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Load a wav as a 1-D float tensor, averaging channels. Returns ``(waveform, sr)``."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), sr


def load_mono_channel(path: str, channel: int = 0) -> Tuple[torch.Tensor, int]:
    """Load one channel of a wav as a 1-D float tensor. Returns ``(waveform, sr)``."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] <= channel:
        raise ValueError(
            f"File has {waveform.shape[0]} channel(s), requested channel {channel}: {path}"
        )
    return waveform[channel], sr


def vad_silero(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.4,
    min_speech_ms: int = 100,
    min_silence_ms: int = 80,
) -> List[Tuple[float, float]]:
    """Run Silero VAD on a mono tensor; returns ``[(t_start, t_end), ...]`` in seconds."""
    model, utils = get_silero_model()
    get_speech_timestamps = utils[0]

    if sample_rate != SILERO_SR:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SILERO_SR
        )(waveform)

    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=SILERO_SR,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )

    return [(s["start"], s["end"]) for s in speech_timestamps]


def plot_activity_bars(
    segs: Dict[str, Sequence[Tuple[float, float]]],
    bar_y: Dict[str, float],
    colors: Dict[str, str],
    duration: float,
    title: str,
    out_path: str,
    fig_size: Tuple[float, float],
    ylim: Tuple[float, float],
    row_labels: Optional[Dict[str, str]] = None,
    divider_y: Optional[float] = None,
    integer_time_ticks: bool = False,
    save_pdf: bool = False,
) -> None:
    """Draw one horizontal activity bar row per track and save a PNG.

    ``segs`` maps a track key to its speech segments; ``bar_y`` and ``colors``
    give each key's row position and colour. With ``row_labels`` the y ticks
    are labelled by name, otherwise they show the row positions. ``divider_y``
    draws a dashed separator; ``save_pdf`` also writes a vector PDF next to
    the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for key, segments in segs.items():
        for (t0, t1) in segments:
            ax.barh(
                bar_y[key], t1 - t0, left=t0,
                height=BAR_HEIGHT, color=colors[key], alpha=0.85, linewidth=0,
            )

    if divider_y is not None:
        ax.axhline(divider_y, color="#cccccc", linewidth=0.5, linestyle="--")

    ax.set_yticks([bar_y[k] for k in bar_y])
    if row_labels is not None:
        ax.set_yticklabels([row_labels[k] for k in bar_y])
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(0, duration)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Time (s)")
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.4, linestyle="--")
    if integer_time_ticks:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_edgecolor("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)

    ax.set_title(title, pad=3)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if save_pdf:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"    Saved PDF -> {pdf_path}")
    plt.close(fig)
    print(f"    Saved -> {out_path}")
