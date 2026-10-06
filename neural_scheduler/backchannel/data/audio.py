"""Audio loading shared by training and inference.

Everything is returned as mono float32 at the HuBERT sample rate, truncated to
a maximum duration. Resampling is linear interpolation; Seamless Interaction is
already 16 kHz, so this only matters for synthesised voice lines at inference.
"""

from typing import Optional

import numpy as np
import soundfile as sf
import torch

from ..constants import HUBERT_SR, MAX_AUDIO_SECONDS_DEFAULT

_SILENCE = np.zeros(1, dtype=np.float32)


def _resample_and_cap(audio: np.ndarray, sr: int, target_sr: int, max_seconds: float) -> np.ndarray:
    if sr != target_sr:
        n_target = int(round(len(audio) * target_sr / sr))
        if n_target <= 1:
            return _SILENCE.copy()
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=n_target, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    max_samples = int(max_seconds * target_sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


def load_audio_span(
    wav_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    target_sr: int = HUBERT_SR,
    max_seconds: float = MAX_AUDIO_SECONDS_DEFAULT,
) -> np.ndarray:
    """Read ``[start_sec, end_sec]`` of a wav file (the whole file when both are
    ``None``) without decoding the rest. An empty span yields one sample of
    silence so downstream tensors are never empty."""
    info = sf.info(wav_path)
    file_sr = info.samplerate

    start_frame = 0 if start_sec is None else max(0, int(round(float(start_sec) * file_sr)))
    end_frame = info.frames if end_sec is None else min(info.frames, int(round(float(end_sec) * file_sr)))
    if end_frame <= start_frame:
        return _SILENCE.copy()

    audio, sr = sf.read(wav_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return _resample_and_cap(audio, sr, target_sr, max_seconds)


def waveform_to_target_sr(
    wav: torch.Tensor,
    src_sr: int,
    target_sr: int = HUBERT_SR,
    max_seconds: float = MAX_AUDIO_SECONDS_DEFAULT,
) -> np.ndarray:
    """Convert an in-memory waveform ``[C, T]`` or ``[T]`` to mono float32 at ``target_sr``."""
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    arr = wav.detach().cpu().to(torch.float32).numpy()
    return _resample_and_cap(arr, src_sr, target_sr, max_seconds)
