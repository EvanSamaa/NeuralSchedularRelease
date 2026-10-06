"""Step 4: turn a schedule back into audio.

No vocoder is involved. Step 2 stored, for every speech token of each
speaker's script, which 80 ms slice of that speaker's original audio it came
from. The schedule is a list of codec ids per channel; each id is matched, in
order, to the next script token with that id and the corresponding audio slice
is copied. Silence in the schedule becomes digital silence.

Files read from ``<cache_dir>/source_map/`` (per channel ``a`` / ``b``):
``concatenated_audio_*.wav`` (16 kHz), ``raw_tokens_*.json`` (codec ids of that
audio), ``script_tokens_*.json`` and ``script_source_map_*.json`` (for each
script position, the index into ``raw_tokens`` or null).
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..constants import SAMPLES_PER_TOKEN, SILENCE_CODE, SPEECH_TOKENIZER_SR

_logger = logging.getLogger(__name__)


@dataclass
class SourceMapData:
    """Source-map files of one speaker."""

    concatenated_audio: np.ndarray  # 1-D float32
    raw_tokens: List[int]  # codec ids, one per SAMPLES_PER_TOKEN samples
    script_tokens: List[int]  # vocab ids of the script
    script_source_map: List[Optional[int]]  # index into raw_tokens per script position
    sr: int = SPEECH_TOKENIZER_SR


def load_source_map(source_map_dir: str) -> Tuple[SourceMapData, SourceMapData]:
    """Load the source maps of channel A and channel B."""

    def load_json(name: str):
        with open(os.path.join(source_map_dir, name)) as f:
            return json.load(f)

    def load_channel(suffix: str) -> SourceMapData:
        audio, sr = sf.read(os.path.join(source_map_dir, f"concatenated_audio_{suffix}.wav"))
        assert sr == SPEECH_TOKENIZER_SR, f"Expected {SPEECH_TOKENIZER_SR} Hz, got {sr}"
        return SourceMapData(
            concatenated_audio=audio.astype(np.float32),
            raw_tokens=load_json(f"raw_tokens_{suffix}.json"),
            script_tokens=load_json(f"script_tokens_{suffix}.json"),
            script_source_map=load_json(f"script_source_map_{suffix}.json"),
            sr=sr,
        )

    return load_channel("a"), load_channel("b")


def _audio_slice(audio: np.ndarray, raw_idx: int) -> np.ndarray:
    """The 80 ms of audio behind raw token ``raw_idx``, zero-padded at the end of the file."""
    start = raw_idx * SAMPLES_PER_TOKEN
    end = start + SAMPLES_PER_TOKEN
    if end <= len(audio):
        return audio[start:end].copy()
    padded = np.zeros(SAMPLES_PER_TOKEN, dtype=np.float32)
    remaining = len(audio) - start
    if remaining > 0:
        padded[:remaining] = audio[start:]
    return padded


def reconstruct_channel_audio(
    scheduled_tokens: List[int],
    source: SourceMapData,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Waveform (16 kHz float32) for one channel of a schedule.

    ``scheduled_tokens`` are codec ids. A pointer walks through the script's
    speech tokens: each scheduled token takes the next script token with the
    same codec id at or after the pointer. If there is none ahead, the first
    match before the pointer is used without moving it; if there is none at
    all, the step is silent.
    """
    logger = logger or _logger
    script_seq = [
        (source.raw_tokens[raw_idx], raw_idx)
        for raw_idx in source.script_source_map
        if raw_idx is not None
    ]
    silence_slice = np.zeros(SAMPLES_PER_TOKEN, dtype=np.float32)

    ptr = 0
    output_slices: List[np.ndarray] = []
    matched = silence_count = fallback_count = unmatched_count = 0

    for tok in scheduled_tokens:
        if tok == SILENCE_CODE:
            output_slices.append(silence_slice)
            silence_count += 1
            continue

        ahead = next((j for j in range(ptr, len(script_seq)) if script_seq[j][0] == tok), None)
        if ahead is not None:
            output_slices.append(_audio_slice(source.concatenated_audio, script_seq[ahead][1]))
            ptr = ahead + 1
            matched += 1
            continue

        behind = next((j for j in range(0, ptr) if script_seq[j][0] == tok), None)
        if behind is not None:
            output_slices.append(_audio_slice(source.concatenated_audio, script_seq[behind][1]))
            fallback_count += 1
            continue

        logger.debug(f"Unmatched token {tok} at output position {len(output_slices)} "
                     f"(pointer {ptr}/{len(script_seq)})")
        output_slices.append(silence_slice)
        unmatched_count += 1

    logger.info(
        f"Reconstruction: {len(scheduled_tokens)} scheduled tokens -> {matched} matched, "
        f"{silence_count} silence, {fallback_count} out of order, {unmatched_count} unmatched"
    )
    if not output_slices:
        return np.zeros(SAMPLES_PER_TOKEN, dtype=np.float32)
    return np.concatenate(output_slices)


def _save(audio: np.ndarray, path: str, sr: int, output_sr: int) -> None:
    if output_sr == sr:
        sf.write(path, audio, sr)
        return
    import torch
    import torchaudio

    resampled = torchaudio.transforms.Resample(sr, output_sr)(torch.from_numpy(audio).unsqueeze(0))
    torchaudio.save(path, resampled, output_sr)


def reconstruct_dialogue(
    channel_a_scheduled: List[int],
    channel_b_scheduled: List[int],
    source_map_dir: str,
    sr: int = SPEECH_TOKENIZER_SR,
    output_dir: Optional[str] = None,
    output_sr: int = SPEECH_TOKENIZER_SR,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct both channels; returns ``(wav_a, wav_b)`` at 16 kHz.

    With ``output_dir``, also writes ``channel_a.wav``, ``channel_b.wav`` and
    their sum ``channel_sum.wav``, resampled to ``output_sr``.
    """
    logger = logger or _logger
    source_a, source_b = load_source_map(source_map_dir)
    wav_a = reconstruct_channel_audio(channel_a_scheduled, source_a, logger)
    wav_b = reconstruct_channel_audio(channel_b_scheduled, source_b, logger)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        min_len = min(len(wav_a), len(wav_b))
        for name, audio in (
            ("channel_a.wav", wav_a),
            ("channel_b.wav", wav_b),
            ("channel_sum.wav", wav_a[:min_len] + wav_b[:min_len]),
        ):
            _save(audio, os.path.join(output_dir, name), sr, output_sr)
            logger.info(f"  Saved {os.path.join(output_dir, name)}")

    return wav_a, wav_b


def reconstruct_dialogue_from_cache(
    channel_a_scheduled: List[int],
    channel_b_scheduled: List[int],
    cache_dir: str,
    output_dir: Optional[str] = None,
    output_sr: int = 22050,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`reconstruct_dialogue` for a segment-cache directory written by steps 1 and 2."""
    source_map_dir = os.path.join(cache_dir, "source_map")
    if not os.path.isdir(source_map_dir):
        raise FileNotFoundError(
            f"Source map directory not found: {source_map_dir}\n"
            f"Run the tokenisation step (neural_scheduler.pipeline.tokenize) on this cache first."
        )
    return reconstruct_dialogue(
        channel_a_scheduled=channel_a_scheduled,
        channel_b_scheduled=channel_b_scheduled,
        source_map_dir=source_map_dir,
        output_dir=output_dir,
        output_sr=output_sr,
        logger=logger,
    )
