"""Turn one 20.48 s two-channel audio window into a training record.

Sequence layout::

    <|start-of-script-A|> script_A <|end-of-script-A|>
    <|start-of-script-B|> script_B <|end-of-script-B|>
    <|start-of-dialogue|> A0 B0 A1 B1 ... <|end-of-dialogue|>

The dialogue region is time aligned (12.5 Hz, A/B interleaved); every frame
outside a VAD segment is the silence token. The script regions hold the same
speech tokens with the timing removed: each turn is wrapped in
``<|start-turn-N|> ... <|end-turn-N|>`` and separated by single silence tokens.
"""

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

from ..constants import (
    GLM4_VOICE_MODEL_ID,
    MAX_TOKEN_SIZE,
    MAX_TURNS,
    NUM_CHANNELS,
    SILENCE_CODE,
    TOKEN_RATE_HZ,
)
from ..vocab import add_scheduler_tokens, audio_token
from .durations import compute_remaining_durations_dual_channel
from .speech_tokenizer import extract_speech_token_batched
from .turns import assign_turn_ids
from .vad import (
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
    vad_timestamps_to_seconds,
    vad_timestamps_to_token_indices,
)

# Tokens of one channel are interleaved with the other channel's.
INTERLEAVE_FACTOR = 2
# The one codec id that stock GLM-4-Voice has no <|audio_k|> token for.
UNSUPPORTED_AUDIO_CODE = 16383
TURN_OVERLAP_THRESHOLD = 0.5


class UnknownAudioCodeError(KeyError):
    """A codec id has no ``<|audio_k|>`` token in the tokenizer."""

    def __init__(self, code: int):
        super().__init__(audio_token(code))
        self.code = code

    def __str__(self) -> str:
        return f"no vocabulary entry for {audio_token(self.code)}"


def load_text_tokenizer(model_id: str = GLM4_VOICE_MODEL_ID, add_audio_16383: bool = False):
    """Load the GLM-4-Voice tokenizer with the scheduler's special tokens.

    ``add_audio_16383`` additionally appends ``<|audio_16383|>`` *after* the turn
    tokens (id 168842), which leaves every other id unchanged. Released data and
    checkpoints were produced without it.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    add_scheduler_tokens(tokenizer)
    if add_audio_16383:
        tokenizer.add_tokens([audio_token(UNSUPPORTED_AUDIO_CODE)])
    return tokenizer


def audio_code_to_token_id(tokenizer, code: int) -> int:
    """Vocab id of ``<|audio_{code}|>``; raises :class:`UnknownAudioCodeError` if absent."""
    try:
        return tokenizer.convert_tokens_to_ids([audio_token(code)])[0]
    except KeyError:
        raise UnknownAudioCodeError(code) from None


def silencing_with_vad(tokens: List[int], vad_timestamps_tokens: List[Dict[str, int]]) -> List[int]:
    """Replace every codec id outside the VAD segments with the silence code."""
    output = [SILENCE_CODE] * len(tokens)

    for ts in vad_timestamps_tokens:
        start = max(0, ts["start"])
        end = min(len(tokens), ts["end"])
        for i in range(start, end):
            output[i] = tokens[i]

    return output


def build_script_tokens_for_channel(
    raw_tokens: List[int],
    channel_segments: List[Dict],
    tokenizer,
    special_tokens: Dict[str, int],
) -> List[int]:
    """Build one channel's script region (vocab ids) from its turns.

    ``raw_tokens`` are the channel's codec ids before silencing and
    ``channel_segments`` its turns from :func:`~.turns.assign_turn_ids`.

    Structure::

        SIL <|start-turn-N|> speech <|end-turn-N|> SIL <|start-turn-M|> speech <|end-turn-M|> SIL

    Sub-segments of a merged turn are separated by a single ``SIL``. Turns with
    an id of ``MAX_TURNS`` or more have no marker tokens and are left out.
    """
    silence_id = special_tokens["silence"]
    script = [silence_id]

    sorted_segments = sorted(channel_segments, key=lambda s: s["sub_segments"][0]["start"])

    for seg in sorted_segments:
        turn_id = seg["turn_id"]

        if turn_id >= MAX_TURNS:
            continue

        script.append(special_tokens[f"start_turn_{turn_id}"])

        for sub_idx, sub_seg in enumerate(seg["sub_segments"]):
            if sub_idx > 0:
                script.append(silence_id)

            start_tok = max(0, sub_seg["start"])
            end_tok = min(len(raw_tokens), sub_seg["end"])

            for i in range(start_tok, end_tok):
                script.append(audio_code_to_token_id(tokenizer, raw_tokens[i]))

        script.append(special_tokens[f"end_turn_{turn_id}"])
        script.append(silence_id)

    return script


def process_audio_window(
    audio_left: torch.Tensor,
    audio_right: torch.Tensor,
    sr: int,
    whisper_model,
    feature_extractor,
    tokenizer,
    special_tokens: Dict[str, int],
    vad_model,
    get_speech_timestamps: Callable,
    logger: logging.Logger,
    example_id,
    window_idx: int,
    window_start_sec: float,
    window_end_sec: float,
    device: str = "cuda",
) -> Optional[Dict]:
    """Process one window of two time-aligned mono waveforms.

    Steps: VAD on both channels; WhisperVQ tokens for both channels; silence the
    dialogue outside VAD; assign turn ids; build the script regions; interleave
    the dialogue; compute remaining-duration labels.

    Returns the record dict, or ``None`` when neither channel contains speech.
    Raises :class:`UnknownAudioCodeError` when a speech frame carries a codec id
    the tokenizer has no token for.
    """
    vad_timestamps_left = get_speech_timestamps(
        audio_left.numpy(), vad_model, sampling_rate=sr,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    )
    vad_timestamps_right = get_speech_timestamps(
        audio_right.numpy(), vad_model, sampling_rate=sr,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    )

    if len(vad_timestamps_left) == 0 and len(vad_timestamps_right) == 0:
        logger.debug(f"Example {example_id}, window {window_idx}: No speech detected, skipping")
        return None

    sample_both = torch.stack([audio_left, audio_right], dim=0)
    tokens_both = extract_speech_token_batched(
        model=whisper_model, feature_extractor=feature_extractor,
        batched_audio=sample_both, sampling_rate=sr,
        model_batch_size=32, device=device, logger=logger,
    )
    tokens_left_raw = tokens_both[0]
    tokens_right_raw = tokens_both[1]

    vad_left_tokens = vad_timestamps_to_token_indices(vad_timestamps_left, sr)
    vad_right_tokens = vad_timestamps_to_token_indices(vad_timestamps_right, sr)
    vad_left_seconds = vad_timestamps_to_seconds(vad_timestamps_left, sr)
    vad_right_seconds = vad_timestamps_to_seconds(vad_timestamps_right, sr)

    tokens_left_silenced = silencing_with_vad(tokens_left_raw, vad_left_tokens)
    tokens_right_silenced = silencing_with_vad(tokens_right_raw, vad_right_tokens)

    all_segments_with_turns = assign_turn_ids(
        vad_left_tokens, vad_right_tokens, overlap_threshold=TURN_OVERLAP_THRESHOLD
    )
    channel_a_segments = [s for s in all_segments_with_turns if s["channel"] == "A"]
    channel_b_segments = [s for s in all_segments_with_turns if s["channel"] == "B"]

    script_a_tokens = build_script_tokens_for_channel(
        tokens_left_raw, channel_a_segments, tokenizer, special_tokens
    )
    script_b_tokens = build_script_tokens_for_channel(
        tokens_right_raw, channel_b_segments, tokenizer, special_tokens
    )

    dialogue_tokens = []
    for t_left, t_right in zip(tokens_left_silenced, tokens_right_silenced):
        dialogue_tokens.append(audio_code_to_token_id(tokenizer, t_left))
        dialogue_tokens.append(audio_code_to_token_id(tokenizer, t_right))

    input_ids = []

    input_ids.append(special_tokens["start_script_a"])
    input_ids.extend(script_a_tokens)
    input_ids.append(special_tokens["end_script_a"])

    input_ids.append(special_tokens["start_script_b"])
    input_ids.extend(script_b_tokens)
    input_ids.append(special_tokens["end_script_b"])

    input_ids.append(special_tokens["start_dialogue"])
    generation_start_index = len(input_ids)
    input_ids.extend(dialogue_tokens)
    generation_end_index = len(input_ids)
    input_ids.append(special_tokens["end_dialogue"])

    remaining_durations = compute_remaining_durations_dual_channel(dialogue_tokens)

    turn_boundaries = []
    for seg in all_segments_with_turns:
        for sub in seg["sub_segments"]:
            turn_boundaries.append({
                "turn_id": seg["turn_id"],
                "channel": seg["channel"],
                "start_sec": sub["start"] / TOKEN_RATE_HZ,
                "end_sec": sub["end"] / TOKEN_RATE_HZ,
            })
    turn_boundaries.sort(key=lambda x: (x["turn_id"], x["start_sec"]))

    return {
        "input_ids": input_ids,
        "generation_start_index": generation_start_index,
        "generation_end_index": generation_end_index,
        "remaining_durations": remaining_durations,
        "turn_boundaries": turn_boundaries,
        "source_file": example_id,
        "window_idx": window_idx,
        "vad_timestamps_left": vad_left_seconds,
        "vad_timestamps_right": vad_right_seconds,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
    }


def window_sample_count(sr: int) -> int:
    """Samples per channel in one window (327,680 at 16 kHz, i.e. 20.48 s)."""
    return int(MAX_TOKEN_SIZE / NUM_CHANNELS / INTERLEAVE_FACTOR / TOKEN_RATE_HZ * sr)


def iter_window_bounds(num_samples: int, sr: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(window_idx, start_sample, end_sample)`` for non-overlapping windows.

    The last window is whatever remains. When ``num_samples`` is an exact
    multiple of the window length the final index would be empty; it is not
    yielded (it never produced a record).
    """
    per_window = window_sample_count(sr)
    total_window_count = int(num_samples / per_window) + 1
    for w in range(total_window_count):
        start_sample = w * per_window
        end_sample = min((w + 1) * per_window, num_samples)
        if end_sample <= start_sample:
            continue
        yield w, start_sample, end_sample


@dataclass
class ConversationStats:
    """Window counts for one conversation."""

    written: int = 0
    failed: int = 0  # all windows that raised, including the unknown-code ones
    dropped_unknown_code: int = 0  # subset of ``failed``: <|audio_k|> missing from the vocabulary
    skipped_done: int = 0  # already present in the output (per-window resume)


def process_conversation(
    example_id,
    audio_left: torch.Tensor,
    audio_right: torch.Tensor,
    sr: int,
    whisper_model,
    feature_extractor,
    tokenizer,
    special_tokens: Dict[str, int],
    vad_model,
    get_speech_timestamps: Callable,
    logger: logging.Logger,
    output_file,
    extra_fields: Optional[Dict] = None,
    done_windows: Optional[set] = None,
    on_record: Optional[Callable[[Dict], None]] = None,
    device: str = "cuda",
) -> ConversationStats:
    """Window a conversation, process each window and append records to ``output_file``.

    Both channels are truncated to the shorter one. A window that raises is
    logged and skipped; it does not stop the conversation. ``extra_fields`` are
    added to every record (e.g. the source audio paths). Windows whose index is
    in ``done_windows`` are not recomputed. ``on_record`` is called with each
    written record.
    """
    stats = ConversationStats()

    min_len = min(len(audio_left), len(audio_right))
    audio_left = audio_left[:min_len]
    audio_right = audio_right[:min_len]

    for w, start_sample, end_sample in iter_window_bounds(len(audio_left), sr):
        if done_windows and w in done_windows:
            stats.skipped_done += 1
            continue
        try:
            result = process_audio_window(
                audio_left=audio_left[start_sample:end_sample],
                audio_right=audio_right[start_sample:end_sample],
                sr=sr,
                whisper_model=whisper_model,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                special_tokens=special_tokens,
                vad_model=vad_model,
                get_speech_timestamps=get_speech_timestamps,
                logger=logger,
                example_id=example_id,
                window_idx=w,
                window_start_sec=start_sample / sr,
                window_end_sec=end_sample / sr,
                device=device,
            )
        except UnknownAudioCodeError as e:
            logger.warning(f"Dropping example {example_id} window {w}: {e}")
            stats.failed += 1
            stats.dropped_unknown_code += 1
            continue
        except Exception as e:
            logger.error(f"Error processing example {example_id} window {w}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            stats.failed += 1
            continue

        if result is None:
            continue
        if extra_fields:
            result.update(extra_fields)

        output_file.write(json.dumps(result) + "\n")
        output_file.flush()
        stats.written += 1
        if on_record is not None:
            on_record(result)

    return stats
