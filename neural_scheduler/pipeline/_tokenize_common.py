"""Shared machinery of the two step-2 tokenisers (script mode and benchmark mode).

Both turn a segment cache into the scheduler's input sequence

    <|start-of-script-A|> script A <|end-of-script-A|>
    <|start-of-script-B|> script B <|end-of-script-B|>
    <|start-of-dialogue|> placeholder dialogue <|end-of-dialogue|>

and into a *source map*: for every audio token of a script, the index of the
speech-tokenizer frame it came from. Step 3 uses the source map to rebuild the
scheduled dialogue from the original audio instead of decoding tokens.
"""

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import soundfile as sf
import torch

from ..constants import (
    GLM4_VOICE_MODEL_ID,
    GLM4_VOICE_TOKENIZER_ID,
    SILENCE_CODE,
    SPEECH_TOKENIZER_SR,
    TOKEN_RATE_HZ,
)
from ..vocab import add_scheduler_tokens, audio_token, get_special_token_ids

# Silence inserted between consecutive segments of one speaker before tokenising.
SEGMENT_GAP_SECONDS = 2.0
# Turns with an id at or above this are left out of the script.
DEFAULT_MAX_TURNS = 20


@dataclass
class TokenizerComponents:
    """The models step 2 needs. Built by :func:`load_components` or supplied directly."""

    speech_model: object
    feature_extractor: object
    tokenizer: object
    vad_model: object
    get_speech_timestamps: Callable
    extract_speech_tokens: Callable


def load_components(audio_tokenizer_path: str, model_id: str, device: str) -> TokenizerComponents:
    """Load the GLM-4-Voice speech tokenizer, the text tokenizer with scheduler tokens, and Silero VAD."""
    from transformers import AutoTokenizer

    from ..data.speech_tokenizer import extract_speech_token_batched, load_speech_tokenizer
    from ..data.vad import load_silero_vad

    speech_model, feature_extractor = load_speech_tokenizer(audio_tokenizer_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    add_scheduler_tokens(tokenizer)
    vad_model, get_speech_timestamps = load_silero_vad()
    return TokenizerComponents(
        speech_model=speech_model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        vad_model=vad_model,
        get_speech_timestamps=get_speech_timestamps,
        extract_speech_tokens=extract_speech_token_batched,
    )


class SourceMapTokenizer:
    """Base class: tokenisation, VAD, script construction and artefact writing."""

    def __init__(
        self,
        audio_tokenizer_path: str = GLM4_VOICE_TOKENIZER_ID,
        model_id: str = GLM4_VOICE_MODEL_ID,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
        vad_threshold: float = 0.05,
        vad_min_speech_ms: int = 50,
        vad_min_silence_ms: int = 50,
        max_turns: int = DEFAULT_MAX_TURNS,
        components: Optional[TokenizerComponents] = None,
    ):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_min_silence_ms = vad_min_silence_ms
        self.max_turns = max_turns

        self.logger.info("Loading speech tokenizer, text tokenizer and VAD...")
        self.components = components or load_components(audio_tokenizer_path, model_id, device)
        self.tokenizer = self.components.tokenizer
        self.special_tokens = get_special_token_ids(self.tokenizer, max_turns)
        self.logger.info("Models loaded.")

    # ----- models ----------------------------------------------------------- #

    def _tokenize_audio(self, audio_1d: torch.Tensor) -> List[int]:
        """Speech-tokenizer codes of a 16 kHz mono waveform, one per 80 ms."""
        return self.components.extract_speech_tokens(
            model=self.components.speech_model,
            feature_extractor=self.components.feature_extractor,
            batched_audio=audio_1d.unsqueeze(0),
            sampling_rate=SPEECH_TOKENIZER_SR,
            model_batch_size=32,
            device=self.device,
            logger=self.logger,
        )[0]

    def _run_vad(self, audio_1d: torch.Tensor) -> List[Dict[str, int]]:
        """Speech regions of a 16 kHz mono waveform, in token indices."""
        from ..data.vad import vad_timestamps_to_token_indices

        timestamps = self.components.get_speech_timestamps(
            audio_1d.numpy(),
            self.components.vad_model,
            sampling_rate=SPEECH_TOKENIZER_SR,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.vad_min_speech_ms,
            min_silence_duration_ms=self.vad_min_silence_ms,
        )
        return vad_timestamps_to_token_indices(timestamps, SPEECH_TOKENIZER_SR)

    def _audio_token_id(self, code: int) -> int:
        return self.tokenizer.convert_tokens_to_ids([audio_token(code)])[0]

    # ----- script construction ---------------------------------------------- #

    @staticmethod
    def _concatenate(pieces: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, List[Dict]]:
        """Join ``(audio, turn_id)`` pieces with gaps; returns the waveform and each piece's sample span."""
        gap_samples = int(SEGMENT_GAP_SECONDS * SPEECH_TOKENIZER_SR)
        boundaries = []
        audio_pieces = []
        current_sample = 0
        for audio, turn_id in pieces:
            if audio_pieces:
                audio_pieces.append(torch.zeros(gap_samples))
                current_sample += gap_samples
            start_sample = current_sample
            audio_pieces.append(audio)
            current_sample += len(audio)
            boundaries.append({"start_sample": start_sample, "end_sample": current_sample, "turn_id": turn_id})
        return torch.cat(audio_pieces), boundaries

    @staticmethod
    def _vad_regions_per_turn(
        boundaries: List[Dict], vad_regions: List[Dict[str, int]], n_raw_tokens: int
    ) -> Dict[int, List[Dict[str, int]]]:
        """Intersect each piece's token span with the VAD regions, grouped by turn id."""
        per_turn: Dict[int, List[Dict[str, int]]] = {}
        for seg in boundaries:
            tok_start = int(seg["start_sample"] / SPEECH_TOKENIZER_SR * TOKEN_RATE_HZ)
            tok_end = int(seg["end_sample"] / SPEECH_TOKENIZER_SR * TOKEN_RATE_HZ)
            tok_start = max(0, min(tok_start, n_raw_tokens))
            tok_end = max(0, min(tok_end, n_raw_tokens))
            regions = per_turn.setdefault(seg["turn_id"], [])
            for region in vad_regions:
                start = max(region["start"], tok_start)
                end = min(region["end"], tok_end)
                if start < end:
                    regions.append({"start": start, "end": end})
        for regions in per_turn.values():
            regions.sort(key=lambda r: r["start"])
        return per_turn

    def _new_script(self) -> Tuple[List[int], List[Optional[int]]]:
        """A script and its source map, opened with one silence token."""
        return [self.special_tokens["silence"]], [None]

    def _append_turn(
        self,
        script: List[int],
        source_map: List[Optional[int]],
        turn_id: int,
        vad_regions: List[Dict[str, int]],
        raw_tokens: List[int],
    ) -> None:
        """Append ``<start-turn-N> speech [SIL speech ...] <end-turn-N> SIL`` for one turn.

        Each VAD region becomes one speech block; blocks are separated by a
        single silence token. ``source_map`` receives the raw-token index of
        every audio token and ``None`` for markers and silences.
        """
        silence_id = self.special_tokens["silence"]
        script.append(self.special_tokens[f"start_turn_{turn_id}"])
        source_map.append(None)
        for block_idx, region in enumerate(vad_regions):
            if block_idx > 0:
                script.append(silence_id)
                source_map.append(None)
            for i in range(max(0, region["start"]), min(len(raw_tokens), region["end"])):
                script.append(self._audio_token_id(raw_tokens[i]))
                source_map.append(i)
        script.append(self.special_tokens[f"end_turn_{turn_id}"])
        source_map.append(None)
        script.append(silence_id)
        source_map.append(None)

    def _script_from_turn_regions(
        self, per_turn: Dict[int, List[Dict[str, int]]], raw_tokens: List[int], channel: str
    ) -> Tuple[List[int], List[Optional[int]]]:
        script, source_map = self._new_script()
        for turn_id in sorted(per_turn.keys()):
            if turn_id >= self.max_turns:
                self.logger.warning(f"  {channel}: turn {turn_id} >= max_turns={self.max_turns}, skipping")
                continue
            if not per_turn[turn_id]:
                continue
            self._append_turn(script, source_map, turn_id, per_turn[turn_id], raw_tokens)
        assert len(script) == len(source_map)
        return script, source_map

    def _build_interleaved_dialogue(self, silenced_a: List[int], silenced_b: List[int]) -> List[int]:
        """Interleave two code sequences as A0 B0 A1 B1 ..., padding the shorter with silence."""
        max_len = max(len(silenced_a), len(silenced_b))
        padded_a = silenced_a + [SILENCE_CODE] * (max_len - len(silenced_a))
        padded_b = silenced_b + [SILENCE_CODE] * (max_len - len(silenced_b))
        dialogue = []
        for code_a, code_b in zip(padded_a, padded_b):
            dialogue.append(self._audio_token_id(code_a))
            dialogue.append(self._audio_token_id(code_b))
        return dialogue

    def _assemble(self, script_a: List[int], script_b: List[int], dialogue: List[int]) -> Dict:
        """Lay out the full input sequence; returns it with the region offsets."""
        from ..data.durations import compute_remaining_durations_dual_channel

        st = self.special_tokens
        input_ids: List[int] = [st["start_script_a"]]
        script_a_start = len(input_ids)
        input_ids.extend(script_a)
        script_a_end = len(input_ids)
        input_ids.append(st["end_script_a"])

        input_ids.append(st["start_script_b"])
        script_b_start = len(input_ids)
        input_ids.extend(script_b)
        script_b_end = len(input_ids)
        input_ids.append(st["end_script_b"])

        input_ids.append(st["start_dialogue"])
        generation_start_index = len(input_ids)
        input_ids.extend(dialogue)
        generation_end_index = len(input_ids)
        input_ids.append(st["end_dialogue"])

        self.logger.info(f"Generated sequence with {len(input_ids)} tokens")
        self.logger.info(f"Dialogue region: [{generation_start_index}, {generation_end_index})")
        return {
            "input_ids": input_ids,
            "generation_start_index": generation_start_index,
            "generation_end_index": generation_end_index,
            "remaining_durations": compute_remaining_durations_dual_channel(dialogue),
            "script_offsets": {
                "script_a_start": script_a_start,
                "script_a_end": script_a_end,
                "script_b_start": script_b_start,
                "script_b_end": script_b_end,
            },
        }

    # ----- artefacts ----------------------------------------------------------- #

    def _write_source_map(
        self,
        cache_dir: str,
        channels: Dict[str, Dict],
        script_offsets: Dict[str, int],
        skip_empty_audio: bool = False,
    ) -> str:
        """Write ``<cache_dir>/source_map/``.

        ``channels`` maps ``"a"`` / ``"b"`` to ``audio`` (16 kHz waveform),
        ``raw_tokens``, ``script_tokens``, ``source_map`` and optionally
        ``silenced_tokens``.
        """
        source_map_dir = os.path.join(cache_dir, "source_map")
        os.makedirs(source_map_dir, exist_ok=True)

        def dump(name: str, payload, **kwargs) -> None:
            with open(os.path.join(source_map_dir, name), "w") as f:
                json.dump(payload, f, **kwargs)

        for key, channel in channels.items():
            if not (skip_empty_audio and len(channel["audio"]) == 0):
                sf.write(
                    os.path.join(source_map_dir, f"concatenated_audio_{key}.wav"),
                    channel["audio"].numpy(),
                    SPEECH_TOKENIZER_SR,
                )
            dump(f"raw_tokens_{key}.json", channel["raw_tokens"])
            if "silenced_tokens" in channel:
                dump(f"silenced_tokens_{key}.json", channel["silenced_tokens"])
            dump(f"script_source_map_{key}.json", channel["source_map"])
            dump(f"script_tokens_{key}.json", channel["script_tokens"])
        dump("script_offsets.json", script_offsets, indent=2)
        self.logger.info(f"Source map written to {source_map_dir}")
        return source_map_dir


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def find_cache_dirs(cache_base_dir: str) -> List[str]:
    """Sub-directories of ``cache_base_dir`` that contain a ``manifest.json``, sorted."""
    return [
        d
        for d in sorted(glob.glob(os.path.join(cache_base_dir, "*")))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "manifest.json"))
    ]


def portable_path(path: str, jsonl_path: str, absolute: bool) -> str:
    """``path`` as stored in a JSONL record: relative to the JSONL's directory unless ``absolute``."""
    if absolute:
        return path
    return os.path.relpath(path, os.path.dirname(os.path.abspath(jsonl_path)))


def add_common_arguments(parser) -> None:
    parser.add_argument("--audio_tokenizer_path", type=str, default=GLM4_VOICE_TOKENIZER_ID)
    parser.add_argument("--model_id", type=str, default=GLM4_VOICE_MODEL_ID, help="Source of the text tokenizer.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vad_threshold", type=float, default=0.2)
    parser.add_argument("--vad_min_speech_ms", type=int, default=50)
    parser.add_argument("--vad_min_silence_ms", type=int, default=50)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Turns with an id at or above this are dropped from the scripts (at most 50).",
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Write absolute cache paths into the JSONL instead of paths relative to it.",
    )


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger


def run_over_caches(
    cache_dirs: List[str], output_path: str, process: Callable[[str], Dict], logger, absolute_paths: bool,
    with_example_id: bool = False,
) -> int:
    """Process every cache directory and write one JSONL record each. Returns the number written."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    written = 0
    with open(output_path, "w") as f_out:
        for idx, cache_dir in enumerate(cache_dirs):
            logger.info(f"[{idx + 1}/{len(cache_dirs)}] Processing: {cache_dir}")
            try:
                result = process(cache_dir)
            except Exception:
                logger.exception(f"  Failed: {cache_dir}")
                continue
            result["source_map_dir"] = portable_path(result["source_map_dir"], output_path, absolute_paths)
            result["source_cache"] = portable_path(cache_dir, output_path, absolute_paths)
            if with_example_id:
                result["example_id"] = os.path.basename(cache_dir)
            result["example_idx"] = idx
            f_out.write(json.dumps(result) + "\n")
            written += 1
            logger.info(
                f"  Total tokens: {len(result['input_ids'])}, "
                f"dialogue: {result['generation_end_index'] - result['generation_start_index']}"
            )
    logger.info(f"Done. {written}/{len(cache_dirs)} written to {output_path}")
    return written
