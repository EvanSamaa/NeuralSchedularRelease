"""Step 2 of the inference pipeline (script mode): segment cache -> scheduler input.

For each speaker the cached segments (speech turns and backchannels) are
concatenated in turn order with 2 s gaps, tokenised once with the GLM-4-Voice
speech tokenizer (12.5 Hz) and passed through Silero VAD. Each turn's speech
regions become one ``<|start-turn-N|> ... <|end-turn-N|>`` block of that
speaker's script. Alongside the JSONL record, ``<cache>/source_map/`` keeps the
concatenated audio, the raw codes and, per script token, the frame it came
from, so that step 3 can rebuild the scheduled dialogue from the original
audio.

    python -m neural_scheduler.pipeline.tokenize --cache_base_dir OUT/segment_cache

writes ``OUT/segment_cache/tokenized_result.jsonl`` with one record per cache::

    input_ids, generation_start_index, generation_end_index, remaining_durations,
    turn_boundaries [{turn_id, channel "A"|"B", type "speech"|"backchannel"}],
    source_map_dir, source_cache, example_idx

``source_cache`` and ``source_map_dir`` are relative to the JSONL file.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch

from ..constants import SPEECH_TOKENIZER_SR
from ._tokenize_common import (
    SourceMapTokenizer,
    add_common_arguments,
    find_cache_dirs,
    make_logger,
    run_over_caches,
)

Segment = Tuple[torch.Tensor, int, int]  # (audio, sample rate, turn id)


class ScriptTokenizer(SourceMapTokenizer):
    """Tokenises a cache whose two speakers are both given as separate segments."""

    def _load_segments(self, cache_dir: str, manifest: Dict) -> Tuple[List[Segment], List[Segment]]:
        audio_dir = os.path.join(cache_dir, "audio_segments")

        def load_speaker(speaker_key: str) -> List[Segment]:
            segments = []
            for entry in manifest[speaker_key]:
                audio_np, sr = sf.read(os.path.join(audio_dir, entry["filename"]))
                audio = torch.from_numpy(audio_np).float()
                if audio.dim() > 1:
                    audio = audio.mean(dim=-1)
                segments.append((audio, sr, entry["turn_id"]))
            segments.sort(key=lambda x: x[2])
            return segments

        segments_s1 = load_speaker("speaker_1")
        segments_s2 = load_speaker("speaker_2")
        self.logger.info(f"Loaded {len(segments_s1)} S1 segments, {len(segments_s2)} S2 segments")
        return segments_s1, segments_s2

    def _build_script_for_speaker(self, segments: List[Segment], channel: str) -> Dict:
        """Tokenise one speaker's segments.

        Returns ``script_tokens`` (vocab ids with turn markers), ``raw_tokens``
        (codes of the whole concatenation), ``source_map`` (parallel to the
        script: raw-token index or ``None``), ``silenced_tokens`` (raw codes
        with non-speech replaced by silence) and ``audio`` (the 16 kHz
        concatenation; raw token ``i`` covers samples ``[1280 i, 1280 (i + 1))``).
        """
        from ..data.sequence import silencing_with_vad

        if not segments:
            script, source_map = self._new_script()
            return {
                "script_tokens": script,
                "raw_tokens": [],
                "source_map": source_map,
                "silenced_tokens": [],
                "audio": torch.zeros(0),
            }

        import torchaudio

        pieces = []
        for audio, sr, turn_id in sorted(segments, key=lambda s: s[2]):
            if sr != SPEECH_TOKENIZER_SR:
                audio = torchaudio.transforms.Resample(sr, SPEECH_TOKENIZER_SR)(audio)
            if audio.dim() > 1:
                audio = audio.mean(dim=0)
            pieces.append((audio, turn_id))
        concatenated, boundaries = self._concatenate(pieces)
        self.logger.info(
            f"  {channel}: concatenated {len(segments)} segments -> {len(concatenated)} samples "
            f"({len(concatenated) / SPEECH_TOKENIZER_SR:.2f}s)"
        )

        raw_tokens = self._tokenize_audio(concatenated)
        self.logger.info(f"  {channel}: {len(raw_tokens)} raw tokens")
        vad_regions = self._run_vad(concatenated)
        per_turn = self._vad_regions_per_turn(boundaries, vad_regions, len(raw_tokens))
        script, source_map = self._script_from_turn_regions(per_turn, raw_tokens, channel)
        return {
            "script_tokens": script,
            "raw_tokens": raw_tokens,
            "source_map": source_map,
            "silenced_tokens": silencing_with_vad(raw_tokens, vad_regions),
            "audio": concatenated,
        }

    def process_from_cache(self, cache_dir: str) -> Dict:
        """Tokenise one cache directory, write its source map and return the JSONL record.

        The dialogue region of the record is a placeholder (both speakers'
        concatenations side by side); the scheduler replaces it.
        """
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            manifest = json.load(f)
        segments_s1, segments_s2 = self._load_segments(cache_dir, manifest)

        self.logger.info("Building script + source map for speaker 1 (channel A)")
        channel_a = self._build_script_for_speaker(segments_s1, "A")
        self.logger.info("Building script + source map for speaker 2 (channel B)")
        channel_b = self._build_script_for_speaker(segments_s2, "B")

        dialogue = self._build_interleaved_dialogue(channel_a["silenced_tokens"], channel_b["silenced_tokens"])
        assembled = self._assemble(channel_a["script_tokens"], channel_b["script_tokens"], dialogue)

        turn_boundaries = [
            {"turn_id": entry["turn_id"], "channel": channel, "type": entry["type"]}
            for speaker_key, channel in (("speaker_1", "A"), ("speaker_2", "B"))
            for entry in manifest[speaker_key]
        ]
        turn_boundaries.sort(key=lambda x: x["turn_id"])

        for channel in (channel_a, channel_b):
            del channel["silenced_tokens"]
        source_map_dir = self._write_source_map(
            cache_dir, {"a": channel_a, "b": channel_b}, assembled["script_offsets"]
        )
        return {
            "input_ids": assembled["input_ids"],
            "generation_start_index": assembled["generation_start_index"],
            "generation_end_index": assembled["generation_end_index"],
            "remaining_durations": assembled["remaining_durations"],
            "turn_boundaries": turn_boundaries,
            "source_map_dir": source_map_dir,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2: tokenise segment caches (script mode).")
    parser.add_argument(
        "--cache_base_dir",
        type=str,
        required=True,
        help="The segment_cache/ folder written by step 1; every sub-directory with a manifest.json is processed.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="JSONL to write (default: <cache_base_dir>/tokenized_result.jsonl)."
    )
    add_common_arguments(parser)
    return parser


def main(argv: Optional[List[str]] = None, components=None) -> None:
    """Run step 2. ``components`` may be injected (tests); otherwise the models are loaded."""
    args = build_parser().parse_args(argv)
    output_path = args.output_path or os.path.join(args.cache_base_dir, "tokenized_result.jsonl")
    logger = make_logger("neural_scheduler.pipeline.tokenize")

    processor = ScriptTokenizer(
        audio_tokenizer_path=args.audio_tokenizer_path,
        model_id=args.model_id,
        device=args.device,
        logger=logger,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        max_turns=args.max_turns,
        components=components,
    )
    cache_dirs = find_cache_dirs(args.cache_base_dir)
    logger.info(f"Found {len(cache_dirs)} cache directories in {args.cache_base_dir}")
    run_over_caches(cache_dirs, output_path, processor.process_from_cache, logger, args.absolute_paths)


if __name__ == "__main__":
    main()
