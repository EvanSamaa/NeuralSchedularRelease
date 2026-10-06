"""Step 2 of the inference pipeline (benchmark mode): backchannel-only scheduling.

Used when speaker 1 is a real recording whose timing is kept and only speaker
2's backchannels are scheduled (Full-Duplex-Bench ``icc_backchannel``).

* Speaker 1 (channel A) is tokenised from the whole recording
  ``<icc_base_dir>/<id>/input.wav``. Each turn's span comes from the
  ``input_wav_slice_{start,end}_sec`` fields step 1 put in the manifest,
  intersected with VAD, so raw-token indices are dialogue time.
* Speaker 2 (channel B) is tokenised from the backchannel clips in the cache.

Each JSONL record carries, in addition to the script-mode fields,
``s1_turn_timing``: ``[{turn_id, start_token, end_token, sub_segments:
[{start_token, end_token}]}]`` in 12.5 Hz token steps, which the scheduler uses
to anchor speaker 1, and ``example_id`` (the cache directory name).

    python -m neural_scheduler.pipeline.tokenize_icc \\
        --segment_cache_dir OUT/segment_cache --icc_base_dir DATA/icc_backchannel
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from ..constants import SAMPLES_PER_TOKEN, SPEECH_TOKENIZER_SR
from ._tokenize_common import (
    SourceMapTokenizer,
    add_common_arguments,
    find_cache_dirs,
    make_logger,
    run_over_caches,
)


def samples_to_token_range(start_sample: int, end_sample: int, n_raw_tokens: int) -> Tuple[int, int]:
    """Token span covering a sample span (start floored, end ceiled, clamped)."""
    tok_start = int(start_sample / SAMPLES_PER_TOKEN)
    tok_end = int(np.ceil(end_sample / SAMPLES_PER_TOKEN))
    tok_start = max(0, min(tok_start, n_raw_tokens))
    tok_end = max(tok_start, min(tok_end, n_raw_tokens))
    return tok_start, tok_end


class BenchmarkTokenizer(SourceMapTokenizer):
    """Tokenises a cache whose speaker 1 is one continuous recording."""

    @staticmethod
    def _load_audio_mono_16k(path: str) -> torch.Tensor:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio_t = torch.from_numpy(audio)
        if sr != SPEECH_TOKENIZER_SR:
            import torchaudio

            audio_t = torchaudio.transforms.Resample(sr, SPEECH_TOKENIZER_SR)(audio_t)
        return audio_t

    def _dropped(self, entries: List[Dict], keep_type: str, speaker: str) -> List[Dict]:
        kept = [e for e in entries if e.get("type") == keep_type]
        if len(kept) < len(entries):
            dropped_types = sorted({e.get("type", "?") for e in entries if e.get("type") != keep_type})
            self.logger.info(
                f"  Dropping {len(entries) - len(kept)} {speaker} manifest entries of type "
                f"{dropped_types}; only '{keep_type}' entries are used."
            )
        return kept

    def _build_script_s1_from_input_wav(self, input_wav_path: str, s1_manifest: List[Dict]) -> Dict:
        """Tokenise the whole recording and cut speaker 1's turns out of it by time.

        Besides the usual channel fields, returns ``s1_turn_timing`` for the
        turns that made it into the script.
        """
        from ..data.sequence import silencing_with_vad

        self.logger.info(f"  Loading input.wav: {input_wav_path}")
        full_audio = self._load_audio_mono_16k(input_wav_path)
        raw_tokens = self._tokenize_audio(full_audio)
        self.logger.info(f"  {len(raw_tokens)} raw tokens from input.wav")
        vad_regions = self._run_vad(full_audio)
        silenced_tokens = silencing_with_vad(raw_tokens, vad_regions)

        # Backchannels step 1 synthesised for speaker 1 are not part of the recording.
        speech_turns = sorted(self._dropped(s1_manifest, "speech", "S1"), key=lambda e: e["turn_id"])

        turn_token_ranges: Dict[int, Tuple[int, int]] = {}
        for entry in speech_turns:
            turn_id = entry["turn_id"]
            if "input_wav_slice_start_sec" not in entry or "input_wav_slice_end_sec" not in entry:
                raise KeyError(
                    f"Turn {turn_id}: manifest entry has no 'input_wav_slice_start_sec' / "
                    f"'input_wav_slice_end_sec'. Step 1 copies them from the benchmark JSONL's "
                    f"'slice_start_sec' / 'slice_end_sec'."
                )
            start_samp = int(round(entry["input_wav_slice_start_sec"] * SPEECH_TOKENIZER_SR))
            end_samp = int(round(entry["input_wav_slice_end_sec"] * SPEECH_TOKENIZER_SR))
            start_samp = max(0, min(start_samp, len(full_audio)))
            end_samp = max(start_samp, min(end_samp, len(full_audio)))
            turn_token_ranges[turn_id] = samples_to_token_range(start_samp, end_samp, len(raw_tokens))
            self.logger.info(
                f"  Turn {turn_id}: samples [{start_samp}, {end_samp}] -> tokens {list(turn_token_ranges[turn_id])}"
            )

        script, source_map = self._new_script()
        s1_turn_timing: List[Dict] = []
        for entry in speech_turns:
            turn_id = entry["turn_id"]
            if turn_id >= self.max_turns:
                self.logger.warning(f"  Turn {turn_id} >= max_turns={self.max_turns}, skipping")
                continue
            tok_start, tok_end = turn_token_ranges[turn_id]
            turn_regions = []
            for region in vad_regions:
                start = max(region["start"], tok_start)
                end = min(region["end"], tok_end)
                if start < end:
                    turn_regions.append({"start": start, "end": end})
            turn_regions.sort(key=lambda r: r["start"])
            if not turn_regions:
                self.logger.info(f"  Turn {turn_id}: no speech detected in its span, skipping")
                continue

            s1_turn_timing.append(
                {
                    "turn_id": turn_id,
                    "start_token": int(turn_regions[0]["start"]),
                    "end_token": int(turn_regions[-1]["end"]),
                    "sub_segments": [{"start_token": int(r["start"]), "end_token": int(r["end"])} for r in turn_regions],
                }
            )
            self._append_turn(script, source_map, turn_id, turn_regions, raw_tokens)
        assert len(script) == len(source_map)

        return {
            "script_tokens": script,
            "raw_tokens": raw_tokens,
            "silenced_tokens": silenced_tokens,
            "source_map": source_map,
            "audio": full_audio,
            "s1_turn_timing": s1_turn_timing,
        }

    def _build_script_s2_from_bc_wavs(self, s2_manifest: List[Dict], audio_dir: str) -> Dict:
        """Tokenise speaker 2's backchannel clips, concatenated with 2 s gaps."""
        from ..data.sequence import silencing_with_vad

        script, source_map = self._new_script()
        empty = {
            "script_tokens": script,
            "raw_tokens": [],
            "silenced_tokens": [],
            "source_map": source_map,
            "audio": torch.zeros(0),
        }
        if not s2_manifest:
            return empty

        # Speaker 2 only backchannels in this task.
        backchannels = sorted(self._dropped(s2_manifest, "backchannel", "S2"), key=lambda e: e["turn_id"])
        pieces = []
        for entry in backchannels:
            wav_path = os.path.join(audio_dir, entry["filename"])
            if not os.path.isfile(wav_path):
                self.logger.warning(f"  S2 backchannel clip not found: {wav_path}, skipping")
                continue
            pieces.append((self._load_audio_mono_16k(wav_path), entry["turn_id"]))
        if not pieces:
            return empty

        concatenated, boundaries = self._concatenate(pieces)
        self.logger.info(
            f"  S2: concatenated {len(backchannels)} backchannel clips -> {len(concatenated)} samples "
            f"({len(concatenated) / SPEECH_TOKENIZER_SR:.2f}s)"
        )
        raw_tokens = self._tokenize_audio(concatenated)
        self.logger.info(f"  S2: {len(raw_tokens)} raw tokens")
        vad_regions = self._run_vad(concatenated)
        per_turn = self._vad_regions_per_turn(boundaries, vad_regions, len(raw_tokens))
        script, source_map = self._script_from_turn_regions(per_turn, raw_tokens, "S2")
        return {
            "script_tokens": script,
            "raw_tokens": raw_tokens,
            "silenced_tokens": silencing_with_vad(raw_tokens, vad_regions),
            "source_map": source_map,
            "audio": concatenated,
        }

    def process_from_cache(self, cache_dir: str, icc_base_dir: str) -> Dict:
        """Tokenise one cache directory, write its source map and return the JSONL record.

        The recording is read from ``<icc_base_dir>/<basename(cache_dir)>/input.wav``.
        """
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            manifest = json.load(f)
        example_id = os.path.basename(cache_dir)
        input_wav_path = os.path.join(icc_base_dir, example_id, "input.wav")
        if not os.path.isfile(input_wav_path):
            raise FileNotFoundError(
                f"input.wav not found at {input_wav_path}; check --icc_base_dir and the id '{example_id}'."
            )

        self.logger.info("Building S1 (channel A) script from input.wav")
        channel_a = self._build_script_s1_from_input_wav(input_wav_path, manifest.get("speaker_1", []))
        s1_turn_timing = channel_a.pop("s1_turn_timing")
        self.logger.info("Building S2 (channel B) script from backchannel clips")
        channel_b = self._build_script_s2_from_bc_wavs(
            manifest.get("speaker_2", []), os.path.join(cache_dir, "audio_segments")
        )

        dialogue = self._build_interleaved_dialogue(channel_a["silenced_tokens"], channel_b["silenced_tokens"])
        assembled = self._assemble(channel_a["script_tokens"], channel_b["script_tokens"], dialogue)

        # Same filtering as the scripts: A holds S1 speech, B holds S2 backchannels.
        turn_boundaries = [
            {"turn_id": entry["turn_id"], "channel": channel, "type": entry["type"]}
            for speaker_key, channel, keep_type in (("speaker_1", "A", "speech"), ("speaker_2", "B", "backchannel"))
            for entry in manifest.get(speaker_key, [])
            if entry.get("type") == keep_type
        ]
        turn_boundaries.sort(key=lambda x: x["turn_id"])

        source_map_dir = self._write_source_map(
            cache_dir, {"a": channel_a, "b": channel_b}, assembled["script_offsets"], skip_empty_audio=True
        )
        return {
            "input_ids": assembled["input_ids"],
            "generation_start_index": assembled["generation_start_index"],
            "generation_end_index": assembled["generation_end_index"],
            "remaining_durations": assembled["remaining_durations"],
            "turn_boundaries": turn_boundaries,
            "source_map_dir": source_map_dir,
            "s1_turn_timing": s1_turn_timing,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2: tokenise segment caches (benchmark, backchannel-only mode).")
    parser.add_argument("--segment_cache_dir", type=str, required=True, help="The segment_cache/ folder written by step 1.")
    parser.add_argument(
        "--icc_base_dir",
        type=str,
        required=True,
        help="Dataset root; the recording of example <id> is <icc_base_dir>/<id>/input.wav.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="JSONL to write (default: <segment_cache_dir>/tokenized_result.jsonl)."
    )
    add_common_arguments(parser)
    return parser


def main(argv: Optional[List[str]] = None, components=None) -> None:
    """Run step 2 in benchmark mode. ``components`` may be injected (tests)."""
    args = build_parser().parse_args(argv)
    output_path = args.output_path or os.path.join(args.segment_cache_dir, "tokenized_result.jsonl")
    logger = make_logger("neural_scheduler.pipeline.tokenize_icc")

    processor = BenchmarkTokenizer(
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
    cache_dirs = find_cache_dirs(args.segment_cache_dir)
    logger.info(f"Found {len(cache_dirs)} cache directories in {args.segment_cache_dir}")
    run_over_caches(
        cache_dirs,
        output_path,
        lambda cache_dir: processor.process_from_cache(cache_dir, args.icc_base_dir),
        logger,
        args.absolute_paths,
        with_example_id=True,
    )


if __name__ == "__main__":
    main()
