"""Step 2 (``neural_scheduler.pipeline.tokenize`` and ``tokenize_icc``).

The equivalence tests run the original research scripts and the refactored
modules on the same segment cache with the same stand-in speech tokenizer, text
tokenizer and VAD, and require identical records and source-map files.
"""

import importlib.util
import json
import math
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from neural_scheduler import vocab
from neural_scheduler.constants import AUDIO_TOKEN_OFFSET, BASE_VOCAB_SIZE, SILENCE_TOKEN_ID
from neural_scheduler.pipeline import tokenize, tokenize_icc
from neural_scheduler.pipeline._tokenize_common import TokenizerComponents

EXPERIMENTS = "ntpp/model/chatGLM_dual_timing_only/experiments"
ORIG_SCRIPT_MODE = f"{EXPERIMENTS}/inference_pipeline_10_examples/step_2_tokenization_with_source_map.py"
ORIG_BENCHMARK = f"{EXPERIMENTS}/inference_pipeline_bc_only_full_duplex_bench/step_2_tokenization_with_source_map.py"
VAD = dict(vad_threshold=0.2, vad_min_speech_ms=50, vad_min_silence_ms=50)


# --------------------------------------------------------------------------- #
# Stand-ins shared by both implementations
# --------------------------------------------------------------------------- #


class FakeTextTokenizer:
    """Maps ``<|audio_k|>`` like GLM-4-Voice and appends added tokens after the base vocabulary."""

    def __init__(self):
        self.added = {}

    def get_vocab(self):
        return dict(self.added)

    def add_tokens(self, tokens):
        new = [t for t in tokens if t not in self.added]
        for token in new:
            self.added[token] = BASE_VOCAB_SIZE + len(self.added)
        return len(new)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._one(tokens)
        return [self._one(t) for t in tokens]

    def _one(self, token):
        if token.startswith("<|audio_"):
            return AUDIO_TOKEN_OFFSET + int(token[len("<|audio_") : -2])
        return self.added[token]


def fake_extract(model, feature_extractor, batched_audio, sampling_rate=16000, model_batch_size=32, device="cpu", logger=None):
    """One code per 1280 samples, derived from the frame's content."""
    all_codes = []
    for audio in batched_audio:
        audio = audio.double()
        n_frames = math.ceil(len(audio) / 1280)
        codes = []
        for i in range(n_frames):
            frame = audio[i * 1280 : (i + 1) * 1280]
            codes.append(int(round(float(frame.abs().sum()) * 1000)) % 16383)
        all_codes.append(codes)
    return all_codes


def fake_get_speech_timestamps(
    audio, model, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100
):
    """Energy VAD on 20 ms windows; returns sample spans like Silero."""
    audio = np.asarray(audio, dtype=np.float64)
    window = sampling_rate // 50
    active = [np.abs(audio[i : i + window]).mean() > threshold * 0.25 for i in range(0, len(audio), window)]
    spans = []
    start = None
    for i, flag in enumerate(active + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if (i - start) * 20 >= min_speech_duration_ms:
                spans.append({"start": start * window, "end": min(i * window, len(audio))})
            start = None
    return spans


class FakeWhisperVQEncoder:
    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def parameters(self):
        return []

    def _freeze_parameters(self):
        pass

    def eval(self):
        return self

    def to(self, device):
        return self


class FakeFeatureExtractor:
    @classmethod
    def from_pretrained(cls, path):
        return cls()


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_id, trust_remote_code=False):
        return FakeTextTokenizer()


def new_components():
    tokenizer = FakeTextTokenizer()
    vocab.add_scheduler_tokens(tokenizer)
    return TokenizerComponents(
        speech_model=object(),
        feature_extractor=object(),
        tokenizer=tokenizer,
        vad_model=object(),
        get_speech_timestamps=fake_get_speech_timestamps,
        extract_speech_tokens=fake_extract,
    )


def _load_original(path: Path, name: str, monkeypatch):
    """Import an original step-2 script with its models replaced by the stand-ins."""
    package = types.ModuleType("speech_tokenizer")
    modeling = types.ModuleType("speech_tokenizer.modeling_whisper")
    modeling.WhisperVQEncoder = FakeWhisperVQEncoder
    monkeypatch.setitem(sys.modules, "speech_tokenizer", package)
    monkeypatch.setitem(sys.modules, "speech_tokenizer.modeling_whisper", modeling)
    monkeypatch.delitem(sys.modules, "step_2_tokenization_util", raising=False)
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the scripts append to sys.path on import

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delitem(sys.modules, "step_2_tokenization_util", raising=False)

    monkeypatch.setattr(module, "WhisperFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(module, "extract_speech_token_batched", fake_extract)
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: (object(), (fake_get_speech_timestamps,)))
    return module


# --------------------------------------------------------------------------- #
# Test caches
# --------------------------------------------------------------------------- #


def _burst(rng, sr, seconds, amplitude=0.5):
    return rng.uniform(-amplitude, amplitude, size=int(sr * seconds))


def _silence(sr, seconds):
    return np.zeros(int(sr * seconds))


def _write_segment(audio_dir, name, pieces, sr):
    sf.write(os.path.join(audio_dir, name), np.concatenate(pieces).astype("float32"), sr)


def make_script_cache(root: Path, name: str, seed: int) -> Path:
    """A cache with speech and backchannels on both speakers, two-region turns and one turn id >= 20."""
    rng = np.random.RandomState(seed)
    cache = root / name
    audio_dir = cache / "audio_segments"
    audio_dir.mkdir(parents=True)
    sr = 24000
    manifest = {"speaker_1": [], "speaker_2": []}

    def add(speaker, turn_id, kind, pieces, seg_sr=sr, **extra):
        fname = f"seg_{len(manifest['speaker_1']) + len(manifest['speaker_2']):04d}_s{speaker}_turn{turn_id}_{kind}.wav"
        _write_segment(audio_dir, fname, pieces, seg_sr)
        manifest[f"speaker_{speaker}"].append({"filename": fname, "turn_id": turn_id, "type": kind, "sample_rate": seg_sr, **extra})

    pad = _silence(sr, 0.5)
    add(1, 0, "speech", [pad, _burst(rng, sr, 0.9), _silence(sr, 0.45), _burst(rng, sr, 0.6), pad])
    add(1, 2, "speech", [pad, _burst(rng, sr, 1.1), pad])
    add(1, 1, "backchannel", [pad, _burst(rng, sr, 0.3), pad], bc_text="yeah")
    add(1, 25, "speech", [pad, _burst(rng, sr, 0.5), pad])
    add(1, 4, "speech", [_silence(16000, 0.8)], seg_sr=16000)
    add(2, 1, "speech", [pad, _burst(rng, sr, 0.7), _silence(sr, 0.5), _burst(rng, sr, 0.4), _silence(sr, 0.4), _burst(rng, sr, 0.5), pad])
    add(2, 0, "backchannel", [_silence(16000, 0.4), _burst(rng, 16000, 0.35), _silence(16000, 0.4)], seg_sr=16000, bc_text="uh-huh")
    add(2, 3, "speech", [pad, _burst(rng, sr, 0.8), pad])
    manifest["speaker_1"].sort(key=lambda e: e["turn_id"])
    manifest["speaker_2"].sort(key=lambda e: e["turn_id"])
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache


def make_benchmark_cache(root: Path, icc_root: Path, example_id: str, seed: int) -> Path:
    """A benchmark cache: speaker 1 is cut from ``input.wav`` by time, speaker 2 has backchannel clips."""
    rng = np.random.RandomState(seed)
    sr = 16000
    spans = [(0.5, 2.1), (3.0, 4.4), (5.2, 6.0), (7.5, 8.0)]
    recording = np.zeros(int(sr * 9.0))
    for k, (start, end) in enumerate(spans[:3]):
        burst = _burst(rng, sr, end - start - 0.3)
        if k == 0:
            burst[int(0.5 * sr) : int(0.9 * sr)] = 0.0  # an internal pause -> two sub-segments
        begin = int((start + 0.15) * sr)
        recording[begin : begin + len(burst)] = burst
    (icc_root / example_id).mkdir(parents=True)
    sf.write(icc_root / example_id / "input.wav", recording.astype("float32"), sr)

    cache = root / example_id
    audio_dir = cache / "audio_segments"
    audio_dir.mkdir(parents=True)
    manifest = {"speaker_1": [], "speaker_2": []}
    for turn_id, (start, end) in enumerate(spans):
        manifest["speaker_1"].append(
            {
                "filename": f"seg_{turn_id:04d}_s1_turn{turn_id}_speech.wav",
                "turn_id": turn_id,
                "type": "speech",
                "sample_rate": 24000,
                "input_wav_slice_start_sec": start,
                "input_wav_slice_end_sec": end,
            }
        )
    manifest["speaker_1"].append({"filename": "seg_0009_s1_turn1_bc_ok.wav", "turn_id": 1, "type": "backchannel", "sample_rate": 24000})
    for turn_id, seg_sr in ((0, 24000), (2, 16000)):
        fname = f"seg_001{turn_id}_s2_turn{turn_id}_bc_yeah.wav"
        _write_segment(audio_dir, fname, [_silence(seg_sr, 0.5), _burst(rng, seg_sr, 0.4), _silence(seg_sr, 0.5)], seg_sr)
        manifest["speaker_2"].append({"filename": fname, "turn_id": turn_id, "type": "backchannel", "bc_text": "yeah", "sample_rate": seg_sr})
    manifest["speaker_2"].append({"filename": "missing.wav", "turn_id": 3, "type": "backchannel", "sample_rate": 24000})
    manifest["speaker_2"].append({"filename": "seg_0020_s2_turn1_speech.wav", "turn_id": 1, "type": "speech", "sample_rate": 24000})
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache


def assert_same_source_map(orig_cache, new_cache):
    orig_dir, new_dir = Path(orig_cache) / "source_map", Path(new_cache) / "source_map"
    names = sorted(p.name for p in orig_dir.iterdir())
    assert names == sorted(p.name for p in new_dir.iterdir())
    for name in names:
        if name.endswith(".json"):
            assert json.load(open(orig_dir / name)) == json.load(open(new_dir / name)), name
        else:
            wav_a, sr_a = sf.read(orig_dir / name)
            wav_b, sr_b = sf.read(new_dir / name)
            assert sr_a == sr_b and np.array_equal(wav_a, wav_b), name
    return names


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


# --------------------------------------------------------------------------- #
# Equivalence with the original scripts
# --------------------------------------------------------------------------- #


@pytest.mark.equivalence
def test_script_mode_matches_original(ntpp_orig, tmp_path, monkeypatch):
    orig_root, new_root = tmp_path / "orig", tmp_path / "new"
    for seed, name in enumerate(("example_0000", "example_0001")):
        make_script_cache(orig_root, name, seed)
    shutil.copytree(orig_root, new_root)

    module = _load_original(ntpp_orig / ORIG_SCRIPT_MODE, "orig_step2_script_mode", monkeypatch)
    processor = module.TTSToTokenProcessorWithSourceMap(device="cpu", **VAD)
    expected = []
    for idx, cache_dir in enumerate(sorted(str(p) for p in orig_root.iterdir())):
        record = processor.process_from_cache(cache_dir)
        record["source_cache"] = cache_dir
        record["example_idx"] = idx
        expected.append(json.loads(json.dumps(record)))

    output = new_root / "tokenized_result.jsonl"
    tokenize.main(
        ["--cache_base_dir", str(new_root), "--device", "cpu", "--absolute_paths", "--vad_threshold", "0.2"],
        components=new_components(),
    )
    records = read_jsonl(output)
    assert len(records) == len(expected) == 2
    for got, want in zip(records, expected):
        assert list(got) == list(want)
        assert got["source_cache"] == want["source_cache"].replace(str(orig_root), str(new_root))
        assert got["source_map_dir"] == want["source_map_dir"].replace(str(orig_root), str(new_root))
        for key in ("input_ids", "generation_start_index", "generation_end_index", "remaining_durations", "turn_boundaries", "example_idx"):
            assert got[key] == want[key], key
        names = assert_same_source_map(want["source_cache"], got["source_cache"])
        assert "silenced_tokens_a.json" not in names

    ids = records[0]["input_ids"]
    assert ids[0] == BASE_VOCAB_SIZE and ids[1] == SILENCE_TOKEN_ID
    turn_markers = {i for i in ids if i >= vocab.FIRST_TURN_TOKEN_ID}
    assert vocab.FIRST_TURN_TOKEN_ID + 2 * 25 not in turn_markers  # turn 25 >= max_turns is dropped
    assert vocab.FIRST_TURN_TOKEN_ID + 2 * 4 not in turn_markers  # turn 4 has no speech
    assert {vocab.FIRST_TURN_TOKEN_ID + 2 * t for t in (0, 1, 2, 3)} <= turn_markers


@pytest.mark.equivalence
def test_benchmark_mode_matches_original(ntpp_orig, tmp_path, monkeypatch):
    orig_root, new_root, icc_root = tmp_path / "orig", tmp_path / "new", tmp_path / "icc"
    for seed, example_id in enumerate(("0", "7")):
        make_benchmark_cache(orig_root, icc_root, example_id, seed)
    shutil.copytree(orig_root, new_root)

    module = _load_original(ntpp_orig / ORIG_BENCHMARK, "orig_step2_benchmark", monkeypatch)
    processor = module.TTSToTokenProcessorICC(device="cpu", **VAD)
    expected = []
    for idx, cache_dir in enumerate(sorted(str(p) for p in orig_root.iterdir())):
        record = processor.process_from_cache(cache_dir=cache_dir, icc_base_dir=str(icc_root))
        record["source_cache"] = cache_dir
        record["example_id"] = os.path.basename(cache_dir)
        record["example_idx"] = idx
        expected.append(json.loads(json.dumps(record)))

    tokenize_icc.main(
        ["--segment_cache_dir", str(new_root), "--icc_base_dir", str(icc_root), "--device", "cpu", "--absolute_paths"],
        components=new_components(),
    )
    records = read_jsonl(new_root / "tokenized_result.jsonl")
    assert len(records) == len(expected) == 2
    for got, want in zip(records, expected):
        assert list(got) == list(want)
        for key in want:
            if key in ("source_cache", "source_map_dir"):
                assert got[key] == want[key].replace(str(orig_root), str(new_root))
            else:
                assert got[key] == want[key], key
        names = assert_same_source_map(want["source_cache"], got["source_cache"])
        assert "silenced_tokens_a.json" in names

    timing = records[0]["s1_turn_timing"]
    assert [t["turn_id"] for t in timing] == [0, 1, 2]  # turn 3 has no speech in its span
    assert len(timing[0]["sub_segments"]) == 2
    assert {b["type"] for b in records[0]["turn_boundaries"] if b["channel"] == "B"} == {"backchannel"}


# --------------------------------------------------------------------------- #
# Behaviour that is new in the refactored modules
# --------------------------------------------------------------------------- #


def test_relative_paths_and_default_output(tmp_path):
    root = tmp_path / "out" / "segment_cache"
    make_script_cache(root, "example_0000", 0)
    tokenize.main(["--cache_base_dir", str(root), "--device", "cpu"], components=new_components())
    (record,) = read_jsonl(root / "tokenized_result.jsonl")
    assert record["source_cache"] == "example_0000"
    assert record["source_map_dir"] == os.path.join("example_0000", "source_map")

    elsewhere = tmp_path / "out" / "tokenized.jsonl"
    tokenize.main(["--cache_base_dir", str(root), "--output_path", str(elsewhere), "--device", "cpu"], components=new_components())
    (record,) = read_jsonl(elsewhere)
    assert record["source_cache"] == os.path.join("segment_cache", "example_0000")
    assert (elsewhere.parent / record["source_cache"] / "manifest.json").is_file()


def test_max_turns_is_configurable(tmp_path):
    root = tmp_path / "segment_cache"
    make_script_cache(root, "example_0000", 0)
    marker = vocab.FIRST_TURN_TOKEN_ID + 2 * 25

    tokenize.main(["--cache_base_dir", str(root), "--device", "cpu"], components=new_components())
    assert marker not in read_jsonl(root / "tokenized_result.jsonl")[0]["input_ids"]
    tokenize.main(["--cache_base_dir", str(root), "--device", "cpu", "--max_turns", "50"], components=new_components())
    assert marker in read_jsonl(root / "tokenized_result.jsonl")[0]["input_ids"]


def test_failed_cache_is_skipped(tmp_path):
    root = tmp_path / "segment_cache"
    make_script_cache(root, "example_0000", 0)
    broken = make_script_cache(root, "example_0001", 1)
    manifest = json.loads((broken / "manifest.json").read_text())
    manifest["speaker_1"][0]["filename"] = "does_not_exist.wav"
    (broken / "manifest.json").write_text(json.dumps(manifest))
    make_script_cache(root, "example_0002", 2)

    tokenize.main(["--cache_base_dir", str(root), "--device", "cpu"], components=new_components())
    records = read_jsonl(root / "tokenized_result.jsonl")
    assert [r["source_cache"] for r in records] == ["example_0000", "example_0002"]
    assert [r["example_idx"] for r in records] == [0, 2]


def test_benchmark_mode_requires_slice_times(tmp_path):
    root, icc_root = tmp_path / "segment_cache", tmp_path / "icc"
    cache = make_benchmark_cache(root, icc_root, "0", 0)
    manifest = json.loads((cache / "manifest.json").read_text())
    del manifest["speaker_1"][0]["input_wav_slice_start_sec"]
    (cache / "manifest.json").write_text(json.dumps(manifest))
    processor = tokenize_icc.BenchmarkTokenizer(device="cpu", components=new_components(), **VAD)
    with pytest.raises(KeyError, match="input_wav_slice_start_sec"):
        processor.process_from_cache(str(cache), str(icc_root))


def test_sample_to_token_range():
    assert tokenize_icc.samples_to_token_range(0, 1280, 10) == (0, 1)
    assert tokenize_icc.samples_to_token_range(1279, 1281, 10) == (0, 2)
    assert tokenize_icc.samples_to_token_range(12800, 99999, 10) == (10, 10)


def test_step1_cache_feeds_step2(tmp_path):
    """A cache written by step 1 (stand-in TTS, random backchannels) tokenises into matching scripts."""
    from neural_scheduler.pipeline import synthesize

    class Backend:
        sr = 24000

        def generate(self, text, audio_prompt_path):
            return torch.rand(1, 4800 + 240 * len(text)) * 0.6 - 0.3

    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(json.dumps({"text": "[S1] First point. [S2] I see. [S1] Second point.\nAnd a third."}) + "\n")
    out_dir = tmp_path / "out"
    argv = ["--input_jsonl", str(scenes), "--out_dir", str(out_dir), "--ref_s1", "a.wav", "--ref_s2", "b.wav"]
    argv += ["--no_classifier", "--bc_probability", "1.0", "--bc_silence_padding_ms", "800", "--intra_turn_pause_ms", "600"]
    synthesize.main(argv, tts=Backend())

    cache_root = out_dir / "segment_cache"
    tokenize.main(["--cache_base_dir", str(cache_root), "--device", "cpu"], components=new_components())
    (record,) = read_jsonl(cache_root / "tokenized_result.jsonl")

    assert sorted((b["turn_id"], b["channel"], b["type"]) for b in record["turn_boundaries"]) == [
        (0, "A", "speech"), (0, "B", "backchannel"),
        (1, "A", "backchannel"), (1, "B", "speech"),
        (2, "A", "speech"), (2, "B", "backchannel"),
    ]  # fmt: skip
    special = vocab.get_special_token_ids(new_components().tokenizer)
    ids = record["input_ids"]
    script_a = ids[ids.index(special["start_script_a"]) + 1 : ids.index(special["end_script_a"])]
    script_b = ids[ids.index(special["start_script_b"]) + 1 : ids.index(special["end_script_b"])]
    for script in (script_a, script_b):
        assert [t for t in script if t >= vocab.FIRST_TURN_TOKEN_ID] == [
            special[f"{edge}_turn_{turn}"] for turn in (0, 1, 2) for edge in ("start", "end")
        ]
    # Turn 2 of speaker 1 has two sentences separated by a pause: two speech blocks, one silence between.
    turn2 = script_a[script_a.index(special["start_turn_2"]) + 1 : script_a.index(special["end_turn_2"])]
    assert turn2.count(special["silence"]) == 1
    source_map = json.load(open(cache_root / record["source_cache"] / "source_map" / "script_source_map_a.json"))
    raw = json.load(open(cache_root / record["source_cache"] / "source_map" / "raw_tokens_a.json"))
    assert len(source_map) == len(script_a)
    for token, index in zip(script_a, source_map):
        if index is not None:
            assert token == AUDIO_TOKEN_OFFSET + raw[index]
