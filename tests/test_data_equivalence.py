"""Equivalence of the preprocessing code with the original research scripts.

The originals are imported from ``$NTPP_ORIG`` with their external module-level
imports (the LMDB reader of another repository and the GLM-4-Voice speech
tokenizer) replaced by placeholders; tokenizer, speech tokenizer and VAD are
CPU stand-ins shared by both sides.
"""

import contextlib
import importlib.util
import io
import json
import logging
import random
import sys
import types

import numpy as np
import pytest
import torch

from neural_scheduler import constants as C
from neural_scheduler.data import durations, sequence, sharding, turns, vad
from neural_scheduler.data import merge as data_merge
from neural_scheduler.data import speech_tokenizer as st_mod
from neural_scheduler.data.sources import fisher as fisher_src
from neural_scheduler.data.sources import seamless as seamless_src

from .data_stubs import StubTokenizer, fake_speech_tokens, make_fake_vad, stub_special_tokens

pytestmark = pytest.mark.equivalence

G3 = "scripts/processing_for_timing_only_with_duration_prediction"


def _load_original(root, relpath, name):
    placeholders = {
        "lmdbdatasets": types.SimpleNamespace(WavLMDBDataset=object),
        "speech_tokenizer": types.ModuleType("speech_tokenizer"),
        "speech_tokenizer.modeling_whisper": types.SimpleNamespace(WhisperVQEncoder=object),
    }
    saved = {k: sys.modules.get(k) for k in placeholders}
    saved_path = list(sys.path)
    sys.modules.update(placeholders)
    try:
        spec = importlib.util.spec_from_file_location(name, root / relpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


@pytest.fixture(scope="module")
def orig_fisher(ntpp_orig):
    return _load_original(
        ntpp_orig, f"{G3}/preprocess_fisher_sharded_DTW_with_duration_and_turnID_tokens.py", "orig_fisher_prep")


@pytest.fixture(scope="module")
def orig_seamless(ntpp_orig):
    return _load_original(
        ntpp_orig, f"{G3}/preprocess_seamless_sharded_DTW_with_duration_and_turnID_tokens.py", "orig_seamless_prep")


@pytest.fixture(scope="module", params=["fisher", "seamless"])
def orig(request, orig_fisher, orig_seamless):
    return orig_fisher if request.param == "fisher" else orig_seamless


def _random_vad_tokens(rng, n_frames=256):
    segs, pos = [], rng.randrange(0, 10)
    while pos < n_frames and rng.random() < 0.93:
        end = min(n_frames, pos + rng.choice([0, 1, 2, 5, 12, 40]))
        segs.append({"start": pos, "end": end})
        pos = end + rng.choice([0, 1, 3, 9, 30])
    return segs


def test_constants_match(orig):
    assert orig.MAX_TOKEN_SIZE == C.MAX_TOKEN_SIZE and orig.NUM_CHANNELS == C.NUM_CHANNELS
    assert orig.INTERLEAVE_FACTOR == sequence.INTERLEAVE_FACTOR and orig.TOKEN_RATE_HZ == C.TOKEN_RATE_HZ
    assert orig.SILENCE_TOKEN_ID == C.SILENCE_CODE and orig.MAX_TURNS == C.MAX_TURNS
    assert orig.TARGET_AUDIO_SR == C.SPEECH_TOKENIZER_SR
    assert (orig.VAD_THRESHOLD, orig.VAD_MIN_SPEECH_MS, orig.VAD_MIN_SILENCE_MS) == (
        vad.VAD_THRESHOLD, vad.VAD_MIN_SPEECH_MS, vad.VAD_MIN_SILENCE_MS)


def test_special_token_ids_match(orig):
    assert orig.get_special_token_ids(StubTokenizer()) == stub_special_tokens()


def test_turn_functions_match(orig):
    rng = random.Random(0)
    for _ in range(500):
        left, right = _random_vad_tokens(rng), _random_vad_tokens(rng)
        expected = orig.assign_turn_ids([dict(s) for s in left], [dict(s) for s in right], overlap_threshold=0.5)
        assert turns.assign_turn_ids(left, right, overlap_threshold=0.5) == expected
        pooled = sorted(
            [dict(s, channel="A") for s in left] + [dict(s, channel="B") for s in right],
            key=lambda x: (x["start"], x["channel"]))
        assert turns.merge_adjacent_same_speaker_segments(pooled) == orig.merge_adjacent_same_speaker_segments(pooled)
        for a in pooled[:5]:
            for b in pooled[:5]:
                assert turns.compute_overlap_ratio(a, b) == orig.compute_overlap_ratio(a, b)


def test_vad_conversions_match(orig):
    rng = random.Random(1)
    for sr in (16000, 8000, 22050):
        ts = [{"start": rng.randrange(0, 400000), "end": rng.randrange(0, 400000)} for _ in range(200)]
        assert vad.vad_timestamps_to_token_indices(ts, sr) == orig.vad_timestamps_to_token_indices(ts, sr)
        assert vad.vad_timestamps_to_seconds(ts, sr) == orig.vad_timestamps_to_seconds(ts, sr)


def test_silencing_durations_and_script_match(orig):
    rng = random.Random(2)
    tok, special = StubTokenizer(), stub_special_tokens()
    for _ in range(300):
        n = rng.choice([0, 1, 7, 100, 256])
        codes = [rng.choice([C.SILENCE_CODE, 5, 5, 9, rng.randrange(16383)]) for _ in range(n)]
        segs = _random_vad_tokens(rng)
        assert sequence.silencing_with_vad(codes, segs) == orig.silencing_with_vad(codes, segs)
        inter = [rng.choice([1, 1, 2, 3]) for _ in range(rng.choice([0, 1, 2, 9, 64, 65]))]
        assert (durations.compute_remaining_durations_dual_channel(inter)
                == orig.compute_remaining_durations_dual_channel(inter))
        assigned = turns.assign_turn_ids(segs, _random_vad_tokens(rng))
        for channel in "AB":
            ch = [s for s in assigned if s["channel"] == channel]
            assert (sequence.build_script_tokens_for_channel(codes, ch, tok, special)
                    == orig.build_script_tokens_for_channel(codes, ch, tok, special))


def test_script_turn_cap_matches(orig):
    """More than MAX_TURNS turns: the surplus turns are left out of the script on both sides."""
    tok, special = StubTokenizer(), stub_special_tokens()
    segs = [{"start": 4 * i, "end": 4 * i + 2} for i in range(60)]
    assigned = turns.assign_turn_ids(segs[0::2], segs[1::2])
    assert max(s["turn_id"] for s in assigned) >= C.MAX_TURNS
    codes = list(range(300))
    for channel in "AB":
        ch = [s for s in assigned if s["channel"] == channel]
        assert (sequence.build_script_tokens_for_channel(codes, ch, tok, special)
                == orig.build_script_tokens_for_channel(codes, ch, tok, special))


@pytest.mark.parametrize("seed", range(12))
def test_process_audio_window_matches(orig, monkeypatch, seed):
    fake = fake_speech_tokens(seed)
    monkeypatch.setattr(orig, "extract_speech_token_batched", fake)
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake)
    tok, special = StubTokenizer(), stub_special_tokens()
    logger = logging.getLogger("test")
    n = random.Random(seed).choice([327680, 327680, 200000, 1280 * 3 + 7])
    left, right = torch.randn(n), torch.randn(n)

    def run(fn, extra):
        try:
            return fn(left, right, 16000, None, None, tok, special, None, make_fake_vad(random.Random(seed)),
                      logger, 11, seed, seed * 20.48, seed * 20.48 + n / 16000, **extra)
        except KeyError as e:
            return ("KeyError", e.args[0])

    expected = run(orig.process_audio_window, {})
    actual = run(sequence.process_audio_window, {"device": "cpu"})
    assert json.dumps(actual) == json.dumps(expected)


def test_unknown_code_fails_on_both_sides(orig, monkeypatch):
    fake = fake_speech_tokens(0, constant=16383)
    monkeypatch.setattr(orig, "extract_speech_token_batched", fake)
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake)
    args = (torch.zeros(327680), torch.zeros(327680), 16000, None, None, StubTokenizer(), stub_special_tokens(), None)
    with pytest.raises(KeyError):
        orig.process_audio_window(*args, make_fake_vad(random.Random(4), True), logging.getLogger("t"), 0, 0, 0.0, 20.48)
    with pytest.raises(KeyError):
        sequence.process_audio_window(
            *args, make_fake_vad(random.Random(4), True), logging.getLogger("t"), 0, 0, 0.0, 20.48, device="cpu")


@pytest.mark.parametrize("num_samples", [327680 * 2 + 4000, 327680 * 2, 5000, 327680 * 3 + 100000])
def test_conversation_records_match_fisher(orig_fisher, monkeypatch, num_samples):
    """Whole-conversation output (windowing, failures, written lines) against the Fisher script."""
    def fake(model, feature_extractor, batched_audio, **kw):
        rows = fake_speech_tokens(9)(model, feature_extractor, batched_audio, **kw)
        if batched_audio.shape[1] == 327680 and float(batched_audio[0, 0]) > 0.5:
            rows[0][5] = 16383  # make some full windows hit the unknown code
        return rows

    monkeypatch.setattr(orig_fisher, "extract_speech_token_batched", fake)
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake)
    torch.manual_seed(num_samples)
    left, right = torch.randn(num_samples), torch.randn(num_samples - 37)
    tok, special = StubTokenizer(), stub_special_tokens()
    logger = logging.getLogger("quiet")
    logger.disabled = True

    expected = io.StringIO()
    ok, failed, _ = orig_fisher.process_example(
        3, {"left": left, "right": right, "sr": 16000}, None, None, tok, special, None,
        (make_fake_vad(random.Random(1), True),), logger, expected)
    actual = io.StringIO()
    stats = sequence.process_conversation(
        3, left, right, 16000, None, None, tok, special, None,
        make_fake_vad(random.Random(1), True), logger, actual, device="cpu")
    assert actual.getvalue() == expected.getvalue()
    assert stats.written == ok
    # The original counts the empty trailing window of an exact multiple as failed or silent; it writes nothing.
    assert stats.failed in (failed, failed - 1)


def test_conversation_records_match_seamless(orig_seamless, monkeypatch, tmp_path):
    import soundfile as sf

    fake = fake_speech_tokens(4)
    monkeypatch.setattr(orig_seamless, "extract_speech_token_batched", fake)
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake)
    rng = np.random.default_rng(0)
    paths = []
    for name, sr, n in (("V01_S0001_I00000001_P0001.wav", 48000, 48000 * 45), ("V01_S0001_I00000001_P0002.wav", 16000, 16000 * 44)):
        path = tmp_path / name
        sf.write(path, (rng.standard_normal(n) * 0.1).astype(np.float32), sr)
        paths.append(str(path))
    tok, special = StubTokenizer(), stub_special_tokens()
    logger = logging.getLogger("quiet")
    logger.disabled = True

    expected = io.StringIO()
    orig_seamless.process_example(
        5, paths, None, None, tok, special, None, (make_fake_vad(random.Random(1), True),), logger, expected)

    from neural_scheduler.data.preprocess import seamless_jobs

    job = seamless_jobs({"examples": [{"example_id": 5, "audio_paths": paths}]}, "improvised", "test", "index")[0]
    left, right, sr = job.load()
    orig_left, orig_sr = orig_seamless.load_and_resample_audio(paths[0], 16000)
    assert sr == orig_sr and torch.equal(left, orig_left)
    actual = io.StringIO()
    sequence.process_conversation(
        job.example_id, left, right, sr, None, None, tok, special, None,
        make_fake_vad(random.Random(1), True), logger, actual, extra_fields=job.extra_fields, device="cpu")
    assert actual.getvalue() == expected.getvalue() != ""
    assert json.loads(actual.getvalue().splitlines()[0])["audio_path_left"] == paths[0]


def test_extract_speech_token_batched_matches(orig):
    """Chunking, batching and mask handling with a fake encoder and the real feature extractor."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=8)

    class FakeEncoder:
        config = types.SimpleNamespace(pooling_kernel_size=4)
        conv1 = types.SimpleNamespace(stride=(1,))
        conv2 = types.SimpleNamespace(stride=(2,))

        def __call__(self, input_features, attention_mask):
            frames = input_features.shape[-1] // 8
            pooled = input_features[:, :, : frames * 8].reshape(input_features.shape[0], 8, frames, 8).mean((1, 3))
            return types.SimpleNamespace(quantized_token_ids=(pooled * 1000).long().abs() % 16384)

    torch.manual_seed(0)
    for shape in [(2, 327680), (2, 16000 * 70 + 333), (3, 5000), (1, 3, 40000)]:
        audio = torch.randn(shape) * 0.1
        expected = orig.extract_speech_token_batched(FakeEncoder(), fe, audio, model_batch_size=2, device="cpu")
        actual = st_mod.extract_speech_token_batched(FakeEncoder(), fe, audio, model_batch_size=2, device="cpu")
        assert actual == expected and len(actual[0]) > 0


def test_shard_indices_match(orig_fisher):
    for total, n in [(5850, 21), (5850, 25), (10, 3), (3, 5), (0, 2)]:
        for s in range(n):
            assert sharding.get_shard_indices(total, n, s) == orig_fisher.get_shard_indices(total, n, s)


def test_resume_ids_match(orig, tmp_path):
    path = tmp_path / "o.jsonl"
    path.write_text('{"source_file": 3, "window_idx": 0}\n{"source_file": 3, "window_idx": 1}\n{"source_fi\n'
                    '{"source_file": 9, "window_idx": 4}\n')
    assert sharding.get_processed_example_ids(str(path)) == orig.get_processed_example_ids(str(path)) == {3, 9}
    assert sharding.get_processed_example_ids(str(tmp_path / "missing")) == set()


def _make_seamless_tree(root):
    rng = random.Random(0)
    for v, s, i, parts in [(1, 1, 1, ["0001", "0002"]), (1, 1, 2, ["0010A", "0002"]), (2, 5, 1, ["0007"]),
                           (2, 6, 9, ["0003", "0001", "0002"]), (10, 2, 3, ["0005", "0004"])]:
        sub = root / "improvised" / "test" / f"{rng.randrange(9999):04d}"
        sub.mkdir(parents=True, exist_ok=True)
        for p in parts:
            (sub / f"V{v:02d}_S{s:04d}_I{i:08d}_P{p}.wav").write_bytes(b"")
    (root / "improvised" / "test" / "notes.txt").write_text("x")
    (root / "improvised" / "test" / "stray.wav").write_bytes(b"")


def test_seamless_discovery_and_manifests_match(orig_seamless, tmp_path):
    root = tmp_path / "data"
    _make_seamless_tree(root)
    split_dir = str(root / "improvised" / "test")

    for name in ["V01_S0223_I00000127_P1505.wav", "/a/b/V3_S4_I5_P6A.wav", "stray.wav", "V1_S2_I3_P4.flac"]:
        assert seamless_src.parse_filename_to_dict(name) == orig_seamless.parse_filename_to_dict(name)

    groups = seamless_src.group_files_by_context(seamless_src.find_wav_files_recursive(split_dir))
    expected_groups = orig_seamless.group_files_by_context(orig_seamless.find_wav_files_recursive(split_dir))
    assert groups == expected_groups and len(groups) == 5
    assert seamless_src.discover_interactions(str(root), "improvised", "test") == groups

    for shard_count in (1, 2, 3, 7):
        ours, theirs = tmp_path / f"ours{shard_count}", tmp_path / f"theirs{shard_count}"
        with contextlib.redirect_stdout(io.StringIO()):
            seamless_src.create_shard_manifests(groups, str(ours), "improvised", "test", shard_count)
            orig_seamless.create_shard_manifests(groups, str(theirs), "improvised", "test", shard_count)
        ours_files = sorted(p.relative_to(ours) for p in ours.rglob("*.json"))
        assert ours_files == sorted(p.relative_to(theirs) for p in theirs.rglob("*.json")) and ours_files
        for rel in ours_files:
            assert (ours / rel).read_bytes() == (theirs / rel).read_bytes()
        meta = seamless_src.load_shard_metadata(str(ours), "improvised", "test")
        for shard_id in range(meta["shard_count"]):
            assert (seamless_src.load_shard_manifest(str(ours), "improvised", "test", shard_id)
                    == orig_seamless.load_shard_manifest(str(theirs), "improvised", "test", shard_id))

    assert seamless_src.build_debug_manifest(groups)["examples"] == [
        {"example_id": i, "original_idx": i, "audio_paths": sorted(g, key=lambda x: x.split("_")[-1])}
        for i, g in enumerate(groups) if len(g) >= 2
    ]


def test_fisher_merge_matches(ntpp_orig, tmp_path, monkeypatch):
    orig_merge = _load_original(ntpp_orig, f"{G3}/merge_shards_fisher.py", "orig_merge_fisher")
    rng = random.Random(7)
    shards = tmp_path / "shards"
    shards.mkdir()
    conv = 0
    for shard in range(5):
        with open(shards / f"fisher_audio_tokens_v2_shard_{shard}_of_5.jsonl", "w") as f:
            for _ in range(rng.randint(150, 260)):
                for w in range(rng.randint(1, 30)):
                    f.write(json.dumps({"source_file": conv, "window_idx": w, "x": rng.random()}) + "\n")
                conv += 1
            f.write("\n")

    for seed, fracs in [(42, (0.98, 0.01, 0.01)), (7, (0.9, 0.05, 0.05))]:
        counts = {i: rng.randint(1, 30) for i in range(conv)}
        assert data_merge.assign_splits(counts, *fracs, seed) == orig_merge.assign_splits(counts, *fracs, seed)

    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    with contextlib.redirect_stdout(io.StringIO()):
        data_merge.main(["--dataset", "fisher", "--input_dir", str(shards), "--output_dir", str(ours)])
        monkeypatch.setattr(sys, "argv", ["x", "--input_dir", str(shards), "--output_dir", str(theirs)])
        orig_merge.main()
    for split in ("train", "val", "test"):
        name = f"fisher_audio_tokens_v2_{split}.jsonl"
        assert (ours / name).read_bytes() == (theirs / name).read_bytes()
    assert (ours / "fisher_audio_tokens_v2_val.jsonl").stat().st_size > 0


def test_seamless_merge_matches(ntpp_orig, tmp_path, monkeypatch):
    orig_merge = _load_original(ntpp_orig, f"{G3}/merge_shards_seamless.py", "orig_merge_seamless")
    shards = tmp_path / "shards"
    shards.mkdir()
    rng = random.Random(3)
    for dtype, split, n in [("improvised", "train", 12), ("naturalistic", "train", 11), ("improvised", "test", 2),
                            ("naturalistic", "test", 2), ("naturalistic", "dev", 1)]:
        for shard in range(n):
            if (dtype, split, shard) == ("improvised", "train", 4):
                continue  # a missing shard
            path = shards / f"seamless_interaction_{dtype}_{split}_audio_tokens_shard_{shard:05d}_of_{n}.jsonl"
            path.write_text("".join(json.dumps({"d": dtype, "s": shard, "r": rng.random()}) + "\n\n" * (i % 3 == 0)
                                    for i in range(rng.randint(0, 9))))
    (shards / "seamless_interaction_improvised_train_audio_tokens_debug.jsonl").write_text("{}\n")

    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    with contextlib.redirect_stdout(io.StringIO()):
        data_merge.main(["--dataset", "seamless", "--input_dir", str(shards), "--output_dir", str(ours)])
        monkeypatch.setattr(sys, "argv", ["x", "--input_dir", str(shards), "--output_dir", str(theirs)])
        orig_merge.main()
    names = sorted(p.name for p in theirs.iterdir())
    assert names == sorted(p.name for p in ours.iterdir()) and len(names) == 3
    for name in names:
        assert (ours / name).read_bytes() == (theirs / name).read_bytes()


def _write_fisher_dir(path, sr=8000):
    import soundfile as sf

    rng = np.random.default_rng(1)
    path.mkdir()
    for i, n in enumerate([sr * 3 + 17, sr * 2, sr * 5 + 1]):
        sf.write(path / f"fe_03_{i + 1:05d}.wav", (rng.standard_normal((n, 2)) * 0.2).astype(np.float32), sr,
                 subtype="FLOAT")


def test_fisher_lmdb_roundtrip_and_directory_source(tmp_path):
    pytest.importorskip("lmdb")
    audio_dir = tmp_path / "audio"
    _write_fisher_dir(audio_dir)
    lmdb_path, keys_path = str(tmp_path / "f.lmdb"), str(tmp_path / "keys.txt")
    assert fisher_src.build_fisher_lmdb(str(audio_dir), lmdb_path, keys_path, map_size_gb=0.1) == 3

    from_lmdb = fisher_src.FisherLMDBSource(lmdb_path, keys_path)
    from_dir = fisher_src.FisherDirectorySource(str(audio_dir))
    assert from_lmdb.keys == from_dir.keys == ["fe_03_00001.wav", "fe_03_00002.wav", "fe_03_00003.wav"]
    for i in range(3):
        a, b = from_lmdb.load(i), from_dir.load(i)
        assert a["sr"] == b["sr"] == 16000
        assert torch.equal(a["left"], b["left"]) and torch.equal(a["right"], b["right"])
        assert not torch.equal(a["left"], a["right"])
    raw, sr = from_lmdb._read_raw(0)
    assert sr == 8000 and raw.shape == (2, 8000 * 3 + 17) and raw.dtype == torch.float32
    assert from_lmdb.load(0)["left"].shape[0] == 2 * (8000 * 3 + 17)


def test_fisher_lmdb_reader_matches_original_reader(tmp_path):
    """Against the reader the released data was produced with (lives in another repository)."""
    import os

    pytest.importorskip("lmdb")
    external = os.environ.get("NTPP_LMDB_READER_DIR")
    if not external or not os.path.isfile(os.path.join(external, "lmdbdatasets.py")):
        pytest.skip("NTPP_LMDB_READER_DIR does not point at the directory containing lmdbdatasets.py")
    saved_path = list(sys.path)
    sys.path.insert(0, external)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from lmdbdatasets import WavLMDBDataset
    finally:
        sys.path[:] = saved_path

    audio_dir = tmp_path / "audio"
    _write_fisher_dir(audio_dir)
    lmdb_path, keys_path = str(tmp_path / "f.lmdb"), str(tmp_path / "keys.txt")
    fisher_src.build_fisher_lmdb(str(audio_dir), lmdb_path, keys_path, map_size_gb=0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        original = WavLMDBDataset(lmdb_path, keys_path, window_size=-1, do_cache=False, output_sr=16000)
    ours = fisher_src.FisherLMDBSource(lmdb_path, keys_path)
    assert len(original) == len(ours) == 3
    for i in range(3):
        a, b = original[i], ours.load(i)
        assert a["sr"] == b["sr"] and torch.equal(a["left"], b["left"]) and torch.equal(a["right"], b["right"])
