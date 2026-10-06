"""Unit tests for the preprocessing core (no model, no GPU, no original tree)."""

import io
import json
import logging
import random

import pytest
import torch

from neural_scheduler import constants as C
from neural_scheduler.data import durations, sequence, sharding, turns, vad
from neural_scheduler.data import inspect as data_inspect
from neural_scheduler.data import merge as data_merge
from neural_scheduler.vocab import FIRST_TURN_TOKEN_ID

from .data_stubs import StubTokenizer, fake_speech_tokens, make_fake_vad, stub_special_tokens


def test_turn_ids_backchannel_shares_host_turn():
    left = [{"start": 0, "end": 40}, {"start": 42, "end": 50}]
    right = [{"start": 10, "end": 14}, {"start": 52, "end": 70}]
    assigned = turns.assign_turn_ids(left, right)
    by_span = {(s["channel"], s["start"]): s["turn_id"] for s in assigned}
    assert by_span[("A", 0)] == 0
    assert by_span[("B", 10)] == 0  # fully inside A's first turn
    assert by_span[("A", 42)] == 1
    assert by_span[("B", 52)] == 2


def test_adjacent_same_speaker_segments_merge():
    merged = turns.merge_adjacent_same_speaker_segments([
        {"start": 0, "end": 5, "channel": "A"},
        {"start": 7, "end": 9, "channel": "A"},
        {"start": 10, "end": 12, "channel": "B"},
    ])
    assert [(m["channel"], m["start"], m["end"], len(m["sub_segments"])) for m in merged] == [
        ("A", 0, 9, 2), ("B", 10, 12, 1),
    ]


def test_remaining_durations_count_down_per_channel():
    s, x, y = 9, 1, 2
    tokens = [s, x, s, s, s, y, x, s]  # A: s s s x   B: x s y s
    assert durations.compute_remaining_durations_dual_channel(tokens) == [2, 0, 1, 0, 0, 0, 0, 0]
    assert durations.validate_remaining_durations(tokens, [2, 0, 1, 0, 0, 0, 0, 0]) == []
    assert durations.validate_remaining_durations(tokens, [1, 0, 1, 0, 0, 0, 0, 0])


def test_vad_conversion_floors():
    ts = [{"start": 1279, "end": 2561}]
    assert vad.vad_timestamps_to_token_indices(ts, 16000) == [{"start": 0, "end": 2}]
    assert vad.vad_timestamps_to_seconds(ts, 16000) == [{"start": 1279 / 16000, "end": 2561 / 16000}]


def test_window_bounds():
    per_window = sequence.window_sample_count(16000)
    assert per_window == 327680
    assert per_window / 16000 == C.WINDOW_SECONDS
    assert list(sequence.iter_window_bounds(per_window + 10, 16000)) == [
        (0, 0, per_window), (1, per_window, per_window + 10),
    ]
    # An exact multiple has no trailing empty window.
    assert list(sequence.iter_window_bounds(2 * per_window, 16000)) == [
        (0, 0, per_window), (1, per_window, 2 * per_window),
    ]


def test_shard_indices_partition_everything():
    for total, n in [(5850, 21), (10, 3), (3, 5)]:
        seen = [i for s in range(n) for i in sharding.get_shard_indices(total, n, s)]
        assert seen == list(range(total))


def test_script_layout_and_turn_cap():
    tok = StubTokenizer()
    st = stub_special_tokens()
    sil = st["silence"]
    raw = list(range(100, 130))
    segs = [
        {"turn_id": 0, "channel": "A", "sub_segments": [{"start": 2, "end": 4}, {"start": 6, "end": 7}]},
        {"turn_id": C.MAX_TURNS, "channel": "A", "sub_segments": [{"start": 10, "end": 12}]},
    ]
    script = sequence.build_script_tokens_for_channel(raw, segs, tok, st)
    a = C.AUDIO_TOKEN_OFFSET
    assert script == [sil, st["start_turn_0"], a + 102, a + 103, sil, a + 106, st["end_turn_0"], sil]


def _run_window(codes_seed=0, n_frames=256, force_code=None):
    sr = 16000
    n = n_frames * C.SAMPLES_PER_TOKEN
    audio = torch.zeros(n)
    rng = random.Random(codes_seed)
    get_ts = make_fake_vad(rng)
    return sequence.process_audio_window(
        audio, audio, sr, None, None, StubTokenizer(), stub_special_tokens(), None, get_ts,
        logging.getLogger("test"), 7, 3, 61.44, 81.92, device="cpu",
    )


def test_process_audio_window_record(monkeypatch):
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake_speech_tokens(seed=1))
    rec = _run_window()
    assert list(rec) == [
        "input_ids", "generation_start_index", "generation_end_index", "remaining_durations",
        "turn_boundaries", "source_file", "window_idx", "vad_timestamps_left", "vad_timestamps_right",
        "window_start_sec", "window_end_sec",
    ]
    ids, start, end = rec["input_ids"], rec["generation_start_index"], rec["generation_end_index"]
    st = stub_special_tokens()
    assert ids[0] == st["start_script_a"] and ids[start - 1] == st["start_dialogue"] and ids[end] == st["end_dialogue"]
    assert end - start == 512 == len(rec["remaining_durations"])
    assert durations.validate_remaining_durations(ids[start:end], rec["remaining_durations"]) == []
    # Turn markers never appear in the dialogue; script speech tokens are a subset of dialogue tokens.
    assert all(t < C.BASE_VOCAB_SIZE for t in ids[start:end])
    script_speech = {t for t in ids[:start] if t < C.BASE_VOCAB_SIZE and t != st["silence"]}
    assert script_speech <= set(ids[start:end])
    assert all(FIRST_TURN_TOKEN_ID <= t for t in ids[:start] if t >= C.BASE_VOCAB_SIZE + 6)


def test_unknown_audio_code_drops_window_and_is_counted(monkeypatch):
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake_speech_tokens(seed=1, constant=16383))
    with pytest.raises(sequence.UnknownAudioCodeError):
        _run_window()

    out = io.StringIO()
    audio = torch.zeros(300 * C.SAMPLES_PER_TOKEN)
    stats = sequence.process_conversation(
        0, audio, audio, 16000, None, None, StubTokenizer(), stub_special_tokens(), None,
        make_fake_vad(random.Random(0)), logging.getLogger("test"), out, device="cpu",
    )
    assert (stats.written, stats.failed, stats.dropped_unknown_code) == (0, 2, 2)
    assert out.getvalue() == ""

    # With the token added after the turn tokens the same windows are kept.
    tok = StubTokenizer(extra_tokens=["<|audio_16383|>"])
    assert tok.convert_tokens_to_ids(["<|audio_16383|>"])[0] == 168842
    stats = sequence.process_conversation(
        0, audio, audio, 16000, None, None, tok, stub_special_tokens(), None,
        make_fake_vad(random.Random(0)), logging.getLogger("test"), out, device="cpu",
    )
    assert (stats.written, stats.failed) == (2, 0)


def test_per_window_resume_completes_interrupted_conversation(tmp_path, monkeypatch):
    from neural_scheduler.data import preprocess

    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake_speech_tokens(seed=3))
    audio = torch.zeros(3 * 327680 + 5000)
    models = preprocess.Models(None, None, StubTokenizer(), stub_special_tokens(), None, None)
    logger = logging.getLogger("test")

    def jobs():
        return [preprocess.ConversationJob(i, lambda: (audio, audio, 16000)) for i in range(2)]

    def run(path, **kw):
        models.get_speech_timestamps = make_fake_vad(random.Random(5), deterministic_per_call=True)
        return preprocess.process_shard(jobs(), models, str(path), logger, device="cpu", **kw)

    full = tmp_path / "full.jsonl"
    run(full)
    lines = full.read_text().splitlines()
    assert len(lines) == 6  # three windows with speech per conversation; the short tail has none

    # Simulate a job killed inside conversation 0: two of its three records written.
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(lines[:2]) + "\n")
    summary = run(partial, resume_per_window=True)
    assert partial.read_text().splitlines() == lines
    assert summary.windows_written == 4

    # Conversation-level resume treats the interrupted conversation as done.
    partial.write_text("\n".join(lines[:2]) + "\n")
    summary = run(partial, resume=True)
    assert summary.skipped == 1 and len(partial.read_text().splitlines()) == 5


def test_default_special_token_ids_match_vocabulary():
    assert data_inspect.default_special_token_ids() == stub_special_tokens()


def test_inspect_runs_on_a_record(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sequence, "extract_speech_token_batched", fake_speech_tokens(seed=1))
    rec = _run_window()
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    data_inspect.main(["show", "--input_file", str(path)])
    data_inspect.main(["diagnose", "--input_file", str(path)])
    out = capsys.readouterr().out
    assert "PASSED" in out and "SCRIPT A REGION" in out and "MISMATCH" not in out and "MATCH" in out


def test_fisher_split_keeps_conversations_together(tmp_path):
    rng = random.Random(0)
    for shard in range(3):
        with open(tmp_path / f"fisher_audio_tokens_v2_shard_{shard}_of_3.jsonl", "w") as f:
            for conv in range(shard * 100, shard * 100 + 100):
                for w in range(rng.randint(1, 4)):
                    f.write(json.dumps({"source_file": conv, "window_idx": w}) + "\n")
    data_merge.main(["--dataset", "fisher", "--input_dir", str(tmp_path)])
    owners = {}
    for split in ("train", "val", "test"):
        for line in open(tmp_path / f"fisher_audio_tokens_v2_{split}.jsonl"):
            owners.setdefault(json.loads(line)["source_file"], set()).add(split)
    assert len(owners) == 300 and all(len(v) == 1 for v in owners.values())
    counts = {s: sum(1 for v in owners.values() if v == {s}) for s in ("train", "val", "test")}
    assert counts == {"train": 294, "val": 3, "test": 3}


def test_seamless_trainer_names(tmp_path):
    for dtype in ("improvised", "naturalistic"):
        for split in ("train", "test"):
            name = f"seamless_interaction_{dtype}_{split}_audio_tokens_shard_00000_of_1.jsonl"
            (tmp_path / name).write_text(json.dumps({"t": dtype, "s": split}) + "\n\n")
    out = tmp_path / "out"
    data_merge.main(["--dataset", "seamless", "--input_dir", str(tmp_path), "--output_dir", str(out), "--trainer_names"])
    val = out / "audio_tokens_val_with_durations_and_turnID_as_tokens.jsonl"
    assert [json.loads(l) for l in val.read_text().splitlines()] == [
        {"t": "improvised", "s": "test"}, {"t": "naturalistic", "s": "test"},
    ]
    assert (out / "audio_tokens_train_with_durations_and_turnID_as_tokens.jsonl").exists()
