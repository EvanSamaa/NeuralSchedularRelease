"""The step-3 command line, end to end, with a stub model in place of the checkpoint."""

import json

import numpy as np
import pytest
import soundfile as sf

from neural_scheduler.constants import SAMPLES_PER_TOKEN
from neural_scheduler.pipeline import schedule
from neural_scheduler.scheduler import config

from .test_scheduler_helpers import CompactVocab, StubSchedulerModel


def _write_cache(cache_dir, tok, a_codes, b_codes):
    """A segment cache whose audio encodes the codec id in the sample values."""
    source_map = cache_dir / "source_map"
    source_map.mkdir(parents=True)
    for suffix, codes in (("a", a_codes), ("b", b_codes)):
        audio = np.concatenate([np.full(SAMPLES_PER_TOKEN, (c + 1) / 1000.0, dtype=np.float32) for c in codes])
        sf.write(str(source_map / f"concatenated_audio_{suffix}.wav"), audio, 16000, subtype="FLOAT")
        (source_map / f"raw_tokens_{suffix}.json").write_text(json.dumps(codes))
        (source_map / f"script_tokens_{suffix}.json").write_text(json.dumps([tok.code_to_id(c) for c in codes]))
        (source_map / f"script_source_map_{suffix}.json").write_text(json.dumps(list(range(len(codes)))))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    tok = CompactVocab()
    st = tok.special_tokens
    sil = st["silence"]
    a_codes, b_codes, bc_codes = [1, 2, 3, 4, 5, 6], [20, 21, 22], [30, 31]
    ids = lambda codes: [tok.code_to_id(c) for c in codes]
    input_ids = (
        [st["start_script_a"], sil, st["start_turn_0"]] + ids(a_codes) + [st["end_turn_0"], sil, st["end_script_a"]]
        + [st["start_script_b"], sil, st["start_turn_0"]] + ids(bc_codes) + [st["end_turn_0"], sil,
           st["start_turn_1"]] + ids(b_codes) + [st["end_turn_1"], sil, st["end_script_b"], st["start_dialogue"]]
    )
    cache = tmp_path / "run" / "segment_cache" / "dialogue_7"
    _write_cache(cache, tok, a_codes, bc_codes + b_codes)
    jsonl = tmp_path / "run" / "tokenized_result.jsonl"
    sample = {
        "input_ids": input_ids,
        "source_cache": "segment_cache/dialogue_7",  # relative to the JSONL
        "s1_turn_timing": [{"turn_id": 0, "start_token": 2, "end_token": 8,
                            "sub_segments": [{"start_token": 2, "end_token": 8}]}],
    }
    jsonl.write_text(json.dumps(sample) + "\n")

    import neural_scheduler.model as model_module
    monkeypatch.setattr(model_module, "load_scheduler",
                        lambda checkpoint, device="cuda": (StubSchedulerModel(tok.vocab_size), tok))
    return tmp_path, jsonl


def _run(tmp_path, jsonl, *extra):
    schedule.main([
        "--tokenized_jsonl", str(jsonl), "--output_root", str(tmp_path / "out"),
        "--checkpoint", "/models/run_name/checkpoint-1", "--device", "cpu",
        "--pass1_search_window", "3", "--pass1_scoring_context", "2", "--no-save_video", *extra,
    ])
    return tmp_path / "out" / "run_name_checkpoint-1"


def test_three_pass_cli_writes_schedule_and_audio(workspace):
    tmp_path, jsonl = workspace
    out = _run(tmp_path, jsonl) / "sample_0"

    for name in ("channel_a.wav", "channel_b.wav", "channel_sum.wav", "timeline.json"):
        assert (out / "segment_search_v2" / name).is_file()
    assert (out / "input_script" / "channel_a.wav").is_file()

    timeline = json.loads((out / "segment_search_v2" / "timeline.json").read_text())
    assert {p["turn_id"] for p in timeline["refined_placements"]} == {0, 1}
    assert len(timeline["placed_backchannels"]) == 1

    audio, sr = sf.read(str(out / "segment_search_v2" / "channel_a.wav"))
    assert sr == 22050 and np.abs(audio).max() > 0


def test_bc_only_cli_names_output_after_cache_and_keeps_speaker_one(workspace):
    tmp_path, jsonl = workspace
    out = _run(tmp_path, jsonl, "--mode", "bc_only", "--output_sub_dir", "bc", "--output_sr", "16000",
               "--pass2_bc_effect", "np", "--no-save_input_scripts") / "sample_dialogue_7" / "bc"

    assert not (out.parent / "input_script").exists()
    audio, sr = sf.read(str(out / "channel_a.wav"), dtype="float32")
    assert sr == 16000
    # Speaker 1 starts at step 2 and its six tokens are copied (16-bit WAV, hence the tolerance).
    assert np.all(audio[: 2 * SAMPLES_PER_TOKEN] == 0)
    expected = np.concatenate([np.full(SAMPLES_PER_TOKEN, (c + 1) / 1000.0, dtype=np.float32) for c in range(1, 7)])
    np.testing.assert_allclose(audio[2 * SAMPLES_PER_TOKEN: 8 * SAMPLES_PER_TOKEN], expected, atol=1e-4)


def test_cli_defaults_and_removed_flags():
    parser = schedule.build_parser()
    args = parser.parse_args(["--tokenized_jsonl", "x", "--output_root", "y", "--checkpoint", "z"])
    assert args.pass2_scoring_context == config.CLI_DEFAULT_OVERRIDES["pass2_scoring_context"] == 0
    assert args.pass2_bc_effect == "hp+fp+np"
    assert args.pass1_inter_turn_gap == 5 and args.pass3_max_extra_gap == 10
    assert args.save_video is None and args.save_input_scripts is True and args.mode == "three_pass"
    for removed in ("--pass1_min_gap", "--pass3_search_window"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--tokenized_jsonl", "x", "--output_root", "y", "--checkpoint", "z", removed, "1"])


def test_source_cache_resolution(tmp_path):
    jsonl = tmp_path / "a" / "t.jsonl"
    assert schedule.resolve_cache_dir({"source_cache": "/abs/cache"}, 0, str(jsonl), None) == "/abs/cache"
    assert schedule.resolve_cache_dir({"source_cache": "c/x"}, 0, str(jsonl), None) == str(tmp_path / "a" / "c" / "x")
    assert schedule.resolve_cache_dir({}, 3, str(jsonl), "/base") == "/base/example_0003"
    with pytest.raises(ValueError):
        schedule.resolve_cache_dir({}, 3, str(jsonl), None)
