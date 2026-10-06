"""The YAML-driven pipeline runner and the shipped pipeline configs."""

import glob
import os
import sys
from pathlib import Path

import pytest
import yaml

from neural_scheduler.pipeline import run as runner
from neural_scheduler.pipeline import synthesize, tokenize, tokenize_icc

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted(glob.glob(str(REPO / "configs" / "pipeline" / "*.yaml")))


def base_config():
    return {
        "mode": "three_pass",
        "variants": ["bc_never", "classifier"],
        "io": {"input_jsonl": "scenes.jsonl", "out_dir": "out"},
        "classifier": {
            "type": "joint",
            "dir": "ckpt/best_model",
            "confidence_threshold": 0.3,
            "vocab_per_class": "vocab.json",
            "vocab": ["yeah", "that's right"],
            "suppress_consecutive_bcs": False,
            "bc_context": True,
            "ensure_at_least_one_bc": False,
        },
        "tts": {"ref_s1": "a.wav", "ref_s2": "b.wav", "seed": 46, "run_naive_only": True, "naive_vad_trim": True},
        "vad": {"threshold": 0.1},
        "scheduler": {
            "checkpoint": "ckpt/scheduler",
            "save_video": False,
            "pass1": {"search_window": 10, "schedule_subturns": True, "use_speaker_gap_constraints": False},
            "pass2": {"bc_effect": "hp+np"},
            "pass3": {"max_extra_gap": 12},
        },
    }


def command_for(config, step, variant=None):
    resolved = runner.config_for_variant(config, variant)
    command, env = runner.step_command(resolved, step, variant)
    return command, env


def test_step_commands_are_exact():
    config = base_config()
    command, env = command_for(config, 1, "classifier")
    assert env == {}
    assert command == [
        sys.executable, "-m", "neural_scheduler.pipeline.synthesize",
        "--out_dir", os.path.join("out", "classifier"),
        "--input_jsonl", "scenes.jsonl",
        "--bc_classifier_type", "joint",
        "--bc_classifier_dir", "ckpt/best_model",
        "--bc_confidence_threshold", "0.3",
        "--no_suppress_consecutive_bcs",
        "--bc_vocab_per_class", "vocab.json",
        "--bc_vocab", "yeah", "that's right",
        "--no-ensure_at_least_one_bc",
        "--ref_s1", "a.wav",
        "--ref_s2", "b.wav",
        "--seed", "46",
        "--run_naive_only",
        "--naive_vad_trim",
    ]  # fmt: skip

    cache = os.path.join("out", "classifier", "segment_cache")
    jsonl = os.path.join(cache, "tokenized_result.jsonl")
    command, _ = command_for(config, 2, "classifier")
    assert command == [
        sys.executable, "-m", "neural_scheduler.pipeline.tokenize",
        "--cache_base_dir", cache, "--output_path", jsonl, "--vad_threshold", "0.1",
    ]  # fmt: skip

    command, _ = command_for(config, 3, "classifier")
    assert command == [
        sys.executable, "-m", "neural_scheduler.pipeline.schedule",
        "--tokenized_jsonl", jsonl, "--cache_base_dir", cache,
        "--output_root", os.path.join("out", "classifier"), "--mode", "three_pass",
        "--checkpoint", "ckpt/scheduler",
        "--no-save_video",
        "--pass1_search_window", "10",
        "--pass_1_schedule_subturns",
        "--pass2_bc_effect", "hp+np",
        "--pass3_max_extra_gap", "12",
    ]  # fmt: skip


def test_bc_never_variant_raises_threshold_only_for_that_variant():
    config = base_config()
    never, _ = command_for(config, 1, "bc_never")
    classifier, _ = command_for(config, 1, "classifier")
    assert never[never.index("--bc_confidence_threshold") + 1] == "1.0"
    assert classifier[classifier.index("--bc_confidence_threshold") + 1] == "0.3"
    assert never[never.index("--out_dir") + 1] == os.path.join("out", "bc_never")
    assert config["classifier"]["confidence_threshold"] == 0.3


def test_variant_overrides_from_config():
    config = base_config()
    config["variants"] = ["dense"]
    config["variant_overrides"] = {"dense": {"classifier.confidence_threshold": 0.05, "scheduler.pass2.bc_effect": "np"}}
    step1, _ = command_for(config, 1, "dense")
    step3, _ = command_for(config, 3, "dense")
    assert step1[step1.index("--bc_confidence_threshold") + 1] == "0.05"
    assert step3[step3.index("--pass2_bc_effect") + 1] == "np"


def test_no_variants_uses_out_dir_directly():
    config = base_config()
    del config["variants"]
    plan = runner.plan(config)
    assert [(p["variant"], p["step"]) for p in plan] == [(None, 1), (None, 2), (None, 3)]
    assert plan[0]["command"][plan[0]["command"].index("--out_dir") + 1] == "out"


def test_plan_orders_steps_within_variants_and_honours_step_selection():
    config = base_config()
    assert [(p["variant"], p["step"]) for p in runner.plan(config)] == [
        ("bc_never", 1), ("bc_never", 2), ("bc_never", 3), ("classifier", 1), ("classifier", 2), ("classifier", 3),
    ]  # fmt: skip
    assert [(p["variant"], p["step"]) for p in runner.plan(config, steps=[3, 2])] == [
        ("bc_never", 2), ("bc_never", 3), ("classifier", 2), ("classifier", 3),
    ]  # fmt: skip
    config["steps"] = [1]
    assert [p["step"] for p in runner.plan(config)] == [1, 1]


def test_bc_only_mode_uses_benchmark_tokenizer_and_layout_overrides():
    config = base_config()
    del config["variants"]
    config["mode"] = "bc_only"
    config["io"] = {
        "input_bc_gt_jsonl": "gt.jsonl",
        "icc_base_dir": "icc",
        "out_dir": "out",
        "tokenized_jsonl": "{out_dir}/tokenized_result.jsonl",
        "output_root": "{out_dir}/inference_output",
    }
    step2, _ = command_for(config, 2)
    assert step2[:9] == [
        sys.executable, "-m", "neural_scheduler.pipeline.tokenize_icc",
        "--segment_cache_dir", os.path.join("out", "segment_cache"), "--icc_base_dir", "icc",
        "--output_path", "out/tokenized_result.jsonl",
    ]  # fmt: skip
    step3, _ = command_for(config, 3)
    assert step3[step3.index("--mode") + 1] == "bc_only"
    assert step3[step3.index("--output_root") + 1] == "out/inference_output"
    assert step3[step3.index("--tokenized_jsonl") + 1] == "out/tokenized_result.jsonl"


def test_per_step_environment():
    config = base_config()
    config["environment"] = {
        "step1": {"env_setup": "source /opt/tts/activate", "env": {"NEURAL_SCHEDULER_STUB_TORCHVISION_META": 1}},
        "step2": {"python": "/opt/py/bin/python", "env": {"GLM4VOICE_REPO": "/opt/GLM-4-Voice"}},
    }
    command, env = command_for(config, 1, "classifier")
    assert command[:2] == ["bash", "-c"]
    assert command[2].startswith("source /opt/tts/activate && exec python -m neural_scheduler.pipeline.synthesize --out_dir ")
    assert "'that'\"'\"'s right'" in command[2]  # arguments are shell-quoted
    assert env == {"NEURAL_SCHEDULER_STUB_TORCHVISION_META": "1"}
    command, env = command_for(config, 2, "classifier")
    assert command[0] == "/opt/py/bin/python" and env == {"GLM4VOICE_REPO": "/opt/GLM-4-Voice"}
    command, env = command_for(config, 3, "classifier")
    assert command[0] == sys.executable and env == {}


def test_set_overrides_parse_yaml_values():
    config = base_config()
    runner.apply_overrides(
        config,
        ["tts.seed=7", "scheduler.pass1.schedule_subturns=false", "classifier.vocab=[mhm, ok]", "io.out_dir=elsewhere", "tokenizer.max_turns=50"],
    )
    assert config["tts"]["seed"] == 7
    assert config["scheduler"]["pass1"]["schedule_subturns"] is False
    assert config["classifier"]["vocab"] == ["mhm", "ok"]
    assert config["tokenizer"] == {"max_turns": 50}
    step3, _ = command_for(config, 3)
    assert "--pass_1_schedule_subturns" not in step3
    with pytest.raises(runner.ConfigError):
        runner.apply_overrides(config, ["no_equals_sign"])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["scheduler"]["pass1"].update(serach_window=3), "serach_window"),
        (lambda c: c.update(mode="two_pass"), "mode"),
        (lambda c: c["io"].pop("out_dir"), "io.out_dir"),
        (lambda c: c["io"].update(input_dialogue_json="d.json"), "exactly one"),
        (lambda c: c["io"].pop("input_jsonl"), "exactly one"),
        (lambda c: c["scheduler"].pop("checkpoint"), "scheduler.checkpoint"),
        (lambda c: c.update(mode="bc_only"), "icc_base_dir"),
        (lambda c: c.update(steps=[1, 4]), "steps"),
    ],
)
def test_validation_errors(mutate, message):
    config = base_config()
    mutate(config)
    with pytest.raises(runner.ConfigError, match=message):
        runner.plan(config)


def test_validation_only_checks_requested_steps():
    config = base_config()
    del config["scheduler"]["checkpoint"]
    del config["io"]["input_jsonl"]
    config["steps"] = [2]
    assert [p["step"] for p in runner.plan(config)] == [2, 2]


def test_dry_run_prints_commands_and_runs_nothing(tmp_path, capsys, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base_config()))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("a step was executed"))
    status = runner.main(["--config", str(path), "--dry_run", "--steps", "1", "--set", "tts.seed=9", "io.out_dir=o"])
    out = capsys.readouterr().out
    assert status == 0
    assert out.count("neural_scheduler.pipeline.synthesize") == 2
    assert "=== step 1 [bc_never] ===" in out and "--seed 9" in out and "--out_dir o/classifier" in out


def test_run_stops_at_first_failure(tmp_path, monkeypatch):
    calls = []

    class Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, env=None):
        calls.append((command, env))
        return Result(3 if len(calls) == 2 else 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    config = base_config()
    config["environment"] = {"step2": {"env": {"GLM4VOICE_REPO": "/opt/GLM-4-Voice"}}}
    assert runner.run(config) == 3
    assert len(calls) == 2
    assert calls[1][1]["GLM4VOICE_REPO"] == "/opt/GLM-4-Voice"
    assert "GLM4VOICE_REPO" not in calls[0][1] or calls[0][1]["GLM4VOICE_REPO"] == os.environ.get("GLM4VOICE_REPO")


def test_config_error_exit_status(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"io": {"out_dir": "o"}}))
    assert runner.main(["--config", str(path), "--dry_run"]) == 2
    assert "Config error" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Every mapped flag exists in the step it is sent to; shipped configs are valid
# --------------------------------------------------------------------------- #


def _option_strings(parser):
    return {option for action in parser._actions for option in action.option_strings}


def _schedule_parser():
    schedule = pytest.importorskip("neural_scheduler.pipeline.schedule")
    return schedule.build_parser()


def test_mapped_flags_exist_in_step_parsers():
    step1 = _option_strings(synthesize.build_parser())
    assert {flag for flag, _ in runner.STEP1_ARGS.values()} <= step1
    for parser in (tokenize.build_parser(), tokenize_icc.build_parser()):
        assert {flag for flag, _ in runner.STEP2_ARGS.values()} <= _option_strings(parser)
    assert {flag for flag, _ in runner.STEP3_ARGS.values()} <= _option_strings(_schedule_parser())


def test_runner_covers_every_step_option():
    """A new option of steps 1-2 must be reachable from the config (or be listed here as runner-managed)."""
    managed = {"-h", "--help", "--out_dir", "--no_classifier", "--no-ensure_at_least_one_bc", "--no-naive_vad_trim"}
    step1 = _option_strings(synthesize.build_parser()) - managed
    assert step1 == {flag for flag, _ in runner.STEP1_ARGS.values()}
    managed = {"-h", "--help", "--cache_base_dir", "--segment_cache_dir", "--icc_base_dir", "--output_path"}
    for parser in (tokenize.build_parser(), tokenize_icc.build_parser()):
        assert _option_strings(parser) - managed == {flag for flag, _ in runner.STEP2_ARGS.values()}


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_shipped_configs_produce_parseable_commands(path):
    config = runner.load_config(path)
    plan = runner.plan(config)
    assert plan, path
    parsers = {
        "neural_scheduler.pipeline.synthesize": synthesize.build_parser(),
        "neural_scheduler.pipeline.tokenize": tokenize.build_parser(),
        "neural_scheduler.pipeline.tokenize_icc": tokenize_icc.build_parser(),
    }
    for item in plan:
        command = item["command"]
        assert command[1] == "-m"
        module, args = command[2], command[3:]
        parser = _schedule_parser() if module == "neural_scheduler.pipeline.schedule" else parsers[module]
        parser.parse_args(args)  # exits on an unknown flag, a bad choice or a missing required argument


def test_expected_configs_are_shipped():
    names = {os.path.basename(p) for p in CONFIGS}
    assert {
        "tts_turn_taking.yaml",
        "tts_with_backchannel.yaml",
        "voice_lines.yaml",
        "strong_overlap.yaml",
        "fdb_backchannel.yaml",
    } <= names


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_shipped_configs_reference_existing_examples_and_no_private_paths(path):
    text = Path(path).read_text()
    assert "/home/" not in text and "evanpan" not in text
    flat = runner.flatten(runner.load_config(path))
    for key, value in flat.items():
        if isinstance(value, str) and value.startswith("examples/"):
            assert (REPO / value).is_file(), f"{key}: {value}"


def test_examples_are_text_only_and_free_of_private_paths():
    files = [p for p in (REPO / "examples").rglob("*") if p.is_file()]
    assert files
    for path in files:
        assert path.suffix in {".json", ".jsonl", ".txt", ".md"}, path
        text = path.read_text()
        assert "/home/" not in text and "evanpan" not in text, path
