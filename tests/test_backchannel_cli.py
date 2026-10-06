"""Dataset statistics, the shard diagnostic and the command-line entry points."""

import json
import subprocess
import sys

import pytest

from neural_scheduler.backchannel import stats
from neural_scheduler.backchannel.labeling import diagnose

from .test_backchannel_support import ORIG_CLASSIFIER, ORIG_PREPROCESS, import_original, make_interactions

CLI_MODULES = [
    "labeling.extract", "labeling.diagnose", "data.prep", "train", "train_gpt2_baseline",
    "train_text_only", "eval", "ablation", "stats",
]


def _write_shard(tmp_path, n=12):
    shard_dir = tmp_path / "labelled" / "train_naturalistic"
    shard_dir.mkdir(parents=True)
    path = shard_dir / "shard0of1_interactions.jsonl"
    with open(path, "w") as f:
        for interaction in make_interactions(seed=21, n=n):
            f.write(json.dumps(interaction) + "\n")
    (shard_dir / "shard0of1_failures.jsonl").write_text(
        json.dumps({"interaction_id": 99, "identifier_left": "a", "identifier_right": "b",
                    "error": "boom", "traceback": "Traceback ..."}) + "\n")
    return path


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_help(module):
    result = subprocess.run([sys.executable, "-m", f"neural_scheduler.backchannel.{module}", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_stats_cli(tmp_path, capsys):
    _write_shard(tmp_path)
    out = tmp_path / "stats.json"
    stats.main(["--input_dir", str(tmp_path / "labelled"), "--output", str(out)])
    result = json.loads(out.read_text())
    assert set(result) == {"bc_density", "class_distribution", "inter_bc_intervals", "bc_durations",
                           "bcs_per_turn", "bc_context", "interaction_level"}
    assert result["interaction_level"]["num_interactions"] == 12
    assert result["class_distribution"]["total_backchannels"] > 0
    assert "SUMMARY" in capsys.readouterr().out


def test_diagnose_cli(tmp_path, capsys):
    path = _write_shard(tmp_path)
    diagnose.main(["--interactions_path", str(path)])
    out = capsys.readouterr().out
    assert "DIAGNOSTICS COMPLETE" in out and "boom" in out


@pytest.mark.equivalence
def test_stats_match_original(ntpp_orig, tmp_path, monkeypatch, capsys):
    orig = import_original(ntpp_orig, ORIG_CLASSIFIER, "analyze_bc_stats")
    _write_shard(tmp_path)
    ours, theirs = tmp_path / "ours.json", tmp_path / "theirs.json"
    stats.main(["--input_dir", str(tmp_path / "labelled"), "--output", str(ours)])
    monkeypatch.setattr("sys.argv", ["stats", "--input_dir", str(tmp_path / "labelled"), "--output", str(theirs)])
    orig.main()
    assert ours.read_text() == theirs.read_text()


@pytest.mark.equivalence
def test_diagnose_matches_original(ntpp_orig, tmp_path):
    orig = import_original(ntpp_orig, ORIG_PREPROCESS, "extract_backchannel_dataset_sharded_v2_diagnostic")
    path = str(_write_shard(tmp_path))
    records, errors = diagnose.load_interactions(path)
    assert (records, errors) == orig.load_interactions(path)
    assert diagnose.check_integrity(records) == orig.check_integrity(records)
    assert diagnose.check_label_distribution(records) == orig.check_label_distribution(records)
    assert diagnose.check_failures(path.replace("_interactions", "_failures")) == \
        orig.check_failures(path.replace("_interactions", "_failures"))
    for name in ("check_turn_id_sanity", "check_bc_quality"):
        import random
        random.seed(0)
        a = getattr(diagnose, name)(records)
        random.seed(0)
        assert a == getattr(orig, name)(records)


def test_config_file_supplies_defaults(tmp_path):
    from neural_scheduler.backchannel import train, train_gpt2_baseline

    args = train.parse_args(["--config", "configs/backchannel/joint_merged.yaml",
                             "--val_file", "v.jsonl", "--output_dir", "out", "--patience", "2"])
    assert (args.lr, args.batch_size, args.patience, args.phase) == (2e-4, 32, 2, "merged")
    assert args.fusion_ffn == 3072 and args.eval_only is False

    args = train_gpt2_baseline.parse_args(["--config", "configs/backchannel/gpt2_paper_fusion.yaml",
                                           "--val_file", "v.jsonl", "--output_dir", "out"])
    assert args.balanced_sampling is True and args.lr == 5e-5 and args.phase == "binary"

    config = tmp_path / "c.yaml"
    config.write_text("val_file: from_config.jsonl\noutput_dir: run\n")
    assert train.parse_args(["--config", str(config)]).val_file == "from_config.jsonl"
    config.write_text("not_an_argument: 1\n")
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(config), "--val_file", "v", "--output_dir", "o"])
