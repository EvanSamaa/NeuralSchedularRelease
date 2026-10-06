"""End-to-end runs of the classifier command lines on tiny models.

The equivalence tests run the original training scripts and the release entry
points with the same seed on the same data and require identical weights and
metrics.
"""

import json
import math
import os

import pytest
import torch

from neural_scheduler.backchannel import ablation, eval as bc_eval
from neural_scheduler.backchannel import predictor, train, train_gpt2_baseline, train_text_only
from neural_scheduler.backchannel.data.audio import load_audio_span

from .test_backchannel_support import (  # noqa: F401  (tiny_models is a fixture)
    ORIG_CLASSIFIER,
    import_original,
    tiny_models,
    write_classification_data,
)

COMMON = ["--epochs", "2", "--batch_size", "4", "--num_workers", "0", "--max_length", "48",
          "--max_audio_seconds", "1.0", "--lr", "1e-3", "--log_interval", "1", "--seed", "7"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return write_classification_data(str(tmp_path_factory.mktemp("bc_data")), seed=3, n_interactions=8)


def _nan_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_nan_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _assert_same_run(ours_dir, theirs_dir):
    """Best and last checkpoints, their configs and the best metrics agree."""
    for ckpt in ("best_model", "last_model"):
        ours = torch.load(os.path.join(ours_dir, ckpt, "model.pt"), map_location="cpu")
        theirs = torch.load(os.path.join(theirs_dir, ckpt, "model.pt"), map_location="cpu")
        assert list(ours) == list(theirs)
        for key in ours:
            assert torch.equal(ours[key], theirs[key]), (ckpt, key)
        cfg = _load_json(os.path.join(ours_dir, ckpt, "config.json"))
        cfg.pop("architecture")
        assert cfg == _load_json(os.path.join(theirs_dir, ckpt, "config.json"))
        assert os.path.isdir(os.path.join(ours_dir, ckpt, "tokenizer"))
    assert _nan_equal(_load_json(os.path.join(ours_dir, "best_model", "best_metrics.json")),
                      _load_json(os.path.join(theirs_dir, "best_model", "best_metrics.json")))
    assert sorted(os.listdir(os.path.join(ours_dir, "tokenizer"))) == \
        sorted(os.listdir(os.path.join(theirs_dir, "tokenizer")))


def _run_original(module, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"] + argv)
    module.main()


def _joint_args(tiny_models, data, out_dir):
    return ["--train_file", data["train"], "--val_file", data["test"], "--output_dir", out_dir,
            "--roberta_name", tiny_models["roberta"], "--hubert_name", tiny_models["hubert"],
            "--fusion_layers", "1", "--fusion_heads", "4", "--fusion_ffn", "48"] + COMMON


def test_train_eval_ablation_predict(tiny_models, data, tmp_path):
    """The whole release workflow on CPU: train, evaluate, ablate, predict."""
    run = str(tmp_path / "run")
    train.main(_joint_args(tiny_models, data, run))
    best = os.path.join(run, "best_model")
    cfg = _load_json(os.path.join(best, "config.json"))
    assert cfg["architecture"] == "joint" and cfg["phase"] == "merged" and cfg["num_classes"] == 3
    assert sorted(os.listdir(best)) == ["best_metrics.json", "config.json", "model.pt", "tokenizer"]

    bc_eval.main(["--checkpoint", best, "--val_file", data["test"], "--output_dir", str(tmp_path / "eval"),
                  "--num_workers", "0", "--batch_size", "4", "--max_length", "48", "--max_audio_seconds", "1.0",
                  "--strict"])
    eval_metrics = _load_json(str(tmp_path / "eval" / "eval_metrics.json"))
    best_metrics = _load_json(os.path.join(best, "best_metrics.json"))
    assert eval_metrics["confusion_matrix"] == best_metrics["confusion_matrix"]
    assert eval_metrics["label_names"] == ["NoBackchannel", "Acknowledgment", "Reactive"]

    out_json = str(tmp_path / "ablation.json")
    ablation.main(["--checkpoint", best, "--val_file", data["test"], "--output_json", out_json,
                   "--num_workers", "0", "--batch_size", "4", "--max_length", "48",
                   "--max_audio_seconds", "1.0", "--num_samples", "12"])
    result = _load_json(out_json)
    assert list(result["ablations"]) == ablation.DEFAULT_ABLATIONS
    assert result["num_samples_evaluated"] == 12

    # tokenizer found inside the checkpoint directory, and beside it
    for model_dir, tokenizer_dir in ((best, None), (os.path.join(best, "model.pt"), os.path.join(run, "tokenizer"))):
        clf = predictor.BackchannelClassifier(model_dir, tokenizer_dir=tokenizer_dir, device="cpu",
                                              confidence_threshold=0.5, max_audio_seconds=1.0)
        example = json.loads(open(data["test"]).readline())
        span = example["audio_span"]
        label, confidence, probs = clf.predict(example["input_text"], audio=span["wav"],
                                               audio_start_sec=span["start"], audio_end_sec=span["end"])
        assert set(probs) == {"NoBackchannel", "Acknowledgment", "Reactive"}
        assert abs(sum(probs.values()) - 1.0) < 1e-5 and confidence == probs[label]

        audio = load_audio_span(span["wav"], span["start"], span["end"], max_seconds=1.0)
        enc = clf.tokenizer(example["input_text"], max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = clf.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               audio=torch.from_numpy(audio).unsqueeze(0),
                               audio_lens=torch.tensor([len(audio)]))
        expected = torch.softmax(logits, -1)[0].tolist()
        assert [probs[n] for n in ("NoBackchannel", "Acknowledgment", "Reactive")] == pytest.approx(expected)
        # an in-memory waveform gives the same answer as the file
        assert clf.predict(example["input_text"], audio=audio, audio_sample_rate=16000)[2] == pytest.approx(probs)
    silent = clf.predict(example["input_text"])
    assert silent[0] in ("NoBackchannel", "Acknowledgment", "Reactive")


# --- equivalence with the research code --------------------------------------

@pytest.fixture(scope="module")
def orig_joint(ntpp_orig):
    return import_original(ntpp_orig, ORIG_CLASSIFIER, "train_classifier_3_class_joint_with_audio")


@pytest.mark.equivalence
def test_joint_training_matches_original(orig_joint, ntpp_orig, tiny_models, data, tmp_path, monkeypatch):
    ours_dir, theirs_dir = str(tmp_path / "ours"), str(tmp_path / "theirs")
    extra = ["--balance_nobc", "--class_weight_method", "inv"]
    train.main(_joint_args(tiny_models, data, ours_dir) + extra)
    _run_original(orig_joint, _joint_args(tiny_models, data, theirs_dir) + extra, monkeypatch)
    _assert_same_run(ours_dir, theirs_dir)

    # evaluation: release CLI on the original checkpoint layout vs the original --eval_only
    eval_args = ["--val_file", data["test"], "--num_workers", "0", "--batch_size", "4",
                 "--max_length", "48", "--max_audio_seconds", "1.0"]
    theirs_best = os.path.join(theirs_dir, "best_model")
    bc_eval.main(["--checkpoint", theirs_best, "--output_dir", str(tmp_path / "eval_ours")] + eval_args)
    _run_original(orig_joint, ["--eval_only", "--checkpoint", theirs_best,
                               "--output_dir", str(tmp_path / "eval_theirs")] + eval_args, monkeypatch)
    assert _nan_equal(_load_json(str(tmp_path / "eval_ours" / "eval_metrics.json")),
                      _load_json(str(tmp_path / "eval_theirs" / "eval_metrics.json")))

    # ablations on the original checkpoint
    orig_ablation = import_original(ntpp_orig, ORIG_CLASSIFIER, "ablation")
    ab_args = ["--checkpoint", theirs_best, "--val_file", data["test"], "--num_workers", "0",
               "--batch_size", "4", "--max_length", "48", "--max_audio_seconds", "1.0", "--num_samples", "14"]
    ablation.main(ab_args + ["--output_json", str(tmp_path / "ab_ours.json")])
    _run_original(orig_ablation, ab_args + ["--output_json", str(tmp_path / "ab_theirs.json")], monkeypatch)
    assert _nan_equal(_load_json(str(tmp_path / "ab_ours.json")), _load_json(str(tmp_path / "ab_theirs.json")))

    # the inference wrapper loads the original layout (tokenizer beside best_model)
    clf = predictor.BackchannelClassifier(theirs_best, device="cpu", max_audio_seconds=1.0)
    assert clf.architecture == "joint"


@pytest.mark.equivalence
def test_paper_fusion_two_stage_matches_original(orig_joint, tiny_models, data, tmp_path, monkeypatch):
    base = ["--use_paper_fusion", "--phase", "binary", "--exclude_repetition", "--paper_audio_proj_dim", "8"]
    dirs = {}
    for name, runner in (("ours", lambda argv: train.main(argv)),
                         ("theirs", lambda argv: _run_original(orig_joint, argv, monkeypatch))):
        stage1, stage2 = str(tmp_path / f"{name}_s1"), str(tmp_path / f"{name}_s2")
        runner(_joint_args(tiny_models, data, stage1) + base + ["--text_only"])
        runner(_joint_args(tiny_models, data, stage2) + base
               + ["--paper_fusion_option", "opt2", "--text_model_checkpoint", os.path.join(stage1, "best_model")])
        dirs[name] = (stage1, stage2)
    _assert_same_run(dirs["ours"][0], dirs["theirs"][0])
    _assert_same_run(dirs["ours"][1], dirs["theirs"][1])
    with pytest.raises(ValueError):
        train.main(_joint_args(tiny_models, data, str(tmp_path / "bad")) + base)


@pytest.mark.equivalence
def test_mmf2f_training_matches_original(orig_joint, tiny_models, data, tmp_path, monkeypatch):
    extra = ["--use_mmf2f", "--gpt2_name", tiny_models["gpt2"], "--mmf2f_hidden_dim", "16",
             "--mmf2f_rank", "4", "--mmf2f_drop_rate", "0.5", "--phase", "full"]
    ours_dir, theirs_dir = str(tmp_path / "ours"), str(tmp_path / "theirs")
    train.main(_joint_args(tiny_models, data, ours_dir) + extra)
    _run_original(orig_joint, _joint_args(tiny_models, data, theirs_dir) + extra, monkeypatch)
    _assert_same_run(ours_dir, theirs_dir)


@pytest.mark.equivalence
def test_gpt2_baseline_matches_original(ntpp_orig, tiny_models, data, tmp_path, monkeypatch):
    orig = import_original(ntpp_orig, ORIG_CLASSIFIER, "train_classifier_3_class_joint_with_audio_with_GPT2")
    for i, extra in enumerate((
        ["--binary", "--balanced_sampling", "--balanced_mode", "sqrt", "--paper_fusion_option", "opt1"],
        ["--text_only"],
    )):
        ours_dir, theirs_dir = str(tmp_path / f"ours{i}"), str(tmp_path / f"theirs{i}")
        argv = ["--train_file", data["train"], "--val_file", data["test"],
                "--gpt2_name", tiny_models["gpt2"], "--hubert_name", tiny_models["hubert"],
                "--paper_audio_proj_dim", "8"] + COMMON + extra
        train_gpt2_baseline.main(argv + ["--output_dir", ours_dir])
        _run_original(orig, argv + ["--output_dir", theirs_dir], monkeypatch)
        _assert_same_run(ours_dir, theirs_dir)

    # both entry points evaluate the original GPT-2 checkpoint (config without an architecture key)
    eval_args = ["--checkpoint", os.path.join(theirs_dir, "best_model"), "--val_file", data["test"],
                 "--num_workers", "0", "--batch_size", "4", "--max_length", "48", "--max_audio_seconds", "1.0"]
    _run_original(orig, ["--eval_only", "--output_dir", str(tmp_path / "e_theirs")] + eval_args, monkeypatch)
    expected = _load_json(str(tmp_path / "e_theirs" / "eval_metrics.json"))
    train_gpt2_baseline.main(["--eval_only", "--output_dir", str(tmp_path / "e_ours")] + eval_args)
    bc_eval.main(["--output_dir", str(tmp_path / "e_generic"), "--strict"] + eval_args)
    assert _nan_equal(_load_json(str(tmp_path / "e_ours" / "eval_metrics.json")), expected)
    assert _nan_equal(_load_json(str(tmp_path / "e_generic" / "eval_metrics.json")), expected)


@pytest.mark.equivalence
def test_text_only_training_matches_original(ntpp_orig, tiny_models, data, tmp_path, monkeypatch):
    from safetensors.torch import load_file

    orig = import_original(ntpp_orig, ORIG_CLASSIFIER, "train_classifier_3_class")
    dirs = {}
    for name, runner in (("ours", lambda argv: train_text_only.main(argv)),
                         ("theirs", lambda argv: _run_original(orig, argv, monkeypatch))):
        warmup, final = str(tmp_path / f"{name}_binary"), str(tmp_path / f"{name}_merged")
        argv = ["--train_file", data["train"], "--val_file", data["test"],
                "--model_name", tiny_models["roberta"], "--epochs", "2", "--batch_size", "4",
                "--num_workers", "0", "--max_length", "48", "--lr", "1e-3", "--seed", "7"]
        runner(argv + ["--output_dir", warmup, "--phase", "binary"])
        runner(argv + ["--output_dir", final, "--phase", "merged",
                       "--init_checkpoint", os.path.join(warmup, "best_model")])
        dirs[name] = (warmup, final)
    for ours_dir, theirs_dir in zip(dirs["ours"], dirs["theirs"]):
        for ckpt in ("best_model", "last_model"):
            ours = load_file(os.path.join(ours_dir, ckpt, "model.safetensors"))
            theirs = load_file(os.path.join(theirs_dir, ckpt, "model.safetensors"))
            assert ours.keys() == theirs.keys()
            for key in ours:
                assert torch.equal(ours[key], theirs[key]), (ckpt, key)
        assert _nan_equal(_load_json(os.path.join(ours_dir, "best_model", "best_metrics.json")),
                          _load_json(os.path.join(theirs_dir, "best_model", "best_metrics.json")))

    clf = predictor.TextOnlyBackchannelClassifier(os.path.join(dirs["theirs"][1], "best_model"), device="cpu")
    label, confidence, probs = clf.predict("A: hello <bc_slot>")
    assert confidence == probs[label] and len(probs) == 3
