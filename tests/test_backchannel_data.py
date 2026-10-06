"""Classification data: prep, audio loading, datasets, weights and samplers."""

import copy
import json
import os
from collections import Counter

import numpy as np
import pytest
import torch

from neural_scheduler.backchannel.constants import BC_SLOT_TOKEN
from neural_scheduler.backchannel.data import audio, dataset, prep

from .test_backchannel_support import (  # noqa: F401  (tiny_models is a fixture)
    ORIG_CLASSIFIER,
    ORIG_STEP1_JOINT,
    extract_definitions,
    import_original,
    make_interactions,
    tiny_models,
    write_classification_data,
)

SCHEMES = {
    "semantic": "prep_classification_data",
    "duration": "prep_classification_data_duration_based",
}


def _interaction(turns):
    return {"interaction_id": 0, "wav_left": "L.wav", "wav_right": "R.wav", "split": "train",
            "dataset_type": "naturalistic", "sr": 16000, "turns": turns}


def _seg(text, start, end, speaker, turn, label="Speech"):
    return {"text": text, "start": start, "end": end, "speaker_id": speaker, "turn_id": turn, "label": label}


def test_input_text_format_and_audio_span():
    interaction = _interaction([
        _seg("my skin is sensitive", 0.0, 2.0, 0, 0),
        _seg("mm", 1.0, 1.2, 1, 0, "backchannel-Neutral"),
        _seg("so I get red", 2.5, 4.0, 1, 1),
        _seg("blotches", 4.2, 5.0, 1, 1),
        _seg("oh no", 3.0, 3.5, 0, 1, "backchannel-Emotional"),
        _seg("wait what", 4.4, 4.9, 0, 1),
        _seg("it hurts", 6.0, 7.0, 0, 2),
        _seg("far away", 60.0, 61.0, 0, 3),
    ])
    pos, neg, stats = prep.generate_examples([interaction])
    by_turn = {e["turn_id"]: e for e in pos + neg}

    target = by_turn[1]
    # main speaker of the target (speaker 1) is A; the hidden backchannel is not in the text
    assert target["input_text"] == (
        f"B: my skin is sensitive [BC:Neutral] </s> A: so I get red blotches {BC_SLOT_TOKEN}"
        " </s> B: wait what </s> B: it hurts"
    )
    assert target["label"] == "Emotional" and target["label_id"] == 3
    assert (target["bc_start"], target["bc_end"]) == (3.0, 3.5)
    assert target["audio_span"] == {"wav": "R.wav", "start": 2.5, "end": 5.0}
    assert by_turn[2]["label"] == "NoBackchannel" and by_turn[2]["audio_span"]["wav"] == "L.wav"
    assert BC_SLOT_TOKEN in by_turn[3]["input_text"] and "skin" not in by_turn[3]["input_text"]
    assert stats["total_positive"] == 2 and stats["total_negative"] == 2


def test_duration_scheme_and_exclusions():
    interaction = _interaction([
        _seg("a long enough turn", 0.0, 2.0, 0, 0),
        _seg("mm", 1.0, 1.25, 1, 0, "backchannel-Emotional"),
        _seg("another long turn", 3.0, 5.0, 0, 1),
        _seg("oh wow really", 3.5, 4.0, 1, 1, "backchannel-Neutral"),
        _seg("two of them", 6.0, 8.0, 0, 2),
        _seg("mm", 6.5, 6.6, 1, 2, "backchannel-Neutral"),
        _seg("ok", 7.0, 7.1, 1, 2, "backchannel-Positive"),
        _seg("hi", 9.0, 9.2, 0, 3),
    ])
    pos, neg, stats = prep.generate_examples([interaction], label_scheme="duration")
    assert [(e["turn_id"], e["label"], e["bc_duration"]) for e in pos] == [(0, "Neutral", 0.25), (1, "Emotional", 0.5)]
    assert stats["excluded_multi_bc"] == 1 and stats["excluded_short"] == 1 and neg == []
    # context tags keep the labeller's class under the duration scheme
    assert "[BC:Emotional]" in pos[1]["input_text"]
    pos_sem, _, _ = prep.generate_examples([interaction])
    assert [e["label"] for e in pos_sem] == ["Emotional", "Neutral"] and "bc_duration" not in pos_sem[0]
    with pytest.raises(ValueError):
        prep.generate_examples([interaction], label_scheme="other")


def test_class_weights_and_samplers():
    counts = Counter({0: 800, 1: 150, 2: 50})
    w = dataset.compute_class_weights(counts, 3, "sqrt_inv")
    assert torch.isclose(w.mean(), torch.tensor(1.0)) and w[2] > w[1] > w[0]
    assert torch.equal(dataset.compute_class_weights(counts, 3, "uniform"), torch.ones(3))
    labels = [0] * 8 + [1] * 2
    sampler = dataset.build_balanced_sampler(labels, 2, mode="uniform")
    assert sampler.num_samples == 10
    assert torch.allclose(sampler.weights, torch.tensor([1 / 8] * 8 + [1 / 2] * 2, dtype=torch.double))


def test_audio_span_loading(tmp_path):
    import soundfile as sf

    sr = 8000
    path = str(tmp_path / "a.wav")
    sf.write(path, np.stack([np.linspace(-1, 1, sr * 3), np.zeros(sr * 3)], axis=1).astype(np.float32), sr)
    out = audio.load_audio_span(path, 1.0, 2.0)
    assert out.dtype == np.float32 and len(out) == 16000
    assert len(audio.load_audio_span(path)) == 48000
    assert len(audio.load_audio_span(path, max_seconds=0.5)) == 8000
    assert audio.load_audio_span(path, 2.0, 2.0).tolist() == [0.0]
    wav = torch.randn(2, 2400)
    assert len(audio.waveform_to_target_sr(wav, 24000)) == 1600


# --- equivalence with the research code --------------------------------------

@pytest.mark.equivalence
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_prep_matches_original(ntpp_orig, tmp_path, scheme):
    orig = import_original(ntpp_orig, ORIG_CLASSIFIER, SCHEMES[scheme])
    interactions = make_interactions(seed=11, n=30)

    for interaction in interactions[:5]:
        ours_tg = prep.extract_turn_groups(copy.deepcopy(interaction), label_scheme=scheme)
        theirs_tg = orig.extract_turn_groups(copy.deepcopy(interaction))
        assert ours_tg == theirs_tg
        for a, b in zip(ours_tg, theirs_tg):
            for window in (3.0, 10.0):
                assert prep.build_input_text(a, ours_tg, window) == orig.build_input_text(b, theirs_tg, window)
            assert prep.compute_audio_span(a, interaction) == orig.compute_audio_span(b, interaction)

    ours = prep.generate_examples(copy.deepcopy(interactions), 10.0, 0.5, label_scheme=scheme)
    theirs = orig.generate_examples(copy.deepcopy(interactions), 10.0, 0.5)
    assert ours == theirs
    assert ours[2]["excluded_multi_bc"] > 0 and ours[2]["excluded_short"] > 0 and len(ours[0]) > 50

    ours_dir, theirs_dir = str(tmp_path / "ours"), str(tmp_path / "theirs")
    prep.balance_and_write(ours[0], ours[1], ours_dir, neg_ratio=1.5, seed=42)
    orig.balance_and_write(theirs[0], theirs[1], theirs_dir, neg_ratio=1.5, seed=42)
    for split in ("train", "test"):
        with open(os.path.join(ours_dir, f"{split}.jsonl")) as fa, \
                open(os.path.join(theirs_dir, f"{split}.jsonl")) as fb:
            assert fa.read() == fb.read()
    summary_a = json.load(open(os.path.join(ours_dir, "data_summary.json")))
    summary_b = json.load(open(os.path.join(theirs_dir, "data_summary.json")))
    for split in summary_a:
        summary_a[split].pop("output_path"), summary_b[split].pop("output_path")
    assert summary_a == summary_b


@pytest.mark.equivalence
def test_prep_cli_matches_original_files(ntpp_orig, tmp_path, monkeypatch):
    """Whole command line, duration scheme, default arguments."""
    orig = import_original(ntpp_orig, ORIG_CLASSIFIER, SCHEMES["duration"])
    in_dir = tmp_path / "in" / "train_naturalistic"
    in_dir.mkdir(parents=True)
    with open(in_dir / "shard0of1_interactions.jsonl", "w") as f:
        for interaction in make_interactions(seed=5, n=20):
            f.write(json.dumps(interaction) + "\n")

    prep.main(["--input_dir", str(tmp_path / "in"), "--output_dir", str(tmp_path / "ours"),
               "--label_scheme", "duration"])
    monkeypatch.setattr("sys.argv", ["prep", "--input_dir", str(tmp_path / "in"),
                                     "--output_dir", str(tmp_path / "theirs")])
    orig.main()
    for split in ("train", "test"):
        assert (tmp_path / "ours" / f"{split}.jsonl").read_text() == \
            (tmp_path / "theirs" / f"{split}.jsonl").read_text()


@pytest.fixture(scope="module")
def orig_train(ntpp_orig):
    return import_original(ntpp_orig, ORIG_CLASSIFIER, "train_classifier_3_class_joint_with_audio")


@pytest.mark.equivalence
def test_weights_and_sampler_match_original(ntpp_orig, orig_train):
    orig_gpt2 = import_original(ntpp_orig, ORIG_CLASSIFIER, "train_classifier_3_class_joint_with_audio_with_GPT2")
    for counts in (Counter({0: 800, 1: 150, 2: 50}), Counter({0: 3, 2: 9}), Counter({1: 5})):
        for method in ("sqrt_inv", "inv", "uniform"):
            assert torch.equal(dataset.compute_class_weights(counts, 3, method),
                               orig_train.compute_class_weights(counts, 3, method))
    labels = [0, 0, 1, 2, 0, 0, 1, 0, 0, 0]
    for mode in ("uniform", "sqrt"):
        a = dataset.build_balanced_sampler(labels, 3, mode)
        b = orig_gpt2.build_balanced_sampler(labels, 3, mode)
        assert torch.equal(a.weights, b.weights) and a.num_samples == b.num_samples


@pytest.mark.equivalence
def test_audio_loading_matches_original(ntpp_orig, orig_train, tmp_path):
    import soundfile as sf

    ns = extract_definitions(
        ntpp_orig / ORIG_STEP1_JOINT, ["_load_audio_span", "_waveform_to_target_sr"],
        {"np": np, "sf": sf, "torch": torch, "Optional": __import__("typing").Optional,
         "HUBERT_SR": 16000, "MAX_AUDIO_SECONDS_DEFAULT": 15.0},
    )
    rng = np.random.default_rng(0)
    files = {}
    for sr, channels, seconds in ((16000, 1, 4.0), (8000, 2, 3.0), (24000, 1, 20.0), (44100, 2, 1.0)):
        path = str(tmp_path / f"{sr}_{channels}.wav")
        sf.write(path, rng.standard_normal((int(sr * seconds), channels)).astype(np.float32).squeeze() * 0.1, sr)
        files[path] = seconds
    for path, seconds in files.items():
        for start, end in ((0.0, seconds), (0.37, 1.91), (0.5, 0.5), (0.9, 0.1), (0.0, 100.0), (0.2, 0.2001)):
            for max_seconds in (15.0, 0.7):
                ours = audio.load_audio_span(path, start, end, 16000, max_seconds)
                assert np.array_equal(ours, orig_train.load_audio_span(path, start, end, 16000, max_seconds))
                assert np.array_equal(ours, ns["_load_audio_span"](path, start, end, 16000, max_seconds))
        assert np.array_equal(audio.load_audio_span(path), ns["_load_audio_span"](path))
        assert np.array_equal(audio.load_audio_span(path, None, 1.0), ns["_load_audio_span"](path, None, 1.0))
    for shape, sr in (((2, 30000), 24000), ((16000,), 16000), ((1, 3), 48000), ((500000,), 22050)):
        wav = torch.randn(*shape)
        assert np.array_equal(audio.waveform_to_target_sr(wav, sr), ns["_waveform_to_target_sr"](wav, sr))


@pytest.mark.equivalence
def test_dataset_and_collate_match_original(orig_train, tiny_models, tmp_path):
    from transformers import RobertaTokenizer

    files = write_classification_data(str(tmp_path))
    tokenizer = RobertaTokenizer.from_pretrained(tiny_models["roberta"])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [orig_train.BC_SLOT_TOKEN] + orig_train.BC_TAG_TOKENS})

    for phase, exclude in (("merged", False), ("binary", True), ("binary", False), ("full", False)):
        ours = dataset.BackchannelJointDataset(files["train"], tokenizer, 64, phase, 1.0, exclude)
        theirs = orig_train.BackchannelJointDataset(files["train"], tokenizer, 64, phase, 1.0, exclude)
        assert len(ours) == len(theirs) > 10
        assert ours.get_label_counts() == theirs.get_label_counts()
        batch_a = dataset.make_collate_fn(tokenizer.pad_token_id)([ours[i] for i in range(6)])
        batch_b = orig_train.make_collate_fn(tokenizer.pad_token_id)([theirs[i] for i in range(6)])
        assert batch_a.keys() == batch_b.keys()
        for key in batch_a:
            assert torch.equal(batch_a[key], batch_b[key]), key
        if phase == "merged":
            a = dataset.make_balanced_sampler(ours)
            b = orig_train.make_balanced_sampler(theirs)
            assert torch.equal(a.weights, b.weights) and a.num_samples == b.num_samples


def test_relative_audio_paths(tiny_models, tmp_path):
    from transformers import RobertaTokenizer

    files = write_classification_data(str(tmp_path))
    wav_dir = os.path.join(str(tmp_path), "wavs")
    relative = str(tmp_path / "relative.jsonl")
    with open(files["train"]) as f_in, open(relative, "w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            ex["audio_span"]["wav"] = os.path.relpath(ex["audio_span"]["wav"], wav_dir)
            f_out.write(json.dumps(ex) + "\n")
    tokenizer = RobertaTokenizer.from_pretrained(tiny_models["roberta"])
    absolute = dataset.BackchannelJointDataset(files["train"], tokenizer, 64)
    rel = dataset.BackchannelJointDataset(relative, tokenizer, 64, audio_root=wav_dir)
    assert torch.equal(absolute[0]["audio"], rel[0]["audio"])
