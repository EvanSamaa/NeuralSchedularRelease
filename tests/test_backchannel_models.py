"""Classifier architectures, checkpoint factory, metrics and the inference wrappers."""

import json
import math
import os
import types

import numpy as np
import pytest
import torch

from neural_scheduler.backchannel import metrics, predictor
from neural_scheduler.backchannel.models import (
    GPT2PaperFusionClassifier,
    JointBCClassifier,
    MMF2FModel,
    PaperFusionClassifier,
    factory,
)

from .test_backchannel_support import (  # noqa: F401  (tiny_models is a fixture)
    ORIG_CLASSIFIER,
    ORIG_STEP1_JOINT,
    ORIG_STEP1_TEXT,
    extract_definitions,
    import_original,
    tiny_models,
)

VOCAB_ROBERTA = 50265 + 7
VOCAB_GPT2 = 50257 + 8


def make_batch(vocab_size, seed=0):
    """Two examples with different text and audio lengths (right-padded)."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(5, vocab_size, (2, 12), generator=g)
    attention_mask = torch.ones(2, 12, dtype=torch.long)
    attention_mask[1, 7:] = 0
    input_ids[1, 7:] = 1
    audio = torch.randn(2, 4000, generator=g) * 0.1
    audio[1, 1700:] = 0
    return dict(input_ids=input_ids, attention_mask=attention_mask, audio=audio,
                audio_lens=torch.tensor([4000, 1700]))


def model_specs(tiny):
    """(name, release class, original module, original class name, kwargs, vocab size)."""
    joint_mod = "train_classifier_3_class_joint_with_audio"
    gpt2_mod = "train_classifier_3_class_joint_with_audio_with_GPT2"
    rob = dict(roberta_name=tiny["roberta"], hubert_name=tiny["hubert"], tokenizer_len=VOCAB_ROBERTA)
    gpt = dict(gpt2_name=tiny["gpt2"], hubert_name=tiny["hubert"], tokenizer_len=VOCAB_GPT2)
    joint = dict(fusion_layers=1, fusion_heads=4, fusion_ffn=48)
    return [
        ("joint", JointBCClassifier, joint_mod, "JointBCClassifier", {**rob, **joint, "num_classes": 3}, VOCAB_ROBERTA),
        ("joint_text_only", JointBCClassifier, joint_mod, "JointBCClassifier",
         {**rob, **joint, "num_classes": 2, "text_only": True}, VOCAB_ROBERTA),
        ("joint_audio_only", JointBCClassifier, joint_mod, "JointBCClassifier",
         {**rob, **joint, "num_classes": 7, "audio_only": True, "audio_stride": 2}, VOCAB_ROBERTA),
        ("paper_opt1", PaperFusionClassifier, joint_mod, "PaperFusionClassifier",
         {**rob, "num_classes": 3, "fusion_option": "opt1", "audio_proj_dim": 8}, VOCAB_ROBERTA),
        ("paper_opt2_frozen", PaperFusionClassifier, joint_mod, "PaperFusionClassifier",
         {**rob, "num_classes": 3, "fusion_option": "opt2", "freeze_text": True}, VOCAB_ROBERTA),
        ("paper_text_only", PaperFusionClassifier, joint_mod, "PaperFusionClassifier",
         {**rob, "num_classes": 2, "text_only": True}, VOCAB_ROBERTA),
        ("paper_audio_only", PaperFusionClassifier, joint_mod, "PaperFusionClassifier",
         {**rob, "num_classes": 2, "audio_only": True}, VOCAB_ROBERTA),
        ("gpt2_paper_opt1", GPT2PaperFusionClassifier, gpt2_mod, "PaperFusionClassifier",
         {**gpt, "num_classes": 2, "fusion_option": "opt1"}, VOCAB_GPT2),
        ("gpt2_paper_frozen", GPT2PaperFusionClassifier, gpt2_mod, "PaperFusionClassifier",
         {**gpt, "num_classes": 3, "freeze_text": True, "audio_proj_dim": 8}, VOCAB_GPT2),
        ("mmf2f", MMF2FModel, joint_mod, "MMF2FModel",
         {**gpt, "num_classes": 2, "hidden_dim": 16, "lmf_rank": 4}, VOCAB_GPT2),
        ("mmf2f_text_only", MMF2FModel, joint_mod, "MMF2FModel",
         {**gpt, "num_classes": 3, "hidden_dim": 16, "lmf_rank": 4, "text_only": True}, VOCAB_GPT2),
        ("mmf2f_audio_only", MMF2FModel, joint_mod, "MMF2FModel",
         {**gpt, "num_classes": 3, "hidden_dim": 16, "lmf_rank": 4, "audio_only": True}, VOCAB_GPT2),
    ]


def test_frozen_parts_and_output_shape(tiny_models):
    model = JointBCClassifier(roberta_name=tiny_models["roberta"], hubert_name=tiny_models["hubert"],
                              fusion_layers=1, fusion_heads=4, fusion_ffn=48, tokenizer_len=VOCAB_ROBERTA)
    model.train()
    assert not model.hubert.training and model.roberta.training
    assert not any(p.requires_grad for p in model.hubert.parameters())
    assert model(**make_batch(VOCAB_ROBERTA)).shape == (2, 3)

    frozen = PaperFusionClassifier(roberta_name=tiny_models["roberta"], hubert_name=tiny_models["hubert"],
                                   freeze_text=True, tokenizer_len=VOCAB_ROBERTA)
    frozen.train()
    assert not frozen.roberta.training
    assert [n for n, p in frozen.named_parameters() if p.requires_grad] == [
        "audio_proj.weight", "audio_proj.bias", "fusion_classifier.weight", "fusion_classifier.bias"]
    assert next(iter(GPT2PaperFusionClassifier(
        gpt2_name=tiny_models["gpt2"], hubert_name=tiny_models["hubert"]).state_dict())).startswith("gpt2.")


def test_architecture_detection():
    assert factory.architecture_of({"roberta_name": "r", "use_paper_fusion": False, "use_mmf2f": False}) == "joint"
    assert factory.architecture_of({"roberta_name": "r", "use_paper_fusion": True}) == "paper_fusion"
    assert factory.architecture_of({"roberta_name": "r", "use_mmf2f": True, "gpt2_name": "g"}) == "mmf2f"
    assert factory.architecture_of({"gpt2_name": "g", "paper_fusion_option": "opt2"}) == "paper_fusion_gpt2"
    assert factory.architecture_of({"architecture": "mmf2f", "roberta_name": "r"}) == "mmf2f"
    assert factory.architecture_of({}) == "joint"


def test_decision_rule():
    probs = {"NoBackchannel": 0.5, "Acknowledgment": 0.2, "Reactive": 0.3}
    assert predictor.decide(probs, 0.0) == ("Reactive", 0.3)
    assert predictor.decide(probs, 0.5) == ("Reactive", 0.3)
    assert predictor.decide(probs, 0.51) == ("NoBackchannel", 0.5)
    tie = {"NoBackchannel": 0.2, "Acknowledgment": 0.4, "Reactive": 0.4}
    assert predictor.decide(tie, 0.8) == ("Acknowledgment", 0.4)
    assert predictor.decide(probs, 1.0) == ("NoBackchannel", 0.5)


def test_build_input_text():
    segments = [
        {"speaker": 1, "text": "hello\nthere", "bc_class": None},
        {"speaker": 2, "text": "hi", "bc_class": "Neutral"},
        {"speaker": 1, "text": "how  are you", "bc_class": "NoBackchannel"},
        {"speaker": 2, "text": "fine", "bc_class": "Emotional"},
        {"speaker": 2, "text": "and you"},
    ]
    assert predictor.build_input_text(segments, 2) == \
        "A: hello there </s> B: hi [BC:Neutral] </s> A: how are you <bc_slot> </s> B: fine </s> B: and you"
    assert predictor.build_input_text(segments, 3, context_window_turns=1, context_window_future_turns=0) == \
        "B: how are you </s> A: fine <bc_slot>"
    assert "[BC:" not in predictor.build_input_text(segments, 2, include_bc_context=False)


def test_predictor_does_not_need_training_dependencies():
    import subprocess
    import sys

    code = ("import sys; import neural_scheduler.backchannel.predictor; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('sklearn', 'tensorboard') "
            "or m.startswith('torch.utils.tensorboard')]; print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# --- equivalence with the research code --------------------------------------

@pytest.fixture(scope="module")
def orig_modules(ntpp_orig):
    return {name: import_original(ntpp_orig, ORIG_CLASSIFIER, name) for name in (
        "train_classifier_3_class_joint_with_audio",
        "train_classifier_3_class_joint_with_audio_with_GPT2",
    )}


@pytest.mark.equivalence
def test_models_match_original(orig_modules, tiny_models):
    for name, cls, orig_mod, orig_cls, kwargs, vocab in model_specs(tiny_models):
        torch.manual_seed(0)
        orig = getattr(orig_modules[orig_mod], orig_cls)(**kwargs)
        torch.manual_seed(0)
        ours = cls(**kwargs)

        # same parameter names, order and (seeded) initialisation
        sd_orig, sd_ours = orig.state_dict(), ours.state_dict()
        assert list(sd_orig) == list(sd_ours), name
        for key in sd_orig:
            assert torch.equal(sd_orig[key], sd_ours[key]), (name, key)
        assert [n for n, p in orig.named_parameters() if p.requires_grad] == \
            [n for n, p in ours.named_parameters() if p.requires_grad], name

        # weights from the original load strictly and give identical logits
        torch.manual_seed(1)
        donor = getattr(orig_modules[orig_mod], orig_cls)(**kwargs)
        ours.load_state_dict(donor.state_dict(), strict=True)
        batch = make_batch(vocab)
        donor.eval(), ours.eval()
        with torch.no_grad():
            assert torch.equal(donor(**batch), ours(**batch)), name

        # training mode: same dropout stream, same gradients
        donor.train(), ours.train()
        for model in (donor, ours):
            torch.manual_seed(2)
            model.zero_grad()
            model(**batch).sum().backward()
        for (n1, p1), (n2, p2) in zip(donor.named_parameters(), ours.named_parameters()):
            assert (p1.grad is None) == (p2.grad is None), (name, n1)
            if p1.grad is not None:
                assert torch.equal(p1.grad, p2.grad), (name, n1)


@pytest.mark.equivalence
def test_warm_start_helpers_match_original(orig_modules, tiny_models, tmp_path):
    joint_mod = orig_modules["train_classifier_3_class_joint_with_audio"]
    gpt2_mod = orig_modules["train_classifier_3_class_joint_with_audio_with_GPT2"]
    rob = dict(roberta_name=tiny_models["roberta"], hubert_name=tiny_models["hubert"], tokenizer_len=VOCAB_ROBERTA)
    gpt = dict(gpt2_name=tiny_models["gpt2"], hubert_name=tiny_models["hubert"], tokenizer_len=VOCAB_GPT2)

    for ours_cls, orig_cls, kwargs in (
        (PaperFusionClassifier, joint_mod.PaperFusionClassifier, rob),
        (GPT2PaperFusionClassifier, gpt2_mod.PaperFusionClassifier, gpt),
    ):
        torch.manual_seed(3)
        stage1 = orig_cls(**kwargs, text_only=True)
        path = str(tmp_path / "stage1.pt")
        torch.save(stage1.state_dict(), path)
        torch.manual_seed(4)
        a = orig_cls(**kwargs, freeze_text=True)
        torch.manual_seed(4)
        b = ours_cls(**kwargs, freeze_text=True)
        assert a.load_text_encoder_from_checkpoint(path) == b.load_text_encoder_from_checkpoint(path)
        for key, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[key]), key

    mm_kwargs = dict(**gpt, hidden_dim=16, lmf_rank=4)
    torch.manual_seed(5)
    a = joint_mod.MMF2FModel(**mm_kwargs)
    torch.manual_seed(5)
    b = MMF2FModel(**mm_kwargs)
    torch.manual_seed(6)
    donor = joint_mod.MMF2FModel(**mm_kwargs).state_dict()
    text_sd = {"transformer." + k[len("gpt2."):]: v for k, v in donor.items() if k.startswith("gpt2.")}
    text_sd.update({"proj.weight": donor["text_proj.weight"], "proj.bias": donor["text_proj.bias"],
                    "other.weight": torch.zeros(1)})
    audio_sd = {k: v for k, v in donor.items() if k.startswith("hubert.")}
    audio_sd.update({"proj.weight": donor["audio_proj.weight"], "proj.bias": torch.zeros(3)})
    text_path, audio_path = str(tmp_path / "t.pt"), str(tmp_path / "a.pt")
    torch.save(text_sd, text_path), torch.save(audio_sd, audio_path)
    assert a.load_unimodal_checkpoints(text_path, audio_path) == b.load_unimodal_checkpoints(text_path, audio_path)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def _nan_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_nan_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


class _FixedLogits(torch.nn.Module):
    """Returns precomputed logits batch by batch."""

    def __init__(self, logits):
        super().__init__()
        self.logits, self.cursor = logits, 0

    def forward(self, input_ids, attention_mask, audio, audio_lens):
        out = self.logits[self.cursor:self.cursor + input_ids.size(0)]
        self.cursor += input_ids.size(0)
        return out


def _fake_loader(labels, batch_size=7):
    batches = []
    for i in range(0, len(labels), batch_size):
        chunk = labels[i:i + batch_size]
        n = len(chunk)
        batches.append({"input_ids": torch.zeros(n, 2, dtype=torch.long),
                        "attention_mask": torch.ones(n, 2, dtype=torch.long),
                        "audio": torch.zeros(n, 4), "audio_lens": torch.full((n,), 4),
                        "label": chunk})
    return batches


@pytest.mark.equivalence
def test_metrics_match_original(ntpp_orig, orig_modules):
    orig = orig_modules["train_classifier_3_class_joint_with_audio"]
    orig_ablation = import_original(ntpp_orig, ORIG_CLASSIFIER, "ablation")
    rng = np.random.default_rng(0)

    for _ in range(20):
        y = rng.integers(0, 2, 50)
        score = rng.random(50)
        assert _nan_equal(metrics.compute_eer(y, score), orig.compute_eer(y, score))
    assert _nan_equal(metrics.compute_eer(np.ones(5, dtype=int), rng.random(5)),
                      orig.compute_eer(np.ones(5, dtype=int), rng.random(5)))

    cases = [(3, [0, 1, 2]), (3, [0, 1]), (2, [0, 1]), (7, list(range(7))), (3, [0])]
    for num_classes, present in cases:
        names = ["NoBackchannel", "Acknowledgment", "Reactive", "d", "e", "f", "g"][:num_classes]
        labels = torch.tensor(rng.choice(present, 60))
        logits = torch.tensor(rng.standard_normal((60, num_classes)), dtype=torch.float32)
        logits[torch.arange(60), labels] += 1.0

        ours = metrics.evaluate(_FixedLogits(logits), _fake_loader(labels), torch.device("cpu"), num_classes, names)
        theirs = orig.evaluate(_FixedLogits(logits), _fake_loader(labels), torch.device("cpu"), num_classes, names)
        assert _nan_equal(ours, theirs) and list(ours) == list(theirs)
        assert metrics.format_eval_table(ours, names) == orig.format_eval_table(theirs, names)

        ours_full = metrics.evaluate_full(
            _FixedLogits(logits), _fake_loader(labels), torch.device("cpu"), num_classes, names)
        theirs_full = orig_ablation.evaluate_full(
            _FixedLogits(logits), _fake_loader(labels), torch.device("cpu"), num_classes, names)
        assert _nan_equal(ours_full, theirs_full) and list(ours_full) == list(theirs_full)


def _stub_wrapper(cls, logits, threshold):
    """An inference wrapper with a stub tokenizer and model that returns ``logits``."""
    wrapper = object.__new__(cls)
    wrapper.device = torch.device("cpu")
    wrapper.confidence_threshold = threshold
    wrapper.label_names = {0: "NoBackchannel", 1: "Acknowledgment", 2: "Reactive"}
    wrapper.num_classes = 3
    wrapper.max_audio_seconds = 15.0
    wrapper.max_text_length = 512
    wrapper._min_audio_samples = 0
    wrapper.tokenizer = lambda text, **kw: {"input_ids": torch.zeros(1, 3, dtype=torch.long),
                                            "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    result = torch.tensor([logits], dtype=torch.float32)
    wrapper.model = lambda **kw: result if "audio" in kw else types.SimpleNamespace(logits=result)
    return wrapper


@pytest.mark.equivalence
def test_inference_wrappers_match_original(ntpp_orig):
    import soundfile as sf
    from typing import Dict, List, Optional, Tuple, Union

    base_ns = {"torch": torch, "np": np, "sf": sf, "os": os, "json": json, "Optional": Optional,
               "Union": Union, "List": List, "Dict": Dict, "Tuple": Tuple, "BC_SLOT_TOKEN": "<bc_slot>",
               "HUBERT_SR": 16000, "MAX_AUDIO_SECONDS_DEFAULT": 15.0}
    joint_ns = extract_definitions(
        ntpp_orig / ORIG_STEP1_JOINT,
        ["_load_audio_span", "_waveform_to_target_sr", "BackchannelClassifier"], dict(base_ns))
    text_ns = extract_definitions(ntpp_orig / ORIG_STEP1_TEXT, ["BackchannelClassifier"], dict(base_ns))
    pairs = [(predictor.BackchannelClassifier, joint_ns["BackchannelClassifier"]),
             (predictor.TextOnlyBackchannelClassifier, text_ns["BackchannelClassifier"])]

    rng = np.random.default_rng(1)
    segments = [{"speaker": int(rng.integers(1, 3)), "text": f"line {i}\n more  text",
                 "bc_class": rng.choice([None, "Neutral", "Emotional", "NoBackchannel"])} for i in range(12)]
    for ours_cls, orig_cls in pairs:
        ours, theirs = _stub_wrapper(ours_cls, [0, 0, 0], 0.0), _stub_wrapper(orig_cls, [0, 0, 0], 0.0)
        for idx in range(len(segments)):
            for past, future, with_bc in ((5, 5, True), (10, 5, True), (0, 0, True), (3, 20, False)):
                assert ours.build_input_text(segments, idx, past, future, with_bc) == \
                    theirs.build_input_text(segments, idx, past, future, with_bc)

        for _ in range(200):
            logits = (rng.standard_normal(3) * 2).tolist()
            threshold = float(rng.choice([0.0, 0.2, 0.38, 0.5, 0.9, 1.0]))
            assert _stub_wrapper(ours_cls, logits, threshold).predict("x") == \
                _stub_wrapper(orig_cls, logits, threshold).predict("x")
        for logits in ([0.0, 1.0, 1.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]):
            for threshold in (0.0, 2 / 3, 1.0):
                assert _stub_wrapper(ours_cls, logits, threshold).predict("x") == \
                    _stub_wrapper(orig_cls, logits, threshold).predict("x")

    assert predictor.MERGED_TO_CANONICAL_BC_TAG == {
        "Acknowledgment": "Neutral", "Reactive": "Emotional", "NoBackchannel": None}
    assert predictor.LABEL_NAMES == joint_ns["BackchannelClassifier"].LABEL_NAMES

    # audio resolution: None, in-memory tensor / array
    ours = _stub_wrapper(predictor.BackchannelClassifier, [0, 0, 0], 0.0)
    theirs = _stub_wrapper(joint_ns["BackchannelClassifier"], [0, 0, 0], 0.0)
    wav = torch.randn(1, 24000)
    for source, sr in ((None, None), (wav, 24000), (wav.numpy()[0], 8000)):
        assert np.array_equal(ours.load_audio_for_target(source, sample_rate=sr),
                              theirs.load_audio_for_target(source, sample_rate=sr))
    with pytest.raises(ValueError):
        ours.load_audio_for_target(wav)
