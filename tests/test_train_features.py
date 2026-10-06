"""Feature preparation, collation and parameter freezing."""

import ast
import copy
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from neural_scheduler.constants import PAD_TOKEN_ID
from neural_scheduler.data.collator import DataCollatorForSeq2SeqWithChannels
from neural_scheduler.data.features import prepare_features
from neural_scheduler.model import ChatGLMConfig, ChatGLMForConditionalGenerationPredDuration
from neural_scheduler.train.args import ModelArguments
from neural_scheduler.train.train import apply_freezing

ORIG_TRAINER = "ntpp/train/sft_glm_4_voice_pred_timing_only_with_duration_pred_stable.py"
ORIG_MODEL_DIR = "ntpp/model/chatGLM_dual_timing_only"


def records(with_end=True, with_durations=True):
    examples = {
        "input_ids": [list(range(100, 112)), list(range(200, 209))],
        "generation_start_index": [4, 3],
        "source_file": ["a", "b"],
    }
    if with_end:
        examples["generation_end_index"] = [11, 8]
    if with_durations:
        ends = examples.get("generation_end_index", [12, 9])
        examples["remaining_durations"] = [
            list(range(end - start)) for start, end in zip(examples["generation_start_index"], ends)
        ]
    return examples


def test_prepare_features():
    out = prepare_features(records())
    assert set(out) == {"input_ids", "labels", "attention_mask", "generation_start_index", "remaining_durations"}
    assert out["labels"][0] == [-100] * 4 + list(range(104, 112))
    assert out["attention_mask"][1] == [1] * 9
    assert out["remaining_durations"][0] == [-100] * 4 + list(range(7)) + [-100]
    assert out["remaining_durations"][1] == [-100] * 3 + list(range(5)) + [-100]
    assert "remaining_durations" not in prepare_features(records(with_durations=False))


def test_prepare_features_rejects_wrong_duration_length():
    bad = records()
    bad["remaining_durations"][0] = bad["remaining_durations"][0][:-1]
    with pytest.raises(ValueError):
        prepare_features(bad)


def test_collator_pads_right():
    features = prepare_features(records())
    rows = [{key: features[key][i] for key in features} for i in range(2)]
    collator = DataCollatorForSeq2SeqWithChannels(SimpleNamespace(pad_token_id=PAD_TOKEN_ID), pad_to_multiple_of=8)
    batch = collator(rows)
    assert batch["input_ids"].shape == (2, 16) and batch["input_ids"].dtype == torch.long
    assert batch["input_ids"][1, 9:].eq(PAD_TOKEN_ID).all()
    assert batch["attention_mask"][1].tolist() == [1] * 9 + [0] * 7
    assert batch["labels"][0, 12:].eq(-100).all() and batch["remaining_durations"][1, 9:].eq(-100).all()
    assert batch["generation_start_index"].tolist() == [4, 3]


@pytest.mark.equivalence
@pytest.mark.parametrize("with_end", [True, False])
@pytest.mark.parametrize("with_durations", [True, False])
def test_prepare_features_matches_original(ntpp_orig, with_end, with_durations):
    """The original function is taken from the trainer's source (importing the trainer has side effects)."""
    tree = ast.parse((ntpp_orig / ORIG_TRAINER).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_features")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), ORIG_TRAINER, "exec"), namespace)
    examples = records(with_end, with_durations)
    assert prepare_features(copy.deepcopy(examples)) == namespace["prepare_features"](copy.deepcopy(examples))


@pytest.mark.equivalence
@pytest.mark.parametrize("with_durations", [True, False])
def test_collator_matches_original(ntpp_orig, with_durations):
    model_dir = str(ntpp_orig / ORIG_MODEL_DIR)
    sys.path.insert(0, model_dir)
    try:
        original = importlib.import_module("custom_collator").DataCollatorForSeq2SeqWithChannels
    finally:
        sys.path.remove(model_dir)
    tokenizer = SimpleNamespace(pad_token_id=PAD_TOKEN_ID)
    features = prepare_features(records(with_durations=with_durations))
    rows = [{key: features[key][i] for key in features} for i in range(2)]
    expected = original(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)(copy.deepcopy(rows))
    actual = DataCollatorForSeq2SeqWithChannels(tokenizer, pad_to_multiple_of=8)(copy.deepcopy(rows))
    assert set(expected) == set(actual)
    for key in expected:
        assert torch.equal(expected[key], actual[key]) and expected[key].dtype == actual[key].dtype


def tiny_model():
    config = ChatGLMConfig(num_layers=4, padded_vocab_size=50, hidden_size=16, ffn_hidden_size=24, kv_channels=4,
                           num_attention_heads=4, torch_dtype="float32", use_duration_prediction=True)
    return ChatGLMForConditionalGenerationPredDuration(config, empty_init=False)


def trainable(model):
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_freezing_default_trains_everything():
    model = tiny_model()
    apply_freezing(model, ModelArguments())
    assert trainable(model) == {name for name, _ in model.named_parameters()}


def test_freezing_selected_layers():
    model = tiny_model()
    apply_freezing(model, ModelArguments(num_of_acoustic_layers_to_unfreeze=1, num_of_output_layers_to_unfreeze=1))
    names = trainable(model)
    assert any(n.startswith("transformer.encoder.layers.0.") for n in names)
    assert any(n.startswith("transformer.encoder.layers.3.") for n in names)
    assert not any(n.startswith("transformer.encoder.layers.1.") for n in names)
    assert not any(n.startswith("transformer.encoder.layers.2.") for n in names)
    for always in ("embedding.word_embeddings", "channel_emb", "output_layer", "final_layernorm", "duration_head"):
        assert any(always in n for n in names), always


def test_freezing_embeddings_only():
    model = tiny_model()
    apply_freezing(model, ModelArguments(train_embeddings=True))
    assert trainable(model) == {
        "transformer.embedding.word_embeddings.weight",
        "transformer.channel_emb.channel_embedding.weight",
    }
