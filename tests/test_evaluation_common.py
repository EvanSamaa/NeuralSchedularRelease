"""neural_scheduler.evaluation.common: unit tests and equivalence with the research code."""

import json
import random

import numpy as np
import pytest
import torch

from neural_scheduler.constants import AUDIO_TOKEN_OFFSET, SILENCE_CODE, SILENCE_TOKEN_ID
from neural_scheduler.evaluation import common

from .test_evaluation_support import (
    EXPERIMENTS,
    MODEL_DIR,
    StubModel,
    StubTokenizer,
    load_definitions,
    load_module,
    make_samples,
    real_offset_tokenizer,
)

MUTUAL_SILENCE_HELPERS = [
    "prepare_inference_input", "compute_channel_ids_for_inference", "generate_dialogue", "_separate_channels",
    "load_random_shard_examples", "extract_script_contents", "score_sequences",
    "get_input_ID_object_attention_mask_and_channel_id",
]


@pytest.fixture(scope="module")
def tok():
    return StubTokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return StubModel(tok.vocab_size)


@pytest.fixture(scope="module")
def orig_ms(ntpp_orig):
    return load_definitions(ntpp_orig / EXPERIMENTS / "experiment_with_mutual_silence.py", MUTUAL_SILENCE_HELPERS)


@pytest.fixture(scope="module")
def orig_dtw(ntpp_orig):
    return load_module(ntpp_orig / MODEL_DIR / "dtw_module.py", "orig_dtw_module")


# --- unit ---------------------------------------------------------------------

def test_channel_ids_layout():
    ids = common.compute_channel_ids(7, 3, "cpu")
    assert ids.tolist() == [[0, 0, 0, 1, 2, 1, 2]]
    assert ids.dtype == torch.long
    assert common.compute_alignment_channel_ids(7, 3, "cpu").tolist() == [[0, 0, 0, 0, 1, 0, 1]]
    assert common.compute_batch_channel_ids(2, 5, [1, 4], "cpu").tolist() == [[0, 1, 2, 1, 2], [0, 0, 0, 0, 1]]


def test_extract_and_prepare_roundtrip_with_real_ids():
    tok = real_offset_tokenizer()
    sample = make_samples(tok, 1, seed=3)[0]
    a, b, d = common.extract_script_contents(tok, sample["input_ids"], return_dialogue=True)
    assert len(d) == 80 and all(AUDIO_TOKEN_OFFSET <= x < AUDIO_TOKEN_OFFSET + 16383 for x in d)
    assert a[0] == SILENCE_TOKEN_ID
    context, start = common.prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
    assert context[0].tolist() == sample["input_ids"][:start]
    assert common.extract_script_contents(tok, [1, 2, 3]) == ([], [])


def test_prepare_from_codec_ids_drops_turn_markers():
    tok = real_offset_tokenizer()
    sample = make_samples(tok, 1, seed=4)[0]
    a, b = common.extract_script_contents(tok, sample["input_ids"])
    codes_a = [x - AUDIO_TOKEN_OFFSET for x in a]
    codes_b = [x - AUDIO_TOKEN_OFFSET for x in b]
    assert max(codes_a) > common.MAX_SCRIPT_CODE  # turn markers are present
    context, start = common.prepare_inference_input(codes_a, codes_b, tok, "cpu")
    kept = [x for x in a if x - AUDIO_TOKEN_OFFSET <= common.MAX_SCRIPT_CODE]
    assert context[0, 1:1 + len(kept)].tolist() == kept
    assert start == context.shape[1]


def test_load_jsonl_examples(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(20)) + "\n\n")
    assert [r["i"] for r in common.load_jsonl_examples(str(path))] == list(range(20))
    first = common.load_jsonl_examples(str(path), 5, seed=1)
    assert first == common.load_jsonl_examples(str(path), 5, seed=1)
    assert len(common.load_jsonl_examples(str(path), 50, seed=1)) == 20
    with pytest.raises(FileNotFoundError):
        common.load_jsonl_examples(str(tmp_path / "missing.jsonl"))


def test_dialogue_nll_matches_per_token_sum(tok, model):
    sample = make_samples(tok, 1, seed=5, frames=12)[0]
    a, b, d = common.extract_script_contents(tok, sample["input_ids"], return_dialogue=True)
    context, start = common.prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
    ppl, nll = common.compute_dialogue_nll(model, context, start, d, device="cpu")
    per_token = common.compute_per_token_nll(model, context[0].tolist(), d, "cpu")
    assert per_token.shape == (len(d),)
    assert nll == pytest.approx(float(per_token.sum()), rel=1e-5)
    assert ppl == pytest.approx(float(np.exp(per_token.mean())), rel=1e-5)
    _, legacy_nll = common.compute_dialogue_nll(model, context, start, d, device="cpu", legacy_channel_ids=True)
    assert legacy_nll != pytest.approx(nll)


def test_to_serializable():
    out = common.to_serializable({"a": np.float32(1.5), "b": [np.int64(2), np.arange(2)], "c": (1, 2)})
    assert json.loads(json.dumps(out)) == {"a": 1.5, "b": [2, [0, 1]], "c": [1, 2]}


# --- equivalence ----------------------------------------------------------------

@pytest.mark.equivalence
def test_extract_script_contents_equivalence(orig_ms, ntpp_orig, tok):
    orig_vs = load_module(ntpp_orig / EXPERIMENTS / "varying_silence_helpers.py", "orig_varying_silence")
    for sample in make_samples(tok, 5, seed=1):
        for ret in (False, True):
            expected = orig_ms["extract_script_contents"](tok, sample["input_ids"], return_dialogue=ret)
            assert common.extract_script_contents(tok, sample["input_ids"], return_dialogue=ret) == expected
            assert orig_vs.extract_script_contents(tok, sample["input_ids"], return_dialogue=ret) == expected


@pytest.mark.equivalence
def test_prepare_inference_input_equivalence(orig_ms, orig_dtw, tok):
    for sample in make_samples(tok, 4, seed=2):
        a, b = common.extract_script_contents(tok, sample["input_ids"])
        new_ids, new_start = common.prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
        for orig in (orig_ms["prepare_inference_input"], orig_dtw.prepare_inference_input):
            ids, start = orig(a, b, tok, "cpu", already_tokenized=True)
            assert start == new_start and torch.equal(ids, new_ids)
        codes_a = [x - tok.offset for x in a]
        codes_b = [x - tok.offset for x in b]
        ids, start = orig_dtw.prepare_inference_input(codes_a, codes_b, tok, "cpu", already_tokenized=False)
        new_ids, new_start = common.prepare_inference_input(codes_a, codes_b, tok, "cpu", already_tokenized=False)
        assert start == new_start and torch.equal(ids, new_ids)


@pytest.mark.equivalence
def test_channel_ids_equivalence(orig_ms, orig_dtw):
    for length, start in [(1, 0), (9, 4), (10, 4), (6, 6), (5, 9)]:
        expected = orig_ms["compute_channel_ids_for_inference"](length, start, torch.device("cpu"))
        assert torch.equal(common.compute_channel_ids(length, start, "cpu"), expected)
        expected = orig_dtw.compute_channel_ids_for_alignment(length, start, torch.device("cpu"))
        assert torch.equal(common.compute_alignment_channel_ids(length, start, "cpu"), expected)
    ids = torch.zeros(3, 11, dtype=torch.long)
    _, mask, expected = orig_ms["get_input_ID_object_attention_mask_and_channel_id"](ids, [2, 5, 11], "cpu")
    inputs, new_mask, new_channels = common.build_scoring_inputs(ids, [2, 5, 11], "cpu")
    assert torch.equal(new_channels, expected) and torch.equal(new_mask, mask)
    assert inputs["input_ids"] is ids and inputs["labels"] is ids


@pytest.mark.equivalence
def test_score_sequences_equivalence(orig_ms, tok, model):
    samples = make_samples(tok, 3, seed=6, frames=10)
    length = min(len(s["input_ids"]) for s in samples)
    ids = torch.tensor([s["input_ids"][:length] for s in samples])
    starts = [length - 9, length - 5, length - 13]
    inputs, mask, channels = common.build_scoring_inputs(ids, starts, "cpu")
    ppl, loss = common.score_sequences(model, inputs, mask, channels)
    exp_ppl, exp_loss = orig_ms["score_sequences"](model, inputs, mask, channels)
    assert torch.equal(ppl, exp_ppl) and torch.equal(loss, exp_loss)


@pytest.mark.equivalence
def test_compute_full_nll_equivalence(orig_dtw, tok, model):
    for sample in make_samples(tok, 3, seed=7, frames=14):
        a, b, d = common.extract_script_contents(tok, sample["input_ids"], return_dialogue=True)
        context, start = common.prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
        da = torch.tensor([d[::2]])
        db = torch.tensor([d[1::2]])
        expected = orig_dtw.compute_full_nll(model, tok, context, start, da, db, device="cpu")
        assert common.compute_full_nll(model, context, start, da, db, device="cpu") == expected


@pytest.mark.equivalence
def test_per_token_nll_equivalence(ntpp_orig, tok, model):
    orig = load_module(ntpp_orig / EXPERIMENTS / "nll_diagnostics.py", "orig_nll_diagnostics")
    for sample in make_samples(tok, 3, seed=8, frames=14):
        a, b, d = common.extract_script_contents(tok, sample["input_ids"], return_dialogue=True)
        context, _ = common.prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
        expected = orig.compute_per_token_nll(model, context[0].tolist(), d, torch.device("cpu"))
        np.testing.assert_array_equal(common.compute_per_token_nll(model, context[0].tolist(), d, "cpu"), expected)


@pytest.mark.equivalence
def test_remove_duplicated_silence_equivalence(orig_dtw):
    rng = np.random.default_rng(0)
    for _ in range(50):
        seq = [SILENCE_CODE if x < 5 else int(x) for x in rng.integers(0, 9, size=int(rng.integers(0, 40)))]
        assert common.remove_duplicated_silence_tokens(seq, target_val=SILENCE_CODE) == \
            orig_dtw.remove_duplicated_silence_tokens(seq, target_val=SILENCE_CODE)
    seq = [SILENCE_TOKEN_ID] * 4 + [5, 6] + [SILENCE_TOKEN_ID] * 2 + [7]
    assert common.remove_duplicated_silence_tokens(seq) == orig_dtw.remove_duplicated_silence_tokens(seq)


@pytest.mark.equivalence
@pytest.mark.parametrize("separated", [True, False])
def test_generate_dialogue_equivalence(orig_ms, tok, model, separated):
    sample = make_samples(tok, 1, seed=9, frames=8)[0]
    a, b = common.extract_script_contents(tok, sample["input_ids"])
    kwargs = dict(max_new_tokens=24, temperature=0.8, top_k=50, top_p=0.95, do_sample=True, device="cpu",
                  return_channel_separated=separated, already_tokenized=True)
    torch.manual_seed(11)
    expected = orig_ms["generate_dialogue"](model, tok, a, b, **kwargs)
    torch.manual_seed(11)
    assert common.generate_dialogue(model, tok, a, b, **kwargs) == expected
    greedy = dict(kwargs, do_sample=False)
    assert common.generate_dialogue(model, tok, a, b, **greedy) == orig_ms["generate_dialogue"](model, tok, a, b, **greedy)


@pytest.mark.equivalence
def test_separate_channels_equivalence(orig_ms, tok):
    t = tok.convert_tokens_to_ids
    generated = [tok.offset + 5, tok.offset + 6, t("<|end-of-dialogue|>"), tok.offset + 7, tok.offset + 8]
    assert common.separate_channels(generated, tok) == orig_ms["_separate_channels"](generated, tok)


@pytest.mark.equivalence
def test_load_examples_matches_seeded_original(orig_ms, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(30)))
    random.seed(123)
    expected = orig_ms["load_random_shard_examples"](str(path), num_examples=7)
    assert common.load_jsonl_examples(str(path), 7, seed=123) == expected
    random.seed(5)
    expected = orig_ms["load_random_shard_examples"](str(path), num_examples=7)
    random.seed(5)
    assert common.load_jsonl_examples(str(path), 7) == expected
