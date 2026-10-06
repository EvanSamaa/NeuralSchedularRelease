"""neural_scheduler.evaluation.turn_order and silence_perturbation."""

import json

import numpy as np
import pytest

from neural_scheduler.constants import SILENCE_CODE
from neural_scheduler.evaluation import segments as seg
from neural_scheduler.evaluation import silence_perturbation as sp
from neural_scheduler.evaluation import turn_order as to
from neural_scheduler.evaluation.common import to_serializable

from .test_evaluation_support import (
    EXPERIMENTS,
    MODEL_DIR,
    StubModel,
    StubTokenizer,
    load_module,
    make_channel,
    make_samples,
)

S = SILENCE_CODE


@pytest.fixture(scope="module")
def originals(ntpp_orig):
    """The original helper modules, importable the way they import each other."""
    import sys

    sys.path.insert(0, str(ntpp_orig / MODEL_DIR))
    try:
        order = load_module(ntpp_orig / EXPERIMENTS / "varying_order_helpers.py", "orig_varying_order")
        silence = load_module(ntpp_orig / EXPERIMENTS / "varying_silence_helpers.py", "orig_varying_silence2")
        import dtw_module  # noqa: F401  (resolved lazily by the originals)
        import experiments.varying_silence_helpers  # noqa: F401
        yield order, silence
    finally:
        sys.path.remove(str(ntpp_orig / MODEL_DIR))


@pytest.fixture(scope="module")
def tok():
    return StubTokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return StubModel(tok.vocab_size)


def _as_tuple(segment):
    return (segment.start_idx, segment.end_idx, segment.pre_silence_len, segment.post_silence_len,
            getattr(segment, "channel", None))


# --- unit ---------------------------------------------------------------------

def test_find_speech_segments():
    found = seg.find_speech_segments([S, S, 1, 2, S, 3, S, S, S])
    assert [_as_tuple(x) for x in found] == [(2, 4, 2, 1, None), (5, 6, 1, 3, None)]
    assert found[0].length == 2
    a, b = seg.find_speech_segments_with_channel([1, S], [S, 2])
    assert a[0].channel == "A" and b[0].channel == "B" and b[0].pre_silence_len == 1
    assert seg.find_speech_segments([S, S]) == []


def test_silence_perturbation_builders():
    tokens = [S, S, 1, 2, S, S, S, 3]
    segment = seg.find_speech_segments(tokens)[0]
    assert sp.perturb_silence_before_segment(tokens, segment, 4, include_post_silence_tokens=1) == [S] * 4 + [1, 2, S]
    assert sp.perturb_silence_before_segment(tokens, segment, 0, truncate_after_segment=False) == [1, 2, S, S, S, 3]
    assert sp.perturb_silence_after_segment(tokens, segment, 1) == [S, S, 1, 2, S]
    assert sp.perturb_silence_after_segment(tokens, segment, 1, truncate_after_silence=False) == [S, S, 1, 2, S, 3]
    assert sp.truncate_to_segment(tokens, segment, include_post_silence_tokens=2) == [S, S, 1, 2, S, S]


def test_plots_run_headless(tok, model, tmp_path):
    samples = make_samples(tok, 3, seed=31, frames=30)
    order = to.run_multi_sample_turn_order_experiment(
        model, tok, samples, tok.offset, "cpu", num_pairs_per_sample=3, min_segment_length=0, verbose=False)
    assert order["experiments"]
    to.plot_turn_order_results(order, save_path=str(tmp_path / "order.png"))
    to.plot_by_first_speaker(order, save_path=str(tmp_path / "speaker.png"))

    silence = sp.run_multi_sample_experiment(
        model, tok, samples, tok.offset, "cpu", perturbation_range=range(-3, 4), num_segments_per_sample=2,
        verbose=False)
    assert silence["experiments"]
    sp.plot_multi_sample_aggregate(silence, save_path=str(tmp_path / "multi.png"))
    sp.plot_aggregate_results(silence, save_path=str(tmp_path / "agg.png"))
    sp.plot_by_segment_length(silence, save_path=str(tmp_path / "length.png"))
    sp.plot_perturbation_results(silence, save_path=str(tmp_path / "single.png"))
    for name in ("order", "speaker", "multi", "agg", "length", "single"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
    json.dumps(to_serializable(order))
    json.dumps(to_serializable(silence))


def test_corrected_scoring_swaps_fields_and_channel_layout(tok, model):
    samples = make_samples(tok, 2, seed=32, frames=24)
    kwargs = dict(perturbation_range=range(-1, 2), num_segments_per_sample=2, verbose=False)
    legacy = sp.run_multi_sample_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    fixed = sp.run_multi_sample_experiment(model, tok, samples, tok.offset, "cpu", legacy_scoring=False, **kwargs)
    exp_legacy, exp_fixed = legacy["experiments"][0], fixed["experiments"][0]
    # Legacy 'perplexities' are summed NLLs: they grow with the sequence, real perplexities do not.
    assert exp_legacy["perplexities"][0] > 10 * exp_legacy["nlls"][0] or exp_legacy["perplexities"][0] > 50
    ppl, nll, length = exp_fixed["perplexities"][0], exp_fixed["nlls"][0], exp_fixed["sequence_lengths"][0]
    assert ppl == pytest.approx(float(np.exp(nll / (2 * length))), rel=1e-4)
    assert exp_legacy["nlls"][0] != pytest.approx(ppl)


# --- equivalence ----------------------------------------------------------------

@pytest.mark.equivalence
def test_segment_finders_equivalence(originals):
    order, silence = originals
    rng = np.random.default_rng(0)
    for _ in range(30):
        a, b = make_channel(rng, 70), make_channel(rng, 70)
        old_a, old_b = order.find_speech_segments_with_channel(a, b, 8520)
        new_a, new_b = seg.find_speech_segments_with_channel(a, b)
        assert [_as_tuple(x) for x in new_a] == [_as_tuple(x) for x in old_a]
        assert [_as_tuple(x) for x in new_b] == [_as_tuple(x) for x in old_b]
        assert [_as_tuple(x)[:4] for x in seg.find_speech_segments(a)] == \
            [_as_tuple(x)[:4] for x in silence.find_speech_segments(a, 8520)]


@pytest.mark.equivalence
def test_turn_order_sequence_builders_equivalence(originals):
    order, _ = originals
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(25):
        a, b = make_channel(rng, 80), make_channel(rng, 80)
        old_pairs = order.find_sequential_segment_pairs(*order.find_speech_segments_with_channel(a, b, 8520),
                                                        max_gap=20, allow_overlap=True)
        new_pairs = to.find_sequential_segment_pairs(*seg.find_speech_segments_with_channel(a, b),
                                                     max_gap=20, allow_overlap=True)
        assert len(new_pairs) == len(old_pairs)
        for new_pair, old_pair in zip(new_pairs, old_pairs):
            assert (new_pair.gap_duration, new_pair.overlap_duration, new_pair.first_channel,
                    new_pair.second_channel, new_pair.is_turn_taking) == \
                   (old_pair.gap_duration, old_pair.overlap_duration, old_pair.first_channel,
                    old_pair.second_channel, old_pair.is_turn_taking)
            for name in ("create_original_sequence", "create_reversed_sequence", "create_simultaneous_sequence"):
                for kwargs in (dict(truncate_after=True, post_tokens=1), dict(truncate_after=False, post_tokens=0)):
                    expected = getattr(order, name)(a, b, old_pair, silence_token=8520, **kwargs)
                    assert getattr(to, name)(a, b, new_pair, silence_token=S, **kwargs) == expected
                    checked += 1
    assert checked > 100


@pytest.mark.equivalence
def test_silence_builders_equivalence(originals):
    _, silence = originals
    rng = np.random.default_rng(2)
    for _ in range(20):
        tokens = make_channel(rng, 60)
        for old_seg, new_seg in zip(silence.find_speech_segments(tokens, 8520), seg.find_speech_segments(tokens)):
            for n in (0, 1, 5):
                for trunc in (True, False):
                    assert sp.perturb_silence_before_segment(tokens, new_seg, n, S, trunc, 1) == \
                        silence.perturb_silence_before_segment(tokens, old_seg, n, 8520, trunc, 1)
                    assert sp.perturb_silence_after_segment(tokens, new_seg, n, S, trunc) == \
                        silence.perturb_silence_after_segment(tokens, old_seg, n, 8520, trunc)
            assert sp.truncate_to_segment(tokens, new_seg, 2) == silence.truncate_to_segment(tokens, old_seg, 2)


@pytest.mark.equivalence
def test_truncated_perplexity_equivalence(originals, tok, model):
    order, _ = originals
    from neural_scheduler.evaluation.common import extract_script_contents, prepare_inference_input

    sample = make_samples(tok, 1, seed=33, frames=16)[0]
    a, b, d = extract_script_contents(tok, sample["input_ids"], return_dialogue=True)
    context, start = prepare_inference_input(a, b, tok, "cpu", already_tokenized=True)
    da = [x - tok.offset for x in d[::2]]
    db = [x - tok.offset for x in d[1::2]]
    expected = order.compute_truncated_perplexity(model, tok, context, start, da, db, tok.offset, "cpu")
    assert to.compute_truncated_perplexity(model, context, start, da, db, tok.offset, "cpu") == expected


@pytest.mark.equivalence
@pytest.mark.parametrize("with_turn_tokens", [True, False])
def test_turn_order_experiment_equivalence(originals, tok, model, with_turn_tokens):
    order, _ = originals
    samples = make_samples(tok, 4, seed=34, frames=36, with_turn_tokens=with_turn_tokens)
    kwargs = dict(max_gap=20, num_pairs_per_sample=4, min_segment_length=0, verbose=False)
    old = order.run_multi_sample_turn_order_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    new = to.run_multi_sample_turn_order_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    assert len(new["experiments"]) > 3
    assert json.dumps(to_serializable(new), sort_keys=True) == json.dumps(to_serializable(old), sort_keys=True)


@pytest.mark.equivalence
@pytest.mark.parametrize("location", ["before", "after"])
def test_silence_experiment_equivalence(originals, tok, model, location):
    _, silence = originals
    samples = make_samples(tok, 3, seed=35, frames=30)
    kwargs = dict(perturbation_range=range(-4, 6), num_segments_per_sample=3, perturb_location=location,
                  min_segment_length=1, max_segment_length=20, verbose=False)
    old = silence.run_multi_sample_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    new = sp.run_multi_sample_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    assert len(new["experiments"]) > 2
    assert json.dumps(to_serializable(new), sort_keys=True) == json.dumps(to_serializable(old), sort_keys=True)


@pytest.mark.equivalence
def test_length_comparison_equivalence(originals, tok, model):
    _, silence = originals
    samples = make_samples(tok, 3, seed=36, frames=30)
    kwargs = dict(perturbation_range=range(-2, 3), num_segments_per_sample=4, verbose=False)
    old = silence.run_length_comparison_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    new = sp.run_length_comparison_experiment(model, tok, samples, tok.offset, "cpu", **kwargs)
    assert new["bin_stats"]
    assert json.dumps(to_serializable(new), sort_keys=True, default=str) == \
        json.dumps(to_serializable(old), sort_keys=True, default=str)
