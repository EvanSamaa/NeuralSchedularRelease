"""The refactored search must make exactly the decisions of the original research code.

Both implementations are driven with the same stub model over random scripts
and random hyperparameters; schedules, token outputs and every recorded score
must be identical.
"""

import copy
import random

import pytest

from neural_scheduler.scheduler import generate_dialogue_bc_only, generate_dialogue_three_pass

from .test_scheduler_helpers import (
    ORIG_BC_ONLY,
    ORIG_THREE_PASS,
    CompactVocab,
    RealVocab,
    StubSchedulerModel,
    load_module,
    make_s1_timing,
    make_script,
    plain,
    random_options,
    random_pass2_options,
)

pytestmark = pytest.mark.equivalence


@pytest.fixture(scope="module")
def orig_three_pass(ntpp_orig):
    return load_module(ntpp_orig / ORIG_THREE_PASS, "orig_search_three_pass")


@pytest.fixture(scope="module")
def orig_bc_only(ntpp_orig):
    return load_module(ntpp_orig / ORIG_BC_ONLY, "orig_search_bc_only")


def _normalise(state, original: bool, anchored_scores_empty: bool = False):
    """State as plain data, with the documented differences removed.

    * ``pass3_sub_segment_details``: the original read a key pass 3 never wrote
      and always exported empty lists; the release exports the breakpoint
      results, so the original side is rebuilt from its own pass-3 diagnostics.
    * ``gap_constraint_applied`` of single-block turns: the original evaluated
      it after appending the turn and always recorded False.
    * ``nll_scores`` of anchored turns: ``[]`` in one original copy, ``[0.0]``
      in the other and in the release.
    """
    data = plain(state)
    pass3 = data["diagnostics"].get("pass3", {})
    if original:
        data["pass3_sub_segment_details"] = {
            host["turn_id"]: host["breakpoint_results"] for host in pass3.get("host_turns_refined", [])
        }
    for entry in data["diagnostics"]["pass1"]["per_turn"]:
        if entry.get("mode") == "single_block":
            entry["gap_constraint_applied"] = None
        if anchored_scores_empty and entry.get("mode") in ("gt_anchored", "fallback_no_gt"):
            entry["nll_scores"] = None
    if anchored_scores_empty:
        data["pass1_nll_scores"] = {k: None for k in data["pass1_nll_scores"]}
    return data


def _assert_same(result_new, result_orig, **normalise_kwargs):
    a_new, b_new, state_new = result_new
    a_orig, b_orig, state_orig = result_orig
    assert a_new == a_orig
    assert b_new == b_orig
    new = _normalise(state_new, original=False, **normalise_kwargs)
    orig = _normalise(state_orig, original=True, **normalise_kwargs)
    assert new.keys() == orig.keys()
    for key in new:
        assert new[key] == orig[key], f"SchedulingState.{key} differs"


def _run_pair(new_fn, orig_fn, tok, input_ids, options, new_only=None, **extra):
    """Run both implementations on separate stub models and check they made the same calls."""
    model_new = StubSchedulerModel(tok.vocab_size)
    model_orig = StubSchedulerModel(tok.vocab_size)
    st = tok.special_tokens
    result_new = new_fn(model_new, tok, list(input_ids), dict(st), device="cpu", verbose=False,
                        **copy.deepcopy(extra), **options, **(new_only or {}))
    result_orig = orig_fn(model_orig, tok, list(input_ids), dict(st), device="cpu", verbose=False,
                          **copy.deepcopy(extra), **options)
    assert model_new.calls == model_orig.calls
    return result_new, result_orig


@pytest.mark.parametrize("seed", range(60))
def test_three_pass_matches_original(seed, orig_three_pass):
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok)
    options = random_options(rng)
    result_new, result_orig = _run_pair(
        generate_dialogue_three_pass, orig_three_pass.generate_dialogue_three_pass,
        tok, script["input_ids"], options,
    )
    _assert_same(result_new, result_orig)


@pytest.mark.parametrize("seed", range(1000, 1012))
def test_three_pass_matches_original_with_backchannels_everywhere(seed, orig_three_pass):
    """Dense backchannels with pause strategies, the setting that exercises pass 3 most."""
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok, bc_only=True, max_turns=4)
    options = random_options(rng)
    options["pass2_bc_effect"] = rng.choice(["hp", "fp", "hp+fp+np", None])
    options["pass2_max_bc_overlap_ratio"] = rng.choice([0.05, 0.5])
    result_new, result_orig = _run_pair(
        generate_dialogue_three_pass, orig_three_pass.generate_dialogue_three_pass,
        tok, script["input_ids"], options,
    )
    _assert_same(result_new, result_orig)


@pytest.mark.parametrize("seed", range(2000, 2020))
def test_three_pass_score_only_pauses_matches_bc_only_copy(seed, orig_bc_only):
    """``pass2_score_only_pauses`` reproduces the pass 2 of the benchmark copy of the search."""
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok)
    options = random_options(rng)
    options["pass2_bc_effect"] = rng.choice(["hp", "fp", "hp+fp", "hp+fp+np", "np"])
    result_new, result_orig = _run_pair(
        generate_dialogue_three_pass, orig_bc_only.generate_dialogue_three_pass,
        tok, script["input_ids"], options, new_only={"pass2_score_only_pauses": True},
    )
    _assert_same(result_new, result_orig)


@pytest.mark.parametrize("seed", range(3000, 3030))
def test_bc_only_matches_benchmark_copy(seed, orig_bc_only):
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok, bc_only=rng.random() < 0.8)
    timing = make_s1_timing(rng, script)
    options = random_pass2_options(rng)
    result_new, result_orig = _run_pair(
        generate_dialogue_bc_only, orig_bc_only.generate_dialogue_bc_only,
        tok, script["input_ids"], options, new_only={"pass2_score_only_pauses": True},
        s1_turn_timing=timing,
    )
    _assert_same(result_new, result_orig)


@pytest.mark.parametrize("seed", range(4000, 4015))
def test_bc_only_matches_three_pass_copy(seed, orig_three_pass):
    """Same entry point in the other original file, where pass 2 commits pauses."""
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok, bc_only=True)
    timing = make_s1_timing(rng, script)
    options = random_pass2_options(rng)
    result_new, result_orig = _run_pair(
        generate_dialogue_bc_only, orig_three_pass.generate_dialogue_bc_only,
        tok, script["input_ids"], options, s1_turn_timing=timing,
    )
    _assert_same(result_new, result_orig, anchored_scores_empty=True)


@pytest.mark.parametrize("seed", range(5000, 5003))
def test_three_pass_matches_original_with_released_vocabulary(seed, orig_three_pass):
    """Same check with the real token ids and a 168960-way output layer."""
    rng = random.Random(seed)
    tok = RealVocab()
    script = make_script(rng, tok, max_turns=3, max_block=4)
    options = random_options(rng)
    options.update(pass1_search_window=2, pass1_scoring_context=3, pass1_inter_turn_gap=3,
                   pass2_search_stride=2, pass3_max_extra_gap=3)
    result_new, result_orig = _run_pair(
        generate_dialogue_three_pass, orig_three_pass.generate_dialogue_three_pass,
        tok, script["input_ids"], options,
    )
    _assert_same(result_new, result_orig)


def test_bc_only_matches_benchmark_copy_with_released_vocabulary(orig_bc_only):
    rng = random.Random(6000)
    tok = RealVocab()
    script = make_script(rng, tok, bc_only=True, max_turns=3, max_block=4)
    timing = make_s1_timing(rng, script)
    options = dict(pass2_search_stride=2, pass2_scoring_context=0, pass2_bc_effect="np",
                   pass2_max_bc_overlap_ratio=0.01, pass2_bc_position_limit=0.6)
    result_new, result_orig = _run_pair(
        generate_dialogue_bc_only, orig_bc_only.generate_dialogue_bc_only,
        tok, script["input_ids"], options, new_only={"pass2_score_only_pauses": True},
        s1_turn_timing=timing,
    )
    _assert_same(result_new, result_orig)
