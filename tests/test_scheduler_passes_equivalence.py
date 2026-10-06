"""Pass 2 and pass 3 called directly, for layouts the entry points never produce.

Through the entry points every host has exactly one backchannel unit. The
passes also support several units per host (ordering constraint, pause only for
the first unit, several breakpoints per host), so those paths are compared with
the original here.
"""

import copy
import random

import pytest

from neural_scheduler.scheduler.pass1 import pass_1_primary_turns
from neural_scheduler.scheduler.pass2 import pass_2_backchannel_placement
from neural_scheduler.scheduler.pass3 import pass_3_refinement
from neural_scheduler.scheduler.script_parse import (
    compute_turn_boundaries_from_script_only,
    parse_turn_structure,
)

from .test_scheduler_helpers import (
    ORIG_THREE_PASS,
    CompactVocab,
    StubSchedulerModel,
    load_module,
    make_script,
    plain,
)

pytestmark = pytest.mark.equivalence


@pytest.fixture(scope="module")
def orig(ntpp_orig):
    return load_module(ntpp_orig / ORIG_THREE_PASS, "orig_search_three_pass_direct")


def _convert(items, cls_by_name):
    """Rebuild dataclass instances as the equally named classes of another module."""
    def rebuild(obj):
        name = type(obj).__name__
        if name in cls_by_name:
            fields = {k: rebuild(v) for k, v in vars(obj).items()}
            return cls_by_name[name](**fields)
        if isinstance(obj, list):
            return [rebuild(v) for v in obj]
        if isinstance(obj, dict):
            return {k: rebuild(v) for k, v in obj.items()}
        return obj
    return rebuild(copy.deepcopy(items))


@pytest.mark.parametrize("seed", range(7000, 7030))
def test_passes_with_several_backchannels_per_host(seed, orig):
    rng = random.Random(seed)
    tok = CompactVocab()
    st = tok.special_tokens
    script = make_script(rng, tok, bc_only=True, max_turns=4, max_block=8)
    input_ids = script["input_ids"]

    boundaries = compute_turn_boundaries_from_script_only(input_ids, st)
    primary, backchannels = parse_turn_structure(input_ids, st, boundaries)
    if not backchannels:
        pytest.skip("script without backchannels")

    # Two or three units per host, in script order.
    units = []
    for bc in backchannels:
        for group in range(rng.choice([2, 2, 3])):
            unit = copy.deepcopy(bc)
            unit.bc_group_index = group
            unit.start_token = bc.start_token + group
            units.append(unit)

    pass1_kwargs = dict(search_window=3, scoring_context=3, inter_turn_gap=rng.choice([0, 4]),
                        schedule_subturns=rng.random() < 0.5)
    pass2_kwargs = dict(
        search_stride=rng.choice([1, 2]),
        scoring_context=rng.choice([0, 5]),
        bc_position_limit=rng.choice([1.0, 0.5]),
        bc_effect=rng.choice([None, "hp", "fp", "np", "hp+fp+np"]),
        max_bc_overlap_ratio=rng.choice([0.5, 0.05, 1.0]),
    )
    pass3_kwargs = dict(scoring_context=rng.choice([0, 10]), max_extra_gap=rng.choice([3, 10]))

    model_new = StubSchedulerModel(tok.vocab_size)
    placed, _ = pass_1_primary_turns(model_new, input_ids, primary, backchannels, st,
                                     device="cpu", **pass1_kwargs)
    new_primary, new_bcs, new_diag2 = pass_2_backchannel_placement(
        model_new, input_ids, placed, units, st, device="cpu", **pass2_kwargs)
    new_refined, new_diag3 = pass_3_refinement(
        model_new, input_ids, new_primary, new_bcs, st, device="cpu", **pass3_kwargs)

    classes = {"TurnInfo": orig.TurnInfo, "PlacedTurn": orig.PlacedTurn}
    model_orig = StubSchedulerModel(tok.vocab_size)
    o_placed, _ = orig.pass_1_primary_turns(
        model_orig, tok, input_ids, _convert(primary, classes), _convert(backchannels, classes), st,
        device="cpu", **pass1_kwargs)
    o_primary, o_bcs, o_diag2 = orig.pass_2_backchannel_placement(
        model_orig, tok, input_ids, o_placed, _convert(units, classes), st, device="cpu", **pass2_kwargs)
    o_refined, o_diag3 = orig.pass_3_refinement(
        model_orig, tok, input_ids, o_primary, o_bcs, st, device="cpu", **pass3_kwargs)

    assert model_new.calls == model_orig.calls
    assert plain(new_primary) == plain(o_primary)
    assert plain(new_bcs) == plain(o_bcs)
    assert plain(new_diag2) == plain(o_diag2)
    assert plain(new_refined) == plain(o_refined)
    assert plain(new_diag3) == plain(o_diag3)
