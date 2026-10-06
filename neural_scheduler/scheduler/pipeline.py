"""Entry points: schedule a whole dialogue.

``generate_dialogue_three_pass`` searches every primary turn onset (pass 1),
places the backchannels (pass 2) and re-times what follows them (pass 3).
``generate_dialogue_bc_only`` keeps one speaker at known positions and only
places the other speaker's backchannels.

Both take the scheduler LM, its tokenizer and an input sequence whose script
region holds what each speaker will say, and return the two channels of the
scheduled dialogue as codec ids plus the :class:`SchedulingState`.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

from . import config
from .dialogue import build_dialogue_from_placements, dialogue_to_channel_codes
from .pass1 import pass_1_primary_turns
from .pass1_anchored import pass_1_anchor_primary_turns
from .pass2 import pass_2_backchannel_placement
from .pass3 import pass_3_refinement
from .script_parse import compute_turn_boundaries_from_script_only, parse_turn_structure
from .types import PlacedTurn, SchedulingState

logger = logging.getLogger(__name__)

_OUTPUT_LENGTH_MARGIN = 5


def _parse_script(
    input_ids: List[int],
    special_tokens: Dict[str, int],
    turn_boundaries: Optional[List[Dict]],
    verbose: bool,
) -> SchedulingState:
    """Read primary turns and backchannels out of the script into a fresh state."""
    if turn_boundaries is None:
        turn_boundaries = compute_turn_boundaries_from_script_only(input_ids, special_tokens)

    state = SchedulingState()
    state.primary_turns, state.backchannel_turns = parse_turn_structure(
        input_ids, special_tokens, turn_boundaries
    )
    # Each backchannel turn is scheduled as one unit.
    for bc in state.backchannel_turns:
        unit = copy.deepcopy(bc)
        unit.bc_group_index = 0
        state.backchannel_units.append(unit)

    if verbose:
        logger.info(f"Script: {len(state.primary_turns)} primary turn(s), "
                    f"{len(state.backchannel_turns)} backchannel(s)")
        for t in state.primary_turns:
            logger.info(f"  Turn {t.turn_id} ch={t.channel} dur={t.duration} sub_segs={len(t.sub_segments)}")
        for t in state.backchannel_turns:
            logger.info(f"  Backchannel {t.turn_id} ch={t.channel} dur={t.duration} host={t.host_turn_id}")
    return state


def _record_pass1(state: SchedulingState, placed_primary: List[PlacedTurn], diag: Dict) -> None:
    state.placed_primary = placed_primary
    state.pass1_nll_scores = {d["turn_id"]: d["nll_scores"] for d in diag["per_turn"]}
    state.diagnostics["pass1"] = diag


def _run_pass2(model, input_ids, special_tokens, state: SchedulingState, device, verbose, **pass2_kwargs) -> None:
    state.placed_primary, state.placed_backchannels, diag = pass_2_backchannel_placement(
        model, input_ids, state.placed_primary, state.backchannel_units, special_tokens,
        device=device, verbose=verbose, **pass2_kwargs,
    )
    state.pass2_nll_scores = {
        (d["turn_id"], d.get("bc_group_index", 0)): d.get("nll_scores", [])
        for d in diag["per_backchannel"]
    }
    state.diagnostics["pass2"] = diag


def _final_channels(state: SchedulingState, tokenizer, special_tokens: Dict[str, int]) -> Tuple[List[int], List[int]]:
    placements = state.refined_placements
    max_end = max((pt.onset_step + pt.turn_info.duration for pt in placements), default=100)
    dialogue = build_dialogue_from_placements(
        placements, max_end + _OUTPUT_LENGTH_MARGIN, special_tokens["silence"]
    )
    return dialogue_to_channel_codes(dialogue, tokenizer)


def generate_dialogue_three_pass(
    model,
    tokenizer,
    input_ids: List[int],
    special_tokens: Dict[str, int],
    turn_boundaries: Optional[List[Dict]] = None,
    pass1_search_window: int = config.PASS1.search_window,
    pass1_search_stride: int = config.PASS1.search_stride,
    pass1_scoring_context: int = config.PASS1.scoring_context,
    pass1_inter_turn_gap: int = config.PASS1.inter_turn_gap,
    pass_1_schedule_subturns: bool = config.PASS1.schedule_subturns,
    pass_1_speaker_1_subturn_min_pause: int = config.PASS1.speaker_1_subturn_min_pause,
    pass_1_speaker_1_subturn_max_pause: int = config.PASS1.speaker_1_subturn_max_pause,
    pass_1_speaker_2_subturn_min_pause: int = config.PASS1.speaker_2_subturn_min_pause,
    pass_1_speaker_2_subturn_max_pause: int = config.PASS1.speaker_2_subturn_max_pause,
    pass_1_subturn_search_stride: int = config.PASS1.subturn_search_stride,
    pass_1_use_speaker_gap_constraints: bool = config.PASS1.use_speaker_gap_constraints,
    pass_1_speaker_1_min_gap: int = config.PASS1.speaker_1_min_gap,
    pass_1_speaker_1_max_gap: int = config.PASS1.speaker_1_max_gap,
    pass_1_speaker_2_min_gap: int = config.PASS1.speaker_2_min_gap,
    pass_1_speaker_2_max_gap: int = config.PASS1.speaker_2_max_gap,
    pass_1_strong_overlap_mode: bool = config.PASS1.strong_overlap_mode,
    pass_1_strong_overlap_min: float = config.PASS1.strong_overlap_min,
    pass_1_strong_overlap_max: float = config.PASS1.strong_overlap_max,
    pass2_search_stride: int = config.PASS2.search_stride,
    pass2_scoring_context: int = config.PASS2.scoring_context,
    pass2_bc_position_limit: float = config.PASS2.bc_position_limit,
    pass2_bc_effect: Optional[str] = config.PASS2.bc_effect,
    pass2_max_bc_overlap_ratio: float = config.PASS2.max_bc_overlap_ratio,
    pass2_score_only_pauses: bool = config.PASS2.score_only_pauses,
    pass3_scoring_context: int = config.PASS3.scoring_context,
    pass3_max_extra_gap: int = config.PASS3.max_extra_gap,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[List[int], List[int], SchedulingState]:
    """Schedule primary turns, backchannels and post-backchannel timing.

    Args:
        model: scheduler LM; called with ``input_ids``, ``attention_mask``,
            ``channel_ids``, ``use_cache=False`` and ``return_dict=True``.
        tokenizer: its tokenizer (used to turn the result into codec ids).
        input_ids: script A, script B and the dialogue marker, as vocab ids.
            Anything after ``<|start-of-dialogue|>`` is ignored.
        special_tokens: from :func:`neural_scheduler.vocab.get_special_token_ids`.
        turn_boundaries: ``{turn_id, channel, start_sec, end_sec}`` per speech
            block; derived from the script when omitted.
        pass1_*, pass_1_*: see :func:`pass_1_primary_turns`.
        pass2_*: see :func:`pass_2_backchannel_placement`.
        pass3_*: see :func:`pass_3_refinement`.

    Returns:
        ``(channel_a_codes, channel_b_codes, state)``; the two lists have equal
        length, one codec id per 80 ms step.
    """
    state = _parse_script(input_ids, special_tokens, turn_boundaries, verbose)

    placed_primary, diag_1 = pass_1_primary_turns(
        model, input_ids, state.primary_turns, state.backchannel_turns, special_tokens,
        search_window=pass1_search_window,
        search_stride=pass1_search_stride,
        scoring_context=pass1_scoring_context,
        inter_turn_gap=pass1_inter_turn_gap,
        device=device,
        verbose=verbose,
        schedule_subturns=pass_1_schedule_subturns,
        speaker_1_subturn_min_pause=pass_1_speaker_1_subturn_min_pause,
        speaker_1_subturn_max_pause=pass_1_speaker_1_subturn_max_pause,
        speaker_2_subturn_min_pause=pass_1_speaker_2_subturn_min_pause,
        speaker_2_subturn_max_pause=pass_1_speaker_2_subturn_max_pause,
        subturn_search_stride=pass_1_subturn_search_stride,
        use_speaker_gap_constraints=pass_1_use_speaker_gap_constraints,
        speaker_1_min_gap=pass_1_speaker_1_min_gap,
        speaker_1_max_gap=pass_1_speaker_1_max_gap,
        speaker_2_min_gap=pass_1_speaker_2_min_gap,
        speaker_2_max_gap=pass_1_speaker_2_max_gap,
        strong_overlap_mode=pass_1_strong_overlap_mode,
        strong_overlap_min=pass_1_strong_overlap_min,
        strong_overlap_max=pass_1_strong_overlap_max,
    )
    _record_pass1(state, placed_primary, diag_1)
    state.pass1_subturn_nll_scores = diag_1.get("subturn_nll_detail", {})

    _run_pass2(
        model, input_ids, special_tokens, state, device, verbose,
        search_stride=pass2_search_stride,
        scoring_context=pass2_scoring_context,
        bc_position_limit=pass2_bc_position_limit,
        bc_effect=pass2_bc_effect,
        max_bc_overlap_ratio=pass2_max_bc_overlap_ratio,
        score_only_pauses=pass2_score_only_pauses,
    )

    state.refined_placements, diag_3 = pass_3_refinement(
        model, input_ids, state.placed_primary, state.placed_backchannels, special_tokens,
        scoring_context=pass3_scoring_context,
        max_extra_gap=pass3_max_extra_gap,
        device=device,
        verbose=verbose,
    )
    state.diagnostics["pass3"] = diag_3
    for host_diag in diag_3.get("host_turns_refined", []):
        tid = host_diag["turn_id"]
        state.pass3_sub_segment_details[tid] = host_diag.get("breakpoint_results", [])
        state.pass3_adjustments[tid] = {
            "original_turn_end": host_diag.get("original_turn_end"),
            "new_turn_end": host_diag.get("new_turn_end"),
            "total_shift": host_diag.get("total_shift"),
        }

    channel_a, channel_b = _final_channels(state, tokenizer, special_tokens)
    return channel_a, channel_b, state


def generate_dialogue_bc_only(
    model,
    tokenizer,
    input_ids: List[int],
    special_tokens: Dict[str, int],
    s1_turn_timing: List[Dict],
    turn_boundaries: Optional[List[Dict]] = None,
    pass2_search_stride: int = config.PASS2.search_stride,
    pass2_scoring_context: int = config.PASS2.scoring_context,
    pass2_bc_position_limit: float = config.PASS2.bc_position_limit,
    pass2_bc_effect: Optional[str] = config.PASS2.bc_effect,
    pass2_max_bc_overlap_ratio: float = config.PASS2.max_bc_overlap_ratio,
    pass2_score_only_pauses: bool = config.PASS2.score_only_pauses,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[List[int], List[int], SchedulingState]:
    """Schedule backchannels only; primary turns stay at known positions.

    Args:
        s1_turn_timing: one ``{'turn_id', 'start_token', 'end_token',
            'sub_segments': [{'start_token', 'end_token'}, ...]}`` per
            channel-A turn, in dialogue steps. See
            :func:`pass_1_anchor_primary_turns`.
        pass2_*: see :func:`pass_2_backchannel_placement`.

    Pass 3 is not run, because it would move the primary turns. Other
    arguments and the return value are as in
    :func:`generate_dialogue_three_pass`.
    """
    state = _parse_script(input_ids, special_tokens, turn_boundaries, verbose)

    placed_primary, diag_1 = pass_1_anchor_primary_turns(
        state.primary_turns, special_tokens, s1_turn_timing, verbose=verbose
    )
    _record_pass1(state, placed_primary, diag_1)

    _run_pass2(
        model, input_ids, special_tokens, state, device, verbose,
        search_stride=pass2_search_stride,
        scoring_context=pass2_scoring_context,
        bc_position_limit=pass2_bc_position_limit,
        bc_effect=pass2_bc_effect,
        max_bc_overlap_ratio=pass2_max_bc_overlap_ratio,
        score_only_pauses=pass2_score_only_pauses,
    )

    state.refined_placements = list(state.placed_primary) + list(state.placed_backchannels)
    state.diagnostics["pass3"] = {
        "pass": 3,
        "skipped": True,
        "reason": "bc_only_mode",
        "host_turns_refined": [],
        "propagated_shifts": [],
        "host_turns_skipped": [],
    }

    channel_a, channel_b = _final_channels(state, tokenizer, special_tokens)
    return channel_a, channel_b, state
