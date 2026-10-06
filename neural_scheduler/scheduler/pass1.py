"""Pass 1: onsets of the primary turns.

Turns are placed one at a time in ``turn_id`` order. Each turn is tried at every
onset in a window around an anchor just after the previously placed turn, and
committed at the onset with the lowest NLL. Turns not yet placed are absent
(silence) while a candidate is scored, and backchannels are removed from the
script the model sees.
"""

import logging
from typing import Dict, List, Tuple

from . import config
from .dialogue import build_dialogue_from_placements
from .scoring import compute_nll_for_placement
from .script_parse import strip_backchannels_from_script
from .search import scan_candidates
from .subsegments import (
    assemble_turn_from_sub_segments,
    split_turn_tokens_by_sub_segments,
    with_sub_segment_layout,
)
from .types import PlacedTurn, TurnInfo

logger = logging.getLogger(__name__)

_DIALOGUE_LENGTH_MARGIN = 20


def _end(placed: PlacedTurn) -> int:
    return placed.onset_step + placed.turn_info.duration


def pass_1_primary_turns(
    model,
    input_ids: List[int],
    primary_turns: List[TurnInfo],
    backchannel_turns: List[TurnInfo],
    special_tokens: Dict[str, int],
    search_window: int = config.PASS1.search_window,
    search_stride: int = config.PASS1.search_stride,
    scoring_context: int = config.PASS1.scoring_context,
    inter_turn_gap: int = config.PASS1.inter_turn_gap,
    device: str = "cuda",
    verbose: bool = False,
    schedule_subturns: bool = config.PASS1.schedule_subturns,
    speaker_1_subturn_min_pause: int = config.PASS1.speaker_1_subturn_min_pause,
    speaker_1_subturn_max_pause: int = config.PASS1.speaker_1_subturn_max_pause,
    speaker_2_subturn_min_pause: int = config.PASS1.speaker_2_subturn_min_pause,
    speaker_2_subturn_max_pause: int = config.PASS1.speaker_2_subturn_max_pause,
    subturn_search_stride: int = config.PASS1.subturn_search_stride,
    use_speaker_gap_constraints: bool = config.PASS1.use_speaker_gap_constraints,
    speaker_1_min_gap: int = config.PASS1.speaker_1_min_gap,
    speaker_1_max_gap: int = config.PASS1.speaker_1_max_gap,
    speaker_2_min_gap: int = config.PASS1.speaker_2_min_gap,
    speaker_2_max_gap: int = config.PASS1.speaker_2_max_gap,
    strong_overlap_mode: bool = config.PASS1.strong_overlap_mode,
    strong_overlap_min: float = config.PASS1.strong_overlap_min,
    strong_overlap_max: float = config.PASS1.strong_overlap_max,
) -> Tuple[List[PlacedTurn], Dict]:
    """Place every primary turn; returns ``(placed_turns, diagnostics)``.

    The onset window of a turn is built in this order:

    1. ``anchor +/- search_window`` where ``anchor`` is the end of the previously
       placed turn plus ``inter_turn_gap``.
    2. With ``use_speaker_gap_constraints``, intersect with ``[prev_end + min,
       prev_end + max]``. On a speaker switch the range is the incoming
       speaker's ``speaker_N_{min,max}_gap``; for a turn following one of the
       same speaker it is ``speaker_N_subturn_{min,max}_pause``. An empty
       intersection falls back to the gap range alone.
    3. With ``strong_overlap_mode``, a speaker switch replaces the window by the
       onsets at which a fraction in ``[strong_overlap_min, strong_overlap_max]``
       of the incoming turn overlaps the previous turn.
    4. The onset is never earlier than the end of any placed turn of the same speaker.

    With ``schedule_subturns``, a turn with several speech blocks is scheduled
    block by block: the first block's onset is searched over the window above,
    then the pause before each later block is searched over the speaker's
    ``[subturn_min_pause, subturn_max_pause]`` (a negative minimum is treated as
    0). The committed turn is a single token list with silence in the pauses.
    Without it, the turn's speech tokens are placed back to back.
    """
    silence_token_id = special_tokens["silence"]

    stripped_ids, gen_start = strip_backchannels_from_script(input_ids, backchannel_turns, special_tokens)
    script_ids = stripped_ids[:gen_start]
    sorted_turns = sorted(primary_turns, key=lambda t: t.turn_id)

    subturn_pause_ranges = {
        "A": (speaker_1_subturn_min_pause, speaker_1_subturn_max_pause),
        "B": (speaker_2_subturn_min_pause, speaker_2_subturn_max_pause),
    }
    speaker_gap_ranges = {
        "A": (speaker_1_min_gap, speaker_1_max_gap),
        "B": (speaker_2_min_gap, speaker_2_max_gap),
    }

    # Upper bound on the dialogue length; every candidate is padded to it.
    extra_subturn_slack = 0
    if schedule_subturns:
        max_pause = max(speaker_1_subturn_max_pause, speaker_2_subturn_max_pause)
        for t in primary_turns:
            extra_subturn_slack += max(0, max(1, len(t.sub_segments)) - 1) * max_pause
    total_dialogue_steps = (
        sum(t.duration for t in primary_turns)
        + len(primary_turns) * (inter_turn_gap + search_window)
        + extra_subturn_slack
        + _DIALOGUE_LENGTH_MARGIN
    )

    placed_turns: List[PlacedTurn] = []
    diagnostics = {
        "pass": 1,
        "num_turns": len(primary_turns),
        "schedule_subturns": schedule_subturns,
        "per_turn": [],
    }
    subturn_nll_detail: Dict[Tuple[int, int], List[float]] = {}

    if verbose:
        logger.info(f"Pass 1: scheduling {len(primary_turns)} primary turns "
                    f"(window +/-{search_window}, stride {search_stride}, "
                    f"schedule_subturns={schedule_subturns})")

    def score_placement(candidate: PlacedTurn, span_start: int, span_end: int) -> float:
        """NLL around ``[span_start, span_end)`` with ``candidate`` added to the placed turns."""
        dialogue = build_dialogue_from_placements(
            placed_turns + [candidate], total_dialogue_steps, silence_token_id
        )
        win_start = max(0, (span_start * 2) - scoring_context * 2)
        win_end = min(len(dialogue), (span_end * 2) + scoring_context * 2)
        total_nll, _ = compute_nll_for_placement(
            model, script_ids, dialogue, gen_start, special_tokens,
            device=device, scoring_window=(win_start, win_end),
        )
        return total_nll

    for turn in sorted_turns:
        ch = turn.channel

        anchor = _end(placed_turns[-1]) + inter_turn_gap if placed_turns else inter_turn_gap
        search_start = max(0, anchor - search_window)
        search_end = anchor + search_window

        if use_speaker_gap_constraints and placed_turns:
            prev_pt = placed_turns[-1]
            prev_end = _end(prev_pt)
            same_speaker = prev_pt.turn_info.channel == ch
            gap_min, gap_max = (subturn_pause_ranges if same_speaker else speaker_gap_ranges)[ch]
            gap_lower = prev_end + gap_min
            gap_upper = prev_end + gap_max

            search_start = max(search_start, gap_lower)
            search_end = min(search_end, gap_upper)
            if search_start > search_end:
                search_start = gap_lower
                search_end = gap_upper

        strong_overlap_applied = False
        if strong_overlap_mode and placed_turns:
            prev_pt = placed_turns[-1]
            if prev_pt.turn_info.channel != ch:
                prev_end = _end(prev_pt)
                prev_duration = prev_pt.turn_info.duration
                # The overlap cannot be longer than the previous turn.
                max_feasible_frac = min(1.0, prev_duration / max(turn.duration, 1))
                frac_max = min(strong_overlap_max, max_feasible_frac)
                frac_min = min(strong_overlap_min, frac_max)
                overlap_min_steps = int(frac_min * turn.duration)
                overlap_max_steps = int(frac_max * turn.duration + 0.9999)

                search_start = prev_end - overlap_max_steps
                search_end = max(prev_end - overlap_min_steps, search_start)
                strong_overlap_applied = True
                if verbose:
                    logger.info(f"  Turn {turn.turn_id} (ch={ch}): strong-overlap window "
                                f"[{search_start}, {search_end}]")

        for pt in placed_turns:
            if pt.turn_info.channel == ch:
                search_start = max(search_start, _end(pt))
        if search_end < search_start:
            search_end = search_start

        prev_end_step = _end(placed_turns[-1]) if placed_turns else 0
        gap_constraint_applied = (
            use_speaker_gap_constraints
            and len(placed_turns) > 0
            and placed_turns[-1].turn_info.channel != ch
        )

        # ---- the turn as one contiguous block ----
        if not schedule_subturns or not turn.sub_segments or len(turn.sub_segments) <= 1:
            candidates = list(range(search_start, search_end + 1, search_stride))
            if not candidates:
                candidates = [anchor]

            best_onset, best_nll, nll_scores = scan_candidates(
                candidates,
                lambda onset: score_placement(
                    PlacedTurn(turn_info=turn, onset_step=onset), onset, onset + turn.duration
                ),
                default=anchor,
            )

            placed_turns.append(PlacedTurn(
                turn_info=turn,
                onset_step=best_onset,
                gap_to_previous=best_onset - prev_end_step,
            ))
            diagnostics["per_turn"].append({
                "turn_id": turn.turn_id,
                "channel": ch,
                "script_onset": anchor,
                "best_onset": best_onset,
                "shift": best_onset - anchor,
                "best_nll": best_nll,
                "nll_scores": nll_scores,
                "candidates": candidates,
                "duration": turn.duration,
                "mode": "single_block",
                "search_bounds": (candidates[0], candidates[-1]),
                "gap_constraint_applied": gap_constraint_applied,
                "strong_overlap_applied": strong_overlap_applied,
            })
            if verbose:
                logger.info(f"  Turn {turn.turn_id} (ch={ch}): anchor={anchor}, "
                            f"searched [{candidates[0]}, {candidates[-1]}] -> {best_onset} "
                            f"(shift={best_onset - anchor:+d}), NLL={best_nll:.4f}")
            continue

        # ---- the turn block by block ----
        allocations = split_turn_tokens_by_sub_segments(turn)
        n_subsegs = len(allocations)

        pause_min, pause_max = subturn_pause_ranges[ch]
        pause_min = max(pause_min, 0)
        pause_max = max(pause_max, pause_min)

        committed_onsets: List[int] = []
        per_subseg_diag: List[Dict] = []

        def score_layout(ss_idx: int, onset: int) -> float:
            """Score block ``ss_idx`` at ``onset``; later blocks follow at the minimum pause.

            The first block is scored over the whole turn, a later block over itself.
            """
            onsets = committed_onsets[:ss_idx] + [onset]
            current_pos = onset + len(allocations[ss_idx])
            for k in range(ss_idx + 1, n_subsegs):
                onsets.append(current_pos + pause_min)
                current_pos = onsets[-1] + len(allocations[k])

            tokens, meta, t_onset, t_end = assemble_turn_from_sub_segments(allocations, onsets, silence_token_id)
            candidate = PlacedTurn(turn_info=with_sub_segment_layout(turn, tokens, meta), onset_step=t_onset)
            if ss_idx == 0:
                return score_placement(candidate, t_onset, t_end)
            return score_placement(candidate, onset, onset + len(allocations[ss_idx]))

        ss0_candidates = list(range(search_start, search_end + 1, search_stride))
        if not ss0_candidates:
            ss0_candidates = [max(anchor, search_start)]

        best_onset_ss0, best_nll_ss0, nll_scores_ss0 = scan_candidates(
            ss0_candidates, lambda onset: score_layout(0, onset), default=ss0_candidates[0]
        )
        committed_onsets.append(best_onset_ss0)
        subturn_nll_detail[(turn.turn_id, 0)] = nll_scores_ss0
        per_subseg_diag.append({
            "sub_seg_index": 0,
            "mode": "onset",
            "candidates": ss0_candidates,
            "nll_scores": nll_scores_ss0,
            "best_onset": best_onset_ss0,
            "best_nll": best_nll_ss0,
        })

        for ss_idx in range(1, n_subsegs):
            prev_block_end = committed_onsets[ss_idx - 1] + len(allocations[ss_idx - 1])

            # The block may not start before any placed turn of the same speaker has ended.
            same_speaker_end = prev_block_end
            for pt in placed_turns:
                if pt.turn_info.channel == ch:
                    same_speaker_end = max(same_speaker_end, _end(pt))
            local_min_pause = max(pause_min, same_speaker_end - prev_block_end)
            local_max_pause = max(pause_max, local_min_pause)

            pause_candidates = list(range(local_min_pause, local_max_pause + 1, max(1, subturn_search_stride)))
            if not pause_candidates:
                pause_candidates = [local_min_pause]

            best_pause, best_nll_ss, nll_scores_ss = scan_candidates(
                pause_candidates,
                lambda pause: score_layout(ss_idx, prev_block_end + pause),
                default=pause_candidates[0],
            )
            committed_onsets.append(prev_block_end + best_pause)
            subturn_nll_detail[(turn.turn_id, ss_idx)] = nll_scores_ss
            per_subseg_diag.append({
                "sub_seg_index": ss_idx,
                "mode": "pause",
                "pause_candidates": pause_candidates,
                "nll_scores": nll_scores_ss,
                "best_pause": best_pause,
                "best_onset": prev_block_end + best_pause,
                "best_nll": best_nll_ss,
            })

        final_tokens, final_meta, final_onset, final_end = assemble_turn_from_sub_segments(
            allocations, committed_onsets, silence_token_id
        )
        placed_turns.append(PlacedTurn(
            turn_info=with_sub_segment_layout(turn, final_tokens, final_meta),
            onset_step=final_onset,
            gap_to_previous=final_onset - prev_end_step,
        ))
        diagnostics["per_turn"].append({
            "turn_id": turn.turn_id,
            "channel": ch,
            "script_onset": anchor,
            "best_onset": final_onset,
            "shift": final_onset - anchor,
            "best_nll": per_subseg_diag[-1]["best_nll"],
            "nll_scores": per_subseg_diag[0]["nll_scores"],
            "candidates": per_subseg_diag[0]["candidates"],
            "duration": len(final_tokens),
            "mode": "sub_turn",
            "pause_range": (pause_min, pause_max),
            "num_sub_segments": n_subsegs,
            "sub_segment_onsets": committed_onsets,
            "sub_segment_details": per_subseg_diag,
            "gap_constraint_applied": gap_constraint_applied,
        })
        if verbose:
            logger.info(f"  Turn {turn.turn_id} (ch={ch}): anchor={anchor}, {n_subsegs} blocks at "
                        f"{committed_onsets}, span [{final_onset}, {final_end}]")

    diagnostics["subturn_nll_detail"] = subturn_nll_detail
    return placed_turns, diagnostics
