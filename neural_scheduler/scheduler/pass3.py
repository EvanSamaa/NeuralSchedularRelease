"""Pass 3: re-timing what follows a backchannel.

Pass 2 leaves the host turn's later speech, and the next speaker's turn, where
pass 1 put them. Pass 3 revisits each host that received backchannels: the
speech block that follows a backchannel (a "breakpoint") is searched again,
moving everything after it rigidly, and the resulting shift is propagated to
later turns.

A breakpoint is either a later speech block of the host itself (``subseg``) or
the next primary turn (``next_turn``), used when the backchannel falls in the
host's last block or after the host's end.
"""

import copy
import logging
from typing import Dict, List, Tuple

from . import config
from .dialogue import build_dialogue_from_sub_segments_and_turns
from .scoring import compute_nll_for_placement
from .script_parse import generation_start_index
from .search import scan_candidates
from .subsegments import (
    assemble_turn_from_sub_segments,
    compute_original_sub_segment_onsets,
    split_turn_into_sub_segments,
)
from .types import PlacedSubSegment, PlacedTurn

logger = logging.getLogger(__name__)

_DIALOGUE_LENGTH_MARGIN = 50


def _end(placed) -> int:
    """End step of a PlacedTurn or PlacedSubSegment."""
    if isinstance(placed, PlacedSubSegment):
        return placed.onset_step + placed.duration
    return placed.onset_step + placed.turn_info.duration


def _claim_backchannels(
    host_bcs: List[PlacedTurn],
    sub_segs: List[PlacedSubSegment],
    anchors: List[Tuple[str, int, int]],
    host_turn_end: int,
    has_next_turn: bool,
    verbose: bool,
) -> Dict[int, List[int]]:
    """Assign each backchannel to the anchor that should be re-timed because of it.

    Returns ``anchor index -> indices into host_bcs``. A backchannel goes to a
    ``subseg`` anchor when it starts in the gap before that block, straddles the
    end of the block before it, or lies inside the block before it (the last of
    these only if the block before it is not the host's last block with a next
    turn available). Otherwise it goes to the ``next_turn`` anchor when it starts
    inside the host's last block, or after the host's end and before the next
    turn. A backchannel matching neither stays where pass 2 put it.
    """
    claims = {i: [] for i in range(len(anchors))}
    last_block = sub_segs[-1]

    for bc_i, pbc in enumerate(host_bcs):
        bc_onset = pbc.onset_step
        bc_end = _end(pbc)
        assigned = False

        for a_idx, (a_type, _, ss_idx) in enumerate(anchors):
            if a_type != "subseg":
                continue
            prev_block = sub_segs[ss_idx - 1]
            prev_end = _end(prev_block)
            in_gap = prev_end <= bc_onset < sub_segs[ss_idx].onset_step
            straddles = bc_onset < prev_end < bc_end
            inside_prev = prev_block.onset_step <= bc_onset < prev_end
            if in_gap or straddles or (inside_prev and (ss_idx < len(sub_segs) - 1 or not has_next_turn)):
                claims[a_idx].append(bc_i)
                assigned = True
                break
        if assigned:
            continue

        for a_idx, (a_type, a_ref_onset, _) in enumerate(anchors):
            if a_type != "next_turn":
                continue
            in_last_block = last_block.onset_step <= bc_onset < _end(last_block)
            after_host = host_turn_end <= bc_onset < a_ref_onset
            if in_last_block or after_host:
                claims[a_idx].append(bc_i)
                assigned = True
                break

        if not assigned and verbose:
            logger.info(f"    Backchannel {bc_i} (onset={bc_onset}) stays where pass 2 placed it")

    return claims


def pass_3_refinement(
    model,
    input_ids: List[int],
    placed_primary: List[PlacedTurn],
    placed_backchannels: List[PlacedTurn],
    special_tokens: Dict[str, int],
    scoring_context: int = config.PASS3.scoring_context,
    max_extra_gap: int = config.PASS3.max_extra_gap,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[List[PlacedTurn], Dict]:
    """Re-time hosts and following turns; returns ``(all_placements, diagnostics)``.

    The returned list holds copies of every primary turn and backchannel, sorted
    by onset. For each breakpoint the onset is searched from the end of whatever
    precedes it on its channel up to ``current onset + duration of the
    backchannels before it + max_extra_gap``.
    """
    silence_token_id = special_tokens["silence"]
    gen_start = generation_start_index(input_ids, special_tokens)
    script_ids = list(input_ids)[:gen_start]

    refined_primary = [copy.deepcopy(pt) for pt in placed_primary]
    refined_bc = [copy.deepcopy(pbc) for pbc in placed_backchannels]
    refined_primary.sort(key=lambda pt: pt.onset_step)

    max_end = max((_end(pt) for pt in refined_primary + refined_bc), default=100)
    total_dialogue_steps = max_end + _DIALOGUE_LENGTH_MARGIN

    diagnostics = {
        "pass": 3,
        "method": "breakpoint_retiming",
        "host_turns_refined": [],
        "propagated_shifts": [],
        "host_turns_skipped": [],
    }
    # Turns placed by a next_turn breakpoint are not shifted again by propagation.
    directly_retimed_turn_ids = set()

    def score(sub_segments: List[PlacedSubSegment], turns: List[PlacedTurn], moving: List) -> float:
        """NLL over the span of ``moving`` (plus context) for the given layout."""
        dialogue = build_dialogue_from_sub_segments_and_turns(
            placed_sub_segments=sub_segments,
            other_placed_turns=turns,
            total_length=total_dialogue_steps,
            silence_token_id=silence_token_id,
        )
        win_start = max(0, (min(m.onset_step for m in moving) * 2) - scoring_context * 2)
        win_end = min(len(dialogue), (max(_end(m) for m in moving) * 2) + scoring_context * 2)
        total_nll, _ = compute_nll_for_placement(
            model, script_ids, dialogue, gen_start, special_tokens,
            device=device, scoring_window=(win_start, win_end),
        )
        return total_nll

    def search_range(lower_bound: int, current_onset: int, frozen_bc_duration: int) -> Tuple[List[int], int]:
        upper_bound = current_onset + frozen_bc_duration + max_extra_gap
        if lower_bound > upper_bound:
            upper_bound = lower_bound + max_extra_gap
        return list(range(lower_bound, upper_bound + 1)) or [current_onset], upper_bound

    for p_idx, pt in enumerate(refined_primary):
        turn_id = pt.turn_info.turn_id
        host_channel = pt.turn_info.channel
        host_bcs = [pbc for pbc in refined_bc if pbc.turn_info.host_turn_id == turn_id]
        if not host_bcs:
            continue
        host_bcs.sort(key=lambda pbc: pbc.onset_step)

        if verbose:
            logger.info(f"  Host turn {turn_id} (ch={host_channel}): backchannels at "
                        f"{[pbc.onset_step for pbc in host_bcs]}")

        split = split_turn_into_sub_segments(pt.turn_info)
        all_sub_segs = [
            PlacedSubSegment(
                turn_id=turn_id,
                channel=host_channel,
                sub_segment_index=ss["index"],
                onset_step=onset,
                token_ids=ss["token_ids"],
                duration=ss["duration"],
                is_frozen=False,
            )
            for ss, onset in zip(split, compute_original_sub_segment_onsets(pt.onset_step, split))
        ]

        next_primary_idx = p_idx + 1 if p_idx + 1 < len(refined_primary) else None

        # Anchors: ('subseg', onset, block index) or ('next_turn', onset, index into refined_primary).
        anchors = [("subseg", all_sub_segs[i].onset_step, i) for i in range(1, len(all_sub_segs))]
        if next_primary_idx is not None:
            anchors.append(("next_turn", refined_primary[next_primary_idx].onset_step, next_primary_idx))

        claims = _claim_backchannels(
            host_bcs, all_sub_segs, anchors, _end(pt), next_primary_idx is not None, verbose
        )
        active_anchors = [(a_idx, anchors[a_idx], claims[a_idx]) for a_idx in range(len(anchors)) if claims[a_idx]]

        if not active_anchors:
            diagnostics["host_turns_skipped"].append({
                "turn_id": turn_id,
                "channel": host_channel,
                "reason": "no_breakpoints",
                "num_sub_segments": len(all_sub_segs),
                "num_bcs": len(host_bcs),
                "sub_segment_onsets": [pss.onset_step for pss in all_sub_segs],
                "sub_segment_ends": [_end(pss) for pss in all_sub_segs],
                "bc_onsets": [pbc.onset_step for pbc in host_bcs],
                "bc_ends": [_end(pbc) for pbc in host_bcs],
            })
            continue

        other_primary_turns = [rp for rp in refined_primary if rp.turn_info.turn_id != turn_id]
        other_bcs = [pbc for pbc in refined_bc if pbc.turn_info.host_turn_id != turn_id]

        frozen_bc_indices = set()
        host_turn_diag = {
            "turn_id": turn_id,
            "channel": host_channel,
            "num_sub_segments": len(all_sub_segs),
            "num_breakpoints": len(active_anchors),
            "num_bcs": len(host_bcs),
            "breakpoint_results": [],
        }

        for bp_order, (a_idx, (a_type, _, a_meta), bc_indices_before) in enumerate(active_anchors):
            frozen_bc_indices.update(bc_indices_before)
            frozen_bc_duration = sum(host_bcs[i].turn_info.duration for i in frozen_bc_indices)

            if a_type == "subseg":
                # Move this block, every later block of the host and the host's
                # not-yet-frozen backchannels as one rigid group.
                ss_idx = a_meta
                bp_orig_onset = all_sub_segs[ss_idx].onset_step
                frozen_sub_segs = all_sub_segs[:ss_idx]
                moveable_sub_segs = all_sub_segs[ss_idx:]
                frozen_host_bcs = [host_bcs[i] for i in range(len(host_bcs)) if i in frozen_bc_indices]
                moveable_host_bcs = [host_bcs[i] for i in range(len(host_bcs)) if i not in frozen_bc_indices]
                sub_seg_offsets = [pss.onset_step - bp_orig_onset for pss in moveable_sub_segs]
                bc_offsets = [pbc.onset_step - bp_orig_onset for pbc in moveable_host_bcs]

                lower_bound = max([0] + [_end(fss) for fss in frozen_sub_segs])
                candidates, upper_bound = search_range(lower_bound, bp_orig_onset, frozen_bc_duration)

                def score_onset(cand_onset: int) -> float:
                    candidate_sub_segs = list(frozen_sub_segs)
                    for pss, offset in zip(moveable_sub_segs, sub_seg_offsets):
                        candidate_sub_segs.append(PlacedSubSegment(
                            turn_id=pss.turn_id,
                            channel=pss.channel,
                            sub_segment_index=pss.sub_segment_index,
                            onset_step=cand_onset + offset,
                            token_ids=pss.token_ids,
                            duration=pss.duration,
                            is_frozen=False,
                        ))
                    candidate_bcs = list(frozen_host_bcs)
                    for pbc, offset in zip(moveable_host_bcs, bc_offsets):
                        shifted_bc = copy.deepcopy(pbc)
                        shifted_bc.onset_step = cand_onset + offset
                        candidate_bcs.append(shifted_bc)
                    return score(
                        candidate_sub_segs,
                        other_primary_turns + other_bcs + candidate_bcs,
                        candidate_sub_segs + candidate_bcs,
                    )

                best_onset, best_nll, nll_scores = scan_candidates(candidates, score_onset, default=bp_orig_onset)

                for pss, offset in zip(moveable_sub_segs, sub_seg_offsets):
                    pss.onset_step = best_onset + offset
                for pbc, offset in zip(moveable_host_bcs, bc_offsets):
                    pbc.onset_step = best_onset + offset

                host_turn_diag["breakpoint_results"].append({
                    "breakpoint_order": bp_order,
                    "anchor_index": a_idx,
                    "anchor_type": "subseg",
                    "sub_segment_index": ss_idx,
                    "bc_indices_before": bc_indices_before,
                    "original_onset": bp_orig_onset,
                    "best_onset": best_onset,
                    "shift": best_onset - bp_orig_onset,
                    "best_nll": best_nll,
                    "nll_scores": nll_scores,
                    "candidates": candidates,
                    "search_range": (lower_bound, upper_bound),
                    "num_moveable_sub_segs": len(moveable_sub_segs),
                    "num_moveable_bcs": len(moveable_host_bcs),
                })

            else:
                # Move the next primary turn and its own backchannels as one
                # rigid group; the whole host is fixed.
                next_pt = refined_primary[a_meta]
                next_turn_id = next_pt.turn_info.turn_id
                next_channel = next_pt.turn_info.channel
                next_turn_orig_onset = next_pt.onset_step
                next_turn_bcs = [pbc for pbc in refined_bc if pbc.turn_info.host_turn_id == next_turn_id]
                next_turn_bc_offsets = [pbc.onset_step - next_turn_orig_onset for pbc in next_turn_bcs]

                # Not before the end of anything already on the next turn's channel.
                same_channel = [rp for rp in refined_primary[:p_idx + 1] if rp.turn_info.channel == next_channel]
                if host_channel == next_channel:
                    same_channel = same_channel + all_sub_segs
                same_channel = same_channel + [pbc for pbc in refined_bc if pbc.turn_info.channel == next_channel]
                lower_bound = max([0] + [_end(item) for item in same_channel])
                candidates, upper_bound = search_range(lower_bound, next_turn_orig_onset, frozen_bc_duration)

                fixed_turns = [
                    rp for rp in refined_primary
                    if rp.turn_info.turn_id != turn_id and rp.turn_info.turn_id != next_turn_id
                ]
                fixed_bcs = [
                    pbc for pbc in refined_bc
                    if pbc.turn_info.host_turn_id != turn_id and pbc.turn_info.host_turn_id != next_turn_id
                ]

                def score_onset(cand_onset: int) -> float:
                    shifted_next = copy.deepcopy(next_pt)
                    shifted_next.onset_step = cand_onset
                    shifted_next_bcs = []
                    for pbc, offset in zip(next_turn_bcs, next_turn_bc_offsets):
                        shifted_bc = copy.deepcopy(pbc)
                        shifted_bc.onset_step = cand_onset + offset
                        shifted_next_bcs.append(shifted_bc)
                    return score(
                        all_sub_segs,
                        fixed_turns + [shifted_next] + fixed_bcs + host_bcs + shifted_next_bcs,
                        all_sub_segs + host_bcs + [shifted_next] + shifted_next_bcs,
                    )

                best_onset, best_nll, nll_scores = scan_candidates(
                    candidates, score_onset, default=next_turn_orig_onset
                )

                next_pt.onset_step = best_onset
                directly_retimed_turn_ids.add(next_turn_id)
                for pbc, offset in zip(next_turn_bcs, next_turn_bc_offsets):
                    pbc.onset_step = best_onset + offset

                host_turn_diag["breakpoint_results"].append({
                    "breakpoint_order": bp_order,
                    "anchor_index": a_idx,
                    "anchor_type": "next_turn",
                    "next_turn_id": next_turn_id,
                    "next_turn_channel": next_channel,
                    "bc_indices_before": bc_indices_before,
                    "original_onset": next_turn_orig_onset,
                    "best_onset": best_onset,
                    "shift": best_onset - next_turn_orig_onset,
                    "best_nll": best_nll,
                    "nll_scores": nll_scores,
                    "candidates": candidates,
                    "search_range": (lower_bound, upper_bound),
                })

            if verbose:
                result = host_turn_diag["breakpoint_results"][-1]
                logger.info(f"    Breakpoint {a_idx} ({a_type}): searched {result['search_range']}, "
                            f"onset {result['original_onset']} -> {result['best_onset']}, "
                            f"NLL={result['best_nll']:.4f}")

        # Rebuild the host as one token list if any of its blocks moved.
        original_turn_end = _end(pt)
        new_turn_end = original_turn_end
        if any(a_type == "subseg" for _, (a_type, _, _), _ in active_anchors):
            all_sub_segs.sort(key=lambda s: s.onset_step)
            tokens, meta, first_onset, _ = assemble_turn_from_sub_segments(
                [pss.token_ids for pss in all_sub_segs],
                [pss.onset_step for pss in all_sub_segs],
                silence_token_id,
            )
            new_turn_end = _end(all_sub_segs[-1])
            pt.onset_step = first_onset
            pt.turn_info.speech_token_ids = tokens
            pt.turn_info.duration = len(tokens)
            pt.turn_info.sub_segments = meta
        total_shift = new_turn_end - original_turn_end

        next_turn_shift = 0
        for result in host_turn_diag["breakpoint_results"]:
            if result["anchor_type"] == "next_turn":
                next_turn_shift = result["shift"]

        host_turn_diag["original_turn_end"] = original_turn_end
        host_turn_diag["new_turn_end"] = new_turn_end
        host_turn_diag["total_shift"] = total_shift
        host_turn_diag["next_turn_shift"] = next_turn_shift
        diagnostics["host_turns_refined"].append(host_turn_diag)

        # Propagate: the host's own shift moves every later turn, the next
        # turn's shift moves every turn after it. Backchannels move with their host.
        propagations = []
        if total_shift != 0:
            propagations.append(("intra_host", total_shift, p_idx + 1, (turn_id,)))
        if next_turn_shift != 0 and next_primary_idx is not None:
            next_turn_id = refined_primary[next_primary_idx].turn_info.turn_id
            propagations.append(("next_turn", next_turn_shift, next_primary_idx + 1, (turn_id, next_turn_id)))

        for shift_type, shift, first_idx, excluded_hosts in propagations:
            affected_turn_ids = []
            for later_pt in refined_primary[first_idx:]:
                if later_pt.turn_info.turn_id in directly_retimed_turn_ids:
                    continue
                later_pt.onset_step += shift
                affected_turn_ids.append(later_pt.turn_info.turn_id)

            affected_set = set(affected_turn_ids)
            for pbc in refined_bc:
                host_id = pbc.turn_info.host_turn_id
                if host_id not in excluded_hosts and host_id in affected_set:
                    pbc.onset_step += shift

            record = {"source_turn": turn_id, "shift_type": shift_type}
            if shift_type == "next_turn":
                record["next_turn_id"] = excluded_hosts[1]
            record["shift"] = shift
            record["affected_turns"] = affected_turn_ids
            diagnostics["propagated_shifts"].append(record)

        if verbose:
            logger.info(f"    Host turn {turn_id}: end {original_turn_end} -> {new_turn_end}, "
                        f"next-turn shift {next_turn_shift:+d}")

    all_refined = refined_primary + refined_bc
    all_refined.sort(key=lambda pt: pt.onset_step)
    return all_refined, diagnostics
