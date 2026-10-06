"""Pass 2: onsets of the backchannels, and optional pauses in the turns that host them.

Each backchannel is tried at every allowed onset inside its host turn, combined
with each allowed pause strategy, and scored by how much it changes the NLL
relative to the same dialogue without it. The best (onset, strategy) pair is
committed before the next backchannel is considered.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

from . import config
from .dialogue import build_dialogue_from_placements
from .scoring import compute_nll_for_placement
from .script_parse import generation_start_index
from .subsegments import build_host_turn_with_pause
from .types import PlacedTurn, TurnInfo

logger = logging.getLogger(__name__)

STRATEGY_NO_PAUSE = "no_pause"  # the host keeps talking over the backchannel
STRATEGY_FULL_PAUSE = "full_pause"  # the host pauses for the backchannel's duration
STRATEGY_HALF_PAUSE = "half_pause"  # the host pauses for half of it

_ALL_STRATEGIES = [STRATEGY_NO_PAUSE, STRATEGY_FULL_PAUSE, STRATEGY_HALF_PAUSE]
_BC_EFFECT_CODE_MAP = {
    "np": STRATEGY_NO_PAUSE,
    "fp": STRATEGY_FULL_PAUSE,
    "hp": STRATEGY_HALF_PAUSE,
}
_DIALOGUE_LENGTH_MARGIN = 30


def parse_bc_effect(bc_effect: Optional[str] = None) -> Optional[List[str]]:
    """Turn ``"fp+hp"`` into ``['full_pause', 'half_pause']``.

    Codes are ``np``, ``fp`` and ``hp``, joined by ``+``. ``None`` or an empty
    string returns ``None``, meaning every strategy is allowed.
    """
    if bc_effect is None or bc_effect.strip() == "":
        return None
    allowed = []
    for code in (c.strip().lower() for c in bc_effect.split("+")):
        if code not in _BC_EFFECT_CODE_MAP:
            raise ValueError(
                f"Unknown bc_effect code '{code}'. Valid codes: {list(_BC_EFFECT_CODE_MAP.keys())}"
            )
        strategy = _BC_EFFECT_CODE_MAP[code]
        if strategy not in allowed:
            allowed.append(strategy)
    return allowed or None


def pass_2_backchannel_placement(
    model,
    input_ids: List[int],
    placed_primary: List[PlacedTurn],
    backchannel_units: List[TurnInfo],
    special_tokens: Dict[str, int],
    search_stride: int = config.PASS2.search_stride,
    scoring_context: int = config.PASS2.scoring_context,
    bc_position_limit: float = config.PASS2.bc_position_limit,
    bc_effect: Optional[str] = config.PASS2.bc_effect,
    max_bc_overlap_ratio: float = config.PASS2.max_bc_overlap_ratio,
    score_only_pauses: bool = config.PASS2.score_only_pauses,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[List[PlacedTurn], List[PlacedTurn], Dict]:
    """Place the backchannels; returns ``(placed_primary, placed_backchannels, diagnostics)``.

    ``placed_primary`` is a modified copy of the input: a host that received a
    pause has silence inserted, and everything that started at or after the
    pause on either channel is shifted by its length.

    Args:
        bc_position_limit: fraction of the host's tail in which a backchannel
            may start (1.0 = anywhere in the host, 0.25 = its last quarter).
        bc_effect: pause strategies to consider, see :func:`parse_bc_effect`.
            Strategies other than ``no_pause`` are only tried for the first
            backchannel on a host.
        max_bc_overlap_ratio: fraction of the backchannel that must lie inside
            the host; the latest onset is ``host_end - round(duration * ratio)``.
        score_only_pauses: when ``bc_effect`` excludes ``np``, use the pause
            only to score onsets and leave the host turn as it was.
    """
    silence_token_id = special_tokens["silence"]
    gen_start = generation_start_index(input_ids, special_tokens)
    script_ids = list(input_ids)[:gen_start]

    placed_primary = [copy.deepcopy(pt) for pt in placed_primary]

    max_end = max((pt.onset_step + pt.turn_info.duration for pt in placed_primary), default=0)
    max_bc_dur = max((bc.duration for bc in backchannel_units), default=0)
    total_dialogue_steps = max(max_end, 0) + max_bc_dur + _DIALOGUE_LENGTH_MARGIN

    allowed_strategies = parse_bc_effect(bc_effect)
    if allowed_strategies is not None:
        active_strategy_names = [s for s in _ALL_STRATEGIES if s in allowed_strategies]
    else:
        active_strategy_names = list(_ALL_STRATEGIES)
    commit_pauses = not (
        score_only_pauses
        and allowed_strategies is not None
        and STRATEGY_NO_PAUSE not in allowed_strategies
    )

    placed_backchannels: List[PlacedTurn] = []
    diagnostics = {
        "pass": 2,
        "scoring_method": "multi_strategy_nll_difference_sequential",
        "strategies": active_strategy_names,
        "bc_effect": bc_effect,
        "num_backchannel_units": len(backchannel_units),
        "per_backchannel": [],
    }
    if verbose:
        logger.info(f"Pass 2: placing {len(backchannel_units)} backchannel(s), "
                    f"strategies {active_strategy_names}")

    primary_by_id = {pt.turn_info.turn_id: pt for pt in placed_primary}
    placed_bc_by_host: Dict[int, List[Tuple[int, int]]] = {}  # host turn id -> [(onset, duration)]
    hosts_with_backchannel = set()

    for bc_unit in backchannel_units:
        host_turn_id = bc_unit.host_turn_id
        if host_turn_id not in primary_by_id:
            if verbose:
                logger.warning(f"  Backchannel {bc_unit.turn_id} (ch={bc_unit.channel}): "
                               f"host turn {host_turn_id} not found, skipping")
            continue

        host = primary_by_id[host_turn_id]
        host_start = host.onset_step
        host_end = host.onset_step + host.turn_info.duration
        bc_duration = bc_unit.duration

        max_bc_start = host_end - int(round(bc_duration * max_bc_overlap_ratio))
        min_bc_start = max(0, host_start + int((host_end - host_start) * (1.0 - bc_position_limit)))
        prev_bcs = placed_bc_by_host.get(host_turn_id, [])
        for prev_onset, prev_dur in prev_bcs:
            min_bc_start = max(min_bc_start, prev_onset + prev_dur)

        candidates = list(range(min_bc_start, max_bc_start + 1, search_stride))
        if not candidates:
            candidates = [max(min_bc_start, bc_unit.start_token)]

        is_first_bc_on_host = host_turn_id not in hosts_with_backchannel
        if is_first_bc_on_host:
            strategies = [
                (STRATEGY_NO_PAUSE, 0),
                (STRATEGY_FULL_PAUSE, bc_duration),
                (STRATEGY_HALF_PAUSE, bc_duration // 2),
            ]
            if allowed_strategies is not None:
                strategies = [(s, d) for s, d in strategies if s in allowed_strategies] or [(STRATEGY_NO_PAUSE, 0)]
        else:
            strategies = [(STRATEGY_NO_PAUSE, 0)]

        # One scoring window for all candidates, wide enough for the latest
        # onset and for a full pause in the host.
        latest_bc_end = max_bc_start + bc_duration
        fixed_win_start = max(0, (host_start * 2) - scoring_context * 2)
        fixed_win_end = (max(host_end + bc_duration, latest_bc_end) + scoring_context) * 2
        effective_total = max(total_dialogue_steps, fixed_win_end // 2 + 5)

        baseline_dialogue = build_dialogue_from_placements(
            list(placed_primary) + list(placed_backchannels), effective_total, silence_token_id
        )
        actual_win_end = min(fixed_win_end, len(baseline_dialogue))
        baseline_nll, _ = compute_nll_for_placement(
            model, script_ids, baseline_dialogue, gen_start, special_tokens,
            device=device, scoring_window=(fixed_win_start, actual_win_end),
        )

        best_delta = float("inf")
        best_onset = candidates[0]
        best_strategy = STRATEGY_NO_PAUSE
        best_pause_duration = 0
        strategy_results = {s: {"nll_deltas": [], "nll_absolutes": []} for s, _ in strategies}
        all_results = []

        for cand_onset in candidates:
            for strategy_name, pause_dur in strategies:
                if pause_dur > 0:
                    modified_host = build_host_turn_with_pause(host, cand_onset, pause_dur, silence_token_id)
                    temp_primary = [
                        modified_host if pp.turn_info.turn_id == host_turn_id else pp
                        for pp in placed_primary
                    ]
                else:
                    temp_primary = list(placed_primary)

                candidate_dialogue = build_dialogue_from_placements(
                    temp_primary + list(placed_backchannels) + [PlacedTurn(turn_info=bc_unit, onset_step=cand_onset)],
                    effective_total, silence_token_id,
                )
                candidate_nll, _ = compute_nll_for_placement(
                    model, script_ids, candidate_dialogue, gen_start, special_tokens,
                    device=device,
                    scoring_window=(fixed_win_start, min(actual_win_end, len(candidate_dialogue))),
                )
                delta = candidate_nll - baseline_nll

                strategy_results[strategy_name]["nll_deltas"].append(delta)
                strategy_results[strategy_name]["nll_absolutes"].append(candidate_nll)
                all_results.append((cand_onset, strategy_name, candidate_nll))

                if delta < best_delta:
                    best_delta = delta
                    best_onset = cand_onset
                    best_strategy = strategy_name
                    best_pause_duration = pause_dur

        if best_pause_duration > 0 and commit_pauses:
            modified_host = build_host_turn_with_pause(host, best_onset, best_pause_duration, silence_token_id)
            for i, pp in enumerate(placed_primary):
                if pp.turn_info.turn_id == host_turn_id:
                    placed_primary[i] = modified_host
                    primary_by_id[host_turn_id] = modified_host
                    break

            # The pause delays everything scheduled at or after it, on both channels.
            shifted_primary_ids = set()
            for pp in placed_primary:
                if pp.turn_info.turn_id == host_turn_id:
                    continue
                if pp.onset_step >= best_onset:
                    pp.onset_step += best_pause_duration
                    shifted_primary_ids.add(pp.turn_info.turn_id)
                    primary_by_id[pp.turn_info.turn_id] = pp
            for pbc in placed_backchannels:
                if pbc.onset_step >= best_onset:
                    pbc.onset_step += best_pause_duration
            for shifted_id in shifted_primary_ids:
                if shifted_id in placed_bc_by_host:
                    placed_bc_by_host[shifted_id] = [
                        (onset + best_pause_duration, dur) for onset, dur in placed_bc_by_host[shifted_id]
                    ]
            if verbose:
                logger.info(f"    Host turn {host_turn_id}: {best_pause_duration}-step pause at offset "
                            f"{best_onset - host_start}; shifted turns {sorted(shifted_primary_ids)}")

        bc_offset = best_onset - host_start
        placed_backchannels.append(PlacedTurn(
            turn_info=bc_unit,
            onset_step=best_onset,
            bc_offset_from_host_start=bc_offset,
        ))
        placed_bc_by_host.setdefault(host_turn_id, []).append((best_onset, bc_duration))
        hosts_with_backchannel.add(host_turn_id)

        winning_absolute = None
        for onset, strategy_name, absolute_nll in all_results:
            if onset == best_onset and strategy_name == best_strategy:
                winning_absolute = absolute_nll
                break

        # 'nll_scores' / 'nll_deltas' / 'nll_absolutes' report one strategy only:
        # no_pause when it was tried, otherwise the winning one.
        reported = STRATEGY_NO_PAUSE if STRATEGY_NO_PAUSE in strategy_results else best_strategy
        diagnostics["per_backchannel"].append({
            "turn_id": bc_unit.turn_id,
            "bc_group_index": bc_unit.bc_group_index,
            "channel": bc_unit.channel,
            "host_turn_id": host_turn_id,
            "host_start": host_start,
            "host_end": host_end,
            "gt_onset": bc_unit.start_token,
            "best_onset": best_onset,
            "bc_offset": bc_offset,
            "best_delta": best_delta,
            "best_strategy": best_strategy,
            "best_pause_duration": best_pause_duration,
            "baseline_nll": baseline_nll,
            "best_absolute_nll": winning_absolute,
            "strategy_results": strategy_results,
            "candidates": candidates,
            "duration": bc_duration,
            "is_first_bc_on_host": is_first_bc_on_host,
            "ordering_constraint_min": min_bc_start,
            "nll_scores": strategy_results[reported]["nll_deltas"],
            "nll_deltas": strategy_results[reported]["nll_deltas"],
            "nll_absolutes": strategy_results[reported]["nll_absolutes"],
            "best_nll": best_delta,
        })
        if verbose:
            logger.info(f"  Backchannel {bc_unit.turn_id} (ch={bc_unit.channel}) on host "
                        f"[{host_start}, {host_end}]: onset {best_onset} (offset={bc_offset}), "
                        f"strategy={best_strategy}, delta NLL={best_delta:+.4f}")

    return placed_primary, placed_backchannels, diagnostics
