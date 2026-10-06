"""Data structures shared by the scheduling passes.

All positions and durations are in token steps (12.5 Hz, 80 ms per step) unless
a field name ends in ``_sec``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TurnInfo:
    """One turn of one speaker, as read from the script region.

    ``speech_token_ids`` are vocab ids. For a turn straight from the script they
    contain speech only; once a pass has inserted a pause they also contain
    silence filler, and ``duration`` is always ``len(speech_token_ids)``.
    ``sub_segments`` is a list of ``{'start': sec, 'end': sec}`` speech blocks.
    """

    turn_id: int
    channel: str
    start_token: int
    end_token: int
    duration: int
    sub_segments: List[Dict]
    is_backchannel: bool = False
    host_turn_id: Optional[int] = None
    speech_token_ids: List[int] = field(default_factory=list)
    bc_group_index: Optional[int] = None


@dataclass
class PlacedTurn:
    """A turn committed to the dialogue timeline at ``onset_step``."""

    turn_info: TurnInfo
    onset_step: int
    gap_to_previous: Optional[int] = None
    bc_offset_from_host_start: Optional[int] = None


@dataclass
class PlacedSubSegment:
    """One speech block of a host turn while pass 3 re-times it."""

    turn_id: int
    channel: str
    sub_segment_index: int
    onset_step: int
    token_ids: List[int]
    duration: int
    is_frozen: bool = False


@dataclass
class SchedulingState:
    """Everything the passes decided, plus the scores behind each decision.

    ``refined_placements`` is the final schedule. ``pass2_nll_scores`` is keyed
    by ``(turn_id, bc_group_index)`` and ``pass1_subturn_nll_scores`` by
    ``(turn_id, sub_segment_index)``.
    """

    primary_turns: List[TurnInfo] = field(default_factory=list)
    backchannel_turns: List[TurnInfo] = field(default_factory=list)
    backchannel_units: List[TurnInfo] = field(default_factory=list)

    placed_primary: List[PlacedTurn] = field(default_factory=list)
    pass1_nll_scores: Dict[int, List[float]] = field(default_factory=dict)
    pass1_subturn_nll_scores: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)

    placed_backchannels: List[PlacedTurn] = field(default_factory=list)
    pass2_nll_scores: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)

    refined_placements: List[PlacedTurn] = field(default_factory=list)
    pass3_adjustments: Dict[int, Dict] = field(default_factory=dict)
    pass3_sub_segment_details: Dict[int, List[Dict]] = field(default_factory=dict)

    diagnostics: Dict = field(default_factory=dict)
