"""Default search hyperparameters: the single place they are defined.

The three dataclasses hold the defaults of :func:`generate_dialogue_three_pass`.
The step-3 command line (``neural_scheduler.pipeline.schedule``) historically
used different defaults for two pass-2 parameters; those are recorded in
``CLI_DEFAULT_OVERRIDES`` so both entry points keep the behaviour they always
had. All step counts are token steps at 12.5 Hz.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Pass1Config:
    search_window: int = 15  # +/- steps around the anchor
    search_stride: int = 1
    scoring_context: int = 15  # extra steps scored on each side of the candidate
    inter_turn_gap: int = 5  # anchor = end of previous turn + this
    schedule_subturns: bool = False  # search the pauses between a turn's speech blocks
    speaker_1_subturn_min_pause: int = 0
    speaker_1_subturn_max_pause: int = 10
    speaker_2_subturn_min_pause: int = 0
    speaker_2_subturn_max_pause: int = 10
    subturn_search_stride: int = 1
    use_speaker_gap_constraints: bool = False
    speaker_1_min_gap: int = -5  # negative = may start before the other speaker ends
    speaker_1_max_gap: int = 20
    speaker_2_min_gap: int = -5
    speaker_2_max_gap: int = 20
    strong_overlap_mode: bool = False
    strong_overlap_min: float = 0.4  # fraction of the incoming turn that must overlap
    strong_overlap_max: float = 0.6


@dataclass(frozen=True)
class Pass2Config:
    search_stride: int = 1
    scoring_context: int = 10
    bc_position_limit: float = 1.0  # tail fraction of the host in which a backchannel may start
    bc_effect: Optional[str] = None  # '+'-joined subset of np / fp / hp; None = all
    max_bc_overlap_ratio: float = 0.5  # fraction of the backchannel that must overlap its host
    score_only_pauses: bool = False


@dataclass(frozen=True)
class Pass3Config:
    scoring_context: int = 10
    max_extra_gap: int = 10  # silence budget beyond the frozen backchannels' duration


PASS1 = Pass1Config()
PASS2 = Pass2Config()
PASS3 = Pass3Config()

# Defaults of the step-3 command line where they differ from the function defaults above.
CLI_DEFAULT_OVERRIDES = {
    "pass2_scoring_context": 0,
    "pass2_bc_effect": "hp+fp+np",
}
