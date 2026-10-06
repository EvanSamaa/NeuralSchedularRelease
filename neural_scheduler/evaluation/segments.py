"""Speech segments of one channel of a dialogue (runs of non-silence tokens)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..constants import SILENCE_CODE


@dataclass
class SpeechSegment:
    """A contiguous run of non-silence tokens, with the silence around it."""

    start_idx: int  # first token of the run (frame index within the channel)
    end_idx: int  # one past the last token
    pre_silence_len: int  # silence tokens directly before the run
    post_silence_len: int  # silence tokens directly after the run
    channel: Optional[str] = None  # 'A' or 'B' when known

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


def find_speech_segments(
    token_sequence: List[int],
    silence_token: int = SILENCE_CODE,
    channel: Optional[str] = None,
) -> List[SpeechSegment]:
    """All speech segments of one channel, in order."""
    segments = []
    n = len(token_sequence)
    i = 0
    while i < n:
        silence_count = 0
        while i < n and token_sequence[i] == silence_token:
            silence_count += 1
            i += 1
        if i >= n:
            break

        segment_start = i
        while i < n and token_sequence[i] != silence_token:
            i += 1
        segment_end = i

        post_silence = 0
        j = i
        while j < n and token_sequence[j] == silence_token:
            post_silence += 1
            j += 1

        segments.append(SpeechSegment(
            start_idx=segment_start,
            end_idx=segment_end,
            pre_silence_len=silence_count,
            post_silence_len=post_silence,
            channel=channel,
        ))
    return segments


def find_speech_segments_with_channel(
    dialogue_a: List[int],
    dialogue_b: List[int],
    silence_token: int = SILENCE_CODE,
) -> Tuple[List[SpeechSegment], List[SpeechSegment]]:
    """Speech segments of both channels, tagged 'A' and 'B'."""
    return (
        find_speech_segments(dialogue_a, silence_token, channel="A"),
        find_speech_segments(dialogue_b, silence_token, channel="B"),
    )
