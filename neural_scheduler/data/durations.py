"""Remaining-duration labels for the interleaved dialogue region."""

from typing import List


def _channel_durations(ch_tokens: List[int]) -> List[int]:
    durations = []
    i = 0
    while i < len(ch_tokens):
        current = ch_tokens[i]
        repeat = 0
        j = i + 1
        while j < len(ch_tokens) and ch_tokens[j] == current:
            repeat += 1
            j += 1
        for k in range(repeat + 1):
            durations.append(repeat - k)
        i = j
    return durations


def compute_remaining_durations_dual_channel(tokens: List[int]) -> List[int]:
    """For each position of an A/B-interleaved sequence, how many more times the
    token repeats consecutively in its own channel.

    A run of length ``r`` in one channel is labelled ``r-1, r-2, ..., 0``. In
    practice only silence forms runs, so this is a per-channel countdown of the
    silence remaining. The result is interleaved like the input.
    """
    if len(tokens) == 0:
        return []

    dur_a = _channel_durations([tokens[i] for i in range(0, len(tokens), 2)])
    dur_b = _channel_durations([tokens[i] for i in range(1, len(tokens), 2)])

    result = []
    for i in range(max(len(dur_a), len(dur_b))):
        if i < len(dur_a):
            result.append(dur_a[i])
        if i < len(dur_b):
            result.append(dur_b[i])
    return result


def validate_remaining_durations(tokens: List[int], durations: List[int]) -> List[str]:
    """Check ``durations`` against ``tokens`` (both interleaved); return error strings."""
    errors = []
    for name, offset in (("A", 0), ("B", 1)):
        ch_tokens = tokens[offset::2]
        ch_durations = durations[offset::2]
        i = 0
        while i < len(ch_tokens):
            actual = 0
            for j in range(i + 1, len(ch_tokens)):
                if ch_tokens[j] == ch_tokens[i]:
                    actual += 1
                else:
                    break
            expected = ch_durations[i] if i < len(ch_durations) else None
            if actual != expected:
                errors.append(
                    f"channel {name} idx {i}: label {expected}, actual run {actual} (token={ch_tokens[i]})"
                )
            i += actual + 1
    return errors
