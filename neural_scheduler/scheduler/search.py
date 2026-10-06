"""The one search primitive every pass uses: score each candidate, keep the first best."""

from typing import Callable, List, Sequence, Tuple, TypeVar

C = TypeVar("C")


def scan_candidates(candidates: Sequence[C], score: Callable[[C], float], default: C) -> Tuple[C, float, List[float]]:
    """Score ``candidates`` in order and return ``(best, best_score, all_scores)``.

    The lowest score wins and ties go to the earliest candidate. ``default`` is
    returned when no candidate scores below infinity.
    """
    best = default
    best_score = float("inf")
    scores = []
    for candidate in candidates:
        value = score(candidate)
        scores.append(value)
        if value < best_score:
            best_score = value
            best = candidate
    return best, best_score, scores
