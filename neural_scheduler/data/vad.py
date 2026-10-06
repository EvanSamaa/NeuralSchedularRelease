"""Silero VAD loading and conversion of its timestamps to token indices and seconds."""

import os
from typing import Callable, Dict, List, Optional, Tuple

from ..constants import TOKEN_RATE_HZ

# Settings used for every released dataset.
VAD_THRESHOLD = 0.2
VAD_MIN_SPEECH_MS = 50
VAD_MIN_SILENCE_MS = 50

# Silero VAD release the published datasets and experiments were produced with.
# Later releases place segment boundaries slightly differently (tens of
# milliseconds), which changes turn boundaries and therefore the token sequences.
SILERO_REPO = "snakers4/silero-vad:v6.0"
SILERO_REPO_ENV = "SILERO_VAD_REPO"


def load_silero_vad(repo_or_dir: Optional[str] = None) -> Tuple[object, Callable]:
    """Load Silero VAD through ``torch.hub``.

    ``repo_or_dir`` is a ``owner/repo[:ref]`` GitHub spec or a local checkout;
    it defaults to the ``SILERO_VAD_REPO`` environment variable and then to the
    pinned :data:`SILERO_REPO`. Returns ``(vad_model, get_speech_timestamps)``.
    The first call downloads the repository into the torch hub cache
    (``TORCH_HOME``).
    """
    import torch

    repo_or_dir = repo_or_dir or os.environ.get(SILERO_REPO_ENV) or SILERO_REPO
    source = "local" if os.path.isdir(repo_or_dir) else "github"
    vad_model, vad_utils = torch.hub.load(repo_or_dir=repo_or_dir, model="silero_vad", source=source)
    return vad_model, vad_utils[0]


def vad_timestamps_to_token_indices(
    vad_timestamps: List[Dict],
    sr: int,
    token_rate: float = TOKEN_RATE_HZ,
) -> List[Dict[str, int]]:
    """Convert VAD timestamps from sample indices to token indices (floor on both ends)."""
    return [
        {
            "start": int(ts["start"] / sr * token_rate),
            "end": int(ts["end"] / sr * token_rate),
        }
        for ts in vad_timestamps
    ]


def vad_timestamps_to_seconds(vad_timestamps: List[Dict], sr: int) -> List[Dict[str, float]]:
    """Convert VAD timestamps from sample indices to seconds."""
    return [{"start": ts["start"] / sr, "end": ts["end"] / sr} for ts in vad_timestamps]
