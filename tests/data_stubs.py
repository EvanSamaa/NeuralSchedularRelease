"""CPU stand-ins for the tokenizer, the speech tokenizer and VAD used by the data tests."""

import random
from typing import List

from neural_scheduler import constants as C
from neural_scheduler.vocab import scheduler_special_tokens


class StubTokenizer:
    """Mimics the GLM-4-Voice tokenizer for audio and scheduler tokens.

    Like the real one it has ``<|audio_0|>`` .. ``<|audio_16382|>`` and raises
    ``KeyError`` for anything it does not know.
    """

    def __init__(self, extra_tokens: List[str] = ()):
        self.vocab = {f"<|audio_{k}|>": C.AUDIO_TOKEN_OFFSET + k for k in range(C.NUM_AUDIO_CODES - 1)}
        assert len(self) == C.BASE_VOCAB_SIZE
        self.add_tokens(scheduler_special_tokens())
        self.add_tokens(list(extra_tokens))

    def __len__(self):
        return max(self.vocab.values()) + 1

    def get_vocab(self):
        return self.vocab

    def add_tokens(self, tokens):
        added = 0
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[t] for t in tokens]


def stub_special_tokens():
    from neural_scheduler.vocab import get_special_token_ids

    return get_special_token_ids(StubTokenizer())


def fake_speech_tokens(seed: int, constant=None):
    """Replacement for ``extract_speech_token_batched``: one pseudo-random code per 1280 samples."""

    def extract(model, feature_extractor, batched_audio, sampling_rate=16000,
                model_batch_size=32, device="cuda", logger=None):
        out = []
        for row in range(batched_audio.shape[0]):
            n = batched_audio.shape[1] // C.SAMPLES_PER_TOKEN
            rng = random.Random(seed * 1000003 + row * 7919 + n)
            if constant is not None:
                out.append([constant] * n)
            else:
                # A small alphabet makes repeats (duration labels > 0) and silence-valued codes likely.
                out.append([rng.choice([C.SILENCE_CODE, 3, 3, 17, 4000, 16382, rng.randrange(16383)]) for _ in range(n)])
        return out

    return extract


def make_fake_vad(rng: random.Random, deterministic_per_call: bool = False):
    """Replacement for Silero's ``get_speech_timestamps``: random segments in sample units."""
    calls = {"n": 0}

    def get_speech_timestamps(audio, model, sampling_rate=16000, threshold=None,
                              min_speech_duration_ms=None, min_silence_duration_ms=None):
        n = len(audio)
        if n == 0:
            return []
        local = random.Random(n * 31 + calls["n"] % 2) if deterministic_per_call else rng
        calls["n"] += 1
        segments = []
        pos = local.randrange(0, 20000)
        while pos < n:
            length = local.choice([400, 1300, 5000, 20000, 60000])
            end = min(n, pos + length)
            segments.append({"start": pos, "end": end})
            pos = end + local.choice([900, 3000, 15000, 50000])
        if deterministic_per_call:
            return segments
        return segments if local.random() > 0.05 else []

    return get_speech_timestamps
