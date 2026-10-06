"""Helpers for the evaluation tests: a stub tokenizer and model that run on CPU,
synthetic dataset records, and loaders for code from the original research tree.

This file holds no tests of its own.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from neural_scheduler.constants import AUDIO_TOKEN_OFFSET, SILENCE_CODE
from neural_scheduler.vocab import scheduler_special_tokens

EXPERIMENTS = "ntpp/model/chatGLM_dual_timing_only/experiments"
MODEL_DIR = "ntpp/model/chatGLM_dual_timing_only"


class StubTokenizer:
    """Maps tokens to ids with the real layout shifted to a small audio offset.

    ``<|audio_k|>`` is ``offset + k`` for k in 0..16382 and the scheduler's
    special tokens follow directly, exactly as in the released vocabulary where
    ``offset`` is 152353.
    """

    def __init__(self, offset: int = 16):
        self.offset = offset
        self.special = {tok: offset + 16383 + i for i, tok in enumerate(scheduler_special_tokens())}
        self.id_to_special = {v: k for k, v in self.special.items()}
        self.vocab_size = offset + 16383 + len(self.special)
        self.eos_token_id = 3

    def _to_id(self, token: str) -> int:
        if token in self.special:
            return self.special[token]
        if token.startswith("<|audio_") and token.endswith("|>"):
            code = int(token[8:-2])
            if code >= 16383:
                raise KeyError(token)
            return self.offset + code
        raise KeyError(token)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._to_id(tokens)
        return [self._to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, token_id):
        token_id = int(token_id)
        if token_id in self.id_to_special:
            return self.id_to_special[token_id]
        if self.offset <= token_id < self.offset + 16383:
            return f"<|audio_{token_id - self.offset}|>"
        return f"<|other_{token_id}|>"


class _Output:
    def __init__(self, logits):
        self.logits = logits


class StubModel(torch.nn.Module):
    """A deterministic causal 'language model': logits at position t depend on
    the tokens and channel ids at positions <= t."""

    def __init__(self, vocab_size: int, dim: int = 12, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Parameter(torch.randn(vocab_size, dim, generator=g), requires_grad=False)
        self.channel = torch.nn.Parameter(torch.randn(3, dim, generator=g), requires_grad=False)
        self.out = torch.nn.Parameter(torch.randn(vocab_size, dim, generator=g) * 0.5, requires_grad=False)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, channel_ids=None, use_cache=None, return_dict=True, **_):
        self.calls += 1
        hidden = self.embed[input_ids]
        if channel_ids is not None:
            hidden = hidden + self.channel[channel_ids.long()]
        steps = torch.arange(1, hidden.shape[1] + 1, dtype=hidden.dtype).view(1, -1, 1)
        hidden = hidden + torch.cumsum(hidden, dim=1) / steps
        return _Output(hidden @ self.out.T)


def make_channel(rng, frames: int, speech_prob: float = 0.5):
    """One channel of codec ids: alternating silence and speech runs."""
    out = []
    speaking = rng.random() < speech_prob
    while len(out) < frames:
        run = int(rng.integers(1, 9))
        if speaking:
            codes = rng.integers(0, 4000, size=run)
            out.extend(int(c) if c != SILENCE_CODE else 1 for c in codes)
        else:
            out.extend([SILENCE_CODE] * run)
        speaking = not speaking
    return out[:frames]


def make_sample(tokenizer, rng, frames: int = 40, with_turn_tokens: bool = True, source_file=0):
    """A dataset record laid out like the released JSONL (script A, script B, dialogue)."""
    offset = tokenizer.offset
    sil = offset + SILENCE_CODE
    chan_a = make_channel(rng, frames)
    chan_b = make_channel(rng, frames)

    def script(channel, first_turn):
        ids = [sil]
        turn = first_turn
        run = []
        for code in channel + [SILENCE_CODE]:
            if code != SILENCE_CODE:
                run.append(offset + code)
            elif run:
                if with_turn_tokens:
                    ids.append(tokenizer.convert_tokens_to_ids(f"<|start-turn-{turn}|>"))
                ids.extend(run)
                if with_turn_tokens:
                    ids.append(tokenizer.convert_tokens_to_ids(f"<|end-turn-{turn}|>"))
                ids.append(sil)
                if not with_turn_tokens:
                    ids.append(sil)
                turn += 2
                run = []
        return ids

    t = tokenizer.convert_tokens_to_ids
    dialogue = []
    for a, b in zip(chan_a, chan_b):
        dialogue.extend([offset + a, offset + b])
    input_ids = (
        [t("<|start-of-script-A|>")] + script(chan_a, 0) + [t("<|end-of-script-A|>")]
        + [t("<|start-of-script-B|>")] + script(chan_b, 1) + [t("<|end-of-script-B|>")]
        + [t("<|start-of-dialogue|>")] + dialogue + [t("<|end-of-dialogue|>")]
    )
    return {"input_ids": input_ids, "source_file": source_file}


def make_samples(tokenizer, n: int, seed: int = 0, **kwargs):
    rng = np.random.default_rng(seed)
    return [make_sample(tokenizer, rng, source_file=i, **kwargs) for i in range(n)]


def real_offset_tokenizer():
    return StubTokenizer(offset=AUDIO_TOKEN_OFFSET)


# --- loading code from the original tree -------------------------------------

def load_module(path: Path, name: str):
    """Import a file of the original tree as a module called ``name``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_definitions(path: Path, names, extra_globals=None):
    """Execute only the named top-level functions/classes of a file.

    Used for scripts whose import would load models or run an experiment. The
    definitions see ``extra_globals`` plus a standard set of imports.
    """
    import json
    import os
    import random
    from collections import Counter
    from dataclasses import dataclass, field
    from typing import Dict, List, Optional, Tuple, Union

    import matplotlib.pyplot as plt
    from scipy import stats

    source = Path(path).read_text()
    tree = ast.parse(source)
    wanted = [node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in set(names)]
    missing = set(names) - {node.name for node in wanted}
    assert not missing, f"{path}: definitions not found: {missing}"
    namespace = dict(
        torch=torch, np=np, json=json, os=os, random=random, Counter=Counter, dataclass=dataclass, field=field,
        Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Union=Union, plt=plt, stats=stats, sys=sys,
        AutoTokenizer=object, Path=Path,
    )
    namespace.update(extra_globals or {})
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace
