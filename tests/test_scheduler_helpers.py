"""Stub model, stub tokenizer and synthetic scripts for the scheduler tests.

The search only touches the model through one forward call and the tokenizer
through id/token lookups, so both can be replaced by cheap deterministic
stand-ins. ``CompactVocab`` uses a few hundred ids to keep the logits small;
``RealVocab`` uses the ids of the released checkpoints.
"""

import importlib.util
import random
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Dict, List

import torch

from neural_scheduler import constants, vocab

SEARCH_DIR = "ntpp/model/chatGLM_dual_timing_only/experiments"
ORIG_THREE_PASS = f"{SEARCH_DIR}/inference_pipeline_10_examples/segment_level_search_v6.py"
ORIG_BC_ONLY = f"{SEARCH_DIR}/inference_pipeline_bc_only_full_duplex_bench/segment_level_search_v6_bc_only.py"
ORIG_RECONSTRUCT = f"{SEARCH_DIR}/inference_pipeline_10_examples/step_4_reconstruct_audio_with_schedule_result.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Vocab:
    """Token <-> id tables with the tokenizer methods the scheduler calls."""

    def __init__(self, audio_offset: int, num_codes: int, first_special: int, vocab_size: int):
        self.audio_offset = audio_offset
        self.num_codes = num_codes
        self.vocab_size = vocab_size
        self._token_to_id = {}
        for i, token in enumerate(vocab.scheduler_special_tokens()):
            self._token_to_id[token] = first_special + i
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    def _audio_id(self, token: str):
        if token.startswith("<|audio_") and token.endswith("|>"):
            return self.code_to_id(int(token[8:-2]))
        return None

    def code_to_id(self, code: int) -> int:
        raise NotImplementedError

    def id_to_code(self, token_id: int):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.convert_tokens_to_ids(t) for t in tokens]
        if tokens in self._token_to_id:
            return self._token_to_id[tokens]
        return self._audio_id(tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        code = self.id_to_code(token_id)
        return f"<|audio_{code}|>" if code is not None else "<unk>"

    @property
    def special_tokens(self) -> Dict[str, int]:
        return vocab.get_special_token_ids(self)

    def speech_ids(self) -> List[int]:
        """Ids usable as speech tokens (never the silence token)."""
        silence = self.code_to_id(constants.SILENCE_CODE)
        return [i for i in (self.code_to_id(c) for c in range(self.num_codes)) if i != silence]


class CompactVocab(_Vocab):
    """120 speech codes + silence + the scheduler tokens in under 300 ids."""

    SILENCE_ID = 5

    def __init__(self):
        super().__init__(audio_offset=10, num_codes=120, first_special=140, vocab_size=260)

    def code_to_id(self, code: int) -> int:
        if code == constants.SILENCE_CODE:
            return self.SILENCE_ID
        return self.audio_offset + code

    def id_to_code(self, token_id: int):
        if token_id == self.SILENCE_ID:
            return constants.SILENCE_CODE
        if self.audio_offset <= token_id < self.audio_offset + self.num_codes:
            return token_id - self.audio_offset
        return None


class RealVocab(_Vocab):
    """The ids of the released tokenizer."""

    def __init__(self):
        super().__init__(
            audio_offset=constants.AUDIO_TOKEN_OFFSET,
            num_codes=constants.NUM_AUDIO_CODES - 1,
            first_special=constants.BASE_VOCAB_SIZE,
            vocab_size=constants.PADDED_VOCAB_SIZE,
        )

    def code_to_id(self, code: int) -> int:
        return self.audio_offset + code

    def id_to_code(self, token_id: int):
        if self.audio_offset <= token_id < self.audio_offset + constants.NUM_AUDIO_CODES:
            return token_id - self.audio_offset
        return None


class StubSchedulerModel(torch.nn.Module):
    """Deterministic stand-in for the scheduler LM.

    The logits at position ``t`` are a fixed function of every input id and
    channel id up to ``t``, so any difference in what two implementations feed
    the model shows up in the NLL.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = 0
        self.register_buffer("frequencies", torch.arange(1, vocab_size + 1, dtype=torch.float64) * 0.6180339887)

    def forward(self, input_ids, attention_mask=None, channel_ids=None, use_cache=False, return_dict=True):
        assert input_ids.shape[0] == 1 and attention_mask is not None and channel_ids is not None
        assert not use_cache and return_dict
        assert int(attention_mask.sum()) == input_ids.shape[1]
        self.calls += 1
        state = 17
        phases = []
        for token_id, channel in zip(input_ids[0].tolist(), channel_ids[0].tolist()):
            state = (state * 1103515245 + 12345 + token_id * 7919 + channel * 104729) % 2147483647
            phases.append(state / 2147483647.0)
        phase = torch.tensor(phases, dtype=torch.float64).unsqueeze(1)
        logits = 4.0 * torch.sin(6.283185307179586 * phase * self.frequencies.unsqueeze(0) + 12.9898 * phase)
        return SimpleNamespace(logits=logits.to(torch.float32).unsqueeze(0))


def make_script(rng: random.Random, tok: _Vocab, bc_only: bool = False, max_turns: int = 5,
                max_block: int = 6) -> Dict:
    """A random two-speaker script as a scheduler input sequence.

    Returns ``{'input_ids', 'turns'}`` where ``turns`` describes what was
    generated (``turn_id``, ``channel``, ``blocks``, optional ``bc_blocks``).
    """
    st = tok.special_tokens
    speech = tok.speech_ids()
    sil = st["silence"]

    def block(lo, hi):
        return [rng.choice(speech) for _ in range(rng.randint(lo, hi))]

    turns = []
    for turn_id in range(rng.randint(2, max_turns)):
        channel = "A" if bc_only else rng.choice("AB")
        blocks = [block(1, max_block) for _ in range(rng.choice([1, 1, 2, 3]))]
        turn = {"turn_id": turn_id, "channel": channel, "blocks": blocks, "bc_blocks": None}
        if rng.random() < (0.7 if bc_only else 0.45):
            # Sometimes longer than a short host, which makes the roles swap.
            long_bc = rng.random() < 0.2
            turn["bc_blocks"] = [block(4, 9) if long_bc else block(1, 3) for _ in range(rng.choice([1, 1, 2]))]
        turns.append(turn)

    def script_for(channel: str) -> List[int]:
        ids = [sil]
        for turn in turns:
            if turn["channel"] == channel:
                blocks = turn["blocks"]
            elif turn["bc_blocks"] is not None:
                blocks = turn["bc_blocks"]
            else:
                continue
            ids.append(st[f"start_turn_{turn['turn_id']}"])
            for i, tokens in enumerate(blocks):
                if i > 0:
                    ids.extend([sil] * rng.randint(1, 3))
                ids.extend(tokens)
            ids.append(st[f"end_turn_{turn['turn_id']}"])
            ids.append(sil)
        return ids

    input_ids = (
        [st["start_script_a"]] + script_for("A") + [st["end_script_a"]]
        + [st["start_script_b"]] + script_for("B") + [st["end_script_b"]]
        + [st["start_dialogue"]]
    )
    # Step 2 appends a placeholder dialogue; the search must ignore it.
    if rng.random() < 0.5:
        input_ids += [rng.choice(speech + [sil]) for _ in range(2 * rng.randint(1, 6))] + [st["end_dialogue"]]
    return {"input_ids": input_ids, "turns": turns}


def make_s1_timing(rng: random.Random, script: Dict) -> List[Dict]:
    """Known dialogue positions for channel-A turns, with the edge cases the anchoring handles."""
    timing = []
    position = rng.randint(0, 6)
    for turn in script["turns"]:
        if turn["channel"] != "A":
            continue
        start = position
        sub_segments = []
        for tokens in turn["blocks"]:
            # Usually room for every token, sometimes too little (tokens get dropped or carried).
            span = max(0, len(tokens) + rng.choice([0, 0, 0, 1, 2, -1]))
            sub_segments.append({"start_token": position, "end_token": position + span})
            position += span + rng.randint(0, 4)
        entry = {"turn_id": turn["turn_id"], "start_token": start, "end_token": position, "sub_segments": sub_segments}
        choice = rng.random()
        if choice < 0.1:
            entry["sub_segments"] = []
        elif choice < 0.2:
            entry["start_token"] = max(0, start - rng.randint(1, 3))
        if rng.random() < 0.9:
            timing.append(entry)
        position += rng.randint(1, 8)
    return timing


def random_options(rng: random.Random) -> Dict:
    """Search hyperparameters drawn from the ranges the run scripts used, kept small for speed."""
    strong = rng.random() < 0.25
    return dict(
        pass1_search_window=rng.choice([2, 4, 6]),
        pass1_search_stride=rng.choice([1, 1, 2]),
        pass1_scoring_context=rng.choice([0, 3, 5]),
        pass1_inter_turn_gap=rng.choice([0, 3, 5, 6]),
        pass_1_schedule_subturns=rng.random() < 0.5,
        pass_1_speaker_1_subturn_min_pause=rng.choice([-5, 0, 1, 3]),
        pass_1_speaker_1_subturn_max_pause=rng.choice([2, 5]),
        pass_1_speaker_2_subturn_min_pause=rng.choice([-5, 0, 1, 3]),
        pass_1_speaker_2_subturn_max_pause=rng.choice([2, 5]),
        pass_1_subturn_search_stride=rng.choice([1, 1, 2]),
        pass_1_use_speaker_gap_constraints=rng.random() < 0.5,
        pass_1_speaker_1_min_gap=rng.choice([-10, -5, -2, 3]),
        pass_1_speaker_1_max_gap=rng.choice([0, 2, 5, 10]),
        pass_1_speaker_2_min_gap=rng.choice([-10, -5, -2, 3]),
        pass_1_speaker_2_max_gap=rng.choice([0, 2, 5, 10]),
        pass_1_strong_overlap_mode=strong,
        pass_1_strong_overlap_min=0.2 if strong and rng.random() < 0.5 else 0.4,
        pass_1_strong_overlap_max=0.4 if strong and rng.random() < 0.5 else 0.6,
        pass3_scoring_context=rng.choice([0, 10]),
        pass3_max_extra_gap=rng.choice([3, 10]),
        **random_pass2_options(rng),
    )


def random_pass2_options(rng: random.Random) -> Dict:
    return dict(
        pass2_search_stride=rng.choice([1, 1, 2]),
        pass2_scoring_context=rng.choice([0, 5, 10]),
        pass2_bc_position_limit=rng.choice([1.0, 0.9, 0.6, 0.3, 0.05]),
        pass2_bc_effect=rng.choice([None, "", "np", "hp", "fp", "hp+fp", "hp+fp+np", "np+fp"]),
        pass2_max_bc_overlap_ratio=rng.choice([0.5, 0.05, 0.01, 1.0, 0.0]),
    )


def plain(value):
    """Dataclasses, tuples and nested containers as comparable plain data."""
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
