"""Turning placements into token sequences.

A dialogue is ``total_length`` steps long and is serialised interleaved:
``[A0, B0, A1, B1, ...]``. Every step not covered by a placed turn is silence.
"""

from typing import List, Tuple

from ..constants import SILENCE_CODE
from .types import PlacedSubSegment, PlacedTurn


def _write(target: List[int], onset: int, tokens: List[int], total_length: int) -> None:
    for j, tok_id in enumerate(tokens):
        pos = onset + j
        if 0 <= pos < total_length:
            target[pos] = tok_id


def _interleave(channel_a: List[int], channel_b: List[int]) -> List[int]:
    dialogue = []
    for tok_a, tok_b in zip(channel_a, channel_b):
        dialogue.append(tok_a)
        dialogue.append(tok_b)
    return dialogue


def build_dialogue_from_sub_segments_and_turns(
    placed_sub_segments: List[PlacedSubSegment],
    other_placed_turns: List[PlacedTurn],
    total_length: int,
    silence_token_id: int,
) -> List[int]:
    """Interleaved dialogue from sub-segments (written first) and whole turns.

    Later writes overwrite earlier ones where placements collide on a channel.
    """
    channels = {"A": [silence_token_id] * total_length, "B": [silence_token_id] * total_length}

    for pss in placed_sub_segments:
        _write(channels["A" if pss.channel == "A" else "B"], pss.onset_step, pss.token_ids, total_length)
    for pt in other_placed_turns:
        turn = pt.turn_info
        _write(channels["A" if turn.channel == "A" else "B"], pt.onset_step, turn.speech_token_ids, total_length)

    return _interleave(channels["A"], channels["B"])


def build_dialogue_from_placements(
    placed_turns: List[PlacedTurn],
    total_length: int,
    silence_token_id: int,
) -> List[int]:
    """Interleaved dialogue with each turn's tokens written at its onset."""
    return build_dialogue_from_sub_segments_and_turns([], placed_turns, total_length, silence_token_id)


def dialogue_to_channel_codes(dialogue: List[int], tokenizer) -> Tuple[List[int], List[int]]:
    """Split an interleaved dialogue into two lists of codec ids.

    Anything that is not an ``<|audio_k|>`` token becomes the silence code.
    """

    def to_code(token_id: int) -> int:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token.startswith("<|audio_") and token.endswith("|>"):
            return int(token[8:-2])
        return SILENCE_CODE

    channel_a = [to_code(tok) for tok in dialogue[0::2]]
    channel_b = [to_code(tok) for tok in dialogue[1::2]]
    return channel_a, channel_b
