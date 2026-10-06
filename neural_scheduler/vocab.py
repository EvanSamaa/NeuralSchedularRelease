"""The scheduler's special tokens: one list, one order.

The order is part of every released checkpoint and dataset: tokens are appended
to the stock GLM-4-Voice tokenizer (168736 entries), so the six structural
tokens take ids 168736..168741 and ``<|start-turn-i|>`` / ``<|end-turn-i|>``
take 168742 + 2i / 168743 + 2i. Do not reorder or insert.
"""

from typing import Dict, List

from .constants import AUDIO_TOKEN_OFFSET, BASE_VOCAB_SIZE, MAX_TURNS, NUM_AUDIO_CODES, SILENCE_CODE

START_SCRIPT_A = "<|start-of-script-A|>"
END_SCRIPT_A = "<|end-of-script-A|>"
START_SCRIPT_B = "<|start-of-script-B|>"
END_SCRIPT_B = "<|end-of-script-B|>"
START_DIALOGUE = "<|start-of-dialogue|>"
END_DIALOGUE = "<|end-of-dialogue|>"

STRUCTURAL_TOKENS: List[str] = [
    START_SCRIPT_A,
    END_SCRIPT_A,
    START_SCRIPT_B,
    END_SCRIPT_B,
    START_DIALOGUE,
    END_DIALOGUE,
]

FIRST_TURN_TOKEN_ID = BASE_VOCAB_SIZE + len(STRUCTURAL_TOKENS)  # 168742


def start_turn_token(turn_id: int) -> str:
    return f"<|start-turn-{turn_id}|>"


def end_turn_token(turn_id: int) -> str:
    return f"<|end-turn-{turn_id}|>"


def audio_token(code: int) -> str:
    return f"<|audio_{code}|>"


def turn_tokens(max_turns: int = MAX_TURNS) -> List[str]:
    tokens = []
    for i in range(max_turns):
        tokens.append(start_turn_token(i))
        tokens.append(end_turn_token(i))
    return tokens


def scheduler_special_tokens(max_turns: int = MAX_TURNS) -> List[str]:
    """All tokens added on top of stock GLM-4-Voice, in vocabulary order."""
    return STRUCTURAL_TOKENS + turn_tokens(max_turns)


def add_scheduler_tokens(tokenizer, max_turns: int = MAX_TURNS) -> int:
    """Append the scheduler tokens to a stock GLM-4-Voice tokenizer.

    Tokens already present (a tokenizer saved with a checkpoint) are left alone,
    so this is safe to call on either. Returns the number of tokens added.
    """
    return tokenizer.add_tokens(scheduler_special_tokens(max_turns))


def get_special_token_ids(tokenizer, max_turns: int = MAX_TURNS) -> Dict[str, int]:
    """Vocab ids keyed ``start_script_a`` .. ``end_dialogue``, ``silence``,
    ``start_turn_{i}`` and ``end_turn_{i}``."""
    ids = {
        "start_script_a": tokenizer.convert_tokens_to_ids(START_SCRIPT_A),
        "end_script_a": tokenizer.convert_tokens_to_ids(END_SCRIPT_A),
        "start_script_b": tokenizer.convert_tokens_to_ids(START_SCRIPT_B),
        "end_script_b": tokenizer.convert_tokens_to_ids(END_SCRIPT_B),
        "start_dialogue": tokenizer.convert_tokens_to_ids(START_DIALOGUE),
        "end_dialogue": tokenizer.convert_tokens_to_ids(END_DIALOGUE),
        "silence": tokenizer.convert_tokens_to_ids(audio_token(SILENCE_CODE)),
    }
    for i in range(max_turns):
        ids[f"start_turn_{i}"] = tokenizer.convert_tokens_to_ids(start_turn_token(i))
        ids[f"end_turn_{i}"] = tokenizer.convert_tokens_to_ids(end_turn_token(i))
    return ids


def code_to_token_id(code: int) -> int:
    """Codec id -> vocab id."""
    return AUDIO_TOKEN_OFFSET + code


def token_id_to_code(token_id: int) -> int:
    """Vocab id -> codec id. Raises for ids outside the audio range."""
    code = token_id - AUDIO_TOKEN_OFFSET
    if not 0 <= code < NUM_AUDIO_CODES:
        raise ValueError(f"{token_id} is not an audio token id")
    return code


def is_audio_token_id(token_id: int) -> bool:
    return AUDIO_TOKEN_OFFSET <= token_id < AUDIO_TOKEN_OFFSET + NUM_AUDIO_CODES
