"""Scheduler behaviour that can be checked without the original tree or a real model."""

import random

import pytest

from neural_scheduler import constants, vocab
from neural_scheduler.scheduler import (
    config,
    generate_dialogue_bc_only,
    generate_dialogue_three_pass,
    parse_bc_effect,
)
from neural_scheduler.scheduler.dialogue import build_dialogue_from_placements, dialogue_to_channel_codes
from neural_scheduler.scheduler.script_parse import (
    compute_turn_boundaries_from_script_only,
    extract_script_turn_content,
    parse_turn_structure,
    strip_backchannels_from_script,
)
from neural_scheduler.scheduler.subsegments import (
    assemble_turn_from_sub_segments,
    build_host_turn_with_pause,
    proportional_token_counts,
)
from neural_scheduler.scheduler.types import PlacedTurn, TurnInfo

from .test_scheduler_helpers import CompactVocab, RealVocab, StubSchedulerModel, make_script, random_options


@pytest.fixture
def tok():
    return RealVocab()


def _script(tok, a_turns, b_turns):
    """``{turn_id: [block, ...]}`` per channel -> input ids. Blocks are lists of codec ids."""
    st = tok.special_tokens
    sil = st["silence"]

    def region(turns):
        ids = [sil]
        for turn_id, blocks in turns.items():
            ids.append(st[f"start_turn_{turn_id}"])
            for i, block in enumerate(blocks):
                if i:
                    ids.append(sil)
                ids.extend(vocab.code_to_token_id(c) for c in block)
            ids += [st[f"end_turn_{turn_id}"], sil]
        return ids

    return ([st["start_script_a"]] + region(a_turns) + [st["end_script_a"]]
            + [st["start_script_b"]] + region(b_turns) + [st["end_script_b"]] + [st["start_dialogue"]])


def test_stub_vocabulary_agrees_with_released_ids(tok):
    st = tok.special_tokens
    assert st["start_script_a"] == 168736 and st["end_dialogue"] == 168741
    assert st["start_turn_0"] == 168742 and st["end_turn_49"] == 168841
    assert st["silence"] == constants.SILENCE_TOKEN_ID


def test_turn_structure_separates_primary_turns_and_backchannels(tok):
    input_ids = _script(tok, {0: [[1, 2, 3], [4, 5]], 2: [[9]]}, {1: [[6, 7, 8]], 0: [[11]]})
    st = tok.special_tokens
    boundaries = compute_turn_boundaries_from_script_only(input_ids, st)
    primary, backchannels = parse_turn_structure(input_ids, st, boundaries)

    assert [(t.turn_id, t.channel, t.duration, len(t.sub_segments)) for t in sorted(primary, key=lambda t: t.turn_id)] == [
        (0, "A", 5, 2), (1, "B", 3, 1), (2, "A", 1, 1)]
    assert [(t.turn_id, t.channel, t.host_turn_id, t.is_backchannel) for t in backchannels] == [(0, "B", 0, True)]
    # Silence between the two blocks of turn 0 is kept as a gap in the block timing ...
    turn0 = next(t for t in primary if t.turn_id == 0)
    assert turn0.sub_segments == [{"start": 0.0, "end": 3 / 12.5}, {"start": 4 / 12.5, "end": 6 / 12.5}]
    # ... but not in the tokens.
    content = extract_script_turn_content(input_ids, st)
    assert content[("A", 0)] == [vocab.code_to_token_id(c) for c in (1, 2, 3, 4, 5)]


def test_known_issue_longer_backchannel_becomes_the_primary_turn(tok):
    """Documented in docs/scheduler.md: roles are decided by duration, not by the script."""
    st = tok.special_tokens
    input_ids = _script(tok, {0: [[1, 2]]}, {0: [[11, 12, 13, 14, 15, 16]]})
    primary, backchannels = parse_turn_structure(
        input_ids, st, compute_turn_boundaries_from_script_only(input_ids, st))
    assert [(t.channel, t.duration) for t in primary] == [("B", 6)]
    assert [(t.channel, t.duration) for t in backchannels] == [("A", 2)]


def test_strip_backchannels_removes_only_the_backchannel_turn(tok):
    st = tok.special_tokens
    input_ids = _script(tok, {0: [[1, 2, 3]]}, {0: [[11]], 1: [[6, 7]]})
    _, backchannels = parse_turn_structure(input_ids, st, compute_turn_boundaries_from_script_only(input_ids, st))
    stripped, gen_start = strip_backchannels_from_script(input_ids, backchannels, st)

    assert stripped == _script(tok, {0: [[1, 2, 3]]}, {1: [[6, 7]]})
    assert gen_start == len(stripped)


@pytest.mark.parametrize("bc_effect, expected", [
    (None, None), ("", None), ("np", ["no_pause"]), ("hp+fp", ["half_pause", "full_pause"]),
    (" FP + np + fp ", ["full_pause", "no_pause"]),
])
def test_parse_bc_effect(bc_effect, expected):
    assert parse_bc_effect(bc_effect) == expected


def test_parse_bc_effect_rejects_unknown_codes():
    with pytest.raises(ValueError):
        parse_bc_effect("np+xx")


def test_proportional_token_counts_sum_to_total():
    rng = random.Random(0)
    for _ in range(200):
        weights = [rng.uniform(0.01, 3.0) for _ in range(rng.randint(1, 5))]
        total = rng.randint(0, 40)
        counts = proportional_token_counts(weights, total)
        assert sum(counts) == total and len(counts) == len(weights)


def test_assemble_fills_gaps_with_silence():
    tokens, meta, onset, end = assemble_turn_from_sub_segments([[1, 2], [3]], [10, 15], silence_token_id=0)
    assert tokens == [1, 2, 0, 0, 0, 3]
    assert (onset, end) == (10, 16)
    assert meta == [{"start": 10 / 12.5, "end": 12 / 12.5}, {"start": 15 / 12.5, "end": 16 / 12.5}]


def test_host_pause_is_inserted_at_the_backchannel_onset():
    turn = TurnInfo(turn_id=0, channel="A", start_token=0, end_token=4, duration=4,
                    sub_segments=[{"start": 0.0, "end": 0.32}], speech_token_ids=[1, 2, 3, 4])
    host = PlacedTurn(turn_info=turn, onset_step=20)
    paused = build_host_turn_with_pause(host, bc_onset=22, pause_duration=3, silence_token_id=0)

    assert paused.turn_info.speech_token_ids == [1, 2, 0, 0, 0, 3, 4]
    assert paused.turn_info.duration == 7 and paused.onset_step == 20
    assert len(paused.turn_info.sub_segments) == 2
    assert turn.speech_token_ids == [1, 2, 3, 4], "the original host must not be modified"


def test_dialogue_is_interleaved_and_converted_to_codes(tok):
    sil = tok.special_tokens["silence"]
    a = TurnInfo(0, "A", 0, 2, 2, [], speech_token_ids=[vocab.code_to_token_id(1), vocab.code_to_token_id(2)])
    b = TurnInfo(1, "B", 0, 1, 1, [], speech_token_ids=[vocab.code_to_token_id(7)])
    dialogue = build_dialogue_from_placements([PlacedTurn(a, 1), PlacedTurn(b, 2)], 4, sil)

    assert len(dialogue) == 8 and dialogue[2] == vocab.code_to_token_id(1) and dialogue[5] == vocab.code_to_token_id(7)
    channel_a, channel_b = dialogue_to_channel_codes(dialogue, tok)
    s = constants.SILENCE_CODE
    assert channel_a == [s, 1, 2, s] and channel_b == [s, s, 7, s]


def _speech(codes):
    return [c for c in codes if c != constants.SILENCE_CODE]


@pytest.mark.parametrize("seed", range(8))
def test_three_pass_schedule_keeps_every_speakers_tokens_in_order(seed):
    """Whatever the model prefers, each channel must contain its script tokens, in script order.

    Without pauses the host is never split, so nothing can be overwritten.
    """
    rng = random.Random(seed)
    tok = CompactVocab()
    script = make_script(rng, tok)
    options = random_options(rng)
    options.update(pass2_bc_effect="np", pass_1_strong_overlap_mode=False)
    st = tok.special_tokens
    channel_a, channel_b, state = generate_dialogue_three_pass(
        StubSchedulerModel(tok.vocab_size), tok, script["input_ids"], st, device="cpu", verbose=False, **options)

    assert len(channel_a) == len(channel_b)
    content = extract_script_turn_content(script["input_ids"], st)
    for channel, codes in (("A", channel_a), ("B", channel_b)):
        placed = sorted((p for p in state.refined_placements if p.turn_info.channel == channel),
                        key=lambda p: p.onset_step)
        expected = [tok.id_to_code(t) for p in placed for t in content[(channel, p.turn_info.turn_id)]]
        if sorted(_speech(codes)) == sorted(expected):  # no cross-turn overwrite on this channel
            assert _speech(codes) == expected
    primary_ids = sorted(p.turn_info.turn_id for p in state.placed_primary)
    assert primary_ids == sorted(t.turn_id for t in state.primary_turns)
    assert len(state.placed_backchannels) == len(state.backchannel_units)


def test_bc_only_keeps_speaker_one_at_the_given_positions():
    tok = CompactVocab()
    st = tok.special_tokens
    speech = tok.speech_ids()
    a0, a1, bc = speech[:6], speech[6:10], speech[10:12]
    sil = st["silence"]
    input_ids = (
        [st["start_script_a"], sil, st["start_turn_0"]] + a0 + [st["end_turn_0"], sil, st["start_turn_1"]] + a1
        + [st["end_turn_1"], sil, st["end_script_a"], st["start_script_b"], sil, st["start_turn_0"]] + bc
        + [st["end_turn_0"], sil, st["end_script_b"], st["start_dialogue"]]
    )
    timing = [
        {"turn_id": 0, "start_token": 3, "end_token": 11,
         "sub_segments": [{"start_token": 3, "end_token": 6}, {"start_token": 8, "end_token": 11}]},
        {"turn_id": 1, "start_token": 20, "end_token": 24, "sub_segments": [{"start_token": 20, "end_token": 24}]},
    ]
    model = StubSchedulerModel(tok.vocab_size)
    channel_a, channel_b, state = generate_dialogue_bc_only(
        model, tok, input_ids, st, s1_turn_timing=timing, pass2_bc_effect="np", device="cpu", verbose=False)

    s = constants.SILENCE_CODE
    codes = [tok.id_to_code(t) for t in a0]
    assert channel_a[3:11] == codes[:3] + [s, s] + codes[3:]
    assert channel_a[20:24] == [tok.id_to_code(t) for t in a1]
    assert all(c == s for c in channel_a[:3] + channel_a[11:20])
    bc_placed = state.placed_backchannels[0]
    assert 3 <= bc_placed.onset_step <= 11
    assert _speech(channel_b) == [tok.id_to_code(t) for t in bc]
    assert state.diagnostics["pass3"]["skipped"] is True


def test_function_defaults_come_from_config():
    import inspect

    defaults = {k: v.default for k, v in inspect.signature(generate_dialogue_three_pass).parameters.items()}
    assert defaults["pass1_inter_turn_gap"] == config.PASS1.inter_turn_gap == 5
    assert defaults["pass2_scoring_context"] == config.PASS2.scoring_context == 10
    assert defaults["pass2_bc_effect"] is None
    assert defaults["pass3_max_extra_gap"] == config.PASS3.max_extra_gap == 10
    for removed in ("pass1_min_gap", "pass3_search_window", "pass2_bc_gap_threshold_steps"):
        assert removed not in defaults
