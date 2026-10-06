"""Backchannel labelling: candidate rules, turn ids, validator, extraction."""

import copy
import json
import logging
import os
import random

import pytest

from neural_scheduler.backchannel.labeling import candidates, discover, extract, turns
from neural_scheduler.backchannel.labeling.prompt import SYSTEM_PROMPT
from neural_scheduler.backchannel.labeling.validator import BackchannelValidator

from .test_backchannel_support import (
    FakeLabeler,
    ORIG_PREPROCESS,
    import_original,
    import_original_extract,
    make_transcript,
)

VALIDATOR_CASES = [
    ('{"classification": "Neutral", "word": "Mm-hmm"}', True),
    ('{"classification": "Positive", "word": "Yeah"}', True),
    ('\n{"classification": "Neutral", "word": "uh-huh"}\n', True),
    ('Here is the result: {"classification": "Positive", "word": "exactly"}', True),
    ("{'classification': 'Neutral', 'word': 'hmm'}", True),
    ('{"classification": "NotBackchannel", "word": "the dog ran"}', True),
    ('{"classification": "NotBackchannel", "word": ""}', False),
    ('', False),
    ('   ', False),
    ('not json at all', False),
    ('{"classification": "Invalid"}', False),
    ('{"word": "hmm"}', False),
    ('{"classification": "InvalidClass", "word": "hmm"}', False),
    ('{"classification": "Neutral", "word": ""}', False),
    ('{"classification": 123, "word": "hmm"}', False),
    ('{"classification": "Neutral", "word": 5}', False),
    ('{"classification": "Neutral", "word": "hmm", "extra": "field"}', False),
    ('[1, 2]', False),
    ('{"classification": "Neutral", "word": "that\'s"}', True),
    ("{'classification': 'Neutral', 'word': 'that's'}", False),
    ('{"classification": "Neutral", "word": "a"} {"classification": "Positive", "word": "b"}', False),
    ('{"classification": "Emotional", "word": "one two three four five six seven"}', True),
]


def _sentence(start, end, text):
    words = [{"word": w, "start": start, "end": end} for w in text.split()]
    return {"words": words, "start": start, "end": end, "transcript": text}


def test_validator_cases():
    validator = BackchannelValidator()
    for response, expected in VALIDATOR_CASES:
        assert validator.validate(response)[0] is expected, response


def test_validator_batch_counts():
    result = BackchannelValidator().validate_batch([c[0] for c in VALIDATOR_CASES])
    assert result["total"] == len(VALIDATOR_CASES)
    assert result["valid"] == sum(c[1] for c in VALIDATOR_CASES) == len(result["valid_data"])
    assert [e["index"] for e in result["errors"]] == [i for i, c in enumerate(VALIDATOR_CASES) if not c[1]]


def test_split_sentence_by_gaps():
    sentence = {
        "words": [
            {"word": "a", "start": 0.0, "end": 0.2},
            {"word": "b", "start": 0.3, "end": 0.5},
            {"word": "c", "start": 1.5, "end": 1.7},
        ],
        "start": 0.0, "end": 1.7, "transcript": "a b c",
    }
    parts = candidates.split_sentence_by_gaps(sentence, gap_threshold=0.8)
    assert [p["transcript"] for p in parts] == ["a b", "c"]
    assert (parts[0]["start"], parts[0]["end"], parts[1]["start"]) == (0.0, 0.5, 1.5)
    assert candidates.split_sentence_by_gaps(sentence, gap_threshold=2.0) == [sentence]


def test_backchannel_shares_turn_id_with_overlapped_speech():
    s0 = [_sentence(0.0, 5.0, "long story here"), _sentence(5.2, 6.0, "next thing")]
    s1 = [_sentence(2.0, 2.4, "yeah"), _sentence(7.0, 9.0, "my reply")]
    records = turns.build_turn_records(s0, s1, {(1, 2.0, 2.4): "backchannel-Neutral"})
    by_text = {r["text"]: r for r in records}
    assert by_text["yeah"]["turn_id"] == by_text["long story here"]["turn_id"]
    assert by_text["yeah"]["label"] == "backchannel-Neutral"
    assert by_text["my reply"]["turn_id"] != by_text["long story here"]["turn_id"]
    assert sorted(set(r["turn_id"] for r in records)) == list(range(len(set(r["turn_id"] for r in records))))


def test_generate_prompt_marks_candidate():
    bc = {"speaker_id": 1, "start": 2.0, "end": 2.3, "transcript": "yeah",
          "words": [{"word": "yeah", "start": 2.0, "end": 2.3}]}
    context = [
        {"speaker_id": 0, **_sentence(0.0, 1.9, "so it went well")},
        {"speaker_id": 0, **_sentence(3.0, 3.2, "yeah")},
    ]
    text, target = candidates.generate_prompt({"backchannel": bc, "context": context})
    assert target == "yeah"
    assert text == "A: so it went well\nB: yeah <|response|>\nA: yeah <|response|>\n"
    text, _ = candidates.generate_prompt({"backchannel": bc, "context": context}, match_by_position=True)
    assert text == "A: so it went well\nB: yeah <|response|>\nA: yeah\n"


def test_discover_and_shard(tmp_path):
    root = tmp_path / "improvised" / "test" / "0000"
    root.mkdir(parents=True)
    names = ["V01_S0002_I00000003_P0007", "V01_S0002_I00000003_P0004", "V01_S0001_I00000009_P0001A",
             "V01_S0001_I00000009_P0002", "V02_S0001_I00000001_P0001", "notes"]
    for name in names:
        (root / f"{name}.wav").write_bytes(b"")
        if name != "V01_S0001_I00000009_P0002":
            (root / f"{name}.json").write_text("{}")
    found = discover.discover_interactions(str(tmp_path), "test", "improvised", logging.getLogger("t"))
    # one group lacks a transcript, one has a single file, one file name does not parse
    assert [(i["interaction_id"], i["identifier_left"], i["identifier_right"]) for i in found] == [
        (0, "V01_S0002_I00000003_P0004", "V01_S0002_I00000003_P0007")
    ]
    assert extract.select_shard([{"interaction_id": i} for i in range(7)], 1, 3) == [
        {"interaction_id": 1}, {"interaction_id": 4}
    ]


# --- equivalence with the research code --------------------------------------

@pytest.fixture(scope="module")
def orig_extract(ntpp_orig):
    return import_original_extract(ntpp_orig)


@pytest.mark.equivalence
def test_system_prompt_verbatim(orig_extract):
    assert SYSTEM_PROMPT == orig_extract.SYSTEM_PROMPT


@pytest.mark.equivalence
def test_validator_matches_original(ntpp_orig, capsys):
    orig = import_original(ntpp_orig, ORIG_PREPROCESS, "bc_validator")
    responses = [c[0] for c in VALIDATOR_CASES] + FakeLabeler.ANSWERS
    ours, theirs = BackchannelValidator(), orig.BackchannelValidator()
    for response in responses:
        assert ours.validate(response) == theirs.validate(response), response
    assert ours.validate_batch(responses) == theirs.validate_batch(responses)


@pytest.mark.equivalence
@pytest.mark.parametrize("seed", range(6))
def test_candidate_pipeline_matches_original(orig_extract, seed):
    rng = random.Random(seed)
    raw0 = make_transcript(rng, 60)
    raw1 = make_transcript(rng, 60, t0=0.7)

    for gap in (0.5, 0.8):
        ours0 = candidates.preprocess_sentences(copy.deepcopy(raw0), gap)
        ours1 = candidates.preprocess_sentences(copy.deepcopy(raw1), gap)
        theirs0 = orig_extract.preprocess_sentences(copy.deepcopy(raw0), gap)
        theirs1 = orig_extract.preprocess_sentences(copy.deepcopy(raw1), gap)
        assert ours0 == theirs0 and ours1 == theirs1

        bounds = candidates.get_conversation_bounds(ours0, ours1)
        assert bounds == orig_extract.get_conversation_bounds(theirs0, theirs1)

        for threshold in (0.3, 1.0):
            ours_c = candidates.identify_backchannel_candidates(ours0, ours1, *bounds, temporal_threshold=threshold)
            theirs_c = orig_extract.identify_backchannel_candidates(
                theirs0, theirs1, *bounds, temporal_threshold=threshold)
            assert ours_c == theirs_c
            assert ours_c, "synthetic transcript should yield candidates"
            for a, b in zip(ours_c, theirs_c):
                ex_a = candidates.create_backchannel_example(a, ours0, ours1, 10.0)
                ex_b = orig_extract.create_backchannel_example(b, theirs0, theirs1, 10.0)
                assert ex_a == ex_b
                assert candidates.generate_prompt(ex_a) == orig_extract.generate_prompt(ex_b)

        lookup = {}
        for i, cand in enumerate(candidates.identify_backchannel_candidates(ours0, ours1, *bounds, 0.3)):
            if i % 2 == 0:
                s = cand["sentence"]
                lookup[(cand["speaker_id"], s["start"], s["end"])] = "backchannel-Positive"
        assert turns.build_turn_records(ours0, ours1, lookup) == \
            orig_extract.build_turn_records(theirs0, theirs1, lookup)


@pytest.mark.equivalence
def test_turn_id_assignment_matches_original(orig_extract):
    rng = random.Random(7)
    for _ in range(50):
        segs = []
        t = 0.0
        for _ in range(rng.randint(1, 25)):
            t += rng.uniform(-1.0, 2.0)
            start = max(0.0, t)
            segs.append({"start": start, "end": start + rng.uniform(0.0, 3.0),
                         "speaker_id": rng.choice([0, 1]), "text": "x"})
        segs.sort(key=lambda x: (x["start"], x["speaker_id"]))
        ours = turns.assign_turn_ids(turns.merge_adjacent_same_speaker(copy.deepcopy(segs)))
        theirs = orig_extract.assign_turn_ids(orig_extract.merge_adjacent_same_speaker(copy.deepcopy(segs)))
        assert ours == theirs


def _write_interaction_files(tmp_path, n=5):
    rng = random.Random(99)
    interactions = []
    for i in range(n):
        left, right = tmp_path / f"{i}_left.json", tmp_path / f"{i}_right.json"
        left.write_text(json.dumps({"metadata:transcript": make_transcript(rng, 50)}))
        right.write_text(json.dumps({"metadata:transcript": make_transcript(rng, 50, t0=0.4)}))
        interactions.append({
            "interaction_id": i, "identifier_left": f"L{i}", "identifier_right": f"R{i}",
            "wav_left": str(tmp_path / f"{i}_left.wav"), "wav_right": str(tmp_path / f"{i}_right.wav"),
            "transcript_left": str(left), "transcript_right": str(right),
        })
    # one interaction with an empty transcript: fails and goes to the failures file
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"metadata:transcript": []}))
    interactions.append({
        "interaction_id": n, "identifier_left": "Lx", "identifier_right": "Rx",
        "wav_left": "a.wav", "wav_right": "b.wav",
        "transcript_left": str(empty), "transcript_right": str(empty),
    })
    return interactions


@pytest.mark.equivalence
def test_process_shard_matches_original(orig_extract, tmp_path):
    interactions = _write_interaction_files(tmp_path)
    logger = logging.getLogger("bc_test")
    outputs = {}
    for name, module, validator in (
        ("ours", extract, BackchannelValidator()),
        ("theirs", orig_extract, orig_extract.BackchannelValidator()),
    ):
        ok_path, fail_path = tmp_path / f"{name}_ok.jsonl", tmp_path / f"{name}_fail.jsonl"
        labeler = FakeLabeler()
        counts = module.process_shard(
            shard_interactions=copy.deepcopy(interactions), qwen_wrapper=labeler, validator=validator,
            split="train", dataset_type="naturalistic",
            output_interactions_path=str(ok_path), output_failures_path=str(fail_path),
            threshold_for_splitting=0.8, threshold_for_bc=0.3, half_window_size=10.0, batch_size=4,
            logger=logger,
        )
        failures = [json.loads(line) for line in fail_path.read_text().splitlines()]
        outputs[name] = (counts, ok_path.read_text(), [(f["interaction_id"], f["error"]) for f in failures],
                         labeler.calls)

    assert outputs["ours"] == outputs["theirs"]
    counts, ok_text, failures, calls = outputs["ours"]
    assert counts == (5, 0, 1) and len(failures) == 1 and calls
    records = [json.loads(line) for line in ok_text.splitlines()]
    assert sum(r["num_backchannels"] for r in records) > 0


def test_resume_and_relative_paths(tmp_path):
    interactions = _write_interaction_files(tmp_path, n=3)
    ok_path, fail_path = str(tmp_path / "ok.jsonl"), str(tmp_path / "fail.jsonl")
    kwargs = dict(
        qwen_wrapper=FakeLabeler(), validator=BackchannelValidator(), split="test",
        dataset_type="improvised", output_interactions_path=ok_path, output_failures_path=fail_path,
        threshold_for_splitting=0.8, threshold_for_bc=0.3, half_window_size=10.0, batch_size=16,
        logger=logging.getLogger("bc_test"), relative_to=str(tmp_path),
    )
    assert extract.process_shard(shard_interactions=interactions, max_interactions=2, **kwargs) == (2, 0, 0)
    assert extract.process_shard(shard_interactions=interactions, resume=True, **kwargs) == (1, 2, 1)
    records = [json.loads(line) for line in open(ok_path)]
    assert [r["interaction_id"] for r in records] == [0, 1, 2]
    assert records[0]["wav_left"] == "0_left.wav" and not os.path.isabs(records[0]["transcript_right"])


@pytest.mark.equivalence
def test_discovery_matches_original(orig_extract, tmp_path):
    pytest.importorskip("pandas", exc_type=ImportError)
    rng = random.Random(3)
    for split_dir in ("0000/0001", "0000/0002", "0001/0000"):
        (tmp_path / "naturalistic" / "train" / split_dir).mkdir(parents=True)
    dirs = sorted(str(p) for p in (tmp_path / "naturalistic" / "train").rglob("*") if p.is_dir())
    for vendor in (1, 2, 10):
        for session in (3, 20, 100):
            for interaction in (5, 40):
                participants = rng.sample(["0001", "0002", "0003A", "0010"], rng.choice([1, 2, 2, 3]))
                for participant in participants:
                    stem = f"V{vendor:02d}_S{session:04d}_I{interaction:08d}_P{participant}"
                    folder = rng.choice(dirs)
                    open(os.path.join(folder, stem + ".wav"), "w").close()
                    if rng.random() < 0.85:
                        open(os.path.join(folder, stem + ".json"), "w").close()
    open(os.path.join(dirs[0], "readme.wav"), "w").close()

    logger = logging.getLogger("bc_test")
    ours = discover.discover_interactions(str(tmp_path), "train", "naturalistic", logger)
    theirs = orig_extract.discover_interactions(str(tmp_path), "train", "naturalistic", logger)
    assert ours == theirs and len(ours) > 5
