"""User-study analysis and the anonymised response data shipped with the repository."""

import csv
import json
import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from neural_scheduler.analysis import user_study  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "analysis_data" / "user_study"
ORIG_DIR = "ntpp/model/chatGLM_dual_timing_only/experiments/inference_qualitative_examples"


def run_original(script: Path, argv, capsys) -> str:
    """Run a research script as ``__main__`` and return what it printed."""
    capsys.readouterr()
    old_argv, old_flag = sys.argv, sys.dont_write_bytecode
    sys.argv, sys.dont_write_bytecode = [str(script)] + [str(a) for a in argv], True
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv, sys.dont_write_bytecode = old_argv, old_flag
    return capsys.readouterr().out


def test_shipped_data_is_anonymous_and_well_formed():
    for name, n_rows, n_items in [("responses_main.csv", 20, 20), ("responses_dramatic.csv", 14, 5)]:
        rows = list(csv.reader(open(DATA / name, newline="", encoding="utf-8")))
        assert rows[0][0] == "respondent" and len(rows[0]) == n_items + 1
        assert [r[0] for r in rows[1:]] == [str(i) for i in range(1, n_rows + 1)]
        assert {v for r in rows[1:] for v in r[1:]} == {"A", "B"}
    for name, n_items in [("ab_key_backchannel.json", 10), ("ab_key_turn_taking.json", 10),
                          ("ab_key_dramatic.json", 5)]:
        key = json.load(open(DATA / name))
        assert [e["item"] for e in key] == list(range(1, n_items + 1))
        assert all(set(e) == {"item", "video_file", "proposed_is_A"} for e in key)
    for path in DATA.parent.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "/home" not in text and "evanpan" not in text and "2026" not in text, path


def test_key_field_resolution():
    assert user_study.proposed_is_A({"proposed_is_A": False, "model_is_A": True}) is False
    assert user_study.proposed_is_A({"model_is_A": True}) is True
    assert user_study.proposed_is_A({"naive_is_A": True}) is False
    assert user_study.proposed_is_A({"first_is_wav_a": True}) is True
    with pytest.raises(KeyError):
        user_study.proposed_is_A({"A": "model"})


def test_statistics_helpers():
    assert user_study.binomial_test(10, 20) == pytest.approx(1.0)
    assert user_study.binomial_test(18, 20) == pytest.approx(0.00040245, rel=1e-4)
    lo, hi = user_study.ci_95(135, 200)
    assert (lo, hi) == pytest.approx((0.6072, 0.7360), abs=5e-4)
    agg = user_study.aggregate([3, 1], 4)
    assert (agg["total"], agg["trials"], agg["rate"]) == (4, 8, 0.5)


def test_main_study_numbers(capsys):
    res = user_study.main([
        "main", "--csv", str(DATA / "responses_main.csv"),
        "--bc", str(DATA / "ab_key_backchannel.json"), "--tt", str(DATA / "ab_key_turn_taking.json"),
    ])
    capsys.readouterr()
    assert res["n_respondents"] == 20
    assert [i["wins"] for i in res["backchannel"]["items"]] == [18, 12, 15, 9, 10, 18, 16, 13, 11, 13]
    assert [i["wins"] for i in res["turn_taking"]["items"]] == [12, 11, 10, 6, 14, 6, 9, 9, 11, 14]
    assert (res["backchannel"]["aggregate"]["total"], res["backchannel"]["aggregate"]["trials"]) == (135, 200)
    assert (res["turn_taking"]["aggregate"]["total"], res["turn_taking"]["aggregate"]["trials"]) == (102, 200)
    assert res["turn_taking"]["first_half"]["total"] + res["turn_taking"]["second_half"]["total"] == 102


def test_dramatic_study_key_file_and_inline_key_agree(capsys):
    csv_path = str(DATA / "responses_dramatic.csv")
    by_file = user_study.main(["dramatic", "--csv", csv_path, "--key", str(DATA / "ab_key_dramatic.json")])
    out_file = capsys.readouterr().out
    inline = user_study.main(["dramatic", "--csv", csv_path, "--model_is_A", "A", "B", "A", "A", "A"])
    assert capsys.readouterr().out == out_file
    assert by_file == inline
    assert [i["wins"] for i in by_file["items"]] == [9, 10, 8, 9, 9]
    assert by_file["aggregate"]["trials"] == 70


@pytest.mark.equivalence
def test_main_study_report_matches_original(ntpp_orig, capsys):
    """Same report, character for character, from the anonymised data and sanitised keys."""
    orig = ntpp_orig / ORIG_DIR
    expected = run_original(orig / "user_study_analysis.py", [
        "--csv", orig / "user_20.csv",
        "--bc", orig / "user_study_backchannel_videos" / "ab_mapping.json",
        "--tt", orig / "user_study_turn_take_videos" / "ab_mapping.json",
    ], capsys)
    user_study.main([
        "main", "--csv", str(DATA / "responses_main.csv"),
        "--bc", str(DATA / "ab_key_backchannel.json"), "--tt", str(DATA / "ab_key_turn_taking.json"),
    ])
    assert capsys.readouterr().out == expected
    assert "135/200" in expected

    # The original, path-carrying mapping files are accepted directly and give the same report.
    user_study.main([
        "main", "--csv", str(orig / "user_20.csv"),
        "--bc", str(orig / "user_study_backchannel_videos" / "ab_mapping.json"),
        "--tt", str(orig / "user_study_turn_take_videos" / "ab_mapping.json"),
    ])
    assert capsys.readouterr().out == expected


@pytest.mark.equivalence
def test_dramatic_study_report_matches_original(ntpp_orig, capsys):
    orig = ntpp_orig / ORIG_DIR
    expected = run_original(orig / "user_study_analysis_dramatic.py", ["--csv", orig / "dramatic_14.csv"], capsys)
    user_study.main(["dramatic", "--csv", str(DATA / "responses_dramatic.csv"),
                     "--key", str(DATA / "ab_key_dramatic.json")])
    assert capsys.readouterr().out == expected

    flipped = ["B", "B", "A", "B", "A"]
    expected = run_original(orig / "user_study_analysis_dramatic.py",
                            ["--csv", orig / "dramatic_14.csv", "--model_is_A"] + flipped, capsys)
    user_study.main(["dramatic", "--csv", str(DATA / "responses_dramatic.csv"), "--model_is_A"] + flipped)
    assert capsys.readouterr().out == expected


@pytest.mark.equivalence
def test_shipped_data_matches_original_exports(ntpp_orig):
    orig = ntpp_orig / ORIG_DIR
    for shipped, source in [("responses_main.csv", "user_20.csv"), ("responses_dramatic.csv", "dramatic_14.csv")]:
        a = list(csv.reader(open(DATA / shipped, newline="", encoding="utf-8")))
        b = list(csv.reader(open(orig / source, newline="", encoding="utf-8")))
        assert [r[1:] for r in a] == [r[1:] for r in b]
    for shipped, source, field, negate in [
        ("ab_key_backchannel.json", "user_study_backchannel_videos", "model_is_A", False),
        ("ab_key_turn_taking.json", "user_study_turn_take_videos", "naive_is_A", True),
        ("ab_key_dramatic.json", "user_study_turn_take_videos_dramatic", "first_is_wav_a", False),
    ]:
        key = json.load(open(DATA / shipped))
        mapping = json.load(open(orig / source / "ab_mapping.json"))
        assert [e["video_file"] for e in key] == [e["video_file"] for e in mapping]
        assert [e["proposed_is_A"] for e in key] == [bool(e[field]) != negate for e in mapping]
