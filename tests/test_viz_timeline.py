"""Scheduling-state export, timeline comparison and three-pass diagnostic plots."""

import glob
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from neural_scheduler.viz import three_pass, timeline  # noqa: E402

PIPELINE_DIR = "ntpp/model/chatGLM_dual_timing_only/experiments/inference_pipeline_10_examples"


@dataclass
class TurnInfo:
    turn_id: int
    channel: str
    start_token: int
    end_token: int
    duration: int
    sub_segments: List[Dict]
    is_backchannel: bool = False
    host_turn_id: Optional[int] = None
    speech_token_ids: List[int] = field(default_factory=list)
    bc_group_index: Optional[int] = None


@dataclass
class PlacedTurn:
    turn_info: TurnInfo
    onset_step: int
    gap_to_previous: Optional[int] = None
    bc_offset_from_host_start: Optional[int] = None


@dataclass
class SchedulingState:
    primary_turns: list = field(default_factory=list)
    backchannel_turns: list = field(default_factory=list)
    backchannel_units: list = field(default_factory=list)
    placed_primary: list = field(default_factory=list)
    pass1_nll_scores: dict = field(default_factory=dict)
    pass1_subturn_nll_scores: dict = field(default_factory=dict)
    placed_backchannels: list = field(default_factory=list)
    pass2_nll_scores: dict = field(default_factory=dict)
    refined_placements: list = field(default_factory=list)
    pass3_adjustments: dict = field(default_factory=dict)
    pass3_sub_segment_details: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def synthetic_state(types=None) -> SchedulingState:
    """Two primary turns (one with two sub-segments) and one backchannel."""
    ti_cls = types.TurnInfo if types else TurnInfo
    pt_cls = types.PlacedTurn if types else PlacedTurn
    st_cls = types.SchedulingState if types else SchedulingState
    t0 = ti_cls(0, "A", 0, 30, 30,
                [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 2.4}], speech_token_ids=list(range(30)))
    t1 = ti_cls(1, "B", 0, 20, 20, [{"start": 0.0, "end": 1.6}], speech_token_ids=list(range(20)))
    bc = ti_cls(0, "B", 30, 35, 5, [{"start": 2.4, "end": 2.8}], is_backchannel=True,
                host_turn_id=0, speech_token_ids=[1, 2, 3, 4, 5], bc_group_index=0)
    p0 = pt_cls(t0, 5)
    p1 = pt_cls(t1, 41, gap_to_previous=6)
    pbc = pt_cls(bc, 20, bc_offset_from_host_start=15)
    return st_cls(
        primary_turns=[t0, t1],
        backchannel_turns=[bc],
        backchannel_units=[bc],
        placed_primary=[p0, p1],
        pass1_nll_scores={0: [3.0, 2.5, float("nan")], 1: [1.0, float("inf")]},
        pass1_subturn_nll_scores={(0, 1): [0.5, 0.25]},
        placed_backchannels=[pbc],
        pass2_nll_scores={(0, 0): [0.1, -0.2]},
        refined_placements=[p0, pbc, p1],
        pass3_adjustments={0: {"original_turn_end": 35, "new_turn_end": 37, "total_shift": 2}},
        pass3_sub_segment_details={0: []},
        diagnostics={"pass1": {}, "pass2": {}, "pass3": {}, "note": ("a", 1), "obj": object},
    )


def state_from_timeline(data: dict) -> SchedulingState:
    """Rebuild a state from an exported ``timeline.json`` dictionary."""
    def ti(d):
        return TurnInfo(d["turn_id"], d["channel"], d["start_token"], d["end_token"], d["duration"],
                        d["sub_segments"], d["is_backchannel"], d["host_turn_id"], [], d["bc_group_index"])

    def pt(d):
        return PlacedTurn(ti(d), d["onset_step"], d["gap_to_previous"], d["bc_offset_from_host_start"])

    return SchedulingState(
        primary_turns=[ti(d) for d in data["primary_turns"]],
        backchannel_turns=[ti(d) for d in data["backchannel_turns"]],
        backchannel_units=[ti(d) for d in data["backchannel_units"]],
        placed_primary=[pt(d) for d in data["placed_primary"]],
        pass1_nll_scores=data["pass1_nll_scores"],
        pass1_subturn_nll_scores=data["pass1_subturn_nll_scores"],
        placed_backchannels=[pt(d) for d in data["placed_backchannels"]],
        pass2_nll_scores=data["pass2_nll_scores"],
        refined_placements=[pt(d) for d in data["refined_placements"]],
        pass3_adjustments=data["pass3_adjustments"],
        pass3_sub_segment_details=data["pass3_sub_segment_details"],
        diagnostics=data["diagnostics"],
    )


def load_module(path, name):
    """Import a file of the research tree without writing bytecode into it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def png_pixels(path):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


# ---------------------------------------------------------------------------
# unit / smoke tests
# ---------------------------------------------------------------------------

def test_export_schema_and_json_safety(tmp_path):
    out = tmp_path / "sub" / "timeline.json"
    text = timeline.export_scheduling_state_json(synthetic_state(), str(out))
    data = json.loads(out.read_text())
    assert json.loads(text) == data
    assert data["meta"] == {"token_rate_hz": 12.5}
    assert set(data) == {
        "meta", "primary_turns", "backchannel_turns", "backchannel_units", "placed_primary",
        "pass1_nll_scores", "pass1_subturn_nll_scores", "placed_backchannels", "pass2_nll_scores",
        "refined_placements", "pass3_adjustments", "pass3_sub_segment_details", "diagnostics",
    }
    placed = data["refined_placements"][1]
    assert placed["is_backchannel"] and placed["host_turn_id"] == 0
    assert placed["end_step"] == placed["onset_step"] + placed["duration"] == 25
    assert "speech_token_ids" not in placed
    assert data["pass1_nll_scores"] == {"0": [3.0, 2.5, None], "1": [1.0, None]}
    assert data["pass1_subturn_nll_scores"] == {"(0, 1)": [0.5, 0.25]}
    assert data["pass2_nll_scores"] == {"(0, 0)": [0.1, -0.2]}
    assert data["diagnostics"]["note"] == ["a", 1]


def test_sub_segment_spans():
    placement = {"onset_step": 5, "duration": 30,
                 "sub_segments": [{"start": 0.0, "end": 1.0}, {"start": 1.4, "end": 2.4}]}
    assert timeline._compute_sub_segment_spans(placement, 12.5) == [(5, 18), (23, 35)]
    assert timeline._compute_sub_segment_spans({"onset_step": 0, "duration": 4, "sub_segments": []}, 12.5) == []


def test_truncate_label():
    assert timeline._truncate_label("  hello   world ", 20) == "hello world"
    assert timeline._truncate_label("one two three four five six", 12).endswith("...")
    assert timeline._truncate_label("", 5) == ""


def test_parse_input_spec(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text("[]")
    assert timeline.parse_input_spec("a/timeline.json") == ("a/timeline.json", "timeline", None)
    assert timeline.parse_input_spec("a.json:fp:hp") == ("a.json", "fp:hp", None)
    assert timeline.parse_input_spec(f"a.json:run:{meta}:") == ("a.json", "run", (str(meta), None))
    assert timeline.parse_input_spec("a.json:x:nofile.json:nofile2.json") == (
        "a.json", "x:nofile.json:nofile2.json", None)


def _write_run(root, name, state):
    """Lay out ``<root>/<name>/ckpt/sample_0/segment_search_v2/timeline.json`` with a segment cache."""
    run = root / name / "ckpt"
    tl = run / "sample_0" / "segment_search_v2" / "timeline.json"
    timeline.export_scheduling_state_json(state, str(tl))
    cache = run / "segment_cache" / "example_0000"
    cache.mkdir(parents=True)
    (cache / "turn_metadata.json").write_text(json.dumps([
        {"turn_id": 0, "text": "So I was thinking about the trip we planned last year"},
        {"turn_id": 1, "text": "Right."},
    ]))
    (cache / "bc_decisions.json").write_text(json.dumps([{"turn_id": 0, "bc_word": "uh-huh"}]))
    return str(tl)


def test_comparison_html_png_and_scan_cli(tmp_path, capsys):
    tl_a = _write_run(tmp_path, "no_bc", synthetic_state())
    tl_b = _write_run(tmp_path, "with_bc", synthetic_state())
    assert timeline._discover_metadata_paths(tl_a)[0].endswith("turn_metadata.json")
    labels = timeline._load_turn_labels(tl_a, max_chars=20)
    assert labels["bc"] == {0: "uh-huh"} and labels["primary"][1] == "Right."
    assert labels["primary"][0].endswith("...")

    assert timeline.find_timelines(str(tmp_path)) == [(tl_a, "no_bc"), (tl_b, "with_bc")]

    timeline.main(["--scan_dir", str(tmp_path)])
    html = (tmp_path / "timeline_comparison.html").read_text()
    assert "__DATASETS_JSON__" not in html and '"name": "with_bc"' in html and "uh-huh" in html
    png = tmp_path / "timeline_comparison.png"
    assert png.stat().st_size > 1000
    assert "Found 2 timeline(s)" in capsys.readouterr().out

    out = tmp_path / "explicit" / "cmp.html"
    timeline.main([f"{tl_a}:baseline", "-o", str(out), "--no-png"])
    assert out.exists() and not (tmp_path / "explicit" / "cmp.png").exists()

    with pytest.raises(SystemExit):
        timeline.main([f"{tl_a}:baseline"])


def test_three_pass_plots_render_from_minimal_state(tmp_path):
    three_pass.visualize_scheduling(synthetic_state(), save_dir=str(tmp_path), show=False)
    assert (tmp_path / "timeline.png").stat().st_size > 1000


# ---------------------------------------------------------------------------
# equivalence with the research tree
# ---------------------------------------------------------------------------

def _real_timelines(ntpp_orig, limit=4):
    paths = sorted(glob.glob(str(ntpp_orig / PIPELINE_DIR / "*" / "*" / "*" / "*" / "sample_*" /
                                 "segment_search_v2" / "timeline.json")))
    with_bc = [p for p in paths if json.load(open(p))["placed_backchannels"]]
    without = [p for p in paths if p not in with_bc]
    chosen = with_bc[:limit - 1] + without[:1]
    if not chosen:
        pytest.skip("no timeline.json outputs found in the original tree")
    return chosen


@pytest.mark.equivalence
def test_export_matches_original(ntpp_orig, tmp_path):
    orig = load_module(ntpp_orig / PIPELINE_DIR / "visualize_multiple_schedule.py", "orig_timeline")
    search = load_module(ntpp_orig / PIPELINE_DIR / "segment_level_search_v6.py", "orig_search_v6")

    state = synthetic_state(types=search)
    assert timeline.export_scheduling_state_json(state) == orig.export_scheduling_state_json(state)

    for path in _real_timelines(ntpp_orig):
        text = open(path).read()
        rebuilt = state_from_timeline(json.loads(text))
        new = timeline.export_scheduling_state_json(rebuilt, str(tmp_path / "t.json"))
        assert new == orig.export_scheduling_state_json(rebuilt)
        # Files written by the research pipeline round-trip through the exporter unchanged.
        assert json.loads(new) == json.loads(text)


@pytest.mark.equivalence
def test_comparison_outputs_match_original(ntpp_orig, tmp_path):
    orig = load_module(ntpp_orig / PIPELINE_DIR / "visualize_multiple_schedule.py", "orig_timeline")
    pairs = [(p, f"run {i}") for i, p in enumerate(_real_timelines(ntpp_orig))]

    new_html = timeline.visualize_timeline_comparison(pairs, str(tmp_path / "new.html"), max_label_chars=30)
    old_html = orig.visualize_timeline_comparison(pairs, str(tmp_path / "old.html"), max_label_chars=30)
    assert open(new_html).read() == open(old_html).read()

    for kwargs in ({}, {"fig_width": 3.5, "max_label_chars": 30, "show_primary_labels": True}):
        new_png = timeline.render_timeline_comparison_png(pairs, str(tmp_path / "new.png"), **kwargs)
        old_png = orig.render_timeline_comparison_png(pairs, str(tmp_path / "old.png"), **kwargs)
        assert (png_pixels(new_png) == png_pixels(old_png)).all()


@pytest.mark.equivalence
def test_three_pass_plots_match_original(ntpp_orig, tmp_path, capsys):
    orig = load_module(ntpp_orig / PIPELINE_DIR / "visualize_3_pass_v4.py", "orig_three_pass")
    names = ["timeline.png", "pass1_nll.png", "pass1_shifts.png", "pass2_nll.png",
             "pass2_strategies.png", "pass3_nll.png", "pass3_summary.png"]
    compared = 0
    for i, path in enumerate(_real_timelines(ntpp_orig)):
        state = state_from_timeline(json.load(open(path)))
        new_dir, old_dir = tmp_path / f"new{i}", tmp_path / f"old{i}"
        three_pass.visualize_scheduling(state, save_dir=str(new_dir), show=False)
        new_out = capsys.readouterr().out
        orig.visualize_scheduling(state, save_dir=str(old_dir), show=False)
        assert new_out == capsys.readouterr().out
        assert sorted(os.listdir(new_dir)) == sorted(os.listdir(old_dir))
        for name in os.listdir(new_dir):
            assert name in names
            assert (png_pixels(new_dir / name) == png_pixels(old_dir / name)).all(), name
            compared += 1
    assert compared >= 7
