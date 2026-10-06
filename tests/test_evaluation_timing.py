"""neural_scheduler.evaluation.timing_distribution."""

import json

import numpy as np
import pytest
import torch

from neural_scheduler.constants import SILENCE_CODE
from neural_scheduler.evaluation import timing_distribution as td

from .test_evaluation_support import EXPERIMENTS, StubModel, StubTokenizer, load_definitions, make_channel, make_samples

ORIGINAL_NAMES = [
    "MutualSilenceStats", "SimultaneousSpeechStats", "DialogueTimingStats",
    "extract_mutual_silence_segments", "extract_simultaneous_speech_segments", "extract_dialogue_timing_stats",
    "_compare_segment_distributions", "compare_dialogue_timing_distributions", "plot_distribution_only",
    "_stats_to_dict", "_dict_to_stats", "_json_safe", "save_experiment_results_json",
    "load_experiment_results_json", "run_dialogue_timing_experiment",
    "prepare_inference_input", "compute_channel_ids_for_inference", "generate_dialogue", "extract_script_contents",
]


@pytest.fixture(scope="module")
def orig(ntpp_orig):
    return load_definitions(ntpp_orig / EXPERIMENTS / "experiment_with_mutual_silence.py", ORIGINAL_NAMES)


def _channels(seed, frames=120):
    rng = np.random.default_rng(seed)
    return make_channel(rng, frames), make_channel(rng, frames)


def test_extraction_on_known_dialogue():
    s = SILENCE_CODE
    a = [s, s, 1, 1, 1, s, s, 2]
    b = [s, s, s, 3, 3, 3, s, s]
    stats = td.extract_dialogue_timing_stats(a, b)
    assert stats.mutual_silence.mutual_silence_lengths == [2, 1]
    assert stats.mutual_silence.mutual_silence_positions == [0, 6]
    assert stats.simultaneous_speech.simultaneous_speech_lengths == [2]
    assert stats.simultaneous_speech.simultaneous_speech_positions == [3]
    assert stats.single_speaker_tokens == 3
    assert stats.mutual_silence.mutual_silence_ratio == pytest.approx(3 / 8)
    with pytest.raises(AssertionError):
        td.extract_dialogue_timing_stats(a, b[:-1])


def test_json_roundtrip_and_plots(tmp_path):
    gt = [td.extract_dialogue_timing_stats(*_channels(i)) for i in range(6)]
    gen = [td.extract_dialogue_timing_stats(*_channels(100 + i)) for i in range(6)]
    comparison = td.compare_dialogue_timing_distributions(gt, gen, save_path=str(tmp_path / "main.png"))
    td.plot_distribution_only(gt, gen, save_path=str(tmp_path / "dist.png"), num_bins=10, title="t")
    assert (tmp_path / "main.png").stat().st_size > 0 and (tmp_path / "dist.png").stat().st_size > 0
    td.save_experiment_results_json(gt, gen, comparison, {"top_k": 50}, str(tmp_path / "r.json"), {"note": 1})
    loaded = td.load_experiment_results_json(str(tmp_path / "r.json"))
    assert loaded["ground_truth_stats"] == gt and loaded["generated_stats"] == gen
    assert loaded["metadata"] == {"note": 1}
    assert loaded["comparison"]["mutual_silence"]["ks_statistic"] == pytest.approx(
        comparison["mutual_silence"]["ks_statistic"])


def test_cli_plot_only(tmp_path):
    gt = [td.extract_dialogue_timing_stats(*_channels(i)) for i in range(4)]
    gen = [td.extract_dialogue_timing_stats(*_channels(50 + i)) for i in range(4)]
    paths = td.output_paths(str(tmp_path), "run")
    paths["dir"].mkdir(parents=True)
    comparison = td.compare_dialogue_timing_distributions(gt, gen)
    td.save_experiment_results_json(gt, gen, comparison, {}, str(paths["json"]))
    td.main(["--plot_only", "--out_dir", str(tmp_path), "--name", "run", "--num_bins", "8"])
    assert paths["main_png"].exists() and paths["distribution_png"].exists()
    with pytest.raises(SystemExit):
        td.main(["--plot_only", "--out_dir", str(tmp_path), "--name", "absent"])
    with pytest.raises(SystemExit):
        td.main(["--out_dir", str(tmp_path)])


@pytest.mark.equivalence
def test_extraction_equivalence(orig):
    for seed in range(25):
        a, b = _channels(seed, frames=int(60 + seed * 7))
        new = td.extract_dialogue_timing_stats(a, b)
        old = orig["extract_dialogue_timing_stats"](a, b, 8520)
        assert td._stats_to_dict(new) == orig["_stats_to_dict"](old)
    empty_new = td.extract_dialogue_timing_stats([1, 2], [SILENCE_CODE, SILENCE_CODE])
    empty_old = orig["extract_dialogue_timing_stats"]([1, 2], [8520, 8520])
    assert td._stats_to_dict(empty_new) == orig["_stats_to_dict"](empty_old)


@pytest.mark.equivalence
def test_comparison_equivalence(orig, tmp_path):
    gt_ch = [_channels(i) for i in range(8)]
    gen_ch = [_channels(200 + i) for i in range(8)]
    new = td.compare_dialogue_timing_distributions(
        [td.extract_dialogue_timing_stats(a, b) for a, b in gt_ch],
        [td.extract_dialogue_timing_stats(a, b) for a, b in gen_ch],
    )
    old = orig["compare_dialogue_timing_distributions"](
        [orig["extract_dialogue_timing_stats"](a, b) for a, b in gt_ch],
        [orig["extract_dialogue_timing_stats"](a, b) for a, b in gen_ch],
    )
    assert json.dumps(td.to_serializable(new), sort_keys=True) == json.dumps(orig["_json_safe"](old), sort_keys=True)


@pytest.mark.equivalence
def test_experiment_equivalence(orig, tmp_path):
    """Same samples, same sampler seed: the saved experiment JSON is identical."""
    tok = StubTokenizer()
    model = StubModel(tok.vocab_size)
    samples = make_samples(tok, 5, seed=21, frames=30)
    samples.insert(2, {"input_ids": samples[0]["input_ids"][:-70]})  # dialogue shorter than 10 tokens
    params = {"max_new_tokens": 40, "temperature": 1.0, "top_k": 50, "top_p": 0.95, "do_sample": True}

    torch.manual_seed(3)
    old = orig["run_dialogue_timing_experiment"](
        model, tok, samples, tok.offset, device="cpu", num_samples=5, generation_params=dict(params),
        verbose=False, save_path=str(tmp_path / "old" / "exp.png"),
    )
    torch.manual_seed(3)
    new = td.run_dialogue_timing_experiment(
        model, tok, samples, tok.offset, device="cpu", num_samples=5, generation_params=dict(params),
        verbose=False, out_dir=str(tmp_path / "new"), name="exp",
    )
    assert len(new["generated_stats"]) == len(old["generated_stats"]) > 0
    old_json = json.loads((tmp_path / "old" / "exp" / "exp.json").read_text())
    new_json = json.loads((tmp_path / "new" / "exp" / "exp.json").read_text())
    assert new_json == old_json
    for name in ("exp.png", "exp_distribution_only.png"):
        assert (tmp_path / "new" / "exp" / name).exists()
