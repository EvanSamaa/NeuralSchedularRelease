"""Command-line entry points of the evaluations, run end to end with a stub model."""

import json

import pytest

from neural_scheduler.evaluation import common, nll_diagnostics, silence_perturbation, timing_distribution, turn_order

from .test_evaluation_support import StubModel, StubTokenizer, make_samples


@pytest.fixture
def stub_setup(tmp_path, monkeypatch):
    tok = StubTokenizer()
    model = StubModel(tok.vocab_size)
    data = tmp_path / "val.jsonl"
    data.write_text("\n".join(json.dumps(s) for s in make_samples(tok, 6, seed=51, frames=30)))
    loaded = []

    def fake_load(checkpoint, device):
        loaded.append((checkpoint, device))
        return model, tok

    monkeypatch.setattr(common, "load_model_and_tokenizer", fake_load)
    return ["--checkpoint", "CKPT", "--data_jsonl", str(data), "--device", "cpu"], tmp_path / "out", loaded


def test_timing_distribution_cli_is_reproducible_with_seed(stub_setup):
    base, out, loaded = stub_setup
    for name in ("first", "second"):
        timing_distribution.main(base + ["--out_dir", str(out), "--name", name, "--num_samples", "3",
                                         "--pool_size", "4", "--max_new_tokens", "30", "--seed", "7"])
    first = json.loads((out / "first" / "first.json").read_text())
    second = json.loads((out / "second" / "second.json").read_text())
    assert first["metadata"]["seed"] == 7 and first["metadata"]["checkpoint"] == "CKPT"
    assert first["generated_stats"] == second["generated_stats"] and first["ground_truth_stats"]
    assert (out / "first" / "first_distribution_only.png").exists()
    assert loaded[0] == ("CKPT", "cpu")


def test_turn_order_cli(stub_setup):
    base, out, _ = stub_setup
    turn_order.main(base + ["--out_dir", str(out), "--num_samples", "3", "--seed", "1", "--num_pairs_per_sample", "2"])
    results = json.loads((out / "turn_order_results.json").read_text())
    assert results["summary"]["n_pairs_tested"] == len(results["experiments"]) > 0
    assert (out / "turn_order.png").exists() and (out / "turn_order_by_speaker.png").exists()


def test_silence_perturbation_cli(stub_setup):
    base, out, _ = stub_setup
    silence_perturbation.main(base + ["--out_dir", str(out), "--num_samples", "2", "--seed", "1", "--runs", "all",
                                      "short", "--num_segments_per_sample", "2", "--delta_min", "-2",
                                      "--delta_max", "3"])
    everything = json.loads((out / "multi_sample_aggregate.json").read_text())
    assert everything["experiments"][0]["perturbations"] == [-2, -1, 0, 1, 2]
    short = json.loads((out / "multi_sample_aggregate_short_segments.json").read_text())
    assert short["filter_params"]["max_segment_length"] == 15
    assert (out / "multi_sample_aggregate.png").exists()
    assert not (out / "multi_sample_aggregate_long_segments.png").exists()


def test_nll_diagnostics_cli(stub_setup):
    base, out, _ = stub_setup
    nll_diagnostics.main(base + ["--out_dir", str(out), "--pool_size", "4", "--num_sequences", "2", "--seed", "1",
                                 "--export_csv", "--onset_benchmark", "--shift_range", "2",
                                 "--max_segments_per_seq", "1"])
    for name in ("nll_diagnostics.html", "nll_events.csv", "onset_benchmark.html", "aggregate_benchmark.html"):
        assert (out / name).stat().st_size > 0
