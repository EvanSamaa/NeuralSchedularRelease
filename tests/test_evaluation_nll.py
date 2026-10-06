"""neural_scheduler.evaluation.nll_diagnostics and onset_benchmark."""

import dataclasses
import json

import numpy as np
import pytest

from neural_scheduler.evaluation import nll_diagnostics as nd
from neural_scheduler.evaluation import onset_benchmark as ob

from .test_evaluation_support import EXPERIMENTS, StubModel, StubTokenizer, load_definitions, load_module, make_samples


@pytest.fixture(scope="module")
def tok():
    return StubTokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return StubModel(tok.vocab_size)


@pytest.fixture(scope="module")
def sequences(tok):
    return nd.build_sequences(tok, make_samples(tok, 3, seed=41, frames=40))


def _plain(obj):
    """Dataclasses -> plain structures with exact float representation."""
    return json.loads(json.dumps(dataclasses.asdict(obj), default=lambda o: float(o) if isinstance(o, np.floating) else int(o)))


def test_build_sequences(tok, sequences):
    t = tok.convert_tokens_to_ids
    assert len(sequences) == 3
    assert sequences[0]["context_ids"][0] == t("<|start-of-script-A|>")
    assert sequences[0]["context_ids"][-1] == t("<|start-of-dialogue|>")
    assert len(sequences[0]["dialogue_ids"]) == 80
    assert sequences[1]["label"] == "sample_1"


def test_outputs(tok, model, sequences, tmp_path):
    results = nd.run_diagnostics(model, tok, sequences, device="cpu", verbose=False)
    assert len(results) == 3 and results[0].total_steps == 40
    assert {e.region for e in results[0].events} <= {"silence", "single_A", "single_B", "overlap"}
    html = nd.generate_visualizations(results, str(tmp_path))
    assert html.endswith("nll_diagnostics.html") and "<html" in open(html).read()
    csv_path = nd.export_events_to_csv(results, str(tmp_path / "events.csv"))
    assert len(open(csv_path).read().splitlines()) == 1 + 3 * 80

    benchmarks = ob.run_onset_benchmark(model, tok, sequences, shift_range=3, max_segments_per_seq=2,
                                        window_padding=4, device="cpu", verbose=False)
    assert benchmarks and any(bm.shifts for bm in benchmarks)
    page = ob.generate_benchmark_viz(benchmarks, str(tmp_path))
    assert "Onset Timing Benchmark" in open(page).read()
    aggregate = ob.generate_aggregate_viz(benchmarks, str(tmp_path / "sub" / "aggregate.html"))
    text = open(aggregate).read()
    assert "__DATA_PLACEHOLDER__" not in text and '"shifts"' in text
    assert ob.AGGREGATE_TEMPLATE.exists()


@pytest.mark.equivalence
def test_build_sequences_equivalence(ntpp_orig, tok):
    """Same context/dialogue split as the driver block of experiment_with_nll.py."""
    orig = load_definitions(ntpp_orig / EXPERIMENTS / "experiment_with_nll.py",
                            ["extract_script_contents", "prepare_inference_input"])
    samples = make_samples(tok, 3, seed=42, frames=20)
    for sample, built in zip(samples, nd.build_sequences(tok, samples)):
        a, b, d = orig["extract_script_contents"](tok, sample["input_ids"], return_dialogue=True)
        context, _ = orig["prepare_inference_input"](a, b, tok, "cpu", already_tokenized=True)
        assert built == {"context_ids": context.tolist()[0], "dialogue_ids": d,
                         "label": "sample_" + str(sample["source_file"])}


@pytest.mark.equivalence
def test_diagnostics_equivalence(ntpp_orig, tok, model, sequences, tmp_path):
    orig = load_module(ntpp_orig / EXPERIMENTS / "nll_diagnostics.py", "orig_nll_diagnostics_b")
    old = orig.run_diagnostics(model, tok, sequences, device="cpu", verbose=False)
    new = nd.run_diagnostics(model, tok, sequences, device="cpu", verbose=False)
    assert [_plain(x) for x in new] == [_plain(x) for x in old]
    old_html = open(orig.generate_visualizations(old, str(tmp_path / "old"))).read()
    new_html = open(nd.generate_visualizations(new, str(tmp_path / "new"))).read()
    assert new_html == old_html
    assert open(nd.export_events_to_csv(new, str(tmp_path / "new.csv"))).read() == \
        open(orig.export_events_to_csv(old, str(tmp_path / "old.csv"))).read()


@pytest.mark.equivalence
def test_onset_benchmark_equivalence(ntpp_orig, tok, model, sequences, tmp_path):
    orig = load_module(ntpp_orig / EXPERIMENTS / "nll_diagnostics_2.py", "orig_nll_diagnostics_2")
    kwargs = dict(shift_range=5, max_segments_per_seq=3, window_padding=6, device="cpu", verbose=False)
    old = orig.run_onset_benchmark(model, tok, sequences, **kwargs)
    new = ob.run_onset_benchmark(model, tok, sequences, **kwargs)
    assert len(new) > 0 and sum(len(bm.shifts) for bm in new) > 10
    assert [_plain(x) for x in new] == [_plain(x) for x in old]
    assert open(ob.generate_benchmark_viz(new, str(tmp_path / "new"))).read() == \
        open(orig.generate_benchmark_viz(old, str(tmp_path / "old"))).read()
    template = ntpp_orig / EXPERIMENTS / "nll_diagnostic_2_template.html"
    old_agg = orig.generate_aggregate_viz(old, template_path=str(template), output_path=str(tmp_path / "old_agg.html"))
    new_agg = ob.generate_aggregate_viz(new, str(tmp_path / "new_agg.html"))
    assert open(new_agg).read() == open(old_agg).read()
