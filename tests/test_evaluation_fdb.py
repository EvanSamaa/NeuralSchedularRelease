"""neural_scheduler.evaluation.fdb: ICC preparation and backchannel metrics."""

import json
import re
import shutil
import wave

import numpy as np
import pytest

from neural_scheduler.evaluation.fdb import eval_backchannel as eb
from neural_scheduler.evaluation.fdb import prepare_icc as pi

from .test_evaluation_support import load_module

BC_ONLY = "ntpp/model/chatGLM_dual_timing_only/experiments/inference_pipeline_bc_only_full_duplex_bench"


# --- synthetic benchmark data ---------------------------------------------------

def _write_wav(path, seconds, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(seconds * sr)) * 3000).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def _words(rng, n_words, tag_prob=0.15):
    words, t = [], 0.3
    for i in range(n_words):
        if rng.random() < tag_prob:
            words.append({"text": "[BACKCHANNEL]", "timestamp": [round(t, 2), round(t + 0.2, 2)]})
        duration = float(rng.uniform(0.15, 0.5))
        words.append({"text": f"w{i}", "timestamp": [round(t, 2), round(t + duration, 2)]})
        t += duration + (float(rng.uniform(0.7, 1.5)) if rng.random() < 0.25 else float(rng.uniform(0.02, 0.2)))
    if rng.random() < 0.5:
        words.append({"text": "[LAUGH]", "timestamp": [round(t, 2), round(t + 0.2, 2)]})
    return words, t + 0.5


def _make_dataset(root, n=4, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        words, seconds = _words(rng, int(rng.integers(6, 40)))
        (root / str(i)).mkdir(parents=True)
        (root / str(i) / "transcription.json").write_text(json.dumps(words))
        _write_wav(root / str(i) / "input.wav", seconds, seed=i)
    (root / "notes").mkdir()  # non-example directory: reported as an error, not fatal
    return root


def test_words_to_sentences_and_merge():
    w = lambda text, a, b: {"text": text, "timestamp": [a, b]}
    words = [w("a", 0, 1), w("b", 1, 2), w("c", 2, 3), w("d", 3, 4.5), w("[T]", 4.6, 4.7),
             w("e", 6, 6.2), w("[U]", 6.3, 6.4), w("f", 6.5, 9), w("g", 9, 9.5), w("h", 9.6, 10), w("i", 10, 10.4)]
    sentences, tags = pi.words_to_sentences(words, 0.6)
    assert [[x["text"] for x in s] for s in sentences] == [["a", "b", "c", "d"], ["e", "f", "g", "h", "i"]]
    assert [(t["tag"], t["sentence_idx"]) for t in tags] == [("[T]", 1), ("[U]", 1)]
    merged, remapped = pi.merge_short_sentences([sentences[0][:2], sentences[0][2:], sentences[1]], tags)
    assert [len(s) for s in merged] == [4, 5] and [t["sentence_idx"] for t in remapped] == [0, 0]
    assert pi.words_to_sentences([]) == ([], [])


def test_prepare_writes_outside_dataset_with_relative_paths(tmp_path):
    dataset = _make_dataset(tmp_path / "icc")
    before = sorted(p.relative_to(dataset) for p in dataset.rglob("*"))
    pi.main(["--dataset_dir", str(dataset), "--out_dir", str(tmp_path / "prepared")])
    assert sorted(p.relative_to(dataset) for p in dataset.rglob("*")) == before
    jsonl = tmp_path / "prepared" / "icc_backchannel_bc_gt.jsonl"
    records = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert [r["dialogue_id"] for r in records] == ["0", "1", "2", "3"]
    for rec in records:
        assert rec["meta_path"] == f"{rec['dialogue_id']}/meta.json"
        meta = json.loads((jsonl.parent / rec["meta_path"]).read_text())
        for turn, meta_turn in zip(rec["turns"], meta["turns"]):
            assert (jsonl.parent / turn["audio_path"]).exists()
            assert meta_turn["audio_path"].startswith("split_audio/")
            assert not turn["audio_path"].startswith("/")
    with pytest.raises(SystemExit):
        pi.main(["--dataset_dir", str(dataset)])
    with pytest.raises(SystemExit):
        pi.main(["--dataset_dir", str(dataset), "--out_dir", str(tmp_path / "x"), "--in_place"])


@pytest.mark.equivalence
def test_prepare_in_place_equivalence(ntpp_orig, tmp_path):
    """--in_place --absolute_paths writes what the research script wrote."""
    _make_dataset(tmp_path / "a", n=6, seed=5)
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    orig = load_module(ntpp_orig / BC_ONLY / "eval_full_duplex_bench_prep_data.py", "orig_fdb_prep")
    orig.DATASET_DIR = str(tmp_path / "a")
    orig.OUTPUT_JSONL = str(tmp_path / "a_gt.jsonl")
    orig.main()
    pi.main(["--dataset_dir", str(tmp_path / "b"), "--in_place", "--absolute_paths",
             "--output_jsonl", str(tmp_path / "b_gt.jsonl")])

    old_text = (tmp_path / "a_gt.jsonl").read_text().replace(str(tmp_path / "a"), "ROOT")
    new_text = (tmp_path / "b_gt.jsonl").read_text().replace(str(tmp_path / "b"), "ROOT")
    assert new_text == old_text and len(new_text.splitlines()) == 6
    old_files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    new_files = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert new_files == old_files and any(f.suffix == ".wav" and "split_audio" in f.parts for f in new_files)
    for rel in new_files:
        old_bytes = (tmp_path / "a" / rel).read_bytes().replace(str(tmp_path / "a").encode(), b"ROOT")
        new_bytes = (tmp_path / "b" / rel).read_bytes().replace(str(tmp_path / "b").encode(), b"ROOT")
        assert new_bytes == old_bytes, rel


# --- metrics --------------------------------------------------------------------

def test_split_and_jsd():
    segments = [{"start": 0.5, "end": 1.0}, {"start": 2.0, "end": 5.5}, {"start": 7.0, "end": 10.0}]
    backchannels, tor = eb.split_backchannels(segments)
    assert backchannels == [[0.5, 1.0], [7.0, 10.0]] and tor == 1
    assert eb.timing_jsd([], 10.0, None) == 1
    uniform = [1 / 51] * 51
    most = eb.timing_jsd([[0.0, 9.0]], 10.0, uniform)
    early = eb.timing_jsd([[0.0, 1.0]], 10.0, uniform)
    assert 0 < most < 0.3 < early < 1
    with pytest.raises(KeyError):
        eb.timing_jsd([[0.0, 1.0]], 10.0, None)
    scored = eb.score_sample(segments, 20.0, [0.1] * 10)
    assert scored["tor"] == 1 and scored["bc_count"] == 2 and scored["frequency"] == pytest.approx(0.1)


def _make_predictions(root, layout, rng, sr):
    """Sample directories plus the VAD segments a stub VAD will report for each wav."""
    vad_by_length, gt = {}, {}
    for i in range(7):
        seconds = 8.0 + i  # distinct lengths identify the file inside the stub
        if layout == "scheduler":
            wav_path = root / f"sample_{i}" / "segment_search_v2" / "channel_b.wav"
        else:
            wav_path = root / str(i) / "output.wav"
        if i == 3:
            wav_path.parent.parent.mkdir(parents=True, exist_ok=True) if layout == "scheduler" else \
                wav_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            _write_wav(wav_path, seconds, sr=sr, seed=i)
        segments, t = [], 0.4
        for _ in range(int(rng.integers(0, 5)) if i != 5 else 0):
            length = float(rng.choice([0.4, 0.9, 3.6]))
            segments.append({"start": round(t, 1), "end": round(t + length, 1)})
            t += length + 0.7
        vad_by_length[int(seconds * sr)] = segments
        dist = rng.random(int(rng.integers(20, 60))) + 0.01
        gt[str(i)] = (dist / dist.sum()).tolist()
    (root / "README.txt").write_text("not a sample")
    (root / "gt.json").write_text(json.dumps(gt))
    return vad_by_length


@pytest.fixture
def stub_vad(monkeypatch):
    import silero_vad

    table = {}
    monkeypatch.setattr(silero_vad, "load_silero_vad", lambda *a, **k: None)
    monkeypatch.setattr(silero_vad, "get_speech_timestamps",
                        lambda wav, model, return_seconds=True, **k: table[wav.shape[-1]])
    return table


def test_layouts_and_missing_modes(tmp_path, stub_vad):
    rng = np.random.default_rng(3)
    stub_vad.update(_make_predictions(tmp_path / "fdb", "fdb", rng, 16000))
    results = eb.eval_backchannel(str(tmp_path / "fdb"), str(tmp_path / "fdb" / "gt.json"), layout="fdb", verbose=False)
    assert results["summary"]["num_samples"] == 6 and results["summary"]["skipped"] == ["3"]
    silent = eb.eval_backchannel(str(tmp_path / "fdb"), str(tmp_path / "fdb" / "gt.json"), layout="fdb",
                                 missing="silent", verbose=False)
    assert silent["summary"]["num_samples"] == 7
    assert silent["per_sample"]["3"] == {"jsd": 1, "tor": 0, "frequency": 0.0, "bc_count": 0}
    assert silent["per_sample"]["5"]["jsd"] == 1  # audio present, no backchannel detected
    out = tmp_path / "scores" / "r.json"
    eb.main(["--pred_root", str(tmp_path / "fdb"), "--gt_distribution", str(tmp_path / "fdb" / "gt.json"),
             "--layout", "fdb", "--output_json", str(out)])
    assert json.loads(out.read_text())["summary"]["num_samples"] == 6


def test_real_vad_runs(tmp_path):
    """The call into silero-vad works with the installed package (no stub)."""
    _write_wav(tmp_path / "sample_0" / "segment_search_v2" / "channel_b.wav", 3.0, sr=22050)
    (tmp_path / "gt.json").write_text(json.dumps({"0": [0.5, 0.5]}))
    for resample in (False, True):
        results = eb.eval_backchannel(str(tmp_path), str(tmp_path / "gt.json"), resample_for_vad=resample, verbose=False)
        assert results["summary"]["num_samples"] == 1


@pytest.mark.equivalence
def test_metrics_equivalence_with_patched_fdb_script(ntpp_orig, tmp_path, stub_vad, capsys):
    """Same per-sample and aggregate numbers as the locally patched Full-Duplex-Bench script."""
    rng = np.random.default_rng(11)
    stub_vad.update(_make_predictions(tmp_path, "scheduler", rng, 22050))

    source = (ntpp_orig / "Full-Duplex-Bench/v1_v1.5/evaluation/eval_backchannel.py").read_text()
    source, n_dir = re.subn(r'(?m)^    data_dir = ".*"$', f'    data_dir = {str(tmp_path)!r}', source)
    source, n_gt = re.subn(r'(?m)^    gt_distribution_path = ".*"$',
                           f'    gt_distribution_path = {str(tmp_path / "gt.json")!r}', source)
    assert n_dir == 1 and n_gt == 1
    namespace = {"__name__": "patched_fdb_eval"}
    exec(compile(source, "patched_eval_backchannel.py", "exec"), namespace)
    capsys.readouterr()
    namespace["eval_backchannel"]("")
    old_out = capsys.readouterr().out

    results = eb.eval_backchannel(str(tmp_path), str(tmp_path / "gt.json"))
    new_out = capsys.readouterr().out

    old_samples = dict(re.findall(r"Speaker (\d+) - (JSD: [\d.]+, BC Count: \d+)", old_out))
    new_samples = dict(re.findall(r"Sample (\d+) - (JSD: [\d.]+, BC Count: \d+)", new_out))
    assert new_samples == old_samples and len(new_samples) == 6
    for metric in ("JSD", "TOR", "Frequency"):
        pattern = rf"{metric} - Mean: [\d.]+ ± [\d.]+"
        assert re.search(pattern, new_out).group(0) == re.search(pattern, old_out).group(0)
    assert "Number of samples: 6" in old_out and results["summary"]["num_samples"] == 6
    assert "Skipping 3" in old_out and results["summary"]["skipped"] == ["3"]
    assert any(r["tor"] == 1 for r in results["per_sample"].values())
