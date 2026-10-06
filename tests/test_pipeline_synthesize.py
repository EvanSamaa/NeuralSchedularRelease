"""Step 1 (``neural_scheduler.pipeline.synthesize``).

The equivalence tests run the original research script and the refactored
module on the same inputs with the same stand-in TTS, classifier and VAD, and
require identical caches (manifests, decisions, metadata, audio).
"""

import importlib.util
import json
import math
import os
import random
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from neural_scheduler.pipeline import synthesize

EXPERIMENTS = "ntpp/model/chatGLM_dual_timing_only/experiments"
ORIG_DIR = f"{EXPERIMENTS}/inference_pipeline_bc_only_full_duplex_bench"
ORIG_STEP1 = f"{ORIG_DIR}/step_1_no_tts_classifier_voiceline_given_with_audio.py"
ORIG_STEP1_TURBO = f"{ORIG_DIR}/step_1_no_tts_classifier_voiceline_given_with_audio_turbo.py"

SR = 24000
TTS_RNG_SEED = 1234
VOCAB = {"Acknowledgment": ["uh-huh", "yeah", "right on"], "Reactive": ["wow", "no way"]}
TIMING = dict(inter_turn_silence_ms=700, intra_turn_pause_ms=400, bc_silence_padding_ms=250, voice_line_padding_ms=150)


# --------------------------------------------------------------------------- #
# Stand-ins shared by both implementations
# --------------------------------------------------------------------------- #


def fake_wave(text: str) -> torch.Tensor:
    """A waveform drawn from torch's global RNG, so seeding order shows up in the output."""
    return torch.rand(900 + 61 * len(text)) * 0.4 - 0.2


class FakeChatterboxTTS:
    """What the original script believes is ``chatterbox.tts.ChatterboxTTS``."""

    sr = SR

    @classmethod
    def from_pretrained(cls, device=None):
        return cls()

    def generate(self, text, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5):
        return fake_wave(text)


class FakeTurboTTS(FakeChatterboxTTS):
    def generate(self, text, audio_prompt_path=None):
        return fake_wave(text)


class FakeBackend:
    """TTS backend for the refactored module."""

    sr = SR

    def generate(self, text, audio_prompt_path):
        return fake_wave(text).unsqueeze(0).cpu()


class FakeClassifier:
    """Deterministic joint classifier: probabilities are a hash of the text context and the audio."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.calls = []

    def build_input_text(
        self, segments, target_turn_idx, context_window_turns=5, context_window_future_turns=5, include_bc_context=True
    ):
        target = segments[target_turn_idx]["speaker"]
        lines = []
        first = max(0, target_turn_idx - context_window_turns)
        last = min(len(segments) - 1, target_turn_idx + context_window_future_turns)
        for i in range(first, last + 1):
            seg = segments[i]
            line = ("A: " if seg["speaker"] == target else "B: ") + " ".join(seg["text"].split())
            if i < target_turn_idx and include_bc_context and seg.get("bc_class", "NoBackchannel") != "NoBackchannel":
                line += f" [BC:{seg['bc_class']}]"
            if i == target_turn_idx:
                line += " <bc_slot>"
            lines.append(line)
        return " </s> ".join(lines)

    def predict(self, input_text, audio=None, audio_start_sec=None, audio_end_sec=None, audio_sample_rate=None):
        if isinstance(audio, str):
            audio_key = f"path:{os.path.getsize(audio)}"
        elif audio is None:
            audio_key = "none"
        else:
            audio_key = f"wave:{tuple(audio.shape)}:{audio_sample_rate}"
        self.calls.append((input_text, audio_key))
        h = zlib.crc32(f"{input_text}|{audio_key}".encode())
        ack = (h % 997) / 997.0
        react = ((h // 997) % 991) / 991.0
        total = ack + react + 0.8
        probs = {"NoBackchannel": 0.8 / total, "Acknowledgment": ack / total, "Reactive": react / total}
        if probs["Acknowledgment"] + probs["Reactive"] >= self.threshold:
            label = "Acknowledgment" if probs["Acknowledgment"] >= probs["Reactive"] else "Reactive"
        else:
            label = "NoBackchannel"
        return label, float(probs[label]), probs


def fake_get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=16000):
    n = len(wav)
    return [{"start": 111, "end": n - 222}] if n > 1000 else []


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _load_original(path: Path, name: str, tts_module: str, tts_class_name: str, tts_class):
    """Import an original step-1 script with ``chatterbox`` replaced by a stand-in."""
    touched = ("chatterbox", tts_module, "torchvision._meta_registrations")
    saved = {k: sys.modules.get(k) for k in touched}
    package = types.ModuleType("chatterbox")
    module = types.ModuleType(tts_module)
    setattr(module, tts_class_name, tts_class)
    sys.modules["chatterbox"] = package
    sys.modules[tts_module] = module
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return loaded


@pytest.fixture(scope="module")
def orig(ntpp_orig):
    return _load_original(ntpp_orig / ORIG_STEP1, "orig_step1_with_audio", "chatterbox.tts", "ChatterboxTTS", FakeChatterboxTTS)


@pytest.fixture(scope="module")
def orig_turbo(ntpp_orig):
    return _load_original(
        ntpp_orig / ORIG_STEP1_TURBO, "orig_step1_turbo", "chatterbox.tts_turbo", "ChatterboxTurboTTS", FakeTurboTTS
    )


@pytest.fixture(autouse=True)
def _fake_vad(monkeypatch):
    import neural_scheduler.data.vad as vad

    monkeypatch.setattr(vad, "load_silero_vad", lambda: (object(), fake_get_speech_timestamps))


@pytest.fixture
def inputs(tmp_path):
    """Voice lines, a dialogue JSON, text JSONLs, a benchmark JSONL and a vocabulary file."""
    rng = np.random.RandomState(7)
    audio_dir = tmp_path / "inputs" / "audio"
    audio_dir.mkdir(parents=True)
    speakers = [1, 2, 2, 1, 2, 1]
    texts = [
        "I finally fixed the bike.",
        "Oh nice, what was wrong with it?",
        "Was it the chain again?",
        "No, the rear brake cable had snapped.",
        "That explains the noise.",
        "Right, it is quiet now.",
    ]
    turns = []
    for i, (speaker, text) in enumerate(zip(speakers, texts)):
        sr = 22050 if i % 2 == 0 else 16000
        samples = rng.uniform(-0.3, 0.3, size=(int(sr * (0.4 + 0.1 * i)), 2 if i == 3 else 1)).astype("float32")
        path = audio_dir / f"turn{i}_s{speaker}.wav"
        sf.write(path, samples.squeeze(), sr)
        turns.append({"turn_id": i, "speaker": speaker, "text": text, "audio_path": str(path)})

    dialogue_json = tmp_path / "inputs" / "bike.json"
    dialogue_json.write_text(json.dumps({"dialogue_id": "bike", "turns": turns}))
    relative = [dict(t, audio_path=os.path.relpath(t["audio_path"], dialogue_json.parent)) for t in turns]
    relative_json = tmp_path / "inputs" / "bike_relative.json"
    relative_json.write_text(json.dumps({"dialogue_id": "bike", "turns": relative}))

    text_jsonl = tmp_path / "inputs" / "scenes.jsonl"
    scenes = [
        "[S1] Did you see the forecast?\nIt looks like rain all week. [S2] Again? [S1] Afraid so. [S2] I'll bring a coat then.",
        "  [S2] Lunch at noon? [S1] Make it one.\n I have a call before. [S1] Actually, one thirty. ",
    ]
    text_jsonl.write_text("\n".join(json.dumps({"text": t}) for t in scenes) + "\n")

    bc_gt_jsonl = tmp_path / "inputs" / "bc_gt.jsonl"
    records = []
    for dialogue_id, subset in (("0", turns[:4]), ("3", turns[2:])):
        rec_turns = [dict(t, slice_start_sec=1.5 * k, slice_end_sec=1.5 * k + 1.0) for k, t in enumerate(subset)]
        gt = [{"turn_id": t["turn_id"], "bc_count": t["turn_id"] % 2, "bc_events": []} for t in subset]
        records.append({"dialogue_id": dialogue_id, "turns": rec_turns, "tags_removed": 2, "gt_bc_per_turn": gt})
    bc_gt_jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    vocab_json = tmp_path / "inputs" / "vocab.json"
    vocab_json.write_text(json.dumps(VOCAB))
    for name in ("ref1.wav", "ref2.wav"):
        (tmp_path / "inputs" / name).write_bytes(b"")
    return {
        "dialogue_json": str(dialogue_json),
        "relative_json": str(relative_json),
        "text_jsonl": str(text_jsonl),
        "bc_gt_jsonl": str(bc_gt_jsonl),
        "vocab_json": str(vocab_json),
        "ref1": str(tmp_path / "inputs" / "ref1.wav"),
        "ref2": str(tmp_path / "inputs" / "ref2.wav"),
        "new_out": str(tmp_path / "new"),
        "orig_out": str(tmp_path / "orig"),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def run_new(inputs, extra, classifier=None, absolute=True):
    argv = ["--out_dir", inputs["new_out"], "--ref_s1", inputs["ref1"], "--ref_s2", inputs["ref2"], "--device", "cpu"]
    for key, value in TIMING.items():
        argv += [f"--{key}", str(value)]
    if absolute:
        argv.append("--absolute_paths")
    torch.manual_seed(TTS_RNG_SEED)
    synthesize.main(argv + extra, tts=FakeBackend(), classifier=classifier)


def make_orig_generator(module, inputs, master_vocab):
    generator = module.ChatterboxDialogueGenerator(
        speaker1_ref_audio=inputs["ref1"],
        speaker2_ref_audio=inputs["ref2"],
        device="cpu",
        bc_all_words=master_vocab,
        exaggeration=0.5,
        cfg_weight=0.5,
        **TIMING,
    )
    generator._silero_vad_model = object()
    generator._silero_vad_utils = (fake_get_speech_timestamps,)
    torch.manual_seed(TTS_RNG_SEED)
    return generator


def master_vocab_with_classifier(module, vocab_path):
    vocab = module.load_bc_vocab_per_class(vocab_path)
    return vocab, sorted({"hmmm", "uhhmm"} | set(module.collect_all_bc_words(vocab)))


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def assert_same_tree(orig_root, new_root):
    def files(root):
        return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())

    assert files(orig_root) == files(new_root)
    assert files(new_root), "nothing was written"
    for rel in files(new_root):
        a, b = os.path.join(orig_root, rel), os.path.join(new_root, rel)
        if rel.endswith(".json"):
            assert json.load(open(a)) == json.load(open(b)), rel
        elif rel.endswith(".wav"):
            wav_a, sr_a = sf.read(a)
            wav_b, sr_b = sf.read(b)
            assert sr_a == sr_b, rel
            assert wav_a.shape == wav_b.shape and np.array_equal(wav_a, wav_b), rel
        else:
            raise AssertionError(f"unexpected file {rel}")


# --------------------------------------------------------------------------- #
# Equivalence with the original script
# --------------------------------------------------------------------------- #


@pytest.mark.equivalence
@pytest.mark.parametrize(
    "threshold, suppress, context_turns",
    [(0.55, True, 5), (0.0, True, 2), (0.0, False, 5), (2.0, True, 5)],
    ids=["mixed", "always_bc_suppressed", "always_bc", "none_predicted_one_forced"],
)
def test_voice_line_dialogue_matches_original(orig, inputs, threshold, suppress, context_turns):
    seed = 42
    vocab, master = master_vocab_with_classifier(orig, inputs["vocab_json"])
    generator = make_orig_generator(orig, inputs, master)
    orig_classifier = FakeClassifier(threshold)
    dialogue = orig.load_dialogue_json(inputs["dialogue_json"])
    cache_dir = os.path.join(inputs["orig_out"], "segment_cache", dialogue["dialogue_id"])
    _, metadata, _ = generator.generate_dialogue_from_turns(
        turns=dialogue["turns"],
        cache_dir=cache_dir,
        classifier=orig_classifier,
        bc_vocab_per_class=vocab,
        context_window_turns=context_turns,
        context_window_future_turns=3,
        include_bc_context=True,
        suppress_consecutive_same_speaker_bcs=suppress,
        seed=seed,
        save_combined=True,
    )
    write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)
    _, naive_metadata = generator.generate_naive_concatenation(turns=dialogue["turns"], cache_dir=os.path.join(cache_dir, "naive"))
    write_json(os.path.join(cache_dir, "naive", "turn_metadata.json"), naive_metadata)

    new_classifier = FakeClassifier(threshold)
    extra = ["--input_dialogue_json", inputs["dialogue_json"], "--bc_vocab_per_class", inputs["vocab_json"]]
    extra += ["--seed", str(seed), "--bc_context_window_turns", str(context_turns)]
    extra += ["--bc_context_window_future_turns", "3", "--run_naive_only"]
    if not suppress:
        extra.append("--no_suppress_consecutive_bcs")
    run_new(inputs, extra, classifier=new_classifier)

    assert new_classifier.calls == orig_classifier.calls
    assert_same_tree(inputs["orig_out"], inputs["new_out"])
    decisions = json.load(open(os.path.join(inputs["new_out"], "segment_cache", "bike", "bc_decisions.json")))
    assert any(d["bc_word"] for d in decisions)


@pytest.mark.equivalence
@pytest.mark.parametrize("threshold, bc_context", [(0.5, True), (0.0, False)], ids=["mixed", "always_bc_no_context"])
def test_text_dialogue_with_classifier_matches_original(orig, inputs, threshold, bc_context):
    seed = 46
    vocab, master = master_vocab_with_classifier(orig, inputs["vocab_json"])
    generator = make_orig_generator(orig, inputs, master)
    orig_classifier = FakeClassifier(threshold)
    texts = [json.loads(line)["text"] for line in open(inputs["text_jsonl"]) if line.strip()]
    for idx, text in enumerate(texts):
        cache_dir = os.path.join(inputs["orig_out"], "segment_cache", f"example_{idx:04d}")
        _, metadata = generator.generate_dialogue_with_classifier(
            transcript=text.strip(),
            cache_dir=cache_dir,
            classifier=orig_classifier,
            bc_vocab_per_class=vocab,
            context_window_turns=5,
            context_window_future_turns=5,
            seed=seed,
            save_combined=True,
            include_bc_context=bc_context,
        )
        write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)

    new_classifier = FakeClassifier(threshold)
    extra = ["--input_jsonl", inputs["text_jsonl"], "--bc_vocab_per_class", inputs["vocab_json"], "--seed", str(seed)]
    if not bc_context:
        extra.append("--no_bc_context")
    run_new(inputs, extra, classifier=new_classifier)

    assert new_classifier.calls == orig_classifier.calls
    assert_same_tree(inputs["orig_out"], inputs["new_out"])


@pytest.mark.equivalence
def test_text_dialogue_random_backchannels_matches_original(orig, inputs):
    seed = 42
    generator = make_orig_generator(orig, inputs, ["hmmm", "uhhmm", "mhm"])
    texts = [json.loads(line)["text"] for line in open(inputs["text_jsonl"]) if line.strip()]
    for idx, text in enumerate(texts):
        cache_dir = os.path.join(inputs["orig_out"], "segment_cache", f"example_{idx:04d}")
        _, metadata = generator.generate_dialogue(
            transcript=text.strip(), cache_dir=cache_dir, bc_probability=0.7, bc_vocab_subset=["hmmm", "mhm"], seed=seed
        )
        write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)

    extra = ["--input_jsonl", inputs["text_jsonl"], "--no_classifier", "--bc_probability", "0.7", "--seed", str(seed)]
    extra += ["--bc_vocab", "hmmm", "uhhmm", "mhm", "--bc_vocab_subset", "hmmm", "mhm"]
    run_new(inputs, extra)
    assert_same_tree(inputs["orig_out"], inputs["new_out"])


@pytest.mark.equivalence
def test_benchmark_input_matches_original(orig, inputs):
    seed = 42
    vocab, master = master_vocab_with_classifier(orig, inputs["vocab_json"])
    generator = make_orig_generator(orig, inputs, master)
    orig_classifier = FakeClassifier(0.5)
    evals = []
    for rec in orig.load_bc_gt_jsonl(inputs["bc_gt_jsonl"]):
        cache_dir = os.path.join(inputs["orig_out"], "segment_cache", rec["dialogue_id"])
        _, metadata, decisions = generator.generate_dialogue_from_turns(
            turns=rec["turns"],
            cache_dir=cache_dir,
            classifier=orig_classifier,
            bc_vocab_per_class=vocab,
            context_window_turns=5,
            context_window_future_turns=5,
            include_bc_context=True,
            suppress_consecutive_same_speaker_bcs=True,
            seed=seed + int(rec["dialogue_id"]),
            save_combined=True,
        )
        write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)
        result = orig.compute_turn_level_eval(decisions, rec["gt_bc_per_turn"])
        result["dialogue_id"] = rec["dialogue_id"]
        result["tags_removed"] = rec.get("tags_removed", "?")
        write_json(os.path.join(cache_dir, "bc_eval.json"), result)
        evals.append(result)
    write_json(
        os.path.join(inputs["orig_out"], "bc_eval_summary.json"),
        {"per_example": evals, "aggregate": orig.aggregate_eval(evals)},
    )

    new_classifier = FakeClassifier(0.5)
    extra = ["--input_bc_gt_jsonl", inputs["bc_gt_jsonl"], "--bc_vocab_per_class", inputs["vocab_json"], "--seed", str(seed)]
    run_new(inputs, extra, classifier=new_classifier)

    assert new_classifier.calls == orig_classifier.calls
    assert_same_tree(inputs["orig_out"], inputs["new_out"])
    manifest = json.load(open(os.path.join(inputs["new_out"], "segment_cache", "0", "manifest.json")))
    assert "input_wav_slice_start_sec" in manifest["speaker_1"][0]


@pytest.mark.equivalence
def test_naive_only_matches_turbo_script(orig_turbo, inputs):
    """``--naive_only --no-naive_vad_trim`` is the turbo script's ``--run_naive_only``."""
    generator = make_orig_generator(orig_turbo, inputs, ["hmmm", "uhhmm"])
    dialogue = orig_turbo.load_dialogue_json(inputs["dialogue_json"])
    cache_dir = os.path.join(inputs["orig_out"], "voice", "segment_cache", dialogue["dialogue_id"])
    _, metadata = generator.generate_naive_concatenation(turns=dialogue["turns"], cache_dir=cache_dir)
    write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)

    torch.manual_seed(TTS_RNG_SEED)
    texts = [json.loads(line)["text"] for line in open(inputs["text_jsonl"]) if line.strip()]
    for idx, text in enumerate(texts):
        cache_dir = os.path.join(inputs["orig_out"], "text", "segment_cache", f"example_{idx:04d}")
        _, metadata = generator.generate_naive_concatenation_from_transcript(
            transcript=text.strip(), cache_dir=cache_dir, seed=42
        )
        write_json(os.path.join(cache_dir, "turn_metadata.json"), metadata)

    root = inputs["new_out"]
    inputs["new_out"] = os.path.join(root, "voice")
    run_new(inputs, ["--input_dialogue_json", inputs["dialogue_json"], "--naive_only", "--no-naive_vad_trim"])
    inputs["new_out"] = os.path.join(root, "text")
    run_new(inputs, ["--input_jsonl", inputs["text_jsonl"], "--naive_only", "--no-naive_vad_trim", "--seed", "42"])
    assert_same_tree(inputs["orig_out"], root)


# --------------------------------------------------------------------------- #
# Behaviour that is new in the refactored module
# --------------------------------------------------------------------------- #


def test_relative_paths_in_and_out(inputs):
    """Relative voice-line paths are resolved against the JSON; the cache stores cache-relative paths."""
    extra = ["--input_dialogue_json", inputs["relative_json"], "--bc_vocab_per_class", inputs["vocab_json"]]
    run_new(inputs, extra, classifier=FakeClassifier(0.5), absolute=False)
    cache_dir = os.path.join(inputs["new_out"], "segment_cache", "bike")
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    metadata = json.load(open(os.path.join(cache_dir, "turn_metadata.json")))
    stored = [e["source_audio_path"] for key in manifest for e in manifest[key] if e["type"] == "speech"]
    stored += [m["audio_path"] for m in metadata]
    assert len(stored) == 12
    for path in stored:
        assert not os.path.isabs(path)
        assert os.path.isfile(os.path.join(cache_dir, path))


def test_voice_lines_without_classifier_have_no_backchannels(inputs):
    run_new(inputs, ["--input_dialogue_json", inputs["dialogue_json"], "--bc_classifier_type", "none"])
    cache_dir = os.path.join(inputs["new_out"], "segment_cache", "bike")
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    assert {e["type"] for key in manifest for e in manifest[key]} == {"speech"}
    assert len(manifest["speaker_1"]) == 3 and len(manifest["speaker_2"]) == 3
    decisions = json.load(open(os.path.join(cache_dir, "bc_decisions.json")))
    assert all(d["pred_label"] == "NoBackchannel" and d["bc_word"] is None for d in decisions)


def test_voice_lines_without_seed_do_not_crash(inputs):
    """The original raised TypeError when a speaker-1 backchannel was synthesised with no seed."""
    tts = FakeBackend()
    synth = synthesize.DialogueSynthesizer(tts, inputs["ref1"], inputs["ref2"], bc_all_words=sum(VOCAB.values(), []))
    turns = synthesize.load_dialogue_json(inputs["dialogue_json"])["turns"]
    random.seed(0)
    _, _, decisions = synth.generate_dialogue_from_turns(
        turns, os.path.join(inputs["new_out"], "cache"), FakeClassifier(0.0), VOCAB, seed=None
    )
    assert sum(d["bc_word"] is not None for d in decisions) >= 2


def test_clean_cache_removes_stale_segments(inputs):
    cache_audio = Path(inputs["new_out"]) / "segment_cache" / "bike" / "audio_segments"
    cache_audio.mkdir(parents=True)
    stale = cache_audio / "seg_0099_s2_turn1_bc_old.wav"
    stale.write_bytes(b"")
    extra = ["--input_dialogue_json", inputs["dialogue_json"], "--bc_classifier_type", "none"]
    run_new(inputs, extra)
    assert stale.exists()
    run_new(inputs, extra + ["--clean_cache"])
    assert not stale.exists()


def test_word_selection_modes():
    random.seed(3)
    assert all(synthesize.pick_bc_word(["a", "b"], "a") == "b" for _ in range(20))
    assert synthesize.pick_bc_word(["a"], "a") == "a"
    random.seed(3)
    drawn = {synthesize.pick_bc_word(["a", "b"], "a", selection="random") for _ in range(40)}
    assert drawn == {"a", "b"}


def test_parse_transcript_keeps_sentences_and_consecutive_turns():
    parsed = synthesize.parse_transcript(" [S1] Hello   there.\n  How are you? [S2] Fine. [S2]  And you? ")
    assert parsed == [
        {"speaker": 1, "text": "Hello there.\nHow are you?"},
        {"speaker": 2, "text": "Fine."},
        {"speaker": 2, "text": "And you?"},
    ]


def test_turn_level_eval_counts():
    decisions = [
        {"turn_id": 0, "pred_label": "Acknowledgment", "bc_word": "yeah", "suppressed": False},
        {"turn_id": 1, "pred_label": "Reactive", "bc_word": None, "suppressed": True},
        {"turn_id": 2, "pred_label": "NoBackchannel", "bc_word": None, "suppressed": False},
        {"turn_id": 3, "pred_label": "Reactive", "bc_word": "wow", "suppressed": False},
    ]
    gt = [{"turn_id": 0, "bc_count": 2, "bc_events": []}, {"turn_id": 1, "bc_count": 1, "bc_events": []}]
    result = synthesize.compute_turn_level_eval(decisions, gt)
    assert (result["TP"], result["FP"], result["FN"], result["TN"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["f1"] == 0.5
    pooled = synthesize.aggregate_eval([result, result])
    assert pooled["num_examples"] == 2 and pooled["TP"] == 2 and math.isclose(pooled["f1"], 0.5)


def test_cli_rejects_missing_or_ambiguous_input(inputs):
    base = ["--out_dir", inputs["new_out"], "--ref_s1", "a.wav", "--ref_s2", "b.wav"]
    with pytest.raises(SystemExit):
        synthesize.main(base, tts=FakeBackend())
    with pytest.raises(SystemExit):
        synthesize.main(
            base + ["--input_jsonl", inputs["text_jsonl"], "--input_dialogue_json", inputs["dialogue_json"]],
            tts=FakeBackend(),
        )
    with pytest.raises(SystemExit):
        synthesize.main(base + ["--input_jsonl", inputs["text_jsonl"]], tts=FakeBackend())
