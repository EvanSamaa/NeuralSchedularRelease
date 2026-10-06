"""VAD figures, listening-test stimuli and the WebQuestions TTS helper."""

import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from neural_scheduler.analysis import figures, stimuli, webquestions_tts  # noqa: E402
from neural_scheduler.viz import vad_plot  # noqa: E402

ORIG_DIR = "ntpp/model/chatGLM_dual_timing_only/experiments/inference_qualitative_examples"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


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


def run_main(module, argv):
    old = sys.argv
    sys.argv = ["prog"] + [str(a) for a in argv]
    try:
        module.main()
    finally:
        sys.argv = old


def fake_get_speech_timestamps(waveform, model, sampling_rate, threshold, min_speech_duration_ms,
                               min_silence_duration_ms, return_seconds):
    """Deterministic stand-in for Silero: 100 ms frames louder than ``threshold / 10`` are speech."""
    frame = sampling_rate // 10
    n = len(waveform) // frame
    active = [float(waveform[i * frame:(i + 1) * frame].abs().mean()) > threshold / 10 for i in range(n)]
    segments, start = [], None
    for i, on in enumerate(active + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append({"start": start / 10, "end": i / 10})
            start = None
    return segments


FAKE_SILERO = (torch.nn.Identity(), (fake_get_speech_timestamps,))


@pytest.fixture
def stub_silero(monkeypatch):
    monkeypatch.setattr(vad_plot, "_silero_model", FAKE_SILERO[0])
    monkeypatch.setattr(vad_plot, "_silero_utils", FAKE_SILERO[1])


def stub_original_silero(module):
    module._silero_model, module._silero_utils = FAKE_SILERO


def speech_like(seconds, sr, bursts, seed):
    """Noise bursts at the given ``(start, end)`` second ranges over a quiet floor."""
    rng = np.random.default_rng(seed)
    audio = 0.001 * rng.standard_normal(int(seconds * sr))
    for start, end in bursts:
        a, b = int(start * sr), int(end * sr)
        audio[a:b] += 0.2 * rng.standard_normal(b - a)
    return audio.astype(np.float32)


@pytest.fixture
def study_inputs(tmp_path):
    """Pipeline-shaped inputs for two backchannel samples and two turn-taking examples."""
    root = tmp_path / "inputs"
    sample_dirs = []
    for k, sample_id in enumerate([3, 12]):
        search = root / "bc_run" / f"sample_{sample_id}" / "segment_search_v2"
        search.mkdir(parents=True)
        sf.write(search / "channel_a.wav", speech_like(5.0, 22050, [(0.5, 2.0), (2.6, 4.5)], k), 22050)
        sf.write(search / "channel_b.wav", speech_like(5.0, 22050, [(1.2, 1.6), (3.3, 3.8)], 10 + k), 22050)
        pp = root / "personaplex" / str(sample_id)
        pp.mkdir(parents=True)
        sf.write(pp / "output.wav", speech_like(6.0, 24000, [(0.5, 1.5), (4.2, 5.4)], 20 + k), 24000)
        sample_dirs.append(str(search.parent))
    sample_list = root / "good_examples.txt"
    sample_list.write_text("\n".join(sample_dirs + [str(root / "bc_run" / "sample_99")]) + "\n")

    for k, ex_id in enumerate([1, 4]):
        cache = root / "naive" / "segment_cache" / f"example_{ex_id:04d}"
        cache.mkdir(parents=True)
        stereo = np.stack([speech_like(4.0, 24000, [(0.2, 1.5)], 30 + k),
                           speech_like(4.0, 24000, [(2.0, 3.6)], 40 + k)], axis=1)
        sf.write(cache / "combined_dual.wav", stereo, 24000)
        search = root / "model" / f"sample_{ex_id}" / "segment_search_v2"
        search.mkdir(parents=True)
        sf.write(search / "channel_a.wav", speech_like(3.5, 22050, [(0.2, 1.5)], 50 + k), 22050)
        sf.write(search / "channel_b.wav", speech_like(3.5, 22050, [(1.3, 2.9)], 60 + k), 22050)
    selections = root / "selections.txt"
    selections.write_text("1\n4\n7\n")

    scene = root / "scene_x" / "cfg" / "bc_never" / "ckpt" / "sample_0" / "segment_search_v2"
    scene.mkdir(parents=True)
    sf.write(scene / "channel_sum.wav", speech_like(3.0, 22050, [(0.2, 2.5)], 70), 22050)
    pairs = root / "pairs.txt"
    pairs.write_text(f"{scene / 'channel_sum.wav'}, "
                     f"{root / 'naive' / 'segment_cache' / 'example_0001' / 'combined_dual.wav'}\n"
                     "not a pair\n")

    return {
        "sample_list": sample_list, "personaplex_dir": root / "personaplex",
        "selections": selections, "naive_dir": root / "naive" / "segment_cache",
        "model_dir": root / "model", "pairs": pairs,
    }


def pixels(path):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def bc_args(inp):
    return ["--sample_list", inp["sample_list"], "--personaplex_dir", inp["personaplex_dir"]]


def tt_args(inp):
    return ["--selections", inp["selections"], "--naive_dir", inp["naive_dir"], "--model_dir", inp["model_dir"]]


def decoded_audio(video, tmp_path):
    from neural_scheduler.viz import ffmpeg

    os.makedirs(tmp_path, exist_ok=True)
    wav = str(tmp_path / (os.path.basename(str(video)) + ".decoded.wav"))
    ffmpeg.run(["ffmpeg", "-y", "-i", str(video), "-vn", wav])
    return sf.read(wav)[0]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def test_vad_silero_wrapper(stub_silero):
    wav = torch.from_numpy(speech_like(3.0, 22050, [(1.0, 2.0)], 0))
    segs = vad_plot.vad_silero(wav, 22050, threshold=0.4)
    assert len(segs) == 1 and segs[0] == pytest.approx((1.0, 2.0), abs=0.11)


def test_figures_render(stub_silero, study_inputs, tmp_path):
    out = tmp_path / "plots"
    figures.main(["bc", *map(str, bc_args(study_inputs)), "--out_dir", str(out / "bc")])
    assert sorted(os.listdir(out / "bc")) == ["vad_sample_01.png", "vad_sample_02.png"]
    figures.main(["turn-take", *map(str, tt_args(study_inputs)), "--out_dir", str(out / "tt"), "--save_pdf"])
    assert sorted(os.listdir(out / "tt")) == [
        "vad_example_01.pdf", "vad_example_01.png", "vad_example_02.pdf", "vad_example_02.png"]


@pytest.mark.equivalence
def test_figures_match_original(ntpp_orig, stub_silero, study_inputs, tmp_path):
    for script, sub, args in [("generate_figure_bc.py", "bc", bc_args(study_inputs)),
                              ("generate_figure_turn_take.py", "turn-take", tt_args(study_inputs))]:
        orig = load_module(ntpp_orig / ORIG_DIR / script, "orig_" + script[:-3])
        stub_original_silero(orig)
        old_dir, new_dir = tmp_path / ("old_" + sub), tmp_path / ("new_" + sub)
        run_main(orig, [*args, "--out_dir", old_dir])
        figures.main([sub, *map(str, args), "--out_dir", str(new_dir)])
        names = sorted(os.listdir(new_dir))
        assert names == sorted(os.listdir(old_dir)) and len(names) == 2
        for name in names:
            assert (pixels(new_dir / name) == pixels(old_dir / name)).all(), name


# ---------------------------------------------------------------------------
# stimuli
# ---------------------------------------------------------------------------

def test_extract_label():
    assert stimuli.extract_label("/x/TTS_scenes/deep/bc_never/segment_cache/e/combined_dual.wav") == "deep"
    assert stimuli.extract_label("/x/scene/cfg/sample_0/segment_search_v2/channel_sum.wav") == "cfg"
    assert stimuli.extract_label("/x/folder/file.wav") == "folder"


def test_mix_average_pads_shorter():
    mixed = stimuli.mix_average(torch.ones(4), torch.ones(2))
    assert mixed.tolist() == [1.0, 1.0, 0.5, 0.5]


def test_audio_exports_without_ffmpeg(study_inputs, tmp_path):
    out_bc, out_tt = tmp_path / "bc", tmp_path / "tt"
    stimuli.main(["audio-bc", *map(str, bc_args(study_inputs)), "--out_dir", str(out_bc)])
    assert sorted(os.listdir(out_bc)) == [
        "sample_12_personaplex.wav", "sample_12_proposed.wav",
        "sample_3_personaplex.wav", "sample_3_proposed.wav"]
    audio, sr = sf.read(out_bc / "sample_3_personaplex.wav")
    assert sr == 22050 and len(audio) == pytest.approx(6.0 * 22050, abs=2)
    stimuli.main(["audio-turn-take", *map(str, tt_args(study_inputs)), "--out_dir", str(out_tt),
                  "--max_samples", "1"])
    assert sorted(os.listdir(out_tt)) == ["example_0001_naive.wav", "example_0001_proposed.wav"]


@needs_ffmpeg
def test_videos_and_seeded_assignment(study_inputs, tmp_path):
    out = tmp_path / "videos"
    stimuli.main(["videos-bc", *map(str, bc_args(study_inputs)), "--out_dir", str(out), "--seed", "5"])
    mapping = json.load(open(out / "ab_mapping.json"))
    assert [m["video_file"] for m in mapping] == ["bc_video_01.mp4", "bc_video_02.mp4"]
    assert all({m["A"], m["B"]} == {"model", "personaplex"} and (m["A"] == "model") == m["model_is_A"]
               for m in mapping)
    from neural_scheduler.viz import ffmpeg

    # clip A (5 s mix) + 2 s pause + clip B (6 s mix, the baseline is longer), in either order
    assert ffmpeg.get_duration(str(out / "bc_video_01.mp4")) == pytest.approx(13.0, abs=0.3)

    again = tmp_path / "videos_again"
    stimuli.main(["videos-bc", *map(str, bc_args(study_inputs)), "--out_dir", str(again), "--seed", "5"])
    assert [m["model_is_A"] for m in json.load(open(again / "ab_mapping.json"))] == \
        [m["model_is_A"] for m in mapping]

    out_tt = tmp_path / "videos_tt"
    stimuli.main(["videos-turn-take", *map(str, tt_args(study_inputs)), "--out_dir", str(out_tt),
                  "--seed", "1", "--video_prefix", "video_"])
    assert sorted(os.listdir(out_tt)) == ["ab_mapping.json", "video_01.mp4", "video_02.mp4"]


@needs_ffmpeg
@pytest.mark.equivalence
def test_videos_match_original(ntpp_orig, study_inputs, tmp_path):
    cases = [
        ("generate_videos_bc.py", "videos-bc", bc_args(study_inputs)),
        ("generate_videos_turn_take.py", "videos-turn-take", tt_args(study_inputs)),
        ("generate_videos_turn_take_dramatic.py", "videos-paired", ["--sample_list", study_inputs["pairs"]]),
    ]
    for script, sub, args in cases:
        orig = load_module(ntpp_orig / ORIG_DIR / script, "orig_" + script[:-3])
        old_dir, new_dir = tmp_path / ("old_" + sub), tmp_path / ("new_" + sub)
        run_main(orig, [*args, "--out_dir", old_dir, "--seed", 11])
        stimuli.main([sub, *map(str, args), "--out_dir", str(new_dir), "--seed", "11"])
        assert json.load(open(new_dir / "ab_mapping.json")) == json.load(open(old_dir / "ab_mapping.json"))
        videos = sorted(n for n in os.listdir(new_dir) if n.endswith(".mp4"))
        assert videos and videos == sorted(n for n in os.listdir(old_dir) if n.endswith(".mp4"))
        for name in videos:
            new_audio = decoded_audio(new_dir / name, tmp_path / "new")
            old_audio = decoded_audio(old_dir / name, tmp_path / "old")
            assert new_audio.shape == old_audio.shape and np.array_equal(new_audio, old_audio), name


@pytest.mark.equivalence
def test_audio_exports_match_original(ntpp_orig, study_inputs, tmp_path):
    cases = [
        ("generate_folder_bc.py", "audio-bc", bc_args(study_inputs), False),
        ("generate_folder_turn_take.py", "audio-turn-take", tt_args(study_inputs), False),
        ("generate_folder_turn_take_dramatic.py", "audio-paired", ["--sample_list", study_inputs["pairs"]], True),
    ]
    for script, sub, args, uses_ffmpeg in cases:
        if uses_ffmpeg and not HAVE_FFMPEG:
            continue
        orig = load_module(ntpp_orig / ORIG_DIR / script, "orig_" + script[:-3])
        old_dir, new_dir = tmp_path / ("old_" + sub), tmp_path / ("new_" + sub)
        run_main(orig, [*args, "--out_dir", old_dir])
        stimuli.main([sub, *map(str, args), "--out_dir", str(new_dir)])
        names = sorted(os.listdir(new_dir))
        assert names and names == sorted(os.listdir(old_dir))
        for name in names:
            new_audio, new_sr = sf.read(new_dir / name)
            old_audio, old_sr = sf.read(old_dir / name)
            assert new_sr == old_sr and np.array_equal(new_audio, old_audio), name


# ---------------------------------------------------------------------------
# WebQuestions TTS
# ---------------------------------------------------------------------------

def test_format_answer():
    fmt = webquestions_tts.format_answer
    assert fmt([]) == "No answer available"
    assert fmt(["http://x"]) == "The answer is a web link"
    assert fmt(["a"]) == "a"
    assert fmt(["a", "b"]) == "a and b"
    assert fmt(["a", "b", "c"]) == "a, b, and c"
    assert fmt(list("abcdef")) == "a, b, c, and others"


def test_dual_channel_layout_and_silence_option(tmp_path, monkeypatch):
    def fake_synthesize(self, text, voice):
        return np.full(len(text) * 100, 0.5 if voice == self.question_voice else -0.5, dtype=np.float32)

    monkeypatch.setattr(webquestions_tts.PiperDualChannelTTS, "synthesize", fake_synthesize)
    data = tmp_path / "wq.json"
    data.write_text(json.dumps([
        {"qId": "q1", "qText": "what is it", "answers": ["this", "that"]},
        {"qId": "q2", "qText": "   ", "answers": ["skipped"]},
    ]))
    for silence, out in [(0.5, tmp_path / "default"), (1.0, tmp_path / "long")]:
        webquestions_tts.main(["--input", str(data), "--output", str(out), "--silence", str(silence)])
        assert os.listdir(out) == ["q1.wav"]
        audio, sr = sf.read(out / "q1.wav")
        q_len, a_len, gap = 1000, len("this and that") * 100, int(silence * 22050)
        assert sr == 22050 and audio.shape == (q_len + gap + a_len, 2)
        assert np.all(audio[:q_len, 0] > 0) and np.all(audio[q_len:, 0] == 0)
        assert np.all(audio[:q_len + gap, 1] == 0) and np.all(audio[q_len + gap:, 1] < 0)


@pytest.mark.equivalence
def test_webquestions_matches_original(ntpp_orig, tmp_path, monkeypatch):
    orig = load_module(ntpp_orig / "ntpp/model/chatGLM_dual_timing_only/experiments/experiment_turnunderstanding"
                       / "generate_turnbased_speech.py", "orig_webquestions")

    def fake_synthesize(self, text, voice):
        rng = np.random.default_rng(len(text) + len(voice))
        return rng.standard_normal(len(text) * 50).astype(np.float32) * 0.1

    monkeypatch.setattr(orig.PiperDualChannelTTS, "synthesize", fake_synthesize)
    monkeypatch.setattr(orig.PiperDualChannelTTS, "_ensure_models", lambda self: None)
    monkeypatch.setattr(webquestions_tts.PiperDualChannelTTS, "synthesize", fake_synthesize)

    items = [{"qId": f"q{i}", "qText": f"question number {i}?", "answers": [f"answer {j}" for j in range(i)]}
             for i in range(8)] + [{"qId": "url", "qText": "where?", "answers": ["http://a", "b"]}]
    for answers in [it["answers"] for it in items]:
        assert webquestions_tts.format_answer(answers) == orig.format_answer(answers)
    data = tmp_path / "wq.json"
    data.write_text(json.dumps(items))
    orig.process_webquestions(str(data), str(tmp_path / "old"), max_items=2000)
    webquestions_tts.main(["--input", str(data), "--output", str(tmp_path / "new")])
    names = sorted(os.listdir(tmp_path / "new"))
    assert len(names) == 9 and names == sorted(os.listdir(tmp_path / "old"))
    for name in names:
        assert (tmp_path / "new" / name).read_bytes() == (tmp_path / "old" / name).read_bytes()
