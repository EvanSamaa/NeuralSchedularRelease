"""Dual-waveform video renderer and the ffmpeg helpers."""

import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("cv2")
pytest.importorskip("librosa")

from neural_scheduler.viz import ffmpeg, waveform_video  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
ORIG = "ntpp/model/chatGLM_dual_timing_only/experiments/render_visualiazation_and_audio.py"


def tone(seconds, sr, freq, start, end, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.001 * rng.standard_normal(len(t))
    mask = (t >= start) & (t < end)
    audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])
    return audio.astype(np.float32)


@pytest.fixture
def two_wavs(tmp_path):
    a, b = tmp_path / "channel_a.wav", tmp_path / "channel_b.wav"
    sf.write(a, tone(2.0, 16000, 220, 0.2, 1.0, 0), 16000)
    sf.write(b, tone(1.5, 22050, 330, 0.8, 1.4, 1), 22050)
    return a, b


def stream_md5(path, stream):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", stream, "-f", "md5", "-"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_draw_waveform_marks_frame():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    waveform_video.draw_waveform(frame, tone(0.5, 16000, 200, 0.0, 0.5, 2), y_center=100, color=(0, 255, 0),
                                 width=320, height=160, envelope_scale=0.2, is_speech=True)
    assert frame.any()
    untouched = np.zeros((10, 10, 3), dtype=np.uint8)
    waveform_video.draw_waveform(untouched, np.array([]), 5, (0, 255, 0), 10, 10, 0.1, False)
    assert not untouched.any()


@needs_ffmpeg
def test_video_is_written_next_to_first_input(two_wavs):
    a, b = two_wavs
    out = waveform_video.create_dual_waveform_video(str(a), str(b), output_name="demo.mp4", frame_rate=5)
    assert out == str(a.parent / "demo.mp4")
    assert ffmpeg.get_duration(out) == pytest.approx(2.0, abs=0.3)
    assert ffmpeg.get_channels(out) == 2
    assert sorted(p.name for p in a.parent.iterdir()) == ["channel_a.wav", "channel_b.wav", "demo.mp4"]


@needs_ffmpeg
def test_ffmpeg_helpers(tmp_path):
    mono, stereo = tmp_path / "mono.wav", tmp_path / "stereo.wav"
    loud = tone(3.0, 22050, 220, 0.0, 3.0, 3)
    sf.write(mono, loud, 22050)
    sf.write(stereo, np.stack([loud, 0.1 * loud], axis=1), 22050)

    assert ffmpeg.get_duration(str(mono)) == pytest.approx(3.0, abs=0.01)
    assert (ffmpeg.get_channels(str(mono)), ffmpeg.get_channels(str(stereo))) == (1, 2)
    left = ffmpeg.measure_lufs(str(stereo), "pan=mono|c0=c0,")
    right = ffmpeg.measure_lufs(str(stereo), "pan=mono|c0=c1,")
    assert left - right == pytest.approx(20.0, abs=0.5)

    norm = tmp_path / "norm.wav"
    ffmpeg.normalise(str(mono), str(norm))
    assert ffmpeg.measure_lufs(str(norm)) == pytest.approx(ffmpeg.TARGET_LUFS, abs=0.5)

    muted = tmp_path / "muted.wav"
    ffmpeg.silence_first_n_seconds(str(mono), str(muted), 1.0)
    audio, sr = sf.read(muted)
    assert np.abs(audio[: sr - 100]).max() == 0 and np.abs(audio[sr + 100:]).max() > 0.1

    mixed = tmp_path / "mixed.wav"
    ffmpeg.mix_dual_wav_to_mono(str(stereo), str(mixed))
    assert sf.info(mixed).channels == 1 and sf.info(mixed).samplerate == ffmpeg.STIMULUS_SR

    video = tmp_path / "ab.mp4"
    ffmpeg.render_ab_video(str(norm), str(muted), str(video), 1.0, str(tmp_path))
    assert ffmpeg.get_duration(str(video)) == pytest.approx(7.0, abs=0.3)

    with pytest.raises(RuntimeError):
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "missing.wav"), str(tmp_path / "x.wav")])


@needs_ffmpeg
@pytest.mark.equivalence
def test_video_matches_original(ntpp_orig, two_wavs):
    spec = importlib.util.spec_from_file_location("orig_waveform_video", ntpp_orig / ORIG)
    orig = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(orig)
    finally:
        sys.dont_write_bytecode = previous

    wave = tone(0.5, 16000, 200, 0.1, 0.4, 5)
    for is_speech in (True, False):
        new_frame = np.zeros((200, 320, 3), dtype=np.uint8)
        old_frame = np.zeros((200, 320, 3), dtype=np.uint8)
        args = dict(y_center=100, color=(0, 255, 0), width=320, height=160, envelope_scale=0.05, is_speech=is_speech)
        waveform_video.draw_waveform(new_frame, wave, **args)
        orig.draw_waveform(old_frame, wave, **args)
        assert np.array_equal(new_frame, old_frame)

    a, b = two_wavs
    new = waveform_video.create_dual_waveform_video(str(a), str(b), output_name="new.mp4", frame_rate=5)
    old = orig.create_dual_waveform_video(str(a), str(b), output_name="old.mp4", frame_rate=5)
    assert stream_md5(new, "0:v") == stream_md5(old, "0:v")
    assert stream_md5(new, "0:a") == stream_md5(old, "0:a")
