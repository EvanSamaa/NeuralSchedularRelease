"""ffmpeg / ffprobe helpers for building listening-test stimuli.

Everything shells out to ``ffmpeg`` and ``ffprobe``, which must be on
``PATH`` (the build needs the ``ebur128`` and ``drawtext`` filters). Loudness
is matched by measuring integrated LUFS (EBU R128) and applying a static gain
to reach :data:`TARGET_LUFS`.
"""

import os
import subprocess
import sys
import tempfile
from typing import List

TARGET_LUFS = -23.0
STIMULUS_SR = 22050


def run(cmd: list, desc: str = "") -> None:
    """Run a command; print its stderr and raise ``RuntimeError`` on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR during: {desc or ' '.join(str(c) for c in cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode})")


def get_duration(path: str) -> float:
    """Duration of a media file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    return float(result.stdout.strip())


def get_channels(path: str) -> int:
    """Channel count of the first audio stream; 1 if it cannot be determined."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 1
    txt = result.stdout.strip()
    return int(txt) if txt else 1


def measure_lufs(wav_path: str, channel_filter: str = "") -> float:
    """Integrated loudness (LUFS) of a wav via the ``ebur128`` filter.

    ``channel_filter`` is an optional leading filter such as
    ``'pan=mono|c0=c0,'`` that selects one channel of a stereo file.
    """
    af = f"{channel_filter}ebur128=peak=none"
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", wav_path, "-af", af, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in reversed(result.stderr.splitlines()):
        if "I:" in line and "LUFS" in line:
            parts = line.split()
            idx = parts.index("LUFS")
            return float(parts[idx - 1])
    raise RuntimeError(f"Could not parse LUFS from ffmpeg output for {wav_path}")


def to_mono(wav_in: str, wav_out: str, target_sr: int = STIMULUS_SR) -> None:
    """Downmix any wav to mono at ``target_sr``."""
    cmd = ["ffmpeg", "-y", "-i", wav_in, "-ac", "1", "-ar", str(target_sr), wav_out]
    run(cmd, desc=f"to_mono {os.path.basename(wav_in)}")


def normalise(wav_in: str, wav_out: str, target_sr: int = STIMULUS_SR, label: str = "") -> None:
    """Write a mono copy of ``wav_in`` gained to :data:`TARGET_LUFS`."""
    lufs = measure_lufs(wav_in)
    gain = TARGET_LUFS - lufs
    print(f"    {label or os.path.basename(wav_in)}: {lufs:.1f} LUFS → {gain:+.1f} dB")
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_in,
        "-af", f"volume={gain:.4f}dB",
        "-ar", str(target_sr), "-ac", "1",
        wav_out,
    ]
    run(cmd, desc=f"normalise {label or os.path.basename(wav_in)}")


def silence_first_n_seconds(wav_in: str, wav_out: str, n: float) -> None:
    """Copy ``wav_in`` with its first ``n`` seconds muted."""
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_in,
        "-af", f"volume=enable='lte(t,{n})':volume=0",
        wav_out,
    ]
    run(cmd, desc=f"silence first {n}s of {os.path.basename(wav_in)}")


def mix_two_mono_to_mono(
    wav_a: str, wav_b: str, out_wav: str,
    label_a: str = "ch_a", label_b: str = "ch_b",
) -> None:
    """Gain each mono file to :data:`TARGET_LUFS` independently, then sum to mono."""
    lufs_a = measure_lufs(wav_a)
    lufs_b = measure_lufs(wav_b)
    gain_a = TARGET_LUFS - lufs_a
    gain_b = TARGET_LUFS - lufs_b
    print(f"    {label_a}: {lufs_a:.1f} LUFS → {gain_a:+.1f} dB")
    print(f"    {label_b}: {lufs_b:.1f} LUFS → {gain_b:+.1f} dB")

    af = (
        f"[0:a]volume={gain_a:.4f}dB[an];"
        f"[1:a]volume={gain_b:.4f}dB[bn];"
        f"[an][bn]amix=inputs=2:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_a,
        "-i", wav_b,
        "-filter_complex", af,
        "-map", "[out]",
        "-ar", str(STIMULUS_SR), "-ac", "1",
        out_wav,
    ]
    run(cmd, desc=f"mix {label_a}+{label_b} → mono")


def mix_dual_wav_to_mono(dual_wav: str, out_wav: str, label: str = "naive") -> None:
    """Gain each channel of a stereo file to :data:`TARGET_LUFS`, then sum to mono."""
    lufs_ch0 = measure_lufs(dual_wav, channel_filter="pan=mono|c0=c0,")
    lufs_ch1 = measure_lufs(dual_wav, channel_filter="pan=mono|c0=c1,")
    gain_ch0 = TARGET_LUFS - lufs_ch0
    gain_ch1 = TARGET_LUFS - lufs_ch1
    print(f"    {label} ch0: {lufs_ch0:.1f} LUFS  → {gain_ch0:+.1f} dB")
    print(f"    {label} ch1: {lufs_ch1:.1f} LUFS  → {gain_ch1:+.1f} dB")

    af = (
        f"[0:a]pan=mono|c0=c0,volume={gain_ch0:.4f}dB[left];"
        f"[0:a]pan=mono|c0=c1,volume={gain_ch1:.4f}dB[right];"
        f"[left][right]amix=inputs=2:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", dual_wav,
        "-filter_complex", af,
        "-map", "[out]",
        "-ar", str(STIMULUS_SR), "-ac", "1",
        out_wav,
    ]
    run(cmd, desc=f"mix {label} dual→mono with gain match")


def make_labelled_video(
    wav_path: str, label: str, out_path: str,
    width: int = 1280, height: int = 720, fontsize: int = 200,
) -> None:
    """Black video with a large centred text label, as long as ``wav_path``."""
    duration = get_duration(wav_path)
    safe_label = label.replace("'", "\\'").replace(":", "\\:")
    vf = (
        f"color=black:size={width}x{height}:rate=25,"
        f"drawtext=text='{safe_label}'"
        f":fontsize={fontsize}:fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
        f":font='DejaVu Sans':box=0"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", vf,
        "-i", wav_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    run(cmd, desc=f"render clip '{label}'")


def make_pause_clip(duration: float, out_path: str, width: int = 1280, height: int = 720) -> None:
    """Black, silent clip used between the A and B clips."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=black:size={width}x{height}:rate=25",
        "-f", "lavfi", "-i", f"anullsrc=r={STIMULUS_SR}:cl=mono",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    run(cmd, desc="render pause clip")


def concat_clips(clips: List[str], out_path: str) -> None:
    """Concatenate mp4 clips, in order, into one file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
        list_path = f.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
        run(cmd, desc="concat clips")
    finally:
        os.unlink(list_path)


def render_ab_video(first_wav: str, second_wav: str, out_video: str, pause_sec: float, tmp_dir: str) -> None:
    """Render ``A: first_wav``, a silent pause, ``B: second_wav`` into one video."""
    clip_a = os.path.join(tmp_dir, "clip_A.mp4")
    clip_pause = os.path.join(tmp_dir, "clip_pause.mp4")
    clip_b = os.path.join(tmp_dir, "clip_B.mp4")
    make_labelled_video(first_wav, "A", clip_a)
    make_pause_clip(pause_sec, clip_pause)
    make_labelled_video(second_wav, "B", clip_b)
    concat_clips([clip_a, clip_pause, clip_b], out_video)
