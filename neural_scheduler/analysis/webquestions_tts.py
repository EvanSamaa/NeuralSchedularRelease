"""Synthetic question/answer dialogues with an unambiguous turn order.

Each WebQuestions item becomes a stereo wav: the question on the left channel
in one Piper voice, then a short silence, then the answer on the right channel
in another voice. Tokenised with the preprocessing pipeline, these files are
the input of the turn-order perturbation evaluation, where the correct order
(question before answer) is known.

Needs the ``piper`` command-line tool and the two voice models::

    python -m neural_scheduler.analysis.webquestions_tts \\
        --input webquestions/trainmodel.json --output voiced_webquestions

The input is a JSON list of ``{"qId": ..., "qText": ..., "answers": [...]}``.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

QUESTION_VOICE = "en_US-lessac-medium"
ANSWER_VOICE = "en_US-amy-medium"


class PiperDualChannelTTS:
    """Two-voice Piper TTS that lays a question and its answer on separate channels.

    ``question_voice`` and ``answer_voice`` are Piper model names or paths to
    ``.onnx`` files. ``piper_data_dir``, if given, is where Piper looks for
    and downloads voice models; otherwise Piper resolves the names itself
    (current directory first).
    """

    def __init__(
        self,
        question_voice: str = QUESTION_VOICE,
        answer_voice: str = ANSWER_VOICE,
        sample_rate: int = 22050,
        piper_data_dir: Optional[str] = None,
    ):
        self.question_voice = question_voice
        self.answer_voice = answer_voice
        self.sample_rate = sample_rate
        self.piper_data_dir = piper_data_dir

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesise ``text`` with ``voice``; returns mono audio at ``sample_rate``."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        cmd = ["piper", "--model", voice, "--output_file", output_path]
        if self.piper_data_dir:
            cmd += ["--data-dir", self.piper_data_dir, "--download-dir", self.piper_data_dir]
        try:
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _stdout, stderr = process.communicate(input=text.encode("utf-8"))
            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

            audio, sr = sf.read(output_path)
            if sr != self.sample_rate:
                from scipy import signal

                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            return audio
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def create_dual_channel(self, question: str, answer: str, output_path, silence_duration: float = 0.5):
        """Write a stereo wav: question on channel 0, then silence, then answer on channel 1.

        Returns the ``[samples, 2]`` array that was written.
        """
        print("  Synthesizing question...")
        q_audio = self.synthesize(question, self.question_voice)
        print("  Synthesizing answer...")
        a_audio = self.synthesize(answer, self.answer_voice)

        silence = np.zeros(int(silence_duration * self.sample_rate), dtype=np.float32)
        channel_1 = np.concatenate([q_audio, silence, np.zeros(len(a_audio), dtype=np.float32)])
        channel_2 = np.concatenate([np.zeros(len(q_audio), dtype=np.float32), silence, a_audio])

        max_len = max(len(channel_1), len(channel_2))
        channel_1 = np.pad(channel_1, (0, max_len - len(channel_1)))
        channel_2 = np.pad(channel_2, (0, max_len - len(channel_2)))

        stereo_audio = np.stack([channel_1, channel_2], axis=1)
        sf.write(output_path, stereo_audio, self.sample_rate)
        print(f"  Saved: {output_path}")
        return stereo_audio


def format_answer(answers: List[str]) -> str:
    """Turn a WebQuestions answer list into one speakable sentence.

    URLs are dropped; two answers are joined with "and", up to five as a
    comma list, and longer lists are cut to the first three plus "and others".
    """
    if not answers:
        return "No answer available"

    valid_answers = [a for a in answers if not a.startswith("http")]
    if not valid_answers:
        return "The answer is a web link"

    if len(valid_answers) == 1:
        return valid_answers[0]
    elif len(valid_answers) == 2:
        return f"{valid_answers[0]} and {valid_answers[1]}"
    elif len(valid_answers) <= 5:
        return ", ".join(valid_answers[:-1]) + f", and {valid_answers[-1]}"
    else:
        return ", ".join(valid_answers[:3]) + ", and others"


def process_webquestions(
    json_file,
    output_dir,
    max_items: Optional[int] = None,
    silence_duration: float = 0.5,
    question_voice: str = QUESTION_VOICE,
    answer_voice: str = ANSWER_VOICE,
    piper_data_dir: Optional[str] = None,
) -> int:
    """Synthesise ``<qId>.wav`` for each item of a WebQuestions JSON file.

    Items with an empty question, and items whose synthesis fails, are
    skipped. Returns the number of files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tts = PiperDualChannelTTS(
        question_voice=question_voice, answer_voice=answer_voice, piper_data_dir=piper_data_dir
    )

    print(f"Loading data from: {json_file}")
    with open(json_file, "r") as f:
        data = json.load(f)

    if max_items:
        data = data[:max_items]
        print(f"Processing first {max_items} items only")

    total = len(data)
    successful = 0
    skipped = 0
    for idx, item in enumerate(data):
        qid = item.get("qId", f"unknown_{idx}")
        question = item.get("qText", "").strip()
        answer = format_answer(item.get("answers", []))

        if not question or not answer:
            print(f"\n[{idx+1}/{total}] Skipping {qid}: empty Q or A")
            skipped += 1
            continue

        output_path = output_dir / f"{qid}.wav"
        print(f"\n[{idx+1}/{total}] Processing {qid}:")
        print(f"  Q: {question[:80]}...")
        print(f"  A: {answer[:80]}...")

        try:
            tts.create_dual_channel(question, answer, output_path, silence_duration=silence_duration)
            successful += 1
        except Exception as e:  # one failed item should not stop a 2000-item run
            print(f"  ERROR: {e}")
            skipped += 1

    print("\n" + "=" * 60)
    print(f"  Successfully generated: {successful} files")
    print(f"  Skipped: {skipped} files")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    return successful


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate a voiced Q&A dataset from WebQuestions.")
    parser.add_argument("--input", required=True, help="WebQuestions JSON file.")
    parser.add_argument("--output", required=True, help="Output directory for the wavs.")
    parser.add_argument("--max-items", type=int, default=2000,
                        help="Maximum number of items to process (default: 2000).")
    parser.add_argument("--silence", type=float, default=0.5,
                        help="Silence between question and answer in seconds (default: 0.5).")
    parser.add_argument("--question_voice", default=QUESTION_VOICE,
                        help=f"Piper voice (name or .onnx path) for questions (default: {QUESTION_VOICE}).")
    parser.add_argument("--answer_voice", default=ANSWER_VOICE,
                        help=f"Piper voice (name or .onnx path) for answers (default: {ANSWER_VOICE}).")
    parser.add_argument("--piper_data_dir", default=None,
                        help="Directory where Piper finds and downloads voice models.")
    args = parser.parse_args(argv)

    process_webquestions(
        json_file=args.input,
        output_dir=args.output,
        max_items=args.max_items,
        silence_duration=args.silence,
        question_voice=args.question_voice,
        answer_voice=args.answer_voice,
        piper_data_dir=args.piper_data_dir,
    )


if __name__ == "__main__":
    main()
