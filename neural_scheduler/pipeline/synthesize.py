"""Step 1 of the inference pipeline: build the per-dialogue segment cache.

For every dialogue this step

1. decides, turn by turn, whether the listener backchannels and with which
   word (backchannel classifier, or a random draw without one),
2. obtains the audio of every turn (recorded voice lines, or TTS from text),
3. synthesises each backchannel in the listener's voice, and
4. writes everything to ``<out_dir>/segment_cache/<dialogue_id>/``::

       manifest.json        segments per speaker; a backchannel on speaker X's
                            turn N is listed under the *other* speaker with the
                            same turn_id N, which is how the scheduler
                            recognises backchannels
       bc_decisions.json    classifier output per turn
       turn_metadata.json   text and naive timing per turn
       audio_segments/      one wav per speech turn and per backchannel
       combined_mono.wav    turns laid back to back (naive layout)
       combined_dual.wav

Four kinds of input are accepted:

``--input_dialogue_json``  one dialogue with a recorded voice line per turn
``--input_jsonl``          one dialogue per line as ``[S1] ... [S2] ...`` text; turns are synthesised
``--bc_vocab_jsonl``       as above, with a per-dialogue ``bc_vocab`` list (random backchannels)
``--input_bc_gt_jsonl``    benchmark records with voice lines and ground-truth
                           backchannels; also writes ``bc_eval.json`` per
                           dialogue and ``bc_eval_summary.json``

Run ``python -m neural_scheduler.pipeline.synthesize --help`` for all options.
"""

import argparse
import inspect
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio

from .tts import TTS_BACKENDS, load_tts

CLASSIFIER_TYPES = ("joint", "text", "none")
BC_WORD_SELECTIONS = ("avoid_repeat", "random")
NO_BACKCHANNEL = "NoBackchannel"
# Sample rate assumed by the start_frame / end_frame fields of turn_metadata.json.
METADATA_FRAME_RATE = 16000
MAX_AUDIO_SECONDS_DEFAULT = 15.0


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #


def load_bc_vocab_per_class(path: str) -> Dict[str, List[str]]:
    """Read ``{"Acknowledgment": [...], "Reactive": [...]}``."""
    with open(path) as f:
        return json.load(f)


def collect_all_bc_words(vocab: Dict[str, List[str]]) -> List[str]:
    all_words = set()
    for words in vocab.values():
        all_words.update(words)
    return sorted(all_words)


def _resolve_audio_paths(turns: List[Dict], base_dir: str) -> List[Dict]:
    """Return copies of ``turns`` whose relative ``audio_path`` is resolved against ``base_dir``."""
    resolved = []
    for turn in turns:
        turn = dict(turn)
        audio_path = turn.get("audio_path")
        if audio_path is not None and not os.path.isabs(audio_path):
            turn["audio_path"] = os.path.normpath(os.path.join(base_dir, audio_path))
        resolved.append(turn)
    return resolved


def load_dialogue_json(path: str) -> Dict:
    """Load one voice-line dialogue.

    ``{"dialogue_id": str (optional), "turns": [{"turn_id", "speaker" (1|2),
    "text", "audio_path"}, ...]}``. A relative ``audio_path`` is taken relative
    to the JSON file.
    """
    with open(path) as f:
        record = json.load(f)
    assert "turns" in record, "Missing 'turns' field"
    turns = record["turns"]
    assert isinstance(turns, list) and len(turns) > 0, "'turns' must be a non-empty list"
    for t_idx, t in enumerate(turns):
        for field in ("turn_id", "speaker", "text", "audio_path"):
            assert field in t, f"Turn {t_idx}: missing '{field}'"
        assert t["speaker"] in (1, 2), f"Turn {t_idx}: 'speaker' must be 1 or 2"
    return {
        "dialogue_id": record.get("dialogue_id", Path(path).stem),
        "turns": _resolve_audio_paths(turns, os.path.dirname(os.path.abspath(path))),
    }


def load_bc_gt_jsonl(path: str) -> List[Dict]:
    """Load benchmark records.

    Each line holds ``dialogue_id``, ``turns`` (voice-line turns, optionally
    with ``slice_start_sec`` / ``slice_end_sec``), ``tags_removed`` and
    ``gt_bc_per_turn`` (``[{turn_id, bc_count, bc_events: [{tag, timestamp}]}]``).
    Relative ``audio_path`` values are taken relative to the JSONL file.
    """
    base_dir = os.path.dirname(os.path.abspath(path))
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                record["turns"] = _resolve_audio_paths(record["turns"], base_dir)
                records.append(record)
    return records


def load_bc_vocab_jsonl(path: str) -> List[Dict]:
    """Load ``{"text": "[S1] ... [S2] ...", "bc_vocab": [...]}`` records."""
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert "text" in record, f"Line {line_no}: missing 'text' field"
            records.append(record)
    return records


def turns_to_segments(turns: List[Dict]) -> List[Dict]:
    return [{"speaker": t["speaker"], "text": t["text"]} for t in turns]


def parse_transcript(transcript: str) -> List[Dict]:
    """Split ``[S1] ... [S2] ...`` text into ``[{"speaker": 1|2, "text": str}]``.

    Newlines inside a turn are kept: they separate sentences that are
    synthesised one by one.
    """
    pieces = re.split(r"(\[S[12]\])", transcript)
    parsed = []
    current_speaker = None
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        m = re.match(r"\[S([12])\]", piece)
        if m:
            current_speaker = int(m.group(1))
        elif current_speaker is not None:
            lines = [" ".join(line.split()) for line in piece.split("\n")]
            text = "\n".join(line for line in lines if line)
            if text:
                parsed.append({"speaker": current_speaker, "text": text})
    return parsed


# --------------------------------------------------------------------------- #
# Backchannel word choice and evaluation
# --------------------------------------------------------------------------- #


def pick_bc_word(available: List[str], last_bc_word: Optional[str], selection: str = "avoid_repeat") -> str:
    """Draw a backchannel word.

    ``avoid_repeat`` excludes the previously used word when the list has an
    alternative; ``random`` draws from the whole list.
    """
    if selection == "random":
        return random.choice(available)
    non_repeat = [w for w in available if w != last_bc_word]
    return random.choice(non_repeat if non_repeat else available)


def compute_turn_level_eval(bc_decisions: List[Dict], gt_bc_per_turn: List[Dict]) -> Dict:
    """Compare predicted and ground-truth backchannels turn by turn.

    A turn counts as predicted when it has a ``bc_word`` and was not
    suppressed, and as ground truth when ``bc_count > 0``. Returns the per-turn
    rows and TP/FP/FN/TN with precision, recall and F1.
    """
    gt_by_turn = {g["turn_id"]: g for g in gt_bc_per_turn}
    pred_by_turn = {d["turn_id"]: d for d in bc_decisions}

    rows = []
    for turn_id in sorted(pred_by_turn.keys()):
        pred = pred_by_turn[turn_id]
        gt = gt_by_turn.get(turn_id, {"bc_count": 0, "bc_events": []})
        predicted = 1 if (pred.get("bc_word") is not None and not pred.get("suppressed", False)) else 0
        rows.append(
            {
                "turn_id": turn_id,
                "predicted_bc": predicted,
                "pred_label": pred["pred_label"],
                "pred_word": pred.get("bc_word"),
                "suppressed": pred.get("suppressed", False),
                "gt_bc": 1 if gt["bc_count"] > 0 else 0,
                "gt_bc_count": gt["bc_count"],
                "gt_bc_events": gt.get("bc_events", []),
            }
        )

    result = {"turns": rows}
    result.update(
        _confusion_metrics(
            tp=sum(1 for r in rows if r["predicted_bc"] == 1 and r["gt_bc"] == 1),
            fp=sum(1 for r in rows if r["predicted_bc"] == 1 and r["gt_bc"] == 0),
            fn=sum(1 for r in rows if r["predicted_bc"] == 0 and r["gt_bc"] == 1),
            tn=sum(1 for r in rows if r["predicted_bc"] == 0 and r["gt_bc"] == 0),
        )
    )
    return result


def aggregate_eval(per_example_evals: List[Dict]) -> Dict:
    """Pool the confusion counts of several dialogues."""
    result = {"num_examples": len(per_example_evals)}
    result.update(
        _confusion_metrics(
            tp=sum(e["TP"] for e in per_example_evals),
            fp=sum(e["FP"] for e in per_example_evals),
            fn=sum(e["FN"] for e in per_example_evals),
            tn=sum(e["TN"] for e in per_example_evals),
        )
    )
    return result


def _confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #


class NeverBackchannel:
    """Stand-in classifier that predicts no backchannel for every turn."""

    def build_input_text(self, segments, target_turn_idx, *args, **kwargs) -> str:
        return ""

    def predict(self, input_text):
        return NO_BACKCHANNEL, 1.0, {NO_BACKCHANNEL: 1.0, "Acknowledgment": 0.0, "Reactive": 0.0}


def load_classifier(
    classifier_type: str,
    model_dir: Optional[str],
    tokenizer_dir: Optional[str] = None,
    device: str = "cuda",
    confidence_threshold: float = 0.0,
    max_audio_seconds: float = MAX_AUDIO_SECONDS_DEFAULT,
):
    """Build the backchannel classifier wrapper for ``joint`` or ``text``."""
    if classifier_type == "joint":
        from ..backchannel.predictor import BackchannelClassifier

        return BackchannelClassifier(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
            confidence_threshold=confidence_threshold,
            max_audio_seconds=max_audio_seconds,
        )
    if classifier_type == "text":
        from ..backchannel.predictor import TextOnlyBackchannelClassifier

        return TextOnlyBackchannelClassifier(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
            confidence_threshold=confidence_threshold,
        )
    raise ValueError(f"No classifier to load for type '{classifier_type}'")


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


def vad_trim(wav: torch.Tensor, sr: int, vad_model, get_speech_timestamps, threshold: float = 0.6) -> torch.Tensor:
    """Cut leading and trailing non-speech from a ``[1, T]`` or ``[T]`` waveform.

    Returns ``[1, T']``. A waveform without detected speech is returned whole.
    """
    wav_1d = wav.squeeze(0) if wav.dim() == 2 else wav
    timestamps = get_speech_timestamps(wav_1d, vad_model, threshold=threshold, sampling_rate=sr)
    if not timestamps:
        return wav if wav.dim() == 2 else wav.unsqueeze(0)
    return wav_1d[timestamps[0]["start"] : timestamps[-1]["end"]].unsqueeze(0)


# --------------------------------------------------------------------------- #
# Synthesiser
# --------------------------------------------------------------------------- #


class DialogueSynthesizer:
    """Builds segment caches: turn audio, backchannel decisions, backchannel audio.

    Args:
        tts: object with ``sr`` and ``generate(text, audio_prompt_path) -> [1, T]``
            (see :mod:`neural_scheduler.pipeline.tts`).
        speaker1_ref_audio / speaker2_ref_audio: reference recordings the TTS clones.
        inter_turn_silence_ms: gap between turns in the naive back-to-back layout.
        intra_turn_pause_ms: silence between sentences of one synthesised turn.
        bc_silence_padding_ms: silence added on both sides of every backchannel
            clip and of every synthesised turn.
        voice_line_padding_ms: silence added on both sides of every recorded voice line.
        bc_all_words: every backchannel word that may be used.
        bc_word_selection: ``avoid_repeat`` or ``random`` (see :func:`pick_bc_word`).
        ensure_at_least_one_bc: for voice-line input, force one Acknowledgment
            backchannel on the most likely turn when none was predicted.
        naive_vad_trim: trim turns with Silero VAD in the naive baselines.
        absolute_paths: write absolute voice-line paths into the cache instead
            of paths relative to the cache directory.
    """

    def __init__(
        self,
        tts,
        speaker1_ref_audio: str,
        speaker2_ref_audio: str,
        inter_turn_silence_ms: int = 300,
        intra_turn_pause_ms: int = 600,
        bc_silence_padding_ms: int = 300,
        voice_line_padding_ms: int = 300,
        bc_all_words: Optional[List[str]] = None,
        bc_word_selection: str = "avoid_repeat",
        ensure_at_least_one_bc: bool = True,
        naive_vad_trim: bool = True,
        naive_vad_threshold: float = 0.6,
        absolute_paths: bool = False,
    ):
        self.tts = tts
        self.sr: int = tts.sr
        self.speaker_refs = {1: speaker1_ref_audio, 2: speaker2_ref_audio}
        self.inter_turn_silence_ms = inter_turn_silence_ms
        self.intra_turn_pause_ms = intra_turn_pause_ms
        self.bc_silence_padding_ms = bc_silence_padding_ms
        self.voice_line_padding_ms = voice_line_padding_ms
        if bc_all_words is None:
            bc_all_words = ["yeah", "okay", "right", "hmmm", "uhhmm"]
        self.bc_vocab = bc_all_words
        self.bc_word_selection = bc_word_selection
        self.ensure_at_least_one_bc = ensure_at_least_one_bc
        self.naive_vad_trim = naive_vad_trim
        self.naive_vad_threshold = naive_vad_threshold
        self.absolute_paths = absolute_paths
        self._vad = None

    # ----- small helpers --------------------------------------------------- #

    def _get_vad(self):
        if self._vad is None:
            print("  [VAD] Loading Silero VAD ...")
            from ..data.vad import load_silero_vad

            self._vad = load_silero_vad()
        return self._vad

    def _stored_path(self, audio_path: str, cache_dir: str) -> str:
        if self.absolute_paths:
            return audio_path
        return os.path.relpath(audio_path, cache_dir)

    def _pad(self, audio: torch.Tensor, pad_ms: int) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        silence = torch.zeros(1, int(pad_ms * self.sr / 1000), dtype=audio.dtype)
        return torch.cat([silence, audio, silence], dim=1)

    def _pad_silence(self, audio: torch.Tensor) -> torch.Tensor:
        return self._pad(audio, self.bc_silence_padding_ms)

    def _get_bc_audio_padded(self, speaker_id: int, word: str, seed: Optional[int] = None) -> torch.Tensor:
        """Synthesise one backchannel in ``speaker_id``'s voice, padded with silence."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  [BC synth] '{word}' for speaker {speaker_id} (seed={seed})")
        audio = self.tts.generate(word, self.speaker_refs[speaker_id])
        return self._pad(audio, self.bc_silence_padding_ms)

    def _load_voice_line(self, audio_path: str) -> torch.Tensor:
        """Load a recorded turn as padded mono ``[1, T]`` at the TTS sample rate."""
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return self._pad(wav, self.voice_line_padding_ms)

    def _synthesize_turn(self, text: str, speaker: int) -> Tuple[torch.Tensor, int]:
        """Synthesise one turn sentence by sentence; returns the waveform and the sentence count."""
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        intra_silence = torch.zeros(1, int(self.intra_turn_pause_ms * self.sr / 1000))
        sentence_wavs = [self.tts.generate(sentence, self.speaker_refs[speaker]) for sentence in sentences]
        parts = []
        for i, wav in enumerate(sentence_wavs):
            parts.append(wav)
            if i < len(sentence_wavs) - 1:
                parts.append(intra_silence)
        return torch.cat(parts, dim=1), len(sentences)

    def _naive_trim(self, wav: torch.Tensor, turn_id: int, speaker: int) -> torch.Tensor:
        if not self.naive_vad_trim:
            return wav
        vad_model, get_speech_timestamps = self._get_vad()
        trimmed = vad_trim(wav, self.sr, vad_model, get_speech_timestamps, threshold=self.naive_vad_threshold)
        if trimmed.shape[1] < wav.shape[1]:
            print(
                f"  [naive/VAD] Turn {turn_id} (S{speaker}): "
                f"{wav.shape[1] / self.sr:.3f}s -> {trimmed.shape[1] / self.sr:.3f}s"
            )
        return trimmed

    def _lay_out_turn(self, turn_audios: List[Dict], fields: Dict, wav: torch.Tensor) -> None:
        """Append a turn to ``turn_audios`` right after the previous one plus the inter-turn gap."""
        silence_samples = int(self.inter_turn_silence_ms * self.sr / 1000)
        if turn_audios:
            start_time = turn_audios[-1]["end_time"] + (silence_samples / self.sr)
        else:
            start_time = 0.0
        duration = wav.shape[1] / self.sr
        end_time = start_time + duration
        entry = dict(fields)
        entry.update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": int(start_time * METADATA_FRAME_RATE),
                "end_frame": int(end_time * METADATA_FRAME_RATE),
                "duration": duration,
                "audio": wav,
            }
        )
        turn_audios.append(entry)

    def _synthesize_all_turns(self, segments: List[Dict], naive: bool = False) -> List[Dict]:
        turn_audios: List[Dict] = []
        for turn_id, seg in enumerate(segments):
            wav, n_sentences = self._synthesize_turn(seg["text"], seg["speaker"])
            if naive:
                wav = self._naive_trim(wav, turn_id, seg["speaker"])
                print(
                    f"  [naive] Turn {turn_id} (S{seg['speaker']}): {n_sentences} sentence(s), "
                    f"{wav.shape[1] / self.sr:.2f}s"
                )
            self._lay_out_turn(
                turn_audios, {"turn_id": turn_id, "speaker_id": seg["speaker"], "text": seg["text"]}, wav
            )
        return turn_audios

    def _load_all_voice_lines(self, turns: List[Dict], naive: bool = False) -> List[Dict]:
        turn_audios: List[Dict] = []
        for turn in turns:
            tag = "[naive] " if naive else ""
            print(f"  {tag}Loading turn {turn['turn_id']} (S{turn['speaker']}): {turn['audio_path']}")
            wav = self._load_voice_line(turn["audio_path"])
            if naive:
                wav = self._naive_trim(wav, turn["turn_id"], turn["speaker"])
            self._lay_out_turn(
                turn_audios,
                {
                    "turn_id": turn["turn_id"],
                    "speaker_id": turn["speaker"],
                    "text": turn["text"],
                    "audio_path": turn["audio_path"],
                    # Position of this voice line inside the original recording,
                    # when the input provides it (benchmark mode).
                    "input_wav_slice_start_sec": turn.get("slice_start_sec"),
                    "input_wav_slice_end_sec": turn.get("slice_end_sec"),
                },
                wav,
            )
        return turn_audios

    def _voice_line_metadata(self, turn_audios: List[Dict], cache_dir: str) -> List[Dict]:
        keys = (
            "turn_id",
            "speaker_id",
            "text",
            "audio_path",
            "start_time",
            "end_time",
            "start_frame",
            "end_frame",
            "duration",
            "input_wav_slice_start_sec",
            "input_wav_slice_end_sec",
        )
        metadata = []
        for t in turn_audios:
            row = {k: t[k] for k in keys}
            row["audio_path"] = self._stored_path(t["audio_path"], cache_dir)
            metadata.append(row)
        return metadata

    @staticmethod
    def _tts_metadata(turn_audios: List[Dict]) -> List[Dict]:
        keys = ("turn_id", "speaker_id", "text", "start_time", "end_time", "start_frame", "end_frame", "duration")
        return [{k: t[k] for k in keys} for t in turn_audios]

    def _write_speech(self, audio_dir: str, cache_dir: str, counter: int, t: Dict, pad: bool) -> Dict:
        """Write one speech segment and return its manifest entry."""
        speaker = t["speaker_id"]
        fname = f"seg_{counter:04d}_s{speaker}_turn{t['turn_id']}_speech.wav"
        audio = self._pad_silence(t["audio"]) if pad else t["audio"]
        sf.write(os.path.join(audio_dir, fname), audio.squeeze(0).numpy(), self.sr)
        entry = {"filename": fname, "turn_id": t["turn_id"], "type": "speech"}
        if "audio_path" in t:
            entry["source_audio_path"] = self._stored_path(t["audio_path"], cache_dir)
        entry["sample_rate"] = self.sr
        if t.get("input_wav_slice_start_sec") is not None:
            entry["input_wav_slice_start_sec"] = t["input_wav_slice_start_sec"]
            entry["input_wav_slice_end_sec"] = t["input_wav_slice_end_sec"]
        return entry

    def _write_bc(
        self,
        audio_dir: str,
        counter: int,
        bc_speaker: int,
        turn_id: int,
        bc_word: str,
        seed: Optional[int],
        decision: Optional[Dict] = None,
        sanitize_filename: bool = True,
    ) -> Dict:
        """Synthesise and write one backchannel; returns its manifest entry."""
        bc_audio = self._get_bc_audio_padded(bc_speaker, bc_word, seed=seed)
        word_in_name = bc_word.replace(" ", "_") if sanitize_filename else bc_word
        fname = f"seg_{counter:04d}_s{bc_speaker}_turn{turn_id}_bc_{word_in_name}.wav"
        sf.write(os.path.join(audio_dir, fname), bc_audio.squeeze(0).numpy(), self.sr)
        entry = {"filename": fname, "turn_id": turn_id, "type": "backchannel", "bc_text": bc_word}
        if decision is not None:
            entry["bc_class"] = decision["pred_label"]
            entry["bc_confidence"] = decision["confidence"]
        entry["sample_rate"] = self.sr
        return entry

    @staticmethod
    def _prepare_cache(cache_dir: str, clean: bool = False) -> str:
        audio_dir = os.path.join(cache_dir, "audio_segments")
        if clean and os.path.isdir(audio_dir):
            shutil.rmtree(audio_dir)
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

    @staticmethod
    def _write_json(path: str, payload) -> None:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _save_combined(self, turn_audios: List[Dict], cache_dir: str) -> None:
        """Write the naive back-to-back layout as mono and as one channel per speaker."""
        total_samples = int(turn_audios[-1]["end_time"] * self.sr) + self.sr
        mono = torch.zeros(1, total_samples)
        dual = torch.zeros(2, total_samples)
        for t in turn_audios:
            start = int(t["start_time"] * self.sr)
            wav = t["audio"]
            end = min(start + wav.shape[1], total_samples)
            length = end - start
            mono[0, start:end] = wav[0, :length]
            dual[t["speaker_id"] - 1, start:end] = wav[0, :length]
        torchaudio.save(os.path.join(cache_dir, "combined_mono.wav"), mono, self.sr)
        torchaudio.save(os.path.join(cache_dir, "combined_dual.wav"), dual, self.sr)
        print(f"Combined audio saved to {cache_dir}")

    # ----- backchannel decisions ------------------------------------------- #

    def _choose_word(self, pred_label: str, bc_vocab_per_class: Dict, last_bc_word: Optional[str]) -> Optional[str]:
        if pred_label == NO_BACKCHANNEL:
            return None
        class_words = bc_vocab_per_class.get(pred_label, [])
        if not class_words:
            return None
        available = [w for w in class_words if w in self.bc_vocab]
        if not available:
            print(f"  WARNING: no usable backchannel word for '{pred_label}': {class_words}")
            return None
        return pick_bc_word(available, last_bc_word, self.bc_word_selection)

    @staticmethod
    def _report_decision(turn_id, seg, pred_label, bc_word, all_probs) -> None:
        ack = all_probs["Acknowledgment"]
        react = all_probs["Reactive"]
        status = f"-> {pred_label}: '{bc_word}'" if bc_word else f"-> {NO_BACKCHANNEL}"
        print(
            f"  Turn {turn_id} (S{seg['speaker']}): {status} "
            f"(bc_prob={ack + react:.3f}, ack={ack:.3f}, react={react:.3f})"
        )

    @staticmethod
    def _suppress_consecutive_same_speaker_bcs(bc_decisions: List[Dict], segments: List[Dict]) -> None:
        """Drop the backchannel on a turn whose same-speaker predecessor also has one."""
        for i in range(1, len(bc_decisions)):
            if segments[i]["speaker"] != segments[i - 1]["speaker"]:
                continue
            prev, curr = bc_decisions[i - 1], bc_decisions[i]
            if prev["bc_word"] is not None and curr["bc_word"] is not None:
                print(
                    f"  [BC suppression] Turn {curr['turn_id']}: "
                    f"suppressed '{curr['bc_word']}' (consecutive same-speaker turn)."
                )
                curr["bc_word"] = None
                curr["suppressed"] = True

    def _force_one_bc(self, bc_decisions: List[Dict], bc_vocab_per_class: Dict, last_bc_word: Optional[str]) -> None:
        """Give the most backchannel-likely turn an Acknowledgment when no turn has one."""
        if any(d.get("bc_word") is not None and not d.get("suppressed", False) for d in bc_decisions):
            return
        best = max(
            bc_decisions,
            key=lambda d: d["all_probs"].get("Acknowledgment", 0.0) + d["all_probs"].get("Reactive", 0.0),
        )
        available = [w for w in bc_vocab_per_class.get("Acknowledgment", []) if w in self.bc_vocab]
        if not available:
            available = list(self.bc_vocab)
        if not available:
            print("  [ensure_bc] WARNING: no backchannel vocabulary available, cannot force one.")
            return
        chosen = pick_bc_word(available, last_bc_word, self.bc_word_selection)
        best["bc_word"] = chosen
        best["pred_label"] = "Acknowledgment"
        best["suppressed"] = False
        print(
            f"  [ensure_bc] Forced BC '{chosen}' on turn {best['turn_id']} "
            f"(bc_prob={best['all_probs'].get('Acknowledgment', 0):.3f}/"
            f"{best['all_probs'].get('Reactive', 0):.3f})"
        )

    # ----- voice-line input ------------------------------------------------- #

    def generate_dialogue_from_turns(
        self,
        turns: List[Dict],
        cache_dir: str,
        classifier,
        bc_vocab_per_class: Dict[str, List[str]],
        context_window_turns: int = 10,
        context_window_future_turns: int = 5,
        include_bc_context: bool = True,
        suppress_consecutive_same_speaker_bcs: bool = True,
        seed: Optional[int] = None,
        save_combined: bool = True,
        clean_cache: bool = False,
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """Build the cache for a dialogue with recorded voice lines.

        The classifier sees each turn's text context and its voice line.
        Returns ``(manifest_path, turn_metadata, bc_decisions)``.
        """
        if seed is not None:
            random.seed(seed)
        audio_dir = self._prepare_cache(cache_dir, clean_cache)
        segments = turns_to_segments(turns)

        bc_decisions: List[Dict] = []
        bc_by_turn_id: Dict[int, Dict] = {}
        last_bc_word: Optional[str] = None

        for list_idx, (seg, turn) in enumerate(zip(segments, turns)):
            input_text = classifier.build_input_text(
                segments,
                list_idx,
                context_window_turns,
                context_window_future_turns=context_window_future_turns,
                include_bc_context=include_bc_context,
            )
            pred_label, confidence, all_probs = _predict(classifier, input_text, audio=turn.get("audio_path"))
            bc_word = self._choose_word(pred_label, bc_vocab_per_class, last_bc_word)
            if bc_word is not None:
                last_bc_word = bc_word

            decision = {
                "turn_id": turn["turn_id"],
                "list_idx": list_idx,
                "pred_label": pred_label,
                "confidence": confidence,
                "all_probs": all_probs,
                "bc_word": bc_word,
                "suppressed": False,
            }
            bc_decisions.append(decision)
            bc_by_turn_id[turn["turn_id"]] = decision

            seg["bc_class"] = _context_tag(pred_label, bc_word)
            self._report_decision(turn["turn_id"], seg, pred_label, bc_word, all_probs)

        if suppress_consecutive_same_speaker_bcs:
            self._suppress_consecutive_same_speaker_bcs(bc_decisions, segments)
        if self.ensure_at_least_one_bc and not isinstance(classifier, NeverBackchannel):
            self._force_one_bc(bc_decisions, bc_vocab_per_class, last_bc_word)

        turn_audios = self._load_all_voice_lines(turns)
        turn_metadata = self._voice_line_metadata(turn_audios, cache_dir)

        counter = 0
        manifest: Dict[str, list] = {"speaker_1": [], "speaker_2": []}
        s1_turns = sorted((t for t in turn_audios if t["speaker_id"] == 1), key=lambda x: x["turn_id"])
        s2_turns = sorted((t for t in turn_audios if t["speaker_id"] == 2), key=lambda x: x["turn_id"])

        for t in s1_turns:
            manifest["speaker_1"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=False))
            counter += 1
        # Speaker 1 backchannels on speaker 2's turns. The seed advances after
        # each of them; speaker 2's backchannels below all reuse the final value.
        for t in s2_turns:
            decision = bc_by_turn_id[t["turn_id"]]
            if decision["bc_word"] is not None:
                manifest["speaker_1"].append(
                    self._write_bc(audio_dir, counter, 1, t["turn_id"], decision["bc_word"], seed, decision)
                )
                if seed is not None:
                    seed += 1
                counter += 1
        manifest["speaker_1"].sort(key=lambda x: x["turn_id"])

        for t in s2_turns:
            manifest["speaker_2"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=False))
            counter += 1
        for t in s1_turns:
            decision = bc_by_turn_id[t["turn_id"]]
            if decision["bc_word"] is not None:
                manifest["speaker_2"].append(
                    self._write_bc(audio_dir, counter, 2, t["turn_id"], decision["bc_word"], seed, decision)
                )
                counter += 1
        manifest["speaker_2"].sort(key=lambda x: x["turn_id"])

        manifest_path = os.path.join(cache_dir, "manifest.json")
        self._write_json(manifest_path, manifest)
        decisions_path = os.path.join(cache_dir, "bc_decisions.json")
        self._write_json(decisions_path, bc_decisions)
        print(f"Saved {counter} segments -> {audio_dir}")
        print(f"Manifest:     {manifest_path}")
        print(f"BC decisions: {decisions_path}")

        if save_combined:
            self._save_combined(turn_audios, cache_dir)
        return manifest_path, turn_metadata, bc_decisions

    # ----- text input, classifier ------------------------------------------ #

    def generate_dialogue_with_classifier(
        self,
        transcript: str,
        cache_dir: str,
        classifier,
        bc_vocab_per_class: Dict[str, List[str]],
        context_window_turns: int = 10,
        context_window_future_turns: int = 5,
        include_bc_context: bool = True,
        seed: Optional[int] = None,
        save_combined: bool = True,
        clean_cache: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """Build the cache for a ``[S1] ... [S2] ...`` transcript.

        All turns are synthesised first because the joint classifier needs the
        audio of the turn it judges. Returns ``(manifest_path, turn_metadata)``.
        """
        if seed is not None:
            random.seed(seed)
        audio_dir = self._prepare_cache(cache_dir, clean_cache)
        segments = parse_transcript(transcript)
        turn_audios = self._synthesize_all_turns(segments)

        bc_decisions: List[Dict] = []
        last_bc_word: Optional[str] = None
        for turn_idx, seg in enumerate(segments):
            input_text = classifier.build_input_text(
                segments,
                turn_idx,
                context_window_turns,
                context_window_future_turns=context_window_future_turns,
                include_bc_context=include_bc_context,
            )
            pred_label, confidence, all_probs = _predict(
                classifier, input_text, audio=turn_audios[turn_idx]["audio"], audio_sample_rate=self.sr
            )
            bc_word = self._choose_word(pred_label, bc_vocab_per_class, last_bc_word)
            if bc_word is not None:
                last_bc_word = bc_word
            bc_decisions.append(
                {
                    "turn_idx": turn_idx,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "all_probs": all_probs,
                    "bc_word": bc_word,
                }
            )
            seg["bc_class"] = _context_tag(pred_label, bc_word)
            self._report_decision(turn_idx, seg, pred_label, bc_word, all_probs)

        counter = 0
        manifest: Dict[str, list] = {"speaker_1": [], "speaker_2": []}
        s1_turns = sorted((t for t in turn_audios if t["speaker_id"] == 1), key=lambda x: x["turn_id"])
        s2_turns = sorted((t for t in turn_audios if t["speaker_id"] == 2), key=lambda x: x["turn_id"])

        for t in s1_turns:
            manifest["speaker_1"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=True))
            counter += 1
        # Speaker 1's backchannels all use the given seed; speaker 2's draw a fresh one each.
        for t in s2_turns:
            decision = bc_decisions[t["turn_id"]]
            if decision["bc_word"] is not None:
                manifest["speaker_1"].append(
                    self._write_bc(audio_dir, counter, 1, t["turn_id"], decision["bc_word"], seed, decision)
                )
                counter += 1
        manifest["speaker_1"].sort(key=lambda x: x["turn_id"])

        for t in s2_turns:
            manifest["speaker_2"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=True))
            counter += 1
        for t in s1_turns:
            decision = bc_decisions[t["turn_id"]]
            if decision["bc_word"] is not None:
                manifest["speaker_2"].append(
                    self._write_bc(audio_dir, counter, 2, t["turn_id"], decision["bc_word"], None, decision)
                )
                counter += 1
        manifest["speaker_2"].sort(key=lambda x: x["turn_id"])

        manifest_path = os.path.join(cache_dir, "manifest.json")
        self._write_json(manifest_path, manifest)
        decisions_path = os.path.join(cache_dir, "bc_decisions.json")
        self._write_json(decisions_path, bc_decisions)
        print(f"Saved {counter} audio segments to {audio_dir}")
        print(f"Manifest saved to {manifest_path}")
        print(f"BC decisions saved to {decisions_path}")

        if save_combined:
            self._save_combined(turn_audios, cache_dir)
        return manifest_path, self._tts_metadata(turn_audios)

    # ----- text input, random backchannels --------------------------------- #

    def generate_dialogue(
        self,
        transcript: str,
        cache_dir: str,
        bc_probability: float = 1.0,
        bc_vocab_subset: Optional[List[str]] = None,
        seed: Optional[int] = None,
        save_combined: bool = True,
        clean_cache: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """Build the cache for a transcript without a classifier.

        Every turn receives a backchannel with probability ``bc_probability``,
        drawn from ``bc_vocab_subset``. Returns ``(manifest_path, turn_metadata)``.
        """
        if seed is not None:
            random.seed(seed)
        bc_words = bc_vocab_subset or self.bc_vocab
        missing = set(bc_words) - set(self.bc_vocab)
        if missing:
            raise ValueError(f"bc_vocab_subset contains words outside the backchannel vocabulary: {missing}")

        audio_dir = self._prepare_cache(cache_dir, clean_cache)
        turn_audios = self._synthesize_all_turns(parse_transcript(transcript))

        counter = 0
        manifest: Dict[str, list] = {"speaker_1": [], "speaker_2": []}
        s1_turns = sorted((t for t in turn_audios if t["speaker_id"] == 1), key=lambda x: x["turn_id"])
        s2_turns = sorted((t for t in turn_audios if t["speaker_id"] == 2), key=lambda x: x["turn_id"])
        last_word = {1: None, 2: None}

        def add_random_bcs(bc_speaker: int, host_turns: List[Dict]) -> None:
            nonlocal counter
            for t in host_turns:
                if random.random() < bc_probability:
                    word = pick_bc_word(bc_words, last_word[bc_speaker], self.bc_word_selection)
                    last_word[bc_speaker] = word
                    manifest[f"speaker_{bc_speaker}"].append(
                        self._write_bc(
                            audio_dir, counter, bc_speaker, t["turn_id"], word, None, sanitize_filename=False
                        )
                    )
                    counter += 1

        for t in s1_turns:
            manifest["speaker_1"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=True))
            counter += 1
        add_random_bcs(1, s2_turns)
        manifest["speaker_1"].sort(key=lambda x: x["turn_id"])

        for t in s2_turns:
            manifest["speaker_2"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=True))
            counter += 1
        add_random_bcs(2, s1_turns)
        manifest["speaker_2"].sort(key=lambda x: x["turn_id"])

        manifest_path = os.path.join(cache_dir, "manifest.json")
        self._write_json(manifest_path, manifest)
        if save_combined:
            self._save_combined(turn_audios, cache_dir)
        return manifest_path, self._tts_metadata(turn_audios)

    # ----- naive baselines (no backchannels) -------------------------------- #

    def _write_naive_cache(self, turn_audios: List[Dict], cache_dir: str, audio_dir: str, pad: bool, save_combined: bool):
        counter = 0
        manifest: Dict[str, list] = {"speaker_1": [], "speaker_2": []}
        for t in sorted(turn_audios, key=lambda x: x["turn_id"]):
            manifest[f"speaker_{t['speaker_id']}"].append(self._write_speech(audio_dir, cache_dir, counter, t, pad=pad))
            counter += 1
        manifest_path = os.path.join(cache_dir, "manifest.json")
        self._write_json(manifest_path, manifest)
        print(f"  [naive] Saved {counter} segments -> {audio_dir}")
        print(f"  [naive] Manifest: {manifest_path}")
        if save_combined:
            self._save_combined(turn_audios, cache_dir)
        return manifest_path

    def generate_naive_concatenation_from_transcript(
        self,
        transcript: str,
        cache_dir: str,
        seed: Optional[int] = None,
        save_combined: bool = True,
        clean_cache: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """Naive baseline for text input: synthesised turns back to back, no backchannels."""
        if seed is not None:
            random.seed(seed)
        audio_dir = self._prepare_cache(cache_dir, clean_cache)
        turn_audios = self._synthesize_all_turns(parse_transcript(transcript), naive=True)
        manifest_path = self._write_naive_cache(turn_audios, cache_dir, audio_dir, pad=True, save_combined=save_combined)
        return manifest_path, self._tts_metadata(turn_audios)

    def generate_naive_concatenation(
        self, turns: List[Dict], cache_dir: str, save_combined: bool = True, clean_cache: bool = False
    ) -> Tuple[str, List[Dict]]:
        """Naive baseline for voice-line input: recorded turns back to back, no backchannels."""
        audio_dir = self._prepare_cache(cache_dir, clean_cache)
        turn_audios = self._load_all_voice_lines(sorted(turns, key=lambda t: t["turn_id"]), naive=True)
        manifest_path = self._write_naive_cache(turn_audios, cache_dir, audio_dir, pad=False, save_combined=save_combined)
        return manifest_path, self._voice_line_metadata(turn_audios, cache_dir)


def _predict(classifier, input_text: str, audio=None, audio_sample_rate: Optional[int] = None):
    """Call ``classifier.predict``, passing the turn audio when the classifier takes it."""
    if "audio" not in inspect.signature(classifier.predict).parameters:
        return classifier.predict(input_text)
    if audio_sample_rate is not None:
        return classifier.predict(input_text, audio=audio, audio_sample_rate=audio_sample_rate)
    return classifier.predict(input_text, audio=audio)


def _context_tag(pred_label: str, bc_word: Optional[str]) -> str:
    """Tag recorded on a turn for the text context of later turns.

    The classifier was trained on fine-grained tags, so a predicted class is
    mapped to one of them; a turn without a backchannel word gets none.
    """
    if not bc_word:
        return NO_BACKCHANNEL
    from ..backchannel.predictor import MERGED_TO_CANONICAL_BC_TAG

    return MERGED_TO_CANONICAL_BC_TAG.get(pred_label) or NO_BACKCHANNEL


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1: backchannel decisions, turn audio and backchannel audio -> segment cache."
    )

    inputs = parser.add_argument_group("input (choose one)")
    inputs.add_argument("--input_dialogue_json", type=str, default=None, help="One dialogue with a voice line per turn.")
    inputs.add_argument("--input_jsonl", type=str, default=None, help='JSONL of {"text": "[S1] ... [S2] ..."}; turns are synthesised.')
    inputs.add_argument("--bc_vocab_jsonl", type=str, default=None, help='JSONL of {"text": ..., "bc_vocab": [...]} (random backchannels).')
    inputs.add_argument("--input_bc_gt_jsonl", type=str, default=None, help="Benchmark JSONL with voice lines and ground-truth backchannels.")
    parser.add_argument("--out_dir", type=str, required=True, help="Caches are written to <out_dir>/segment_cache/<dialogue_id>/.")

    clf = parser.add_argument_group("backchannel classifier")
    clf.add_argument("--bc_classifier_type", choices=CLASSIFIER_TYPES, default="joint",
                     help="joint: text+audio model; text: RoBERTa text-only model; none: no classifier "
                          "(random backchannels for text input, no backchannels for voice-line input).")
    clf.add_argument("--no_classifier", action="store_true", help="Same as --bc_classifier_type none.")
    clf.add_argument("--bc_classifier_dir", type=str, default=None)
    clf.add_argument("--bc_tokenizer_dir", type=str, default=None)
    clf.add_argument("--bc_confidence_threshold", type=float, default=0.0,
                     help="Backchannel when P(Acknowledgment)+P(Reactive) >= threshold.")
    clf.add_argument("--bc_context_window_turns", type=int, default=5)
    clf.add_argument("--bc_context_window_future_turns", type=int, default=5)
    clf.add_argument("--no_bc_context", action="store_true", help="Do not tag earlier turns with their backchannel class.")
    clf.add_argument("--no_suppress_consecutive_bcs", action="store_true",
                     help="Voice-line input: keep backchannels on consecutive same-speaker turns.")
    clf.add_argument("--bc_max_audio_seconds", type=float, default=MAX_AUDIO_SECONDS_DEFAULT,
                     help="Joint classifier: cap on the target turn's audio.")
    clf.add_argument("--bc_vocab_per_class", type=str, default=None, help="JSON mapping class name to backchannel words.")
    clf.add_argument("--bc_vocab", type=str, nargs="+", default=["hmmm", "uhhmm"],
                     help="Additional backchannel words made available.")
    clf.add_argument("--bc_word_selection", choices=BC_WORD_SELECTIONS, default="avoid_repeat")
    clf.add_argument("--ensure_at_least_one_bc", action=argparse.BooleanOptionalAction, default=True,
                     help="Voice-line input: force one backchannel when the classifier predicts none.")
    clf.add_argument("--bc_probability", type=float, default=1.0, help="Without a classifier: backchannel probability per turn.")
    clf.add_argument("--bc_vocab_subset", type=str, nargs="+", default=["hmmm", "uhhmm"],
                     help="Without a classifier: words to draw from.")

    tts = parser.add_argument_group("speech synthesis")
    tts.add_argument("--tts_backend", choices=TTS_BACKENDS, default="chatterbox")
    tts.add_argument("--ref_s1", type=str, required=True, help="Reference recording for speaker 1's voice.")
    tts.add_argument("--ref_s2", type=str, required=True, help="Reference recording for speaker 2's voice.")
    tts.add_argument("--exaggeration", type=float, default=0.5, help="Chatterbox only.")
    tts.add_argument("--cfg_weight", type=float, default=0.5, help="Chatterbox only.")
    tts.add_argument("--seed", type=int, default=42)
    tts.add_argument("--device", type=str, default="cuda")

    timing = parser.add_argument_group("silence and padding")
    timing.add_argument("--inter_turn_silence_ms", type=int, default=1000)
    timing.add_argument("--intra_turn_pause_ms", type=int, default=1000)
    timing.add_argument("--bc_silence_padding_ms", type=int, default=5000)
    timing.add_argument("--voice_line_padding_ms", type=int, default=5000)

    naive = parser.add_argument_group("naive baseline")
    naive.add_argument("--run_naive_only", action="store_true",
                       help="Voice-line and benchmark input: in addition to the normal run, write a "
                            "back-to-back baseline without backchannels to <cache>/naive/.")
    naive.add_argument("--naive_only", action="store_true",
                       help="Skip the classifier and backchannels entirely and write only the "
                            "back-to-back baseline into the cache directory.")
    naive.add_argument("--naive_vad_trim", action=argparse.BooleanOptionalAction, default=True,
                       help="Trim leading/trailing silence of each turn in the baseline.")

    cache = parser.add_argument_group("cache")
    cache.add_argument("--absolute_paths", action="store_true",
                       help="Store absolute voice-line paths instead of paths relative to the cache.")
    cache.add_argument("--clean_cache", action="store_true",
                       help="Delete an existing audio_segments/ folder before writing.")
    return parser


def _write_metadata(cache_dir: str, metadata: List[Dict]) -> str:
    path = os.path.join(cache_dir, "turn_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def _run_naive(synth: DialogueSynthesizer, turns: List[Dict], cache_dir: str, clean: bool) -> None:
    print(f"\n  [naive] Generating naive output -> {cache_dir}")
    manifest, metadata = synth.generate_naive_concatenation(turns=turns, cache_dir=cache_dir, clean_cache=clean)
    print(f"  [naive] Manifest: {manifest}")
    print(f"  [naive] Metadata: {_write_metadata(cache_dir, metadata)}")


def _read_texts(args) -> Tuple[List[str], Optional[List[Optional[List[str]]]]]:
    """Transcripts and, for ``--bc_vocab_jsonl``, the per-dialogue vocabularies."""
    if args.bc_vocab_jsonl is not None:
        records = load_bc_vocab_jsonl(args.bc_vocab_jsonl)
        return [r["text"] for r in records], [r.get("bc_vocab") for r in records]
    with open(args.input_jsonl) as f:
        return [json.loads(line)["text"] for line in f if line.strip()], None


def main(argv: Optional[List[str]] = None, tts=None, classifier=None) -> None:
    """Run step 1. ``tts`` and ``classifier`` may be injected (tests); otherwise they are loaded."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_classifier:
        args.bc_classifier_type = "none"
    n_inputs = sum(
        x is not None for x in (args.input_dialogue_json, args.input_jsonl, args.bc_vocab_jsonl, args.input_bc_gt_jsonl)
    )
    if n_inputs != 1:
        parser.error("give exactly one of --input_dialogue_json, --input_jsonl, --bc_vocab_jsonl, --input_bc_gt_jsonl")
    voice_line_input = args.input_dialogue_json is not None or args.input_bc_gt_jsonl is not None
    use_classifier = args.bc_classifier_type != "none" and not args.naive_only

    # Backchannel vocabulary.
    bc_vocab_per_class = None
    master_vocab = list(args.bc_vocab)
    if use_classifier:
        if classifier is None and args.bc_classifier_dir is None:
            parser.error("--bc_classifier_dir is required unless --bc_classifier_type none or --naive_only")
        if args.bc_vocab_per_class is None:
            parser.error("--bc_vocab_per_class is required with a classifier")
        bc_vocab_per_class = load_bc_vocab_per_class(args.bc_vocab_per_class)
        master_vocab = sorted(set(args.bc_vocab) | set(collect_all_bc_words(bc_vocab_per_class)))
        print(f"Master BC vocab ({len(master_vocab)} words): {master_vocab}")

    texts, per_dialogue_vocabs = (None, None) if voice_line_input else _read_texts(args)
    if not use_classifier and per_dialogue_vocabs and per_dialogue_vocabs[0] is not None:
        extra = set()
        for vocab in per_dialogue_vocabs:
            if vocab:
                extra.update(vocab)
        master_vocab = sorted(set(master_vocab) | extra)

    if tts is None:
        tts = load_tts(args.tts_backend, device=args.device, exaggeration=args.exaggeration, cfg_weight=args.cfg_weight)
    synth = DialogueSynthesizer(
        tts=tts,
        speaker1_ref_audio=args.ref_s1,
        speaker2_ref_audio=args.ref_s2,
        inter_turn_silence_ms=args.inter_turn_silence_ms,
        intra_turn_pause_ms=args.intra_turn_pause_ms,
        bc_silence_padding_ms=args.bc_silence_padding_ms,
        voice_line_padding_ms=args.voice_line_padding_ms,
        bc_all_words=master_vocab,
        bc_word_selection=args.bc_word_selection,
        ensure_at_least_one_bc=args.ensure_at_least_one_bc,
        naive_vad_trim=args.naive_vad_trim,
        absolute_paths=args.absolute_paths,
    )

    if use_classifier and classifier is None:
        classifier = load_classifier(
            args.bc_classifier_type,
            model_dir=args.bc_classifier_dir,
            tokenizer_dir=args.bc_tokenizer_dir,
            device=args.device,
            confidence_threshold=args.bc_confidence_threshold,
            max_audio_seconds=args.bc_max_audio_seconds,
        )
    if not use_classifier:
        classifier = None

    cache_root = os.path.join(args.out_dir, "segment_cache")
    classifier_kwargs = dict(
        context_window_turns=args.bc_context_window_turns,
        context_window_future_turns=args.bc_context_window_future_turns,
        include_bc_context=not args.no_bc_context,
    )

    # ----- naive baseline only -------------------------------------------- #
    if args.naive_only:
        if args.input_bc_gt_jsonl is not None:
            dialogues = [(r["dialogue_id"], r["turns"]) for r in load_bc_gt_jsonl(args.input_bc_gt_jsonl)]
        elif args.input_dialogue_json is not None:
            dialogue = load_dialogue_json(args.input_dialogue_json)
            dialogues = [(dialogue["dialogue_id"], dialogue["turns"])]
        else:
            dialogues = None
        if dialogues is not None:
            for dialogue_id, turns in dialogues:
                _run_naive(synth, turns, os.path.join(cache_root, dialogue_id), args.clean_cache)
        else:
            for idx, text in enumerate(texts):
                cache_dir = os.path.join(cache_root, f"example_{idx:04d}")
                manifest, metadata = synth.generate_naive_concatenation_from_transcript(
                    transcript=text.strip(), cache_dir=cache_dir, seed=args.seed, clean_cache=args.clean_cache
                )
                print(f"  [naive] Example {idx}: {manifest}, {_write_metadata(cache_dir, metadata)}")
        return

    # ----- benchmark input with evaluation --------------------------------- #
    if args.input_bc_gt_jsonl is not None:
        if classifier is None:
            parser.error("--input_bc_gt_jsonl needs a classifier")
        records = load_bc_gt_jsonl(args.input_bc_gt_jsonl)
        print(f"Loaded {len(records)} examples from {args.input_bc_gt_jsonl}")
        per_example_evals = []
        for rec in records:
            dialogue_id = rec["dialogue_id"]
            turns = rec["turns"]
            tags_removed = rec.get("tags_removed", "?")
            cache_dir = os.path.join(cache_root, dialogue_id)
            print(f"\n{'=' * 50}")
            print(f"Example {dialogue_id}  ({len(turns)} turns, {tags_removed} GT tags)")

            _, metadata, bc_decisions = synth.generate_dialogue_from_turns(
                turns=turns,
                cache_dir=cache_dir,
                classifier=classifier,
                bc_vocab_per_class=bc_vocab_per_class,
                suppress_consecutive_same_speaker_bcs=not args.no_suppress_consecutive_bcs,
                seed=args.seed + int(dialogue_id),
                clean_cache=args.clean_cache,
                **classifier_kwargs,
            )
            _write_metadata(cache_dir, metadata)

            eval_result = compute_turn_level_eval(bc_decisions, rec["gt_bc_per_turn"])
            eval_result["dialogue_id"] = dialogue_id
            eval_result["tags_removed"] = tags_removed
            eval_path = os.path.join(cache_dir, "bc_eval.json")
            with open(eval_path, "w") as f:
                json.dump(eval_result, f, indent=2)
            per_example_evals.append(eval_result)

            pred_count = sum(1 for d in bc_decisions if d["bc_word"] is not None and not d.get("suppressed", False))
            print(f"  GT BCs (tags removed): {tags_removed}")
            print(f"  Predicted BCs:         {pred_count}")
            print(f"  TP={eval_result['TP']}  FP={eval_result['FP']}  FN={eval_result['FN']}  TN={eval_result['TN']}")
            print(f"  P={eval_result['precision']:.3f}  R={eval_result['recall']:.3f}  F1={eval_result['f1']:.3f}")
            print(f"  bc_eval.json -> {eval_path}")

            if args.run_naive_only:
                _run_naive(synth, turns, os.path.join(cache_dir, "naive"), args.clean_cache)

        summary = aggregate_eval(per_example_evals)
        os.makedirs(args.out_dir, exist_ok=True)
        summary_path = os.path.join(args.out_dir, "bc_eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"per_example": per_example_evals, "aggregate": summary}, f, indent=2)
        print(f"\n{'=' * 50}")
        print(f"AGGREGATE  ({summary['num_examples']} examples)")
        print(f"  TP={summary['TP']}  FP={summary['FP']}  FN={summary['FN']}  TN={summary['TN']}")
        print(f"  Precision : {summary['precision']:.4f}")
        print(f"  Recall    : {summary['recall']:.4f}")
        print(f"  F1        : {summary['f1']:.4f}")
        print(f"Summary -> {summary_path}")
        return

    # ----- one voice-line dialogue ------------------------------------------ #
    if args.input_dialogue_json is not None:
        dialogue = load_dialogue_json(args.input_dialogue_json)
        cache_dir = os.path.join(cache_root, dialogue["dialogue_id"])
        manifest_path, metadata, _ = synth.generate_dialogue_from_turns(
            turns=dialogue["turns"],
            cache_dir=cache_dir,
            classifier=classifier if classifier is not None else NeverBackchannel(),
            bc_vocab_per_class=bc_vocab_per_class or {},
            suppress_consecutive_same_speaker_bcs=not args.no_suppress_consecutive_bcs,
            seed=args.seed,
            clean_cache=args.clean_cache,
            **classifier_kwargs,
        )
        _write_metadata(cache_dir, metadata)
        print(f"Cache: {cache_dir}")
        print(f"Manifest: {manifest_path}")
        if args.run_naive_only:
            _run_naive(synth, dialogue["turns"], os.path.join(cache_dir, "naive"), args.clean_cache)
        return

    # ----- text input -------------------------------------------------------- #
    for idx, text in enumerate(texts):
        cache_dir = os.path.join(cache_root, f"example_{idx:04d}")
        text = text.strip()
        if classifier is not None:
            manifest_path, metadata = synth.generate_dialogue_with_classifier(
                transcript=text,
                cache_dir=cache_dir,
                classifier=classifier,
                bc_vocab_per_class=bc_vocab_per_class,
                seed=args.seed,
                clean_cache=args.clean_cache,
                **classifier_kwargs,
            )
        else:
            bc_subset = args.bc_vocab_subset
            if per_dialogue_vocabs is not None and per_dialogue_vocabs[idx] is not None:
                bc_subset = per_dialogue_vocabs[idx]
            manifest_path, metadata = synth.generate_dialogue(
                transcript=text,
                cache_dir=cache_dir,
                bc_probability=args.bc_probability,
                bc_vocab_subset=bc_subset,
                seed=args.seed,
                clean_cache=args.clean_cache,
            )
        meta_path = _write_metadata(cache_dir, metadata)
        print(f"\nExample {idx}:")
        print(f"  Cache:    {cache_dir}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Metadata: {meta_path}")
        if classifier is not None:
            print(f"  BC decisions: {os.path.join(cache_dir, 'bc_decisions.json')}")


if __name__ == "__main__":
    main()
