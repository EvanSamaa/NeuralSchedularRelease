"""Backchannel classifiers as used by the inference pipeline.

Both wrappers decide, for one turn of a scripted dialogue, whether the listener
should backchannel and with which class::

    clf = BackchannelClassifier("runs/joint_merged/best_model", device="cuda")
    text = clf.build_input_text(segments, target_turn_idx=3)
    label, confidence, probs = clf.predict(text, audio="turn3.wav")

``segments`` is the dialogue as ``[{"speaker": 1 | 2, "text": str, "bc_class":
<tag or None>}, ...]``. ``label`` is ``NoBackchannel``, ``Acknowledgment`` or
``Reactive``. To feed a decision back into the context of later turns, set the
turn's ``bc_class`` to ``MERGED_TO_CANONICAL_BC_TAG[label]``.

Decision rule: with ``p_bc = P(Acknowledgment) + P(Reactive)``, predict the
more probable of the two backchannel classes when ``p_bc >=
confidence_threshold``, else ``NoBackchannel``.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .constants import (
    BC_SLOT_TOKEN,
    HUBERT_SR,
    LINE_SEP,
    MAX_AUDIO_SECONDS_DEFAULT,
    MERGED_LABEL_NAMES,
    MERGED_TO_CANONICAL_BC_TAG,
)
from .data.audio import load_audio_span, waveform_to_target_sr
from .models.factory import (
    ARCH_CLASS_NAMES,
    architecture_of,
    build_model_from_config,
    load_tokenizer,
    load_weights,
    tokenizer_dir_candidates,
)

LABEL_NAMES = {i: name for i, name in enumerate(MERGED_LABEL_NAMES)}

__all__ = [
    "BC_SLOT_TOKEN",
    "BackchannelClassifier",
    "LABEL_NAMES",
    "MERGED_TO_CANONICAL_BC_TAG",
    "TextOnlyBackchannelClassifier",
    "build_input_text",
    "decide",
    "load_audio_span",
    "waveform_to_target_sr",
]

AudioSource = Union[str, np.ndarray, torch.Tensor, None]


def build_input_text(
    segments: List[Dict],
    target_turn_idx: int,
    context_window_turns: int = 5,
    context_window_future_turns: int = 5,
    include_bc_context: bool = True,
) -> str:
    """Render the classifier input for one target turn.

    One ``A: ...`` / ``B: ...`` line per turn, from ``context_window_turns``
    before the target to ``context_window_future_turns`` after it, joined by
    ``" </s> "``. The target's speaker is ``A``. The target line ends with
    ``<bc_slot>``; earlier lines whose turn received a backchannel end with
    ``[BC:<bc_class>]`` unless ``include_bc_context`` is off. Later lines are
    never tagged.
    """
    target_speaker = segments[target_turn_idx]["speaker"]
    speaker_tags = {target_speaker: "A", (2 if target_speaker == 1 else 1): "B"}

    start_idx = max(0, target_turn_idx - context_window_turns)
    end_idx = min(len(segments) - 1, target_turn_idx + context_window_future_turns)

    lines = []
    for i in range(start_idx, end_idx + 1):
        seg = segments[i]
        text = " ".join(seg["text"].replace("\n", " ").split())
        line = f"{speaker_tags[seg['speaker']]}: {text}"
        if i < target_turn_idx:
            if include_bc_context and seg.get("bc_class") and seg["bc_class"] != "NoBackchannel":
                line += f" [BC:{seg['bc_class']}]"
        elif i == target_turn_idx:
            line += f" {BC_SLOT_TOKEN}"
        lines.append(line)
    return LINE_SEP.join(lines)


def decide(all_probs: Dict[str, float], confidence_threshold: float) -> Tuple[str, float]:
    """Apply the decision rule to class probabilities; returns ``(label, confidence)``.

    The confidence is the probability of the returned label. Classes missing
    from ``all_probs`` count as probability 0.
    """
    p_ack = all_probs.get("Acknowledgment", 0.0)
    p_reactive = all_probs.get("Reactive", 0.0)
    bc_prob = p_ack + p_reactive
    if bc_prob >= confidence_threshold:
        pred_label = "Acknowledgment" if p_ack >= p_reactive else "Reactive"
        return pred_label, float(all_probs[pred_label])
    return "NoBackchannel", float(all_probs.get("NoBackchannel", 1.0 - bc_prob))


def min_audio_samples(model: torch.nn.Module) -> int:
    """Fewest input samples the model's HuBERT feature extractor can process (0 without HuBERT)."""
    hubert = getattr(model, "hubert", None)
    if hubert is None:
        return 0
    length = 1
    for kernel, stride in zip(reversed(hubert.config.conv_kernel), reversed(hubert.config.conv_stride)):
        length = (length - 1) * stride + kernel
    return length


def _find_tokenizer_dir(model_dir: str) -> str:
    candidates = tokenizer_dir_candidates(model_dir)
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"Could not find tokenizer directory. Searched: {candidates}")


class BackchannelClassifier:
    """Joint text + audio classifier loaded from a training checkpoint.

    ``model_dir`` is a checkpoint directory with ``model.pt`` and
    ``config.json`` (or the path of the ``.pt`` file). The architecture is read
    from the config. The tokenizer is taken from ``tokenizer_dir``, else
    ``<model_dir>/tokenizer``, else ``<model_dir>/../tokenizer``.

    The pipeline expects a model trained with ``--phase merged``; with another
    phase the label names of the config are used and a warning is printed.
    """

    LABEL_NAMES = LABEL_NAMES
    NUM_CLASSES = 3

    def __init__(self, model_dir, tokenizer_dir=None, device="cuda",
                 confidence_threshold=0.0,
                 max_audio_seconds: float = MAX_AUDIO_SECONDS_DEFAULT,
                 max_text_length: int = 512):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.label_names = self.LABEL_NAMES
        self.num_classes = self.NUM_CLASSES
        self.max_audio_seconds = max_audio_seconds
        self.max_text_length = max_text_length

        if os.path.isdir(model_dir):
            sd_path = os.path.join(model_dir, "model.pt")
            cfg_path = os.path.join(model_dir, "config.json")
        else:
            sd_path = model_dir
            cfg_path = os.path.join(os.path.dirname(model_dir), "config.json")
        if not os.path.isfile(sd_path):
            raise FileNotFoundError(f"model.pt not found at {sd_path}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"config.json not found at {cfg_path}. The joint classifier "
                "needs the saved architecture metadata to be reconstructed."
            )
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.cfg = cfg

        phase = cfg.get("phase", "merged")
        if phase != "merged":
            print(f"  WARNING: classifier phase is '{phase}', expected 'merged'. "
                  f"NoBackchannel/Acknowledgment/Reactive label mapping may be wrong.")
        if "label_names" in cfg:
            self.label_names = {i: name for i, name in enumerate(cfg["label_names"])}
            self.num_classes = cfg.get("num_classes", len(cfg["label_names"]))

        if tokenizer_dir is None:
            tokenizer_dir = _find_tokenizer_dir(model_dir)

        self.architecture = architecture_of(cfg)
        self.text_only = bool(cfg.get("text_only", False))
        self.audio_only = bool(cfg.get("audio_only", False))
        print(f"Loading joint BC classifier from {model_dir}")
        print(f"  Tokenizer: {tokenizer_dir}")
        print(f"  Architecture: {ARCH_CLASS_NAMES[self.architecture]}")
        print(f"  text_only={self.text_only}  audio_only={self.audio_only}")

        self.tokenizer = load_tokenizer(self.architecture, tokenizer_dir)
        self.model = build_model_from_config(
            cfg, tokenizer_len=len(self.tokenizer), defaults={"num_classes": self.num_classes}
        )
        load_weights(self.model, sd_path, self.device, warn=lambda msg: print(f"  WARNING: {msg}"))
        self.model.to(self.device)
        self.model.eval()
        self._min_audio_samples = min_audio_samples(self.model)
        print(f"  Loaded weights from {sd_path}")

    def build_input_text(self, segments, target_turn_idx,
                         context_window_turns=5, context_window_future_turns=5,
                         include_bc_context=True):
        """See :func:`build_input_text`."""
        return build_input_text(
            segments, target_turn_idx, context_window_turns,
            context_window_future_turns, include_bc_context,
        )

    def load_audio_for_target(
        self,
        audio_source: AudioSource,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Mono float32 audio at 16 kHz for the target turn's speaker.

        ``audio_source`` is a wav path (optionally sliced to ``[start_sec,
        end_sec]``), an in-memory waveform (``sample_rate`` required), or
        ``None`` for one sample of silence.
        """
        if audio_source is None:
            return np.zeros(1, dtype=np.float32)
        if isinstance(audio_source, str):
            return load_audio_span(
                audio_source, start_sec=start_sec, end_sec=end_sec,
                target_sr=HUBERT_SR, max_seconds=self.max_audio_seconds,
            )
        if isinstance(audio_source, np.ndarray):
            wav = torch.from_numpy(audio_source.astype(np.float32))
        elif isinstance(audio_source, torch.Tensor):
            wav = audio_source
        else:
            raise TypeError(
                f"audio_source must be str | np.ndarray | torch.Tensor | None, got {type(audio_source)}"
            )
        if sample_rate is None:
            raise ValueError("When passing an in-memory waveform, sample_rate must be provided.")
        return waveform_to_target_sr(
            wav, src_sr=int(sample_rate), target_sr=HUBERT_SR, max_seconds=self.max_audio_seconds
        )

    @torch.no_grad()
    def predict(
        self,
        input_text: str,
        audio: AudioSource = None,
        audio_start_sec: Optional[float] = None,
        audio_end_sec: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
    ):
        """Classify one turn.

        ``input_text`` comes from :meth:`build_input_text`; ``audio`` is the
        target turn's audio as accepted by :meth:`load_audio_for_target`.
        Returns ``(pred_label, confidence, all_probs)`` with ``all_probs``
        keyed by label name.

        Audio shorter than the HuBERT receptive field (25 ms for HuBERT-base;
        this includes ``None``) is zero-padded to that length, i.e. the audio
        branch sees one frame of near-silence.
        """
        encoding = self.tokenizer(
            input_text, max_length=self.max_text_length,
            padding=False, truncation=True, return_tensors="pt",
        )
        audio_np = self.load_audio_for_target(
            audio, start_sec=audio_start_sec, end_sec=audio_end_sec, sample_rate=audio_sample_rate
        )
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        if audio_tensor.size(1) < self._min_audio_samples:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self._min_audio_samples - audio_tensor.size(1))
            )
        logits = self.model(
            input_ids=encoding["input_ids"].to(self.device),
            attention_mask=encoding["attention_mask"].to(self.device),
            audio=audio_tensor.to(self.device),
            audio_lens=torch.tensor([audio_tensor.size(1)], dtype=torch.long, device=self.device),
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        all_probs = {self.label_names[i]: float(probs[i]) for i in range(self.num_classes)}
        pred_label, confidence = decide(all_probs, self.confidence_threshold)
        return pred_label, confidence, all_probs


class TextOnlyBackchannelClassifier:
    """Text-only RoBERTa classifier (three merged classes).

    ``model_dir`` is a Hugging Face ``RobertaForSequenceClassification``
    directory written by :mod:`neural_scheduler.backchannel.train_text_only`.
    The tokenizer is taken from ``tokenizer_dir``, else
    ``<model_dir>/tokenizer``, else ``<model_dir>/../tokenizer``.
    """

    LABEL_NAMES = LABEL_NAMES
    NUM_CLASSES = 3

    def __init__(
        self,
        model_dir: str,
        tokenizer_dir: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.0,
    ):
        from transformers import RobertaForSequenceClassification, RobertaTokenizer

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.label_names = self.LABEL_NAMES
        self.num_classes = self.NUM_CLASSES

        if tokenizer_dir is None:
            tokenizer_dir = _find_tokenizer_dir(model_dir)

        print(f"Loading BC classifier from {model_dir}")
        print(f"  Tokenizer: {tokenizer_dir}")
        print("  3-class merged: NoBackchannel / Acknowledgment / Reactive")

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir)
        self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def build_input_text(
        self,
        segments: List[Dict],
        target_turn_idx: int,
        context_window_turns: int = 5,
        context_window_future_turns: int = 5,
        include_bc_context: bool = True,
    ) -> str:
        """See :func:`build_input_text`."""
        return build_input_text(
            segments, target_turn_idx, context_window_turns,
            context_window_future_turns, include_bc_context,
        )

    @torch.no_grad()
    def predict(self, input_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify one turn; returns ``(pred_label, confidence, all_probs)``."""
        encoding = self.tokenizer(
            input_text, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        outputs = self.model(
            input_ids=encoding["input_ids"].to(self.device),
            attention_mask=encoding["attention_mask"].to(self.device),
        )
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        all_probs = {self.label_names[i]: float(probs[i]) for i in range(self.num_classes)}
        pred_label, confidence = decide(all_probs, self.confidence_threshold)
        return pred_label, confidence, all_probs
