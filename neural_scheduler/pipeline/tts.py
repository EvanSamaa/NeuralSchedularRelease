"""Text-to-speech backends for step 1.

Chatterbox is imported only when a backend is constructed, so the rest of the
pipeline package can be imported in an environment without it.
"""

import importlib.util
import os
import sys
import types

TTS_BACKENDS = ("chatterbox", "chatterbox_turbo")

# Some Chatterbox environments ship a torchvision build that does not match
# torch; importing it fails while registering its operators and takes
# ``transformers`` down with it. Setting this variable to 1 replaces the
# offending torchvision submodule with an empty one before anything imports it.
STUB_TORCHVISION_ENV = "NEURAL_SCHEDULER_STUB_TORCHVISION_META"


def stub_torchvision_meta_registrations_if_requested() -> bool:
    """Install an empty ``torchvision._meta_registrations`` when the environment asks for it.

    Must run before ``torch`` / ``transformers`` import torchvision. Returns
    True when the stub was installed.
    """
    if os.environ.get(STUB_TORCHVISION_ENV, "") not in ("1", "true", "True"):
        return False
    if importlib.util.find_spec("torchvision") is None:
        return False
    sys.modules["torchvision._meta_registrations"] = types.ModuleType("torchvision._meta_registrations")
    return True


class ChatterboxBackend:
    """Chatterbox TTS with voice cloning from a reference recording.

    ``generate`` returns a ``[1, T]`` float tensor on the CPU at ``self.sr`` Hz.
    """

    name = "chatterbox"

    def __init__(self, device: str = "cuda", exaggeration: float = 0.5, cfg_weight: float = 0.5):
        try:
            from chatterbox.tts import ChatterboxTTS
        except RuntimeError as err:
            raise RuntimeError(
                f"Importing chatterbox failed ({err}). If the message mentions torchvision, "
                f"set {STUB_TORCHVISION_ENV}=1 and retry."
            ) from err
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sr: int = self.model.sr
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

    def generate(self, text: str, audio_prompt_path: str):
        wav = self.model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
        )
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav.cpu()


class ChatterboxTurboBackend:
    """Chatterbox-Turbo: faster, and without the exaggeration / cfg-weight controls."""

    name = "chatterbox_turbo"

    def __init__(self, device: str = "cuda"):
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except RuntimeError as err:
            raise RuntimeError(
                f"Importing chatterbox failed ({err}). If the message mentions torchvision, "
                f"set {STUB_TORCHVISION_ENV}=1 and retry."
            ) from err
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)
        self.sr: int = self.model.sr

    def generate(self, text: str, audio_prompt_path: str):
        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav.cpu()


def load_tts(backend: str = "chatterbox", device: str = "cuda", exaggeration: float = 0.5, cfg_weight: float = 0.5):
    """Build a TTS backend by name (one of ``TTS_BACKENDS``)."""
    if backend == "chatterbox":
        return ChatterboxBackend(device=device, exaggeration=exaggeration, cfg_weight=cfg_weight)
    if backend == "chatterbox_turbo":
        return ChatterboxTurboBackend(device=device)
    raise ValueError(f"Unknown TTS backend '{backend}'; expected one of {TTS_BACKENDS}")
