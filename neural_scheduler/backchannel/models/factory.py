"""Checkpoint layout, model construction from ``config.json``, tokenizers.

A checkpoint directory holds ``model.pt`` (state dict) and ``config.json``.
The tokenizer is saved in ``<run_dir>/tokenizer`` next to ``best_model`` and
``last_model``, and also inside each checkpoint directory; either location is
accepted when loading.
"""

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, GPT2Tokenizer, RobertaTokenizer

from ..constants import BC_SLOT_TOKEN, BC_TAG_TOKENS, GPT2_PAD_TOKEN
from .baselines import GPT2PaperFusionClassifier, MMF2FModel, PaperFusionClassifier
from .joint import JointBCClassifier

ARCH_JOINT = "joint"
ARCH_PAPER_FUSION = "paper_fusion"
ARCH_PAPER_FUSION_GPT2 = "paper_fusion_gpt2"
ARCH_MMF2F = "mmf2f"
ARCHITECTURES = (ARCH_JOINT, ARCH_PAPER_FUSION, ARCH_PAPER_FUSION_GPT2, ARCH_MMF2F)

ARCH_CLASS_NAMES = {
    ARCH_JOINT: "JointBCClassifier",
    ARCH_PAPER_FUSION: "PaperFusionClassifier",
    ARCH_PAPER_FUSION_GPT2: "GPT2PaperFusionClassifier",
    ARCH_MMF2F: "MMF2FModel",
}

# Values assumed for keys a config.json does not carry.
CONFIG_DEFAULTS = {
    "roberta_name": "roberta-base",
    "hubert_name": "facebook/hubert-base-ls960",
    "gpt2_name": "gpt2",
    "fusion_layers": 2,
    "fusion_heads": 8,
    "fusion_ffn": 3072,
    "fusion_dropout": 0.1,
    "head_dropout": 0.1,
    "audio_stride": 4,
    "text_only": False,
    "audio_only": False,
    "paper_fusion_option": "opt2",
    "paper_audio_proj_dim": 256,
    "mmf2f_hidden_dim": 256,
    "mmf2f_rank": 16,
}


def architecture_of(cfg: Dict) -> str:
    """Architecture a config describes.

    Configs written by this package carry an ``architecture`` key. Older ones
    are recognised by ``use_mmf2f`` / ``use_paper_fusion``; a config with a
    ``gpt2_name`` and a ``paper_fusion_option`` but no ``roberta_name`` is the
    GPT-2 late-fusion baseline; anything else is the joint classifier.
    """
    if cfg.get("architecture") in ARCHITECTURES:
        return cfg["architecture"]
    if cfg.get("use_mmf2f"):
        return ARCH_MMF2F
    if cfg.get("use_paper_fusion"):
        return ARCH_PAPER_FUSION
    if "gpt2_name" in cfg and "paper_fusion_option" in cfg and "roberta_name" not in cfg:
        return ARCH_PAPER_FUSION_GPT2
    return ARCH_JOINT


def build_model_from_config(
    cfg: Dict,
    tokenizer_len: Optional[int] = None,
    defaults: Optional[Dict] = None,
    text_only: Optional[bool] = None,
    audio_only: Optional[bool] = None,
) -> torch.nn.Module:
    """Instantiate the architecture described by ``cfg`` for evaluation.

    Missing keys fall back to ``defaults`` and then :data:`CONFIG_DEFAULTS`.
    ``num_classes`` must be in ``cfg`` or ``defaults``. ``tokenizer_len`` is
    used when the config has none. The late-fusion text encoder is frozen iff
    the recipe is ``opt2``; modality dropout is off. ``text_only`` /
    ``audio_only`` override the config when given.
    """
    full = {**CONFIG_DEFAULTS, **(defaults or {}), **cfg}
    arch = architecture_of(cfg)
    tok_len = cfg.get("tokenizer_len", tokenizer_len)
    text_only = bool(full["text_only"]) if text_only is None else text_only
    audio_only = bool(full["audio_only"]) if audio_only is None else audio_only
    common = dict(
        hubert_name=full["hubert_name"],
        num_classes=full["num_classes"],
        tokenizer_len=tok_len,
        text_only=text_only,
        audio_only=audio_only,
    )

    if arch == ARCH_MMF2F:
        return MMF2FModel(
            gpt2_name=full["gpt2_name"],
            hidden_dim=full["mmf2f_hidden_dim"],
            lmf_rank=full["mmf2f_rank"],
            drop_rate=0.0,
            **common,
        )
    if arch in (ARCH_PAPER_FUSION, ARCH_PAPER_FUSION_GPT2):
        fusion_kwargs = dict(
            audio_proj_dim=full["paper_audio_proj_dim"],
            fusion_option=full["paper_fusion_option"],
            freeze_text=(full["paper_fusion_option"] == "opt2"),
            **common,
        )
        if arch == ARCH_PAPER_FUSION_GPT2:
            return GPT2PaperFusionClassifier(gpt2_name=full["gpt2_name"], **fusion_kwargs)
        return PaperFusionClassifier(roberta_name=full["roberta_name"], **fusion_kwargs)
    return JointBCClassifier(
        roberta_name=full["roberta_name"],
        fusion_layers=full["fusion_layers"],
        fusion_heads=full["fusion_heads"],
        fusion_ffn=full["fusion_ffn"],
        fusion_dropout=full["fusion_dropout"],
        head_dropout=full["head_dropout"],
        audio_stride=full["audio_stride"],
        **common,
    )


def resolve_checkpoint(ckpt_path: str) -> Tuple[str, Optional[Dict]]:
    """Return ``(state_dict_path, config_or_None)`` for a checkpoint directory
    or a path to a ``.pt`` file (whose ``config.json`` is looked up beside it)."""
    if os.path.isdir(ckpt_path):
        sd_path = os.path.join(ckpt_path, "model.pt")
        cfg_path = os.path.join(ckpt_path, "config.json")
    else:
        sd_path = ckpt_path
        cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")

    if not os.path.isfile(sd_path):
        raise FileNotFoundError(f"Checkpoint file not found: {sd_path}")

    cfg = None
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    return sd_path, cfg


def load_weights(model: torch.nn.Module, sd_path: str, device, strict: bool = False,
                 warn: Callable[[str], None] = print, max_listed: int = 8) -> None:
    """Load a state dict. With ``strict=False`` (the default) missing and
    unexpected keys are reported through ``warn`` instead of raising."""
    state = torch.load(sd_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    for kind, keys in (("Missing", missing), ("Unexpected", unexpected)):
        if keys:
            warn(f"{kind} keys when loading state dict: {keys[:max_listed]}"
                 f"{' ...' if len(keys) > max_listed else ''}")


def save_checkpoint(model: torch.nn.Module, out_dir: str, cfg: Dict, tokenizer=None) -> None:
    """Write ``model.pt`` and ``config.json`` (and ``tokenizer/`` if given) to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))


def tokenizer_dir_candidates(checkpoint: str, extra: Iterable[str] = ()) -> List[str]:
    """Places a checkpoint's tokenizer may be, in lookup order: inside the
    checkpoint directory, beside it, then ``extra``."""
    return [
        os.path.join(checkpoint, "tokenizer"),
        os.path.join(os.path.dirname(checkpoint), "tokenizer"),
        *extra,
    ]


def find_tokenizer_dir(checkpoint: str, extra: Iterable[str] = ()) -> Optional[str]:
    for cand in tokenizer_dir_candidates(checkpoint, extra):
        if os.path.isdir(cand):
            return cand
    return None


def add_bc_tokens(tokenizer) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": [BC_SLOT_TOKEN] + BC_TAG_TOKENS})


def build_gpt2_paper_tokenizer(gpt2_name: str) -> GPT2Tokenizer:
    """GPT-2 tokenizer for the late-fusion baseline: a real pad token (not the
    EOS token, so the last non-pad position is unambiguous), the backchannel
    tokens, and left truncation so the most recent context is kept."""
    tok = GPT2Tokenizer.from_pretrained(gpt2_name)
    tok.add_special_tokens({
        "pad_token": GPT2_PAD_TOKEN,
        "additional_special_tokens": [BC_SLOT_TOKEN] + BC_TAG_TOKENS,
    })
    tok.truncation_side = "left"
    return tok


def build_tokenizer(arch: str, name: str):
    """Fresh tokenizer for an architecture, with the backchannel tokens added.
    The LMF baseline reuses EOS as its pad token."""
    if arch == ARCH_PAPER_FUSION_GPT2:
        return build_gpt2_paper_tokenizer(name)
    if arch == ARCH_MMF2F:
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = RobertaTokenizer.from_pretrained(name)
    add_bc_tokens(tokenizer)
    return tokenizer


def load_tokenizer(arch: str, tokenizer_dir: str):
    """Load a tokenizer saved during training."""
    if arch == ARCH_PAPER_FUSION_GPT2:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
        tokenizer.truncation_side = "left"
        return tokenizer
    if arch == ARCH_MMF2F:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    return RobertaTokenizer.from_pretrained(tokenizer_dir)
