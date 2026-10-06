"""Baseline classifiers: late fusion by concatenation and low-rank multimodal fusion."""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2Model, HubertModel, RobertaModel

from .joint import audio_attention_mask, frame_mask


def last_non_pad_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last attended position of each right-padded row, ``[B, D]``."""
    last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]


def masked_mean_pool(hidden: torch.Tensor, frame_lens: torch.Tensor) -> torch.Tensor:
    """Mean over the valid frames of each row, ``[B, D]``."""
    mask = frame_mask(hidden.size(1), frame_lens, hidden.device).float().unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class _PaperFusionBase(nn.Module):
    """Late fusion of Wang et al., "Turn-Taking and Backchannel Prediction with
    Acoustic and Large Language Model Fusion" (ICASSP 2024).

    * Audio: HuBERT (frozen) last hidden state → mean over valid frames →
      ``Linear(→ audio_proj_dim)``.
    * Text: last non-pad token of the text encoder.
    * Fusion: ``Linear(concat([audio, text]) → num_classes)``.

    ``fusion_option`` records the training recipe of the paper: ``opt1`` trains
    the text encoder jointly with the fusion layer; ``opt2`` loads a text
    encoder fine-tuned on its own (a ``text_only`` run) and freezes it
    (``freeze_text=True``). ``text_only`` / ``audio_only`` replace the other
    modality's embedding with zeros.

    Subclasses name the text encoder attribute, which fixes the state-dict prefix.
    """

    text_encoder_attr: str

    def _build(
        self,
        text_encoder: nn.Module,
        hubert_name: str,
        num_classes: int,
        audio_proj_dim: int,
        tokenizer_len: Optional[int],
        fusion_option: str,
        freeze_text: bool,
        text_only: bool,
        audio_only: bool,
    ):
        assert fusion_option in ("opt1", "opt2"), \
            f"fusion_option must be 'opt1' or 'opt2', got {fusion_option}"
        assert not (text_only and audio_only), "Cannot set both text_only and audio_only"

        self.fusion_option = fusion_option
        self.num_classes = num_classes
        self.audio_proj_dim = audio_proj_dim
        self.text_only = text_only
        self.audio_only = audio_only

        setattr(self, self.text_encoder_attr, text_encoder)
        if tokenizer_len is not None:
            text_encoder.resize_token_embeddings(tokenizer_len)
        if freeze_text:
            text_encoder.eval()
            for p in text_encoder.parameters():
                p.requires_grad = False
        self._freeze_text = freeze_text

        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False

        self.audio_proj = nn.Linear(self.hubert.config.hidden_size, audio_proj_dim)
        self.fusion_classifier = nn.Linear(audio_proj_dim + text_encoder.config.hidden_size, num_classes)

    @property
    def text_encoder(self) -> nn.Module:
        return getattr(self, self.text_encoder_attr)

    def train(self, mode: bool = True):
        """Switch modes while keeping frozen modules in eval mode."""
        super().train(mode)
        self.hubert.eval()
        if self._freeze_text:
            self.text_encoder.eval()
        return self

    @torch.no_grad()
    def _hubert_forward_nograd(self, audio, audio_attn_mask):
        out = self.hubert(
            input_values=audio,
            attention_mask=audio_attn_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def _hubert_frame_lengths(self, audio_lens: torch.Tensor) -> torch.Tensor:
        return self.hubert._get_feat_extract_output_lengths(audio_lens)

    def _encode_audio(self, audio, audio_lens):
        hidden = self._hubert_forward_nograd(audio, audio_attention_mask(audio, audio_lens))
        return self.audio_proj(masked_mean_pool(hidden, self._hubert_frame_lengths(audio_lens)))

    def _encode_text(self, input_ids, attention_mask):
        with torch.set_grad_enabled(not self._freeze_text and torch.is_grad_enabled()):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return last_non_pad_token(out.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask, audio, audio_lens):
        batch_size = input_ids.size(0)
        device = input_ids.device

        if self.audio_only:
            text_emb = torch.zeros(
                batch_size, self.text_encoder.config.hidden_size, device=device, dtype=audio.dtype
            )
        else:
            text_emb = self._encode_text(input_ids, attention_mask)

        if self.text_only:
            audio_emb = torch.zeros(batch_size, self.audio_proj_dim, device=device, dtype=text_emb.dtype)
        else:
            audio_emb = self._encode_audio(audio, audio_lens)

        return self.fusion_classifier(torch.cat([audio_emb, text_emb], dim=-1))

    @torch.no_grad()
    def load_text_encoder_from_checkpoint(self, ckpt_path: str) -> Tuple[int, int]:
        """Copy the text-encoder tensors of a saved state dict into this model
        (stage 2 of ``opt2``). Returns ``(copied, skipped)`` tensor counts."""
        state = torch.load(ckpt_path, map_location="cpu")
        own = self.state_dict()
        prefix = self.text_encoder_attr + "."
        copied, skipped = 0, 0
        for k, v in state.items():
            if k.startswith(prefix) and k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        return copied, skipped


class PaperFusionClassifier(_PaperFusionBase):
    """Late-fusion baseline with a RoBERTa text encoder (state-dict prefix ``roberta.``)."""

    text_encoder_attr = "roberta"

    def __init__(
        self,
        roberta_name: str = "roberta-base",
        hubert_name: str = "facebook/hubert-base-ls960",
        num_classes: int = 3,
        audio_proj_dim: int = 256,
        tokenizer_len: Optional[int] = None,
        fusion_option: str = "opt2",
        freeze_text: bool = False,
        text_only: bool = False,
        audio_only: bool = False,
    ):
        super().__init__()
        self._build(
            RobertaModel.from_pretrained(roberta_name), hubert_name, num_classes, audio_proj_dim,
            tokenizer_len, fusion_option, freeze_text, text_only, audio_only,
        )


class GPT2PaperFusionClassifier(_PaperFusionBase):
    """Late-fusion baseline with a GPT-2 text encoder, as in the paper
    (state-dict prefix ``gpt2.``). Expects right-padded input with a real pad token."""

    text_encoder_attr = "gpt2"

    def __init__(
        self,
        gpt2_name: str = "gpt2",
        hubert_name: str = "facebook/hubert-base-ls960",
        num_classes: int = 3,
        audio_proj_dim: int = 256,
        tokenizer_len: Optional[int] = None,
        fusion_option: str = "opt2",
        freeze_text: bool = False,
        text_only: bool = False,
        audio_only: bool = False,
    ):
        super().__init__()
        self._build(
            GPT2Model.from_pretrained(gpt2_name), hubert_name, num_classes, audio_proj_dim,
            tokenizer_len, fusion_option, freeze_text, text_only, audio_only,
        )


class MMF2FLMF(nn.Module):
    """Low-rank multimodal fusion (Liu et al., 2018) over text and audio.

    ``forward(text_x, audio_x)`` takes ``[B, hidden_dim]`` embeddings, either of
    which may be ``None`` to drop that modality, and returns logits
    ``[B, num_classes]``. Each embedding is prefixed with a constant 1 so the
    product also carries the unimodal terms.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2, rank: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rank = rank

        self.factor_text = nn.Parameter(torch.empty(rank, hidden_dim + 1, num_classes))
        self.factor_audio = nn.Parameter(torch.empty(rank, hidden_dim + 1, num_classes))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, num_classes))

        nn.init.xavier_normal_(self.factor_text)
        nn.init.xavier_normal_(self.factor_audio)
        nn.init.xavier_normal_(self.fusion_weights)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([ones, x], dim=1)

    def forward(self, text_x, audio_x):
        assert (text_x is not None) or (audio_x is not None), "MMF2FLMF requires at least one modality"

        # [B, D+1] @ [R, D+1, C] -> [R, B, C]
        f_text = torch.matmul(self._augment(text_x), self.factor_text) if text_x is not None else None
        f_audio = torch.matmul(self._augment(audio_x), self.factor_audio) if audio_x is not None else None

        if f_text is None:
            fused = f_audio
        elif f_audio is None:
            fused = f_text
        else:
            fused = f_text * f_audio

        # [1, R] @ [B, R, C] -> [B, 1, C]
        return torch.matmul(self.fusion_weights, fused.permute(1, 0, 2)).squeeze(1) + self.fusion_bias


class MMF2FModel(nn.Module):
    """GPT-2 + HuBERT with low-rank multimodal fusion.

    Both encoders are trainable. Text is the last non-pad GPT-2 token, audio the
    mean over valid HuBERT frames; each is projected to ``hidden_dim`` and fused
    by :class:`MMF2FLMF`. During training, with probability ``drop_rate`` one
    of the two modalities (chosen 50/50) is dropped for the batch.
    """

    def __init__(
        self,
        gpt2_name: str = "openai-community/gpt2",
        hubert_name: str = "facebook/hubert-base-ls960",
        num_classes: int = 2,
        hidden_dim: int = 256,
        lmf_rank: int = 16,
        tokenizer_len: Optional[int] = None,
        text_only: bool = False,
        audio_only: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        assert not (text_only and audio_only), "Cannot set both text_only and audio_only"
        self.text_only = text_only
        self.audio_only = audio_only
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        if tokenizer_len is not None:
            self.gpt2.resize_token_embeddings(tokenizer_len)
        self.text_proj = nn.Linear(self.gpt2.config.n_embd, hidden_dim)

        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_proj = nn.Linear(self.hubert.config.hidden_size, hidden_dim)

        self.fusion = MMF2FLMF(hidden_dim=hidden_dim, num_classes=num_classes, rank=lmf_rank)

    def _encode_text(self, input_ids, attention_mask):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(last_non_pad_token(out.last_hidden_state, attention_mask))

    def _encode_audio(self, audio, audio_lens):
        out = self.hubert(
            input_values=audio,
            attention_mask=audio_attention_mask(audio, audio_lens),
            return_dict=True,
        )
        frame_lens = self.hubert._get_feat_extract_output_lengths(audio_lens)
        return self.audio_proj(masked_mean_pool(out.last_hidden_state, frame_lens))

    def forward(self, input_ids, attention_mask, audio, audio_lens):
        use_text = not self.audio_only
        use_audio = not self.text_only

        if self.training and self.drop_rate > 0 and use_text and use_audio:
            if random.random() < self.drop_rate:
                if random.random() < 0.5:
                    use_text = False
                else:
                    use_audio = False

        text_emb = self._encode_text(input_ids, attention_mask) if use_text else None
        audio_emb = self._encode_audio(audio, audio_lens) if use_audio else None
        return self.fusion(text_emb, audio_emb)

    @torch.no_grad()
    def load_unimodal_checkpoints(
        self,
        text_ckpt: Optional[str] = None,
        audio_ckpt: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Warm-start the encoders from unimodal state dicts.

        The text checkpoint is expected to hold GPT-2 under ``transformer.*``
        and its projection under ``proj.*``; the audio checkpoint HuBERT under
        ``hubert.*`` and its projection under ``proj.*``. Only tensors with a
        matching name and shape are copied. Returns ``(text, audio)`` counts.
        """
        n_text = n_audio = 0
        if text_ckpt is not None:
            own = self.state_dict()
            for k, v in torch.load(text_ckpt, map_location="cpu").items():
                if k.startswith("transformer."):
                    new_k = "gpt2." + k[len("transformer."):]
                elif k in ("proj.weight", "proj.bias"):
                    new_k = "text_" + k
                else:
                    continue
                if new_k in own and own[new_k].shape == v.shape:
                    own[new_k].copy_(v)
                    n_text += 1
            self.load_state_dict(own)
        if audio_ckpt is not None:
            own = self.state_dict()
            for k, v in torch.load(audio_ckpt, map_location="cpu").items():
                if k.startswith("hubert."):
                    new_k = k
                elif k in ("proj.weight", "proj.bias"):
                    new_k = "audio_" + k
                else:
                    continue
                if new_k in own and own[new_k].shape == v.shape:
                    own[new_k].copy_(v)
                    n_audio += 1
            self.load_state_dict(own)
        return n_text, n_audio
