"""Joint text + audio backchannel classifier with transformer fusion."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, RobertaModel

MODALITY_CLS, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2


def audio_attention_mask(audio: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` mask over raw samples, 1 where the sample index is below the length."""
    positions = torch.arange(audio.size(1), device=audio.device).unsqueeze(0)
    return (positions < audio_lens.unsqueeze(1)).long()


def frame_mask(num_frames: int, frame_lens: torch.Tensor, device) -> torch.Tensor:
    """``[B, num_frames]`` boolean mask of valid frames."""
    return torch.arange(num_frames, device=device).unsqueeze(0) < frame_lens.unsqueeze(1)


class WeightedLayerSum(nn.Module):
    """Softmax-weighted sum over a stack of layer outputs ``[num_layers, B, T, D]``."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (w * layers).sum(dim=0)


class JointBCClassifier(nn.Module):
    """RoBERTa + frozen HuBERT, fused by a small transformer.

    * Text: RoBERTa (trainable) token states ``[B, T_text, D]``.
    * Audio: HuBERT (frozen, always in eval mode) → learned weighted sum of all
      hidden layers → strided convolution (50 Hz → 50 / ``audio_stride`` Hz)
      → GELU → LayerNorm, ``[B, T_audio, D]``.
    * Fusion: ``[CLS ; text ; audio]`` plus a modality embedding → pre-norm
      transformer encoder → the CLS position.
    * Head: LayerNorm → Dropout → Linear.

    ``text_only`` / ``audio_only`` leave the other modality's tokens out of the
    fusion sequence.

    ``forward(input_ids, attention_mask, audio, audio_lens)`` returns logits
    ``[B, num_classes]``; ``audio`` is ``[B, samples]`` at 16 kHz and
    ``audio_lens`` the unpadded sample counts.
    """

    def __init__(
        self,
        roberta_name: str = "roberta-base",
        hubert_name: str = "facebook/hubert-base-ls960",
        num_classes: int = 3,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_ffn: int = 3072,
        fusion_dropout: float = 0.1,
        head_dropout: float = 0.1,
        audio_stride: int = 4,
        tokenizer_len: Optional[int] = None,
        text_only: bool = False,
        audio_only: bool = False,
    ):
        super().__init__()
        assert not (text_only and audio_only), "Cannot set both text_only and audio_only"

        self.text_only = text_only
        self.audio_only = audio_only
        self.num_classes = num_classes
        self.audio_stride = audio_stride

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        if tokenizer_len is not None:
            self.roberta.resize_token_embeddings(tokenizer_len)
        d_model = self.roberta.config.hidden_size

        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False
        # One hidden state per transformer layer plus the embedding output.
        self.audio_layer_mixer = WeightedLayerSum(self.hubert.config.num_hidden_layers + 1)
        self.audio_downsampler = nn.Conv1d(
            in_channels=self.hubert.config.hidden_size,
            out_channels=d_model,
            kernel_size=audio_stride * 2,
            stride=audio_stride,
            padding=audio_stride,
        )
        self.audio_norm = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.modality_embed = nn.Embedding(3, d_model)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=fusion_heads,
            dim_feedforward=fusion_ffn,
            dropout=fusion_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, num_classes),
        )

    def train(self, mode: bool = True):
        """Switch modes while keeping the frozen HuBERT in eval mode."""
        super().train(mode)
        self.hubert.eval()
        return self

    def _encode_text(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, attention_mask

    @torch.no_grad()
    def _hubert_forward_nograd(self, audio, audio_attn_mask):
        out = self.hubert(
            input_values=audio,
            attention_mask=audio_attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return torch.stack(out.hidden_states, dim=0)

    def _hubert_frame_lengths(self, audio_lens: torch.Tensor) -> torch.Tensor:
        """HuBERT output frame counts for raw-sample input lengths."""
        return self.hubert._get_feat_extract_output_lengths(audio_lens)

    def _encode_audio(self, audio, audio_lens):
        """Return downsampled audio features ``[B, Td, D]`` and their mask ``[B, Td]``."""
        layers = self._hubert_forward_nograd(audio, audio_attention_mask(audio, audio_lens))
        mixed = self.audio_layer_mixer(layers)

        # Zero padded frames before the convolution so they cannot leak into valid ones.
        frame_lens = self._hubert_frame_lengths(audio_lens)
        valid = frame_mask(mixed.size(1), frame_lens, mixed.device)
        mixed = mixed * valid.unsqueeze(-1).float()

        x = self.audio_downsampler(mixed.transpose(1, 2)).transpose(1, 2)
        x = self.audio_norm(F.gelu(x))

        kernel = self.audio_downsampler.kernel_size[0]
        stride = self.audio_downsampler.stride[0]
        pad = self.audio_downsampler.padding[0]
        down_lens = ((frame_lens + 2 * pad - kernel) // stride + 1).clamp(min=1)
        down_lens = down_lens.clamp(max=x.size(1))
        return x, frame_mask(x.size(1), down_lens, x.device).long()

    def forward(self, input_ids, attention_mask, audio, audio_lens):
        batch_size = input_ids.size(0)
        device = input_ids.device

        parts = [self.cls_token.expand(batch_size, -1, -1)]
        masks = [torch.ones(batch_size, 1, dtype=torch.long, device=device)]
        modality_ids = [torch.full((batch_size, 1), MODALITY_CLS, dtype=torch.long, device=device)]

        if not self.audio_only:
            text_h, text_mask = self._encode_text(input_ids, attention_mask)
            parts.append(text_h)
            masks.append(text_mask)
            modality_ids.append(
                torch.full((batch_size, text_h.size(1)), MODALITY_TEXT, dtype=torch.long, device=device)
            )

        if not self.text_only:
            audio_h, audio_mask = self._encode_audio(audio, audio_lens)
            parts.append(audio_h)
            masks.append(audio_mask)
            modality_ids.append(
                torch.full((batch_size, audio_h.size(1)), MODALITY_AUDIO, dtype=torch.long, device=device)
            )

        x = torch.cat(parts, dim=1) + self.modality_embed(torch.cat(modality_ids, dim=1))
        key_padding_mask = torch.cat(masks, dim=1) == 0
        h = self.fusion(x, src_key_padding_mask=key_padding_mask)
        return self.head(h[:, 0, :])
