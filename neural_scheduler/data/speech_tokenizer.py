"""GLM-4-Voice speech tokenizer (WhisperVQ, 12.5 Hz) loading and batched tokenisation.

The ``WhisperVQEncoder`` class is not on PyPI; it lives in the GLM-4-Voice
repository (``speech_tokenizer/modeling_whisper.py``). Point the ``GLM4VOICE_REPO``
environment variable at a checkout of https://github.com/THUDM/GLM-4-Voice, or
pass ``glm4voice_repo`` explicitly.
"""

import importlib
import logging
import os
import sys
from typing import List, Optional

import torch

from ..constants import GLM4_VOICE_TOKENIZER_ID

GLM4VOICE_REPO_ENV = "GLM4VOICE_REPO"


def import_whisper_vq_encoder(glm4voice_repo: Optional[str] = None):
    """Return the ``WhisperVQEncoder`` class from a GLM-4-Voice checkout."""
    try:
        return importlib.import_module("speech_tokenizer.modeling_whisper").WhisperVQEncoder
    except ImportError:
        pass

    repo = glm4voice_repo or os.environ.get(GLM4VOICE_REPO_ENV)
    if not repo:
        raise ImportError(
            "Cannot import speech_tokenizer.modeling_whisper. Clone "
            "https://github.com/THUDM/GLM-4-Voice and set the "
            f"{GLM4VOICE_REPO_ENV} environment variable to the checkout."
        )
    if not os.path.isfile(os.path.join(repo, "speech_tokenizer", "modeling_whisper.py")):
        raise ImportError(
            f"{repo!r} does not contain speech_tokenizer/modeling_whisper.py; "
            f"{GLM4VOICE_REPO_ENV} must point at a GLM-4-Voice checkout."
        )
    if repo not in sys.path:
        sys.path.append(repo)
    return importlib.import_module("speech_tokenizer.modeling_whisper").WhisperVQEncoder


def load_speech_tokenizer(
    tokenizer_path: str = GLM4_VOICE_TOKENIZER_ID,
    device: str = "cuda",
    glm4voice_repo: Optional[str] = None,
):
    """Load the frozen WhisperVQ encoder and its feature extractor.

    Returns ``(whisper_vq_model, feature_extractor)`` with the model in eval mode
    on ``device``.
    """
    from transformers import WhisperFeatureExtractor

    encoder_cls = import_whisper_vq_encoder(glm4voice_repo)
    model = encoder_cls.from_pretrained(tokenizer_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    for param in model.parameters():
        param.requires_grad = False
    model._freeze_parameters()
    model.eval()
    model.to(device)
    return model, feature_extractor


def extract_speech_token_batched(
    model,
    feature_extractor,
    batched_audio: torch.Tensor,
    sampling_rate: int = 16000,
    model_batch_size: int = 32,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> List[List[int]]:
    """Tokenise a batch of equally long waveforms into codec ids.

    ``batched_audio`` is ``[num_utterances, samples]`` (a 3-D input uses its first
    channel). Each utterance is cut into 30 s chunks, the chunks are run through
    the encoder ``model_batch_size`` at a time, and padded frames are removed with
    the downsampled attention mask. Returns one list of codec ids (0..16383) per
    utterance, one id per 80 ms.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if batched_audio.dim() == 3:
        logger.warning(f"Input audio has 3 dimensions ({batched_audio.shape}). Using first channel.")
        batched_audio = batched_audio[:, 0, :]
    elif batched_audio.dim() != 2:
        raise ValueError(f"Input audio must have 2 or 3 dimensions, got {batched_audio.dim()}")

    batched_audio = batched_audio.to(device)
    num_utts = batched_audio.shape[0]

    all_chunks = []
    indices = []
    chunk_len_samples = 30 * sampling_rate

    for idx in range(num_utts):
        audio = batched_audio[idx]
        num_samples = audio.shape[0]
        if num_samples == 0:
            logger.warning(f"Utterance {idx} is empty.")
            continue
        time_step_samples = 0
        while time_step_samples < num_samples:
            audio_segment = audio[time_step_samples:time_step_samples + chunk_len_samples]
            all_chunks.append(audio_segment.cpu().numpy())
            indices.append(idx)
            time_step_samples += chunk_len_samples

    if not all_chunks:
        logger.warning("No audio chunks were created.")
        return [[] for _ in range(num_utts)]

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length

    all_speech_tokens = [[] for _ in range(num_utts)]

    for start in range(0, len(all_chunks), model_batch_size):
        chunk_batch = all_chunks[start:start + model_batch_size]
        features = feature_extractor(
            chunk_batch, sampling_rate=sampling_rate,
            return_attention_mask=True, return_tensors="pt",
            device=device, padding="longest", pad_to_multiple_of=stride,
        )
        features.to(device=device)
        outputs = model(**features)
        speech_tokens = outputs.quantized_token_ids

        attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
        attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]

        if attention_mask.shape != speech_tokens.shape:
            target_len = speech_tokens.shape[1]
            attention_mask = attention_mask[:, :target_len]
            assert attention_mask.shape == speech_tokens.shape

        for i in range(len(speech_tokens)):
            original_utt_idx = indices[start + i]
            speech_token_list = speech_tokens[i][attention_mask[i].bool()].tolist()
            all_speech_tokens[original_utt_idx].extend(speech_token_list)

    return all_speech_tokens
