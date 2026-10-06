"""Load a trained scheduler checkpoint for inference."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from ..constants import MAX_TURNS
from ..vocab import add_scheduler_tokens
from .configuration import ChatGLMConfig
from .modeling import scheduler_model_class


def load_scheduler(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[torch.dtype] = None,
    max_turns: int = MAX_TURNS,
) -> Tuple[torch.nn.Module, object]:
    """Load a scheduler checkpoint directory and its tokenizer.

    The directory is a Hugging Face Trainer checkpoint: ``config.json``,
    ``*.safetensors`` shards and the tokenizer files. The model class follows
    ``config.use_duration_prediction``; the model is built directly from the
    config (so it runs with the default attention implementation), filled from
    the safetensors shards, cast to ``dtype`` and put in eval mode.

    Args:
        checkpoint_path: checkpoint directory.
        device: where to put the model.
        dtype: parameter dtype. Defaults to float16 on CUDA and float32 on CPU.
        max_turns: number of turn-marker pairs the tokenizer must know.

    Returns:
        ``(model, tokenizer)``.
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    add_scheduler_tokens(tokenizer, max_turns)

    config = ChatGLMConfig.from_pretrained(checkpoint_path)
    if getattr(config, "use_turn_embeddings", False):
        raise ValueError(
            f"{checkpoint_path} was trained with turn embeddings, a model variant this package does not include"
        )
    config.num_channel = 3

    model = scheduler_model_class(config)(config)
    model = model.to(dtype)

    shards = sorted(checkpoint_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {checkpoint_path}")
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(dtype)
    model.load_state_dict(state, strict=True)

    vocab_rows = model.transformer.embedding.word_embeddings.weight.shape[0]
    if len(tokenizer) > vocab_rows:
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens but the model embeds only {vocab_rows}")

    model.eval()
    model = model.to(device)
    return model, tokenizer
