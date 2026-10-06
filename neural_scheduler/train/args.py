"""Command-line arguments of the trainer."""

from dataclasses import dataclass, field
from typing import Optional

import transformers

from ..constants import GLM4_VOICE_MODEL_ID, NUM_DURATION_CLASSES


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=GLM4_VOICE_MODEL_ID,
        metadata={
            "help": "Weights to start from: a Hugging Face id or a local directory. Either stock "
                    "GLM-4-Voice or a scheduler checkpoint to continue from.",
            "aliases": ["--init_from"],
        },
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation: flash_attention_2, sdpa or eager."},
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "dtype the weights are loaded in (float16, bfloat16 or float32). Mixed-precision "
                          "training (--bf16) casts on the fly."},
    )
    train_embeddings: bool = field(
        default=False,
        metadata={"help": "Train only the token and channel embeddings (ignored when layers are unfrozen selectively)."},
    )
    num_of_acoustic_layers_to_unfreeze: int = field(
        default=-1,
        metadata={"help": "Number of bottom transformer layers to train. Negative together with "
                          "--num_of_output_layers_to_unfreeze: train everything."},
    )
    num_of_output_layers_to_unfreeze: int = field(
        default=-1,
        metadata={"help": "Number of top transformer layers to train."},
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Training JSONL file(s), comma-separated."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Validation JSONL file(s), comma-separated."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for the datasets cache (prepared features). Defaults to the HF cache."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the training set to this many records (debugging)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the validation set to this many records (debugging)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Recompute the prepared features instead of using the cache."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "Processes used to prepare features."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_duration_prediction: bool = field(
        default=False, metadata={"help": "Train the model with the duration head."}
    )
    num_duration_classes: int = field(
        default=NUM_DURATION_CLASSES, metadata={"help": "Number of duration classes; the last is open-ended."}
    )
    duration_loss_weight: float = field(
        default=0.5, metadata={"help": "Weight of the duration loss in the total loss."}
    )
    optim: str = field(default="adamw_torch")
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to a DeepSpeed config file."})
    ddp_timeout: int = field(default=10800, metadata={"help": "Timeout of the distributed backend, in seconds."})
