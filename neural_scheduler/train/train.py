"""Fine-tune the scheduler LM on preprocessed dialogue windows.

    python -m neural_scheduler.train.train \\
        --train_file train.jsonl --validation_file val.jsonl --output_dir runs/scheduler \\
        --use_duration_prediction --do_train --do_eval ...

All ``transformers.TrainingArguments`` flags are accepted. See docs/training.md.
"""

import logging
import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer
from transformers.trainer_utils import get_last_checkpoint

from ..data.collator import DataCollatorForSeq2SeqWithChannels
from ..data.features import prepare_features, setup_tokenizer
from ..model import ChatGLMConfig, scheduler_model_class
from .args import DataArguments, ModelArguments, TrainingArguments
from .trainer import DurationPredictionTrainer

logger = logging.getLogger(__name__)

# Parameters that stay trainable whenever only some transformer layers are trained.
ALWAYS_TRAINABLE = ("embed", "lm_head", "final_layernorm", "output_layer", "channel", "duration_head")


def allow_full_torch_load():
    """Let ``torch.load`` unpickle the RNG states stored in Trainer checkpoints.

    torch >= 2.6 defaults to ``weights_only=True``, which rejects the NumPy
    state in ``rng_state*.pth`` when a run is resumed.
    """
    original_load = torch.load
    if getattr(original_load, "_allows_full_load", False):
        return

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load._allows_full_load = True
    torch.load = load


def use_torch_adamw_in_deepspeed():
    """Make DeepSpeed build ``torch.optim.AdamW`` for Adam/AdamW configs.

    With ZeRO optimizer offload DeepSpeed would otherwise pick its CPU Adam,
    which needs a compiled extension.
    """
    from deepspeed.runtime.engine import DeepSpeedEngine

    original = DeepSpeedEngine._configure_basic_optimizer
    if getattr(original, "_forces_torch_adamw", False):
        return

    def configure(self, model_parameters):
        if self.optimizer_name() in [None, "adam", "adamw"]:
            params = self.optimizer_params()
            return torch.optim.AdamW(
                model_parameters,
                lr=params.get("lr", 0.001),
                betas=params.get("betas", (0.9, 0.999)),
                eps=params.get("eps", 1e-8),
                weight_decay=params.get("weight_decay", 0),
            )
        return original(self, model_parameters)

    configure._forces_torch_adamw = True
    DeepSpeedEngine._configure_basic_optimizer = configure


def init_distributed(timeout_seconds: int):
    """Join the process group early when launched by ``torch.distributed.run``.

    Doing it here sets a long timeout, so ranks waiting for rank 0 to prepare
    the dataset do not time out.
    """
    launched_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ
    if launched_distributed and torch.cuda.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))


def build_model(model_args: ModelArguments, training_args: TrainingArguments, tokenizer):
    """Load the scheduler LM from ``model_args.model_name_or_path``.

    Starting from stock GLM-4-Voice, the channel embedding and the duration
    head are new and randomly initialised; all other weights are kept. The
    scheduler tokens fit in GLM-4-Voice's padded vocabulary, so the embedding
    matrices are not resized.
    """
    config = ChatGLMConfig.from_pretrained(model_args.model_name_or_path)
    config.num_channel = 3
    config.use_duration_prediction = bool(training_args.use_duration_prediction)
    config.num_duration_classes = training_args.num_duration_classes if training_args.use_duration_prediction else 32

    if len(tokenizer) > config.padded_vocab_size:
        raise ValueError(
            f"Tokenizer size ({len(tokenizer)}) exceeds the model's padded vocabulary ({config.padded_vocab_size})."
        )

    model_class = scheduler_model_class(config)
    logger.info(f"Loading {model_class.__name__} from {model_args.model_name_or_path}")
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
    )
    return model


def apply_freezing(model, model_args: ModelArguments):
    """Restrict which parameters are trained, according to the unfreeze arguments."""
    bottom = model_args.num_of_acoustic_layers_to_unfreeze
    top = model_args.num_of_output_layers_to_unfreeze

    if bottom > 0 or top > 0:
        logger.info(f"Training the bottom {max(bottom, 0)} and top {max(top, 0)} transformer layers")
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(key in name for key in ALWAYS_TRAINABLE):
                param.requires_grad = True

        layers = model.transformer.encoder.layers
        total_layers = len(layers)
        selected = set(range(min(max(bottom, 0), total_layers)))
        selected.update(i for i in range(total_layers - max(top, 0), total_layers) if i >= 0)
        for i in sorted(selected):
            for param in layers[i].parameters():
                param.requires_grad = True
    elif model_args.train_embeddings:
        logger.info("Training embedding parameters only")
        for name, param in model.named_parameters():
            if "embed" not in name:
                param.requires_grad = False
    else:
        logger.info("Training all parameters")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable}")


def load_datasets(data_args: DataArguments, training_args: TrainingArguments):
    """Load the JSONL files and turn them into model features."""
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file.split(",")
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file.split(",")
    if not data_files:
        raise ValueError("Give --train_file and/or --validation_file")
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=data_args.cache_dir)

    column_names = list(next(iter(raw_datasets.values())).features)
    with training_args.main_process_first(desc="dataset map features preparation"):
        return raw_datasets.map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preparing features and labels from JSONL",
        )


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if training_args.should_log else logging.WARNING,
    )

    allow_full_torch_load()
    if training_args.deepspeed:
        use_torch_adamw_in_deepspeed()
    init_distributed(training_args.ddp_timeout)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    tokenizer = setup_tokenizer(model_args.model_name_or_path)
    tokenized_datasets = load_datasets(data_args, training_args)

    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    data_collator = DataCollatorForSeq2SeqWithChannels(tokenizer, pad_to_multiple_of=8)

    model = build_model(model_args, training_args, tokenizer)
    apply_freezing(model, model_args)

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if training_args.use_duration_prediction:
        trainer = DurationPredictionTrainer(
            use_duration_prediction=True,
            duration_loss_weight=training_args.duration_loss_weight,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        save_final_model(trainer, training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def save_final_model(trainer: transformers.Trainer, output_dir: str):
    """Write the final weights to ``output_dir`` from a CPU copy of the state dict."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


if __name__ == "__main__":
    train()
