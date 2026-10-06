"""Trainer subclass that feeds duration labels to the model and logs duration metrics."""

import logging

import torch
from transformers import Trainer

from ..constants import SILENCE_TOKEN_ID

logger = logging.getLogger(__name__)

_RUNNING_KEYS = (
    "total_loss",
    "token_loss",
    "duration_loss",
    "duration_accuracy_overall",
    "duration_accuracy_silence",
    "duration_accuracy_speech",
    "silence_ratio",
)


class DurationPredictionTrainer(Trainer):
    """Hugging Face ``Trainer`` for the scheduler with a duration head.

    ``compute_loss`` passes ``remaining_durations`` and the loss weight to the
    model. While training, every ``loss_log_frequency`` steps it logs running
    means of the losses and of the duration accuracy, split into silence and
    speech positions, under ``train/*``.
    """

    def __init__(self, *args, use_duration_prediction=False, duration_loss_weight=0.5,
                 silence_token=SILENCE_TOKEN_ID, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_duration_prediction = use_duration_prediction
        self.duration_loss_weight = duration_loss_weight
        self.silence_token = silence_token
        self.loss_log_frequency = 10
        self.step_count = 0
        self.running_metrics = {key: [] for key in _RUNNING_KEYS}

    def _compute_duration_metrics(self, duration_logits, duration_labels, input_ids):
        """Duration accuracy overall, on silence tokens and on speech tokens.

        Args:
            duration_logits: ``[batch, seq, classes]``.
            duration_labels: ``[batch, seq]`` raw labels (-100 ignored). They are
                compared unclamped, so a label above the last class never counts
                as correct.
            input_ids: ``[batch, seq]``, used to tell silence from speech.
        """
        with torch.no_grad():
            duration_preds = duration_logits.argmax(dim=-1)
            valid_mask = duration_labels != -100
            silence_mask = (input_ids == self.silence_token) & valid_mask
            speech_mask = (input_ids != self.silence_token) & valid_mask

            metrics = {}

            if valid_mask.sum() > 0:
                overall_correct = (duration_preds == duration_labels) & valid_mask
                metrics["overall_accuracy"] = overall_correct.sum().float() / valid_mask.sum().float()
                metrics["total_valid_tokens"] = valid_mask.sum().item()
            else:
                metrics["overall_accuracy"] = torch.tensor(0.0, device=duration_logits.device)
                metrics["total_valid_tokens"] = 0

            if silence_mask.sum() > 0:
                silence_correct = (duration_preds == duration_labels) & silence_mask
                metrics["silence_accuracy"] = silence_correct.sum().float() / silence_mask.sum().float()
                metrics["silence_tokens"] = silence_mask.sum().item()
                metrics["silence_ratio"] = silence_mask.sum().float() / valid_mask.sum().float()
            else:
                metrics["silence_accuracy"] = torch.tensor(0.0, device=duration_logits.device)
                metrics["silence_tokens"] = 0
                metrics["silence_ratio"] = torch.tensor(0.0, device=duration_logits.device)

            if speech_mask.sum() > 0:
                speech_correct = (duration_preds == duration_labels) & speech_mask
                metrics["speech_accuracy"] = speech_correct.sum().float() / speech_mask.sum().float()
                metrics["speech_tokens"] = speech_mask.sum().item()
            else:
                metrics["speech_accuracy"] = torch.tensor(0.0, device=duration_logits.device)
                metrics["speech_tokens"] = 0

            if silence_mask.sum() > 0:
                silence_durations_pred = duration_preds[silence_mask]
                silence_durations_true = duration_labels[silence_mask]
                metrics["silence_duration_mean_pred"] = silence_durations_pred.float().mean()
                metrics["silence_duration_mean_true"] = silence_durations_true.float().mean()
                metrics["silence_duration_std_pred"] = silence_durations_pred.float().std()
                metrics["silence_duration_std_true"] = silence_durations_true.float().std()

            return metrics

    def _track_and_log(self, outputs, duration_metrics):
        """Update the running means and log them every ``loss_log_frequency`` training steps."""
        self.step_count += 1
        values = {
            "total_loss": outputs.loss.item(),
            "token_loss": outputs.token_loss.item(),
            "duration_loss": outputs.duration_loss.item(),
            "duration_accuracy_overall": duration_metrics["overall_accuracy"].item(),
            "duration_accuracy_silence": duration_metrics["silence_accuracy"].item(),
            "duration_accuracy_speech": duration_metrics["speech_accuracy"].item(),
            "silence_ratio": duration_metrics["silence_ratio"].item(),
        }
        for key, value in values.items():
            history = self.running_metrics[key]
            history.append(value)
            if len(history) > self.loss_log_frequency:
                self.running_metrics[key] = history[-self.loss_log_frequency:]

        if self.step_count % self.loss_log_frequency != 0:
            return

        avg = {key: sum(history) / len(history) for key, history in self.running_metrics.items()}
        logger.info(
            f"Step {self.step_count}: "
            f"Total Loss = {avg['total_loss']:.4f}, "
            f"Token Loss = {avg['token_loss']:.4f}, "
            f"Duration Loss = {avg['duration_loss']:.4f}"
        )
        logger.info(
            f"  Duration Accuracy: "
            f"Overall = {avg['duration_accuracy_overall']:.4f}, "
            f"Silence = {avg['duration_accuracy_silence']:.4f}, "
            f"Speech = {avg['duration_accuracy_speech']:.4f}"
        )
        has_silence_stats = "silence_duration_mean_pred" in duration_metrics
        if not self.state.is_world_process_zero:
            return

        log_dict = {
            "train/total_loss": avg["total_loss"],
            "train/token_loss": avg["token_loss"],
            "train/duration_loss": avg["duration_loss"],
            "train/duration_accuracy_overall": avg["duration_accuracy_overall"],
            "train/duration_accuracy_silence": avg["duration_accuracy_silence"],
            "train/duration_accuracy_speech": avg["duration_accuracy_speech"],
            "train/silence_ratio": avg["silence_ratio"],
            "train/silence_tokens": duration_metrics["silence_tokens"],
            "train/speech_tokens": duration_metrics["speech_tokens"],
        }
        if has_silence_stats:
            log_dict.update({
                "train/silence_duration_mean_pred": duration_metrics["silence_duration_mean_pred"].item(),
                "train/silence_duration_mean_true": duration_metrics["silence_duration_mean_true"].item(),
                "train/silence_duration_std_pred": duration_metrics["silence_duration_std_pred"].item(),
                "train/silence_duration_std_true": duration_metrics["silence_duration_std_true"].item(),
            })
        self.log(log_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get("input_ids", None)
        remaining_durations = inputs.pop("remaining_durations", None)

        if self.use_duration_prediction and remaining_durations is not None:
            outputs = model(
                **inputs,
                remaining_durations=remaining_durations,
                duration_loss_weight=self.duration_loss_weight,
            )
            total_loss = outputs.loss
            if model.training:
                duration_metrics = self._compute_duration_metrics(
                    outputs.duration_logits, remaining_durations, input_ids
                )
                self._track_and_log(outputs, duration_metrics)
        else:
            outputs = model(**inputs)
            total_loss = outputs.loss

        return (total_loss, outputs) if return_outputs else total_loss
