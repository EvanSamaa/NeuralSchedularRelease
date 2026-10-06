# Training the scheduler

Code: `neural_scheduler/train/` (`train.py`, `trainer.py`, `args.py`),
`neural_scheduler/data/features.py`, `neural_scheduler/data/collator.py`.
Launch templates: `scripts/slurm/train_*.sh`. DeepSpeed config:
`configs/deepspeed_zero3.json`.

Training fine-tunes all of GLM-4-Voice (9B parameters) with the Hugging Face
`Trainer` and DeepSpeed ZeRO-3. The released checkpoint was trained on Seamless
Interaction on 4 nodes × 4 L40S GPUs (effective batch 64) for 15,200 steps.

## Data format

Plain JSONL, one dialogue window per line, produced by the preprocessing
pipeline (see `docs/data.md`). The trainer reads:

| Field | Required | Meaning |
|---|---|---|
| `input_ids` | yes | the full sequence as vocab ids (script A, script B, dialogue; see `docs/model.md`) |
| `generation_start_index` | yes | index of the first dialogue token |
| `generation_end_index` | no | index of `<|end-of-dialogue|>`; defaults to the sequence length |
| `remaining_durations` | no | duration labels for the dialogue region only; length must equal `generation_end_index - generation_start_index` |

Every other field (`turn_boundaries`, `source_file`, `window_idx`,
`vad_timestamps_*`, ...) is dropped. `prepare_features` turns a record into

* `labels`: `input_ids` with everything before `generation_start_index` set to
  -100, so the loss covers the dialogue only;
* `attention_mask`: all ones;
* `remaining_durations`: padded with -100 over the script and after the
  dialogue, to the sequence length;
* `generation_start_index`: passed through, the model derives channel ids from it.

The collator right-pads each batch to a multiple of 8 (pad id 151329, labels
and durations -100). Sequences are not truncated: preprocessing already limits
a window to about 1024 tokens.

The prepared features are cached by `datasets`; pass `--cache_dir` to choose
where (otherwise the Hugging Face cache under `HF_HOME`).

## Running

```bash
python -m neural_scheduler.train.train \
    --model_name_or_path THUDM/glm-4-voice-9b \
    --train_file train.jsonl --validation_file val.jsonl \
    --output_dir /scratch/$USER/scheduler_run \
    --use_duration_prediction --num_duration_classes 32 --duration_loss_weight 0.5 \
    --do_train --do_eval ...
```

### Arguments

All `transformers.TrainingArguments` flags are accepted. The ones specific to
this trainer:

| Flag | Default | Meaning |
|---|---|---|
| `--model_name_or_path`, `--init_from` | `THUDM/glm-4-voice-9b` | weights to start from: stock GLM-4-Voice, or a scheduler checkpoint directory to continue from (for example Seamless → Fisher) |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2`, `sdpa` or `eager` |
| `--torch_dtype` | `float16` | dtype the weights are loaded in |
| `--train_file`, `--validation_file` | – | JSONL file(s), comma-separated |
| `--cache_dir` | HF cache | datasets cache for the prepared features |
| `--max_train_samples`, `--max_eval_samples` | all | truncate the datasets (debugging) |
| `--overwrite_cache` | off | recompute the prepared features |
| `--preprocessing_num_workers` | 4 | processes for feature preparation |
| `--use_duration_prediction` | off | train `ChatGLMForConditionalGenerationPredDuration` (with the duration head) |
| `--num_duration_classes` | 32 | number of duration classes |
| `--duration_loss_weight` | 0.5 | weight of the duration loss |
| `--num_of_acoustic_layers_to_unfreeze` | -1 | train only this many bottom layers ... |
| `--num_of_output_layers_to_unfreeze` | -1 | ... and this many top layers (both negative: train everything) |
| `--train_embeddings` | off | train only the token and channel embeddings |
| `--deepspeed` | none | path to a DeepSpeed config |
| `--ddp_timeout` | 10800 | distributed backend timeout in seconds |

When layers are unfrozen selectively, the token embedding, channel embedding,
output layer, final layer norm and duration head stay trainable.

### The settings of the released checkpoint

```
--deepspeed configs/deepspeed_zero3.json
--num_train_epochs 2 --per_device_train_batch_size 2 --gradient_accumulation_steps 2
--learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.01
--gradient_checkpointing --optim adamw_torch --bf16 --weight_decay 0.0 --max_grad_norm 1.0
--seed 42 --logging_steps 1 --report_to tensorboard
--save_strategy steps --save_steps 100 --eval_strategy steps --eval_steps 100
--save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_loss
--no_remove_unused_columns --dataloader_num_workers 8
--use_duration_prediction --num_duration_classes 32 --duration_loss_weight 0.5
```

`scripts/slurm/train_worker.sh` passes exactly these.
`--no_remove_unused_columns` is required: it keeps `generation_start_index` and
`remaining_durations` in the batch.

### Multi-node (Slurm)

```bash
export TRAIN_FILE=/data/seamless_train.jsonl
export VALIDATION_FILE=/data/seamless_val.jsonl
export OUTPUT_DIR=/scratch/$USER/scheduler_run
export ENV_SETUP=/path/to/activate_env.sh        # optional: module loads + virtualenv, sourced on every node
sbatch --nodes=4 --output=/scratch/$USER/logs/train-%j.out scripts/slurm/train_multinode.sh
```

`train_multinode.sh` picks the first node as rendezvous host and runs
`train_worker.sh` once per node with `srun --overlap`; each worker starts
`torch.distributed.run` with `--nproc_per_node=$GPUS_PER_NODE` (default 4) and
its node rank. Add `--account` / `--partition` on the `sbatch` command line as
your cluster requires, and submit from project or scratch storage. Other
variables (`INIT_FROM`, `DEEPSPEED_CONFIG`, `CACHE_DIR`, `NUM_EPOCHS`,
`BATCH_SIZE`, `GRAD_ACCUM`, `LEARNING_RATE`, `SAVE_STEPS`, `EVAL_STEPS`,
`MASTER_PORT`) are documented at the top of `train_worker.sh`.

To continue training on a second corpus from a finished run:

```bash
export INIT_FROM=/scratch/$USER/scheduler_run/checkpoint-15200
export OUTPUT_DIR=/scratch/$USER/scheduler_run_fisher
```

### Single GPU (debugging)

```bash
TRAIN_FILE=train.jsonl VALIDATION_FILE=val.jsonl OUTPUT_DIR=/scratch/$USER/debug_run \
    bash scripts/slurm/train_local.sh
```

One process, no DeepSpeed, 16 training samples. No process group is created
unless the script runs under `torch.distributed.run`.

### Resuming

If `--output_dir` already contains `checkpoint-N` directories, the run resumes
from the latest one (optimizer, scheduler and RNG state included) unless
`--overwrite_output_dir` is given. A job that hits its time limit is continued
by submitting the same command again. With `--save_total_limit 1` and
`--load_best_model_at_end` the directory keeps the best and the most recent
checkpoint; if a job is killed while a checkpoint is being written, check that
the newest one is complete (all safetensors shards and
`model.safetensors.index.json`) before relying on it.

## What is logged

TensorBoard events go to `<output_dir>/runs`. In addition to the `Trainer`'s
own `loss` / `eval_loss`, every 10 training steps the duration trainer logs
running means over the last 10 steps:

`train/total_loss`, `train/token_loss`, `train/duration_loss`,
`train/duration_accuracy_overall`, `train/duration_accuracy_silence`,
`train/duration_accuracy_speech`, `train/silence_ratio`, and for the current
batch `train/silence_tokens`, `train/speech_tokens`,
`train/silence_duration_{mean,std}_{pred,true}`.

`eval_loss` is the combined loss (`token + 0.5 × duration`), and the final
`perplexity` in `eval_results.json` is its exponential, so it is not a pure
token perplexity.

## DeepSpeed specifics

* `configs/deepspeed_zero3.json`: ZeRO stage 3 with parameter and optimizer
  offload to CPU, bf16, gradient clipping 1.0, AdamW and `WarmupLR` with
  `"auto"` values taken from the command line.
* With optimizer offload DeepSpeed would use its CPU Adam, which needs a
  compiled extension. When `--deepspeed` is given, the trainer makes DeepSpeed
  build `torch.optim.AdamW` instead.
* `torch.load` is allowed to unpickle full objects during the run, because
  resuming reads NumPy RNG states from `rng_state_*.pth`, which torch ≥ 2.6
  rejects by default. Only resume from checkpoints you trust.

## Requirements

`torch` 2.6, `transformers` 4.46.x (the trainer passes `tokenizer=` to
`Trainer`, and the model relies on 4.46 internals), `datasets`, `safetensors`,
`sentencepiece`, `tensorboard`; for the released configuration also
`deepspeed` (0.18) and `flash-attn` (2.8). `pip install -e ".[train]"`.

## Migration from the research tree

| Research file | Here |
|---|---|
| `ntpp/train/sft_glm_4_voice_pred_timing_only_with_duration_pred_stable.py` (and `..._with_duration_pred.py`, identical but for one import) | `train/train.py`, `train/trainer.py`, `train/args.py` |
| `prepare_features`, `setup_tokenizer_with_new_tokens` in the trainer | `data/features.py` (`prepare_features`, `setup_tokenizer`) |
| `ntpp/model/chatGLM_dual_timing_only/custom_collator.py` | `data/collator.py` |
| `ntpp/config/ds_sft_glm4_voice.json` | `configs/deepspeed_zero3.json` (unchanged) |
| `launch_..._duration_pred_turn_id_as_tokens.sh` + `_worker.sh` (and the `_fisher` pair) | `scripts/slurm/train_multinode.sh` + `train_worker.sh`, with `INIT_FROM` for the Fisher case |
| `launch_training_L40_glm4_timinig_only_local.sh` | `scripts/slurm/train_local.sh` |
| `sft_glm_4_voice_pred_timing_only.py`, `..._and_logic_clock.py`, `..._logic_clock_cold_start.py`, `scheduled_sampling.py`, `duration_weighted_loss.py`, the other launch scripts and DeepSpeed configs | not included |

### Differences from the research trainer

* `--model_name_or_path` is used. The research trainer parsed it and always
  loaded `THUDM/glm-4-voice-9b`; see known issue 3.
* Removed flags: `--block_size`, `--corruption_rate`, `--data_path`,
  `--streaming`, `--validation_split_percentage`, `--use_text` (never read), and
  the `--use_scheduled_sampling` / `--ss_*` and `--use_duration_weighted_loss` /
  `--duration_loss_*` experiment flags. Old launch commands that pass
  `--block_size 1024` must drop it.
* `--cache_dir` now sets the datasets cache directory (it was ignored).
* New flags: `--attn_implementation`, `--torch_dtype` (defaults equal the
  values that were hardcoded).
* The duration metrics are accumulated and logged only while the model is in
  training mode. Previously evaluation batches were mixed into the `train/*`
  running means.
* Partial fine-tuning keeps the duration head trainable (it used to be frozen
  at its random initialisation), and `--train_embeddings` takes effect
  with the default unfreeze arguments (it used to be reachable only when neither was
  negative-negative nor positive).
* The DeepSpeed optimizer override is applied only with `--deepspeed`, and the
  process group is created only under a distributed launcher, so `deepspeed` is
  not needed for a local run.
* The config and model are loaded with this package's classes instead of
  `trust_remote_code`; saved checkpoints no longer carry the stock `auto_map`.

None of these change what the canonical launch command computes.

## Known issues

1. **Learning-rate schedule.** Because the DeepSpeed config has a `scheduler`
   block (`WarmupLR`), DeepSpeed owns the schedule and `--lr_scheduler_type
   cosine` has no effect: the learning rate warms up and then stays constant at
   `--learning_rate`. The released checkpoint was trained this way, so the
   config is shipped unchanged. Remove the `scheduler` block to get the
   `Trainer`'s schedule.
2. **Rotary positions under padding.** Samples that are right-padded in a batch
   see shifted dialogue positions during training (`docs/model.md`, known issue
   1). Inference is unpadded.
3. **The Fisher "continued" run was not a continuation.** The research trainer
   ignored `--model_name_or_path`, so the run named
   `..._continued_off_seamless` started from stock GLM-4-Voice. With this
   trainer `--init_from <checkpoint>` does continue, which is new behaviour, not
   a reproduction of that run.
4. **Duration accuracy in the logs is understated.** `train/duration_accuracy_*`
   compares predictions with the raw labels, so a label above 31 can never be
   matched. The model's own `duration_accuracy` output uses the clamped labels.
   Kept so curves stay comparable with existing runs.
5. **Validation is loss only.** There is no `compute_metrics`; model selection
   uses the combined `eval_loss`. checkpoint-15200 is the last checkpoint of
   its run, not the best by `eval_loss` (14700).
6. **Token-level cross-entropy, not pair prediction.** See `docs/model.md`,
   known issue 2.
7. **`duration_loss` with `--use_duration_prediction` but no
   `remaining_durations` in the data** silently falls back to the token loss
   only.

## Tests

* `tests/test_train_features.py`: feature preparation and collation, including
  equality with the research implementations (needs `NTPP_ORIG`), and the
  freezing rules.
* `tests/test_train_smoke.py`: runs the real entry point on CPU with a 2-layer
  model and a few synthetic windows: trains, evaluates, writes checkpoints,
  loads one with `load_scheduler`, and starts a second run from it with
  `--init_from`. Needs the GLM-4-Voice tokenizer in the local Hugging Face cache.
