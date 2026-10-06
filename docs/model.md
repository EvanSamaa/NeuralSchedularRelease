# The scheduler model

Code: `neural_scheduler/model/` (`configuration.py`, `modeling.py`, `loading.py`).

The scheduler is GLM-4-Voice (a 9B ChatGLM decoder, 40 layers, hidden size
4096) fine-tuned to answer one question: given *what* two speakers are going to
say, *when* does each of them say it? It keeps the GLM-4-Voice weights and
vocabulary and adds two small things, a channel embedding and a duration head.

## Sequence layout

```
<|start-of-script-A|> script A <|end-of-script-A|>
<|start-of-script-B|> script B <|end-of-script-B|>
<|start-of-dialogue|> A0 B0 A1 B1 A2 B2 ... <|end-of-dialogue|>
```

* **Script region.** What each speaker will say, without timing: the speech
  tokens of each turn wrapped in `<|start-turn-i|> ... <|end-turn-i|>`, with a
  single silence token between segments.
* **Dialogue region.** The time-aligned conversation at 12.5 Hz (80 ms per
  frame), channel A and channel B interleaved. Frames where a speaker is silent
  hold the silence token `<|audio_8520|>`.
* `generation_start_index` is the index of `A0`; `generation_end_index` is the
  index of `<|end-of-dialogue|>`.

A training window holds 256 frames per channel (512 dialogue tokens, 20.48 s).
The loss is computed on the dialogue region only, so the model learns to place
script content in time.

## Vocabulary

Tokens are appended to the stock GLM-4-Voice tokenizer (168,736 entries). They
fit in the model's padded vocabulary (168,960 rows), so the embedding and
output matrices are never resized. `neural_scheduler/vocab.py` is the single
definition.

| Token | Vocab id |
|---|---|
| `<|audio_k|>`, k = 0..16382 (speech codec ids) | 152353 + k |
| silence, `<|audio_8520|>` | 160873 |
| `<|start-of-script-A|>`, `<|end-of-script-A|>` | 168736, 168737 |
| `<|start-of-script-B|>`, `<|end-of-script-B|>` | 168738, 168739 |
| `<|start-of-dialogue|>`, `<|end-of-dialogue|>` | 168740, 168741 |
| `<|start-turn-i|>`, `<|end-turn-i|>`, i = 0..49 | 168742 + 2i, 168743 + 2i |
| pad | 151329 |

## Architecture

### Channel embedding

Every position has a channel id: 0 for the script region, 1 for dialogue
channel A, 2 for dialogue channel B (`compute_channel_ids`). A learned table
`transformer.channel_emb.channel_embedding` of shape `[3, 128]` is looked up
with the channel id and **added to the query and key of every attention layer**,
after the rotary embedding and before the KV cache is concatenated
(`apply_chatglm_channel_emb`). It is one table shared by all layers and heads.

If `channel_ids` is not passed to `forward`, it is derived from
`generation_start_index` and multiplied by the attention mask, so padded
positions get channel 0.

### Rotary positions

Positions are not the raw token index:

* a script token at index *j* gets position *j*;
* the two dialogue tokens of frame *t* share position
  `num_script_tokens + t`.

So A<sub>t</sub> and B<sub>t</sub> see each other at relative distance zero,
and a frame advances the position by one regardless of the interleaving.

### Duration head

`ChatGLMForConditionalGenerationPredDuration` adds `duration_head`:
`Linear(4096, 512) → ReLU → Dropout(0.1) → Linear(512, 32)` applied to the
final hidden state of every position. Class *d* means "the token at this
position repeats *d* more frames in its own channel"; class 31 means 31 or
more. Because only silence produces long runs, the head is in effect a
per-channel "frames of silence remaining" countdown. The labels are aligned
with the inputs (not shifted) and clamped to `[0, 31]`. The last linear layer is
initialised with std 0.02 and zero bias.

`ChatGLMForConditionalGeneration` is the same model without the head.
`scheduler_model_class(config)` picks between them from
`config.use_duration_prediction`.

### Loss

```
token_loss    = CE(logits[:, :-1], labels[:, 1:])          # ordinary next-token loss, -100 ignored
duration_loss = CE(duration_logits, clamp(remaining_durations))
loss          = token_loss + duration_loss_weight * duration_loss   # weight 0.5 in the released model
```

The attention is plain causal attention over the interleaved sequence: B<sub>t</sub>
attends to A<sub>t</sub>. See "Known issues".

### Forward interface

```python
out = model(
    input_ids,                      # [B, L] vocab ids
    attention_mask=...,             # [B, L]; all ones if omitted
    generation_start_index=...,     # [B]; or pass channel_ids [B, L] directly
    labels=...,                     # [B, L], script region = -100
    remaining_durations=...,        # [B, L], -100 outside the dialogue (duration model only)
    duration_loss_weight=0.5,
    past_key_values=..., use_cache=...,
)
out.logits            # [B, L, 168960]
out.duration_logits   # [B, L, 32], float32 (duration model only)
out.loss, out.token_loss, out.duration_loss, out.duration_accuracy
```

With a KV cache, pass the `channel_ids` of the *whole* sequence so far (the last
`q_len` entries are used for the new tokens) and the `position_ids` of the new
tokens.

Hugging Face `generate()` is not supported and raises: the model needs channel
ids and positions that `generate()` does not supply. The scheduler search and
the evaluation code drive the model with explicit forward passes.

## Loading a checkpoint

```python
from neural_scheduler.model import load_scheduler

model, tokenizer = load_scheduler("/path/to/checkpoint-15200", device="cuda")
```

`load_scheduler(checkpoint_path, device="cuda", dtype=None, max_turns=50)`:

1. loads the tokenizer saved with the checkpoint and makes sure it has the
   scheduler tokens;
2. reads `config.json` into `ChatGLMConfig` and builds the class named by
   `use_duration_prediction` directly from the config;
3. reads all `*.safetensors` shards, casts them to `dtype` (float16 on CUDA,
   float32 on CPU by default) and loads them with `strict=True`;
4. puts the model in eval mode on `device`.

Building the model directly from the config (rather than through
`from_pretrained`) is deliberate: it is what the research pipeline did, and it
selects the SDPA attention path, so scores match the ones the pipeline produced.

A checkpoint directory written by the trainer contains `config.json`,
`model-0000x-of-0000y.safetensors`, `model.safetensors.index.json`, the
tokenizer files (`tokenizer.model`, `tokenizer_config.json`,
`added_tokens.json`, `special_tokens_map.json`), `trainer_state.json`,
`training_args.bin`, and, for DeepSpeed runs, `global_stepN/`, `latest`,
`rng_state_*.pth` and `zero_to_fp32.py`. Only the config, the safetensors and
the tokenizer files are needed for inference.

`config.json` files written by the research code carry an `auto_map` that
points at the stock GLM-4-Voice classes. `ChatGLMConfig` drops that entry on
load, and checkpoints saved from this package no longer contain it. Do not load
a scheduler checkpoint with `AutoModel.from_pretrained(..., trust_remote_code=True)`:
for an old checkpoint that returns a stock model without the channel embedding
and the duration head.

## Migration from the research tree

| Research file (`ntpp/model/chatGLM_dual_timing_only/`) | Here |
|---|---|
| `configuration_chatglm_timing.py` | `model/configuration.py` |
| `modeling_chatglm_timing_duration_pred_stable.py` | `model/modeling.py` (`ChatGLMForConditionalGenerationPredDuration`) |
| `modeling_chatglm_timing_duration_pred.py` | same class; differs from `_stable` only in the duration head's initial values |
| `modeling_chatglm_timing.py` | `model/modeling.py` (`ChatGLMForConditionalGeneration`) |
| `load_corrupted_checkpoint` (16 copies in the experiment scripts) | `model/loading.py::load_scheduler` |
| `get_special_tokens` and the token lists in trainers and pipelines | `neural_scheduler/vocab.py` |
| `modeling_chatglm_timing_duration_pred_logic_clock.py` (turn embedding) | not included |
| `load_chatglm.py`, `unit_test.py`, `tokenization_chatglm.py`, `config.json` | not included (unused copies) |

Class names are unchanged, so existing checkpoints load. State-dict keys are
identical (289 tensors for checkpoint-15200).

What was removed from the model file: the legacy `generate` /
`_generate_no_beam_search` / `postprocess_next_token_scores` block and the
`prepare_inputs_for_generation` hooks (they referenced names that do not
exist), the unused `position_ids` argument threaded through the attention
layers, a rotary cache that was computed and immediately overwritten, a message
printed on every forward without an attention mask, and the JIT fusion flags.

## Known issues

1. **Rotary positions shift under right padding.** When channel ids are derived
   from `generation_start_index`, padded positions get channel 0 and are counted
   as script tokens. For a sample with *p* pad tokens, the first *p* dialogue
   tokens collapse onto one position and the rest of the dialogue is shifted by
   *p*. Training batches are padded to a multiple of 8, so most batches contain
   affected samples; inference (batch of one, no padding) is not affected. The
   released checkpoint was trained this way, so the behaviour is kept;
   `tests/test_model_unit.py::test_right_padding` pins it. Passing explicit
   `channel_ids` that are non-zero on the padding gives unshifted positions.
2. **Not strict next-token-*pair* prediction.** The original NTPP pair mask
   (B<sub>t</sub> may not attend A<sub>t</sub>) and per-channel loss are not
   used: B<sub>t</sub> sees A<sub>t</sub>, and the loss is ordinary shift-by-one
   cross-entropy over the interleaved sequence. This suits scoring a fully
   specified schedule; free generation is sequential within a frame.
3. **Attention mask on the eager / SDPA paths.** The research code's
   `get_masks` was partly commented out and raised a shape error for any padded
   batch outside flash attention. It is restored to the stock GLM form (causal
   plus padding, boolean). Training with flash attention and unpadded inference
   never used this function, so no reported number depends on it.
4. **Channel-embedding dtype at construction.** The table is created in
   bfloat16 when the GPU supports it and float16 otherwise, independent of
   `config.torch_dtype`, and is then cast with the rest of the model. Copy
   weights *after* casting a freshly built model, or values are rounded to half
   precision.
5. **Parameters without an initialiser.** RMSNorm weights are allocated
   uninitialised and `_init_weights` is a no-op, as in GLM-4-Voice: a model
   built from a config alone is only meaningful after weights are loaded.
6. **`use_turn_embeddings` checkpoints** (the "logic clock" ablation) are
   rejected by `load_scheduler`; that variant is not part of this package. Its
   vocabulary also differs: id 168742 is `<|audio_16383|>` there and
   `<|start-turn-0|>` here.
7. **Which checkpoint.** checkpoint-15200 is the last checkpoint of the
   Seamless run (step 15200 of 22098, the job hit its time limit), not the one
   with the best validation loss (checkpoint-14700). All reported results use
   15200.

## Tests

`tests/test_model_equivalence.py` builds the research implementation and this
one on a tiny random configuration, copies the weights across with
`strict=True`, and requires bit-identical logits, duration logits, losses and
duration accuracy for: unpadded batches on every attention path, the batch-of-one
inference call, step-by-step decoding with the KV cache, and right-padded
batches (with the restored mask on both sides). It needs
`NTPP_ORIG=/path/to/NTPP`. `tests/test_model_unit.py` runs without it.
