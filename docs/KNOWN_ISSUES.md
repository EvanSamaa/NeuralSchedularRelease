# Known issues

This code was restructured from a research tree under one rule: for the same
inputs it must compute what the research code computed, because the released
checkpoint and the reported results depend on it. Behaviour that looks wrong
but affects numbers is therefore kept by default and listed here. Where a fix
was cheap it exists as an option that is off by default.

Each area's page has the full list with details; this page collects the ones
that matter most.

## Model and training — [model.md](model.md#known-issues), [training.md](training.md#known-issues)

| Issue | Status |
|---|---|
| Rotary positions shift when a sample is right-padded: pad positions are counted as script tokens, so the dialogue's positions move by the number of pads. Training batches are padded; inference is not. | Kept. The released checkpoint was trained this way. |
| `--lr_scheduler_type` has no effect with the shipped DeepSpeed config: its `WarmupLR` block owns the schedule, so the rate is constant after warm-up. | Kept (this is how the checkpoint was trained). Remove the `scheduler` block from the JSON to use the Trainer's schedule. |
| The model is not strict next-token-pair prediction: speaker B at frame t attends to speaker A at frame t, and the loss is ordinary shift-by-one cross-entropy. | By design. |
| `generate()` is not supported. | Raises `NotImplementedError`; the pipeline scores candidates with `forward`. |
| Train-time duration accuracy compares against unclamped labels and so understates accuracy for runs longer than 31 frames. | Kept. |

## Data — [data.md](data.md#known-issues)

| Issue | Status |
|---|---|
| Speech code 16383 has no token in the vocabulary; a window containing it is dropped (1.7 % of Fisher windows in the released data, none for Seamless). | Kept; dropped windows are now counted and logged. `--add_audio_16383_token` keeps them (the trainer's tokenizer needs the same token). |
| Voice-activity boundaries depend on the Silero VAD release. | The release used for the published data is pinned; override with `SILERO_VAD_REPO`. |
| Turns with id ≥ 50 are left out of the script while their speech stays in the dialogue. | Kept. |
| Seamless `source_file` ids are unique only within one subset. | Kept; `--source_id_style name` gives unique ids. |
| Resume skips a conversation that was interrupted half-way. | Kept; `--resume_per_window` completes it. |

## Scheduler search — [scheduler.md](scheduler.md#9-known-issues)

| Issue | Status |
|---|---|
| Pass 3 mis-splits a host turn that already contains inserted silence (from a pass-2 pause or sub-turn scheduling): the filler is spread over the sub-segments and the gap is added again. | Kept. |
| The channel with the longer script duration is taken as the primary turn, so a backchannel longer than its host swaps roles with it. | Kept. |
| Pauses inside a turn are removed unless `--pass_1_schedule_subturns` is set. | Kept. |
| Scoring windows clipped at the start of the dialogue contain fewer tokens, favouring early onsets for the first turn. | Kept. |
| Function defaults and command-line defaults differ for `pass2_scoring_context` and `pass2_bc_effect`. | Kept and documented. |
| One forward pass per candidate, no batching or prefix reuse. | Kept; scheduling a scene takes many model calls. |

## Pipeline steps 1–2 — [pipeline.md](pipeline.md#known-issues-steps-1-and-2)

| Issue | Status |
|---|---|
| Backchannel audio seeds advance for speaker 1 only. | Kept. |
| Old backchannel clips stay in `audio_segments/` between runs. | Kept; `--clean_cache` removes them. |
| Step 2 keeps turn markers for ids below 20 only; the search and the model know 50. | Kept; `--max_turns`. |

## Backchannel classifier — [backchannel.md](backchannel.md#known-issues)

| Issue | Status |
|---|---|
| Training text uses ±10 s of context with past *and future* backchannel tags from six classes; inference uses a number of turns with past-only tags from two classes. Training audio is natural speech, inference audio is synthetic. | Kept. |
| Validation and test used the same file, and its negatives are down-sampled 4:1. | Kept; reported accuracy does not reflect the natural class prior. |
| The same class names mean different things under the semantic and the duration label scheme. | `--label_scheme` selects one explicitly. |

## Evaluation — [evaluation.md](evaluation.md#known-issues)

| Issue | Status |
|---|---|
| The research runs drew their evaluation records under an implicit seed of 1986 (set as a side effect of loading an unused module). | `--seed 1986` reproduces the draw; without `--seed` runs are unseeded. |
| Silence-perturbation scoring used channel ids 0/0/1 instead of the trained 0/1/2, and stores summed NLL under `perplexities` and perplexity under `nlls`. | Kept by default; `--corrected_scoring` fixes both. Results of the two modes are not comparable. |
| The Full-Duplex-Bench adapter decides turn-over without ASR (any voiced segment over 3 s), skips missing samples, and runs VAD on 22.05 kHz audio as if it were 16 kHz. | Kept by default so numbers match; `--missing silent` and `--resample_for_vad` change it. Scores are not protocol-identical to the upstream benchmark. |

## Analysis — [analysis.md](analysis.md#known-issues)

| Issue | Status |
|---|---|
| A/B side assignment for listening-test stimuli was unseeded. | `--seed` available; the released keys record what was used. |
| Aggregate win rates treat respondent × item trials as independent. | Kept. |

## Not verified in this repository

The restructuring was checked on CPU against the research implementation
(see `tests/`). The following were not exercised: training with
flash-attention / bf16 / DeepSpeed on GPUs and multi-node launch; Chatterbox
synthesis; the backchannel classifier end to end with a trained checkpoint;
the full pipeline and the evaluations on the released checkpoint on a GPU;
the Slurm templates beyond a syntax check.
