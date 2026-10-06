# Evaluation

Everything under `neural_scheduler/evaluation/` scores a trained scheduler
checkpoint. Each module is a library plus a command line
(`python -m neural_scheduler.evaluation.<module> --help`).

| Module | Question | Output |
|---|---|---|
| `timing_distribution` | Do gaps and overlaps in sampled dialogue have the same duration distribution as real dialogue? | 3×3 figure, single-panel figure, JSON |
| `turn_order` | Does the model prefer the true speaking order over a swapped or simultaneous one? | JSON, two figures |
| `silence_perturbation` | Does the model notice when a pause is lengthened or shortened? | JSON and figure per run |
| `nll_diagnostics` | Where in a real dialogue is the model surprised (regions, onsets, offsets)? | interactive HTML, optional CSV |
| `onset_benchmark` | Is the true onset of a segment the NLL minimum when the segment is shifted in time? | two interactive HTML pages |
| `fdb.prepare_icc`, `fdb.eval_backchannel` | Full-Duplex-Bench backchannel metrics for the backchannel-only pipeline | JSONL / split audio; JSD, TOR, frequency |

Shared code lives in `common.py` (data loading, context construction, channel
ids, NLL scoring, sampling) and `segments.py` (speech-segment finder).

## Inputs

* **Checkpoint**: a scheduler checkpoint directory, loaded with
  `neural_scheduler.model.load_scheduler` (half precision on GPU).
* **Data**: a tokenized JSONL as produced by preprocessing; only `input_ids`
  is read (`source_file`, when present, labels sequences in the NLL pages).
  The reported runs used the Seamless Interaction validation file
  `audio_tokens_val_with_durations_and_turnID_as_tokens.jsonl` (9,700 windows).
* All tools need `--out_dir`; nothing is written to the working directory.
* Plots are headless-safe; set `MPLBACKEND=Agg` on a cluster.

### Seeds

All four GPU evaluations draw records at random from `--data_jsonl`.
`--seed N` makes the draw reproducible; without it Python's global RNG is used.

The research scripts never set a seed themselves, but they constructed a
GLM-4-Voice audio decoder (which they did not use) before drawing records, and
that decoder's `config.yaml` seeds `random`, `numpy` and `torch` with **1986**.
The published runs were therefore drawn with seed 1986:
`--seed 1986` reproduces their record draw exactly. This was checked against
the three saved timing-distribution result sets below: the 99 ground-truth
dialogues are identical, value for value.

For `timing_distribution`, `--seed` also seeds the token sampler (after the
model is loaded, as in the research runs). Whether this reproduces the sampled
dialogues of the published runs bit for bit has not been verified; it needs a
GPU run.

## Gap and overlap distributions (`timing_distribution`)

For each record, the two scripts are given to the model and a dialogue is
sampled token by token (512 new tokens, temperature 1.0, top-k 50, top-p 0.95).
Runs of mutual silence (gaps) and of simultaneous speech (overlaps) are
extracted from the sampled and from the ground-truth dialogue. Reported:

* two-sample KS test on the pooled run lengths (gaps and overlaps separately);
* t-test on per-dialogue mean run length;
* share of frames that are mutual silence / overlap / single speaker.

```bash
python -m neural_scheduler.evaluation.timing_distribution \
    --checkpoint $CKPT --data_jsonl $VAL_JSONL \
    --out_dir $OUT --name with_duration_loss --num_samples 100 --seed 1986

# re-plot from the saved JSON, no GPU needed
python -m neural_scheduler.evaluation.timing_distribution \
    --plot_only --out_dir $OUT --name with_duration_loss --num_bins 25 --x_limit_ms 2000
```

`--pool_size` (200) records are drawn and the first `--num_samples` are used.
Outputs in `$OUT/<name>/`: `<name>.png` (histograms, CDFs, ratio box plots,
composition, summary), `<name>_distribution_only.png` (300 dpi, signed axis:
overlap negative, gap positive, 80 ms per token) and `<name>.json`.

A record is skipped when its dialogue has fewer than 10 tokens, when either
sampled channel has fewer than 10 tokens, or on any error. A record whose
generation is skipped still counts in the ground-truth set.

### Reference results

Saved in the research tree under
`experiments/mutual_silence_experiment/<name>/<name>.json`; 99 ground-truth and
99 generated dialogues each, run lengths in tokens.

| Run | Gap KS D (p) | Overlap KS D (p) | Mean gap GT / gen | Mean overlap GT / gen | Gap share GT / gen | Overlap share GT / gen | Single-speaker share GT / gen |
|---|---|---|---|---|---|---|---|
| `dialogue_timing_comparison_with_duration_loss` | 0.093 (0.068) | 0.038 (0.96) | 11.48 / 18.09 | 7.70 / 7.80 | 0.201 / 0.262 | 0.103 / 0.101 | 0.696 / 0.637 |
| `dialogue_timing_comparison_no_duration_loss` | 0.074 (0.59) | 0.716 (1.7e-43) | 11.48 / 15.72 | 7.70 / 84.35 | 0.201 / 0.122 | 0.103 / 0.505 | 0.696 / 0.372 |
| `dialogue_timing_comparison_no_duration_loss_dtw_ver3` | 0.451 (3.3e-36) | 0.171 (0.021) | 11.48 / 36.65 | 7.70 / 11.65 | 0.201 / 0.549 | 0.103 / 0.046 | 0.696 / 0.406 |

All three used the validation file named above regardless of which data format
the checkpoint was trained on; for the two checkpoints without the duration
head that means scripts with turn-marker tokens they were not trained with.
Which checkpoint produced `..._with_duration_loss` is not recorded in its JSON.

## Turn-order perturbation (`turn_order`)

In a ground-truth dialogue, pairs of speech segments from different channels
are found where the second starts within `--max_gap` tokens of the first
ending (overlap allowed). For each selected pair the dialogue is truncated one
token after the pair and scored three ways: recorded order, the two segments
swapped, both starting together. Reported per pair and aggregated: perplexity
of each variant, ratios to the recorded order, and which variant scored best.

```bash
# synthetic question/answer set, where the order is unambiguous
python -m neural_scheduler.evaluation.turn_order \
    --checkpoint $CKPT --data_jsonl $WEBQUESTIONS_JSONL --out_dir $OUT/turn_order \
    --num_samples 1500 --seed 1986 --max_gap 20 --num_pairs_per_sample 10 --min_segment_length 0

# real conversations
python -m neural_scheduler.evaluation.turn_order \
    --checkpoint $CKPT --data_jsonl $VAL_JSONL --out_dir $OUT/turn_order_val --num_samples 200 --seed 1986
```

Outputs: `turn_order_results.json`, `turn_order.png`,
`turn_order_by_speaker.png`. Pair selection uses `--pair_seed + sample index`
(default 42, as in the research code).

The synthetic set is made by synthesising WebQuestions question/answer pairs
with two TTS voices (`neural_scheduler.analysis.webquestions_tts`) and
tokenizing the result with the preprocessing pipeline. The research figures
are `turn_order_fixed_input_script.png` (synthetic set, 1,500 records,
checkpoint `fisher_and_seamless_timing_duration_predction/checkpoint-10500`)
and `turn_order_with_turn_id.png` (validation set, 200 records, checkpoint
`seamless_timing_duration_predction_turn_ID_as_token/checkpoint-7900`); their
numbers were not saved, only the figures.

## Silence perturbation (`silence_perturbation`)

A speech segment is chosen in one channel, the silence before it (or after,
`--perturb_location after`) is changed by each delta in a range, the dialogue
is truncated one token after the segment, and each variant is scored. The
curve of score against delta should have its minimum at 0.

```bash
python -m neural_scheduler.evaluation.silence_perturbation \
    --checkpoint $CKPT --data_jsonl $VAL_JSONL --out_dir $OUT/silence --num_samples 200 --seed 1986
```

Three runs, selectable with `--runs`:

| Run | Segments | Deltas (tokens) | Files |
|---|---|---|---|
| `all` | any length | `--delta_min` .. `--delta_max`−1 (−15 .. 14) | `multi_sample_aggregate.{json,png}` |
| `long` | length ≥ `--long_min_length` (30) | −10 .. 20 | `multi_sample_aggregate_long_segments.{json,png}` |
| `short` | length ≤ `--short_max_length` (15) | −10 .. 20 | `multi_sample_aggregate_short_segments.{json,png}` |

Each uses `--num_segments_per_sample` (8) segments per record, chosen with
`--segment_seed + sample index` (42). The script region given to the model is
the record's script with turn markers removed and silence runs of length 2 or
4 (the segment delimiters of the older data format) collapsed to one token.

### Scoring compatibility

The research code scored this experiment with two quirks. Both are reproduced
by default so that curves stay comparable:

1. channel ids were 0 for the script **and** dialogue A and 1 for dialogue B,
   not the 0 / 1 / 2 layout the model is trained with;
2. the scorer's two return values were swapped when unpacked, so the result
   field `perplexities` (and `original_perplexity`) holds the **summed NLL**
   of the truncated dialogue and `nlls` holds its perplexity. The plots, which
   normalise `perplexities` by the value at delta 0, therefore show NLL ratios.

`--corrected_scoring` uses the trained channel layout and stores each quantity
under its own name. Results from the two modes are not comparable.

In the research tree this experiment was last run in January 2026 on an early
checkpoint (the three PNGs were written to the working directory); it was
switched off in the script afterwards. No numbers were saved.

## NLL diagnostics and onset benchmark (`nll_diagnostics`, `onset_benchmark`)

```bash
python -m neural_scheduler.evaluation.nll_diagnostics \
    --checkpoint $CKPT --data_jsonl $VAL_JSONL --out_dir $OUT/nll \
    --pool_size 200 --num_sequences 20 --seed 1986 --export_csv --onset_benchmark
```

* `nll_diagnostics.html`: per-token NLL over time per channel, NLL by region
  (mutual silence, single speaker A/B, overlap), NLL around speech onsets, NLL
  against preceding silence length and against position inside a segment.
  `--export_csv` adds `nll_events.csv` with one row per dialogue token.
* `--onset_benchmark` additionally shifts up to `--max_segments_per_seq` (5)
  segments per sequence by ±`--shift_range` (20) steps and scores a window of
  `--window_padding` (15) steps around each: `onset_benchmark.html` (one chart
  per segment) and `aggregate_benchmark.html` (all segments together).

The HTML pages are self-contained except for Chart.js, loaded from a CDN.
These are diagnostics; the research tree holds one set of pages
(`experiments/experiment_NLL/`, checkpoint-15200, first 20 of 200 records).

## Full-Duplex-Bench backchannel task (`fdb/`)

The benchmark ([Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench),
CC BY-NC 4.0) and its data are **not** part of this repository. The research
results used commit `d7336bd` and the v1.0 `icc_backchannel` subset (55
examples; each `<id>/input.wav` is the speaker and `<id>/transcription.json`
holds word timestamps with listener backchannels as `[TAG]` words).

### Flow

1. Download the `icc_backchannel` subset from the benchmark's data release,
   and take `icc_gt_distribution.json` from `v1_v1.5/evaluation/` of the
   benchmark repository (identical to the `all_data_distribution.json` shipped
   with the data).
2. Prepare it:

   ```bash
   python -m neural_scheduler.evaluation.fdb.prepare_icc \
       --dataset_dir $FDB/v1_v1.5/dataset/icc_backchannel --out_dir $OUT/icc_prepared
   ```

   This writes `$OUT/icc_prepared/<id>/split_audio/NNN.wav`, `<id>/meta.json`
   and `icc_backchannel_bc_gt.jsonl`. Words are split into sentences at pauses
   of `--pause_threshold` (0.6 s); sentences with ≤ `--min_words` (3) words or
   shorter than `--min_duration` (2.0 s) are merged into the next (else the
   previous) one; each clip gets `--padding_sec` (0.15 s) on both sides. Every
   tag becomes a ground-truth backchannel event on a sentence.

   Paths in the JSONL are relative to the JSONL's directory and paths in
   `meta.json` to the example directory; `--absolute_paths` stores absolute
   ones. `--in_place` writes into the benchmark directory instead, which is
   what the research script did.
3. Run the backchannel-only pipeline on the JSONL (steps 1–3, see
   `docs/pipeline.md`). Its output is
   `<output_root>/<model>/sample_<id>/segment_search_v2/channel_b.wav`, where
   channel B is the listener and `<id>` is the benchmark id.
4. Score:

   ```bash
   python -m neural_scheduler.evaluation.fdb.eval_backchannel \
       --pred_root <output_root>/<model> --gt_distribution icc_gt_distribution.json \
       --output_json $OUT/fdb_backchannel.json
   ```

   For a baseline stored in the benchmark's own layout (`<id>/output.wav`) add
   `--layout fdb`.

Metrics: **JSD**, the Jensen-Shannon distance between the predicted
backchannel timing histogram (0.2 s bins) and the ground-truth distribution
resampled to the same length, 1 when nothing is predicted; **TOR**, 1 if the
listener took the turn; **frequency**, backchannels per second. Means and
standard deviations over samples are printed and written to `--output_json`.

### Deviations from the upstream protocol

`eval_backchannel` reproduces the locally modified script the research results
were computed with, not upstream Full-Duplex-Bench. Scores are **not**
protocol-identical to published benchmark numbers:

* **No ASR.** Upstream transcribes the output and decides turn-over from the
  words in each segment. Here a VAD segment longer than 3 s sets TOR = 1 and
  every segment of 3 s or less counts as one backchannel.
* **Missing predictions are skipped** and drop out of the means
  (`--missing skip`). Upstream raises an error. `--missing silent` scores a
  missing sample as a silent listener (JSD 1, TOR 0, frequency 0).
* **Sampling rate.** The VAD is called without the file's sampling rate and
  assumes 16 kHz, while pipeline output is 22.05 kHz, so segment times (and the
  3 s threshold) are stretched by 22050/16000. `--resample_for_vad` resamples
  to 16 kHz first; it changes the scores.

Sample coverage of the research runs (55 examples): the `v4_min_1`, `long_bc`
and `longer_bc` variants scored 54 samples (`sample_11` never reached
tokenization) and the `friendly` variant 53. State the coverage with any
number you report.

How the PersonaPlex baseline was scored in the research is not recorded; with
this script it would be `--layout fdb` on its `<id>/output.wav` files.

## Slurm templates

`scripts/slurm/eval_timing_distribution.sh` and
`scripts/slurm/eval_perturbation.sh` wrap the GPU evaluations. They read
`CHECKPOINT`, `DATA_JSONL` and `OUT_DIR` from the environment (plus optional
`ENV_SETUP`, a script to source for the Python environment) and contain no
site-specific paths; pass `--output`, account and partition to `sbatch`.

## Migration from the research tree

Paths are relative to `ntpp/model/chatGLM_dual_timing_only/experiments/`.

| Research file | Here |
|---|---|
| `experiment_with_mutual_silence.py` | `evaluation/timing_distribution.py`; shared helpers in `evaluation/common.py` |
| `varying_order_helpers.py` | `evaluation/turn_order.py`, `evaluation/segments.py` |
| `experiment_turnunderstanding/run_turn_perturbation_experiment.py` (driver block) | `python -m neural_scheduler.evaluation.turn_order` |
| `experiment_with_perplexity.py`, `TEST_SENTENCE_REORDERING` block | same, with the validation JSONL |
| `varying_silence_helpers.py` | `evaluation/silence_perturbation.py`, `evaluation/segments.py` |
| `experiment_with_perplexity.py`, `TEST_SILENCE_PERTURBATION_EXPERIMENT_MULTI_SAMPLE` block | `python -m neural_scheduler.evaluation.silence_perturbation` |
| `nll_diagnostics.py` | `evaluation/nll_diagnostics.py` |
| `nll_diagnostics_2.py` (two modules in one file) | `evaluation/onset_benchmark.py` |
| `nll_diagnostic_2_template.html` | `evaluation/templates/aggregate_benchmark.html` |
| `experiment_with_nll.py` (driver block) | `python -m neural_scheduler.evaluation.nll_diagnostics` |
| `../dtw_module.py`: `prepare_inference_input`, `compute_full_nll`, `remove_duplicated_silence_tokens`, `compute_channel_ids_for_alignment` | `evaluation/common.py` |
| `prepare_inference_input`, `extract_script_contents`, `score_sequences`, `generate_dialogue`, `_separate_channels`, `load_random_shard_examples`, `compute_channel_ids_for_inference`, `get_input_ID_object_attention_mask_and_channel_id` (15–17 copies) | one copy each in `evaluation/common.py` (the last three are `load_jsonl_examples`, `compute_channel_ids`, `build_scoring_inputs`) |
| `load_corrupted_checkpoint` | `neural_scheduler.model.load_scheduler` |
| `inference_pipeline_bc_only_full_duplex_bench/eval_full_duplex_bench_prep_data.py` | `evaluation/fdb/prepare_icc.py` |
| `Full-Duplex-Bench/v1_v1.5/evaluation/eval_backchannel.py` (local modification) | `evaluation/fdb/eval_backchannel.py` |

Not carried over:

| Research file | Reason |
|---|---|
| `experiment_with_inferencing.py` | broken (undefined variable); superseded by the pipeline |
| `experiment_with_perplexity.py` (rest), `experiment_with_nll.py` (rest) | copies of the helpers plus disabled probes |
| `experiment_turnunderstanding/run_inferencing_naive*.py`, `generate_turnbased_speech_dia.py` | exploratory generation scripts |
| `experiment_turnunderstanding/generate_token_dataset.py` | duplicates preprocessing |
| `dtw_module.py` alignment search (`greedy_alignment_search`, `beam_alignment_search`, `align_dialogue_channels`, …) | used only by superseded pipelines |
| `run_mutual_silence_experiment`, `compare_mutual_silence_distributions` | older gap-only versions of the timing experiment |
| `inference_pipeline_bc_only_full_duplex_bench/visualize_gt_distribution.py` | one-off sanity plot of benchmark audio |
| audio decoder set-up, `get_wav_from_tokens`, `get_cosyvoice_mel` | never used by the evaluations |

Interface changes: every tool takes explicit output locations; the timing
experiment takes `--out_dir` and `--name` instead of a `save_path` whose stem
named a folder; `compute_full_nll` and `compute_truncated_perplexity` no
longer take the tokenizer argument they never used; plots are closed instead
of shown.

## Known issues

* **Implicit seed 1986** in the research runs (see Seeds). The release makes
  it explicit; default behaviour without `--seed` is an unseeded draw.
* **Silence perturbation scoring** (channel ids and swapped metric names), kept
  by default, see above.
* **`score_sequences`** averages over script tokens as well as dialogue tokens.
  None of the kept command lines call it; it is provided for parity with the
  research helpers.
* **Fixed validation file.** The timing reference runs evaluated every
  checkpoint on the turn-marker validation set, including checkpoints trained
  without turn markers.
* **Independence.** KS tests pool run lengths across dialogues, and t-tests
  use per-dialogue means of dialogues drawn from overlapping conversations;
  p-values are indicative.
* **Sampling cost.** `generate_dialogue` re-encodes the whole sequence for
  every new token (no key/value cache), as the research code did.
* **Identical distributions** give `nan` from `scipy`'s Jensen-Shannon
  distance through rounding; this can only happen for a prediction that
  matches the resampled ground truth exactly.
* **Full-Duplex-Bench protocol deviations** and sample coverage, see above.

## Tests

`tests/test_evaluation_*.py` run on CPU with a stub model. Tests marked
`equivalence` import the research implementations and require identical
results (segment extraction, perturbation builders, per-pair and aggregate
experiment outputs, per-token NLL, sampling under a fixed seed, HTML and CSV
output, ICC preparation byte for byte, backchannel metrics against the
modified benchmark script); they run when `NTPP_ORIG` points at the research
tree and are skipped otherwise.
