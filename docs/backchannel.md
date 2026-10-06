# Backchannel classifier

The scheduler decides *when* things are said. Whether a turn should receive a
listener backchannel ("mm-hmm", "wow") at all, and of which kind, is decided
beforehand by a separate classifier. This page covers how its training data is
made, the models, and how the inference pipeline uses it.

Everything lives in `neural_scheduler/backchannel/`:

| Module | Purpose |
|---|---|
| `labeling/` | Label backchannels in Seamless Interaction transcripts with Qwen3-Omni |
| `data/prep.py` | Turn labelled interactions into classifier examples |
| `data/dataset.py`, `data/audio.py` | Datasets, collation, samplers, audio loading |
| `models/` | `JointBCClassifier` (main model), baselines, checkpoint factory |
| `train.py`, `eval.py`, `ablation.py` | Train / evaluate / ablate the joint model and the RoBERTa and LMF baselines |
| `train_gpt2_baseline.py` | GPT-2 late-fusion baseline |
| `train_text_only.py` | Text-only RoBERTa classifier |
| `stats.py` | Statistics of the labelled dataset |
| `predictor.py` | Wrappers used by pipeline step 1 |

## What counts as a backchannel

A backchannel is a short listener response that could be removed without
changing the meaning of the conversation. Operationally it is a transcript
fragment that

1. passes a rule pre-filter:
   - shorter than 0.8 s **or** at most two timestamped words;
   - at least 0.3 s away from any other fragment of the same speaker;
   - the fragment that starts immediately before it belongs to the other
     speaker and overlaps it or ended at most 2 s earlier;
   - not in the first or last 10 s of the conversation;
2. and is assigned one of six classes by Qwen3-Omni-30B-A3B-Instruct (text
   only, 4-bit, greedy): `Neutral`, `Positive`, `Emotional`, `Anticipation`,
   `Confirmation`, `Repetition`. `NotBackchannel` and unparsable answers leave
   the fragment labelled as ordinary speech.

The system prompt, with its eleven worked examples, is in
`labeling/prompt.py`. Its wording is part of the dataset definition.

## Step 1: labelling

```bash
python -m neural_scheduler.backchannel.labeling.extract \
    --dataset_root /path/to/seamless-interaction \
    --split train --dataset_type naturalistic \
    --output_dir /path/to/backchannel_v2 \
    --qwen_model_path /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --shard_id 0 --num_shards 72
```

Inputs are the corpus's own transcripts
(`V{vendor}_S{session}_I{interaction}_P{participant}.json`, key
`metadata:transcript`); no ASR is run and no audio is opened. Per interaction:

1. split each sentence wherever two words are ≥ 0.8 s apart;
2. select candidates with the pre-filter;
3. show the labeller the candidate with ±10 s of context (`A:` / `B:` lines,
   the candidate's speaker always `B`, candidate suffixed `<|response|>`);
4. validate the JSON answer (`labeling/validator.py`);
5. merge consecutive same-speaker fragments into segments and assign turn ids:
   a segment covered for ≥ 50 % of its duration by an earlier segment inherits
   that segment's id. This is what ties a backchannel to the turn it overlaps.

Output, one line per interaction in
`{output_dir}/{split}_{dataset_type}/shard{N}of{M}_interactions.jsonl`:

```
interaction_id, identifier_left, identifier_right, wav_left, wav_right,
transcript_left, transcript_right, split, dataset_type, sr (16000),
num_turns, num_sentences, num_backchannels,
turns: [{text, start, end, speaker_id (0 = left, 1 = right), turn_id, label}]
```

`label` is `"Speech"` or `"backchannel-<Class>"`. Failures go to
`shard{N}of{M}_failures.jsonl`. `--resume` appends and skips finished
interactions. `--relative_paths` stores paths relative to `--dataset_root`
(then pass `--audio_root` to the trainers). Needs a GPU, `bitsandbytes`,
`flash-attn` and a `transformers` release that includes Qwen3-Omni; this is
newer than the 4.46 the rest of the package is pinned to, so labelling runs in
its own environment. `scripts/slurm/backchannel_label.sh` runs all shards as a
job array (24 improvised-train, 72 naturalistic-train, 2 + 2 test).

Check a shard with

```bash
python -m neural_scheduler.backchannel.labeling.diagnose \
    --interactions_path /path/to/shard0of72_interactions.jsonl
```

and compute corpus statistics with
`python -m neural_scheduler.backchannel.stats --input_dir ... --output bc_stats.json`.

### Reference statistics

From a statistics run of the research labelling output (which split or subset
it covered was not recorded; treat the numbers as indicative):

| Quantity | Value |
|---|---|
| Interactions | 7,378 (mean length 196 s) |
| Backchannels / speech segments | 65,833 / 510,520 (11.4 % of segments) |
| Class shares | Positive 42.4 %, Neutral 34.3 %, Emotional 7.8 %, Confirmation 7.0 %, Repetition 4.7 %, Anticipation 3.8 % |
| Backchannel duration | mean 0.57 s, median 0.34 s |
| Backchannels per minute | mean 2.6, median 2.4 |
| Interval between a speaker's backchannels | median 12.6 s (p25 5.6 s, p75 29.6 s) |
| Backchannels per 10 s window | mean 0.47; 66 % of windows have none |
| Turns with a backchannel that have exactly one | 96.4 % |
| Position within the accompanied speech | median 26 % of the way through |

## Step 2: classification data

```bash
python -m neural_scheduler.backchannel.data.prep \
    --input_dir /path/to/backchannel_v2/train_naturalistic \
    --output_dir /path/to/bc_classification_data/train_naturalistic_with_audio
```

Segments with the same turn id form a *turn group*: the main speaker (most
speech in the group) plus what the listener did meanwhile. Each group is one
example. Its text is every turn group whose midpoint lies within ±10 s of the
target's midpoint, rendered one line per speaker and joined by ` </s> `:

```
B: my skin is sensitive [BC:Neutral] </s> A: so I get red blotches <bc_slot> </s> B: wait what
```

- the target's main speaker is `A`;
- the target line ends in `<bc_slot>` and its backchannel is hidden;
- a context group with exactly one backchannel carries `[BC:<Class>]`;
- the audio is the main speaker's own speech in the target group
  (`audio_span = {wav, start, end}` on that speaker's mono file), loaded at
  training time and capped at 15 s.

Not used as targets: groups with several backchannels, groups with less than
0.5 s of main speech. Negatives are down-sampled to 4 per positive.

Label schemes (`--label_scheme`):

| Scheme | Target label of a group with one backchannel |
|---|---|
| `semantic` (default) | the labeller's class |
| `duration` | `Neutral` if the backchannel lasts ≤ `--bc_duration_threshold` (0.32 s), else `Emotional`; adds a `bc_duration` field |

The classifier is then trained on one of three **phases**, chosen at training
time:

| Phase | Classes |
|---|---|
| `merged` (default) | `NoBackchannel`; `Acknowledgment` = Neutral + Positive; `Reactive` = Emotional + Anticipation + Confirmation. Repetition examples are dropped. |
| `binary` | `NoBackchannel`, `Backchannel` |
| `full` | `NoBackchannel` + the six classes |

With duration-scheme data the merged classes therefore mean "short" and "long"
backchannel. The pipeline expects a `merged` model.

Reference size of the research training set (semantic scheme, naturalistic
train): 878,925 examples (703,140 without and 175,785 with a backchannel);
11,310 validation examples.

## Models

**`JointBCClassifier`** — the model used by the pipeline.

- Text: RoBERTa-base, trainable, all token states.
- Audio: HuBERT-base (frozen) → learned softmax-weighted sum of its 13 hidden
  states → `Conv1d(kernel 8, stride 4)` (50 Hz → 12.5 Hz) → GELU → LayerNorm.
- Fusion: `[CLS ; text tokens ; audio frames]` + modality embedding → 2-layer
  pre-norm transformer (8 heads, FFN 3072) → CLS.
- Head: LayerNorm → Dropout → Linear.
- `text_only` / `audio_only` leave one modality out of the fusion sequence.

**Baselines** (`models/baselines.py`)

- `PaperFusionClassifier` / `GPT2PaperFusionClassifier`: late fusion after
  Wang et al., *Turn-Taking and Backchannel Prediction with Acoustic and Large
  Language Model Fusion* (ICASSP 2024). Mean-pooled frozen HuBERT → 256-d,
  concatenated with the last non-pad token of RoBERTa or GPT-2 → one linear
  layer. Recipe `opt1` trains the text encoder jointly; `opt2` first fine-tunes
  it alone (`--text_only`) and then freezes it.
- `MMF2FModel`: GPT-2 and HuBERT (both trainable) with low-rank multimodal
  fusion (Liu et al., 2018), rank 16.
- Text-only `RobertaForSequenceClassification` (`train_text_only.py`).

## Training, evaluation, ablation

```bash
# main model, hyper-parameters of the research run
python -m neural_scheduler.backchannel.train \
    --config configs/backchannel/joint_merged.yaml \
    --train_file train.jsonl --val_file val.jsonl --output_dir runs/joint_merged

python -m neural_scheduler.backchannel.eval \
    --checkpoint runs/joint_merged/best_model --val_file test.jsonl --output_dir runs/joint_merged/eval

python -m neural_scheduler.backchannel.ablation \
    --checkpoint runs/joint_merged/best_model --val_file test.jsonl \
    --output_json runs/joint_merged/ablation_results.json --num_samples 11000
```

- Loss: cross-entropy with class weights `sqrt(N / count)` normalised to mean 1
  (`--class_weight_method`); optional `--balance_nobc` sampler.
- AdamW, one learning rate, linear warm-up (6 %) and decay, gradient clipping,
  early stopping on validation macro-F1.
- `configs/backchannel/joint_merged.yaml`: batch 32, lr 2e-4, 10 epochs,
  patience 5. These are taken from the last research launch script; the
  arguments of the run that produced the pipeline checkpoint were not
  preserved.
- A run directory contains `best_model/` and `last_model/` (`model.pt`,
  `config.json`, `tokenizer/`), `best_model/best_metrics.json`, `tokenizer/`,
  `args.json`, `train.log`, `tensorboard/`.
- Reported metrics: accuracy, macro / weighted F1, precision, recall;
  backchannel-vs-none F1 / P / R; per-class and macro one-vs-rest AUC and EER.
- Ablations (inputs perturbed, model fixed): `no_audio`, `no_future_text`,
  `no_current_text`, `no_past_text`, `no_past_bc`, `audio_only`.

Baselines:

```bash
# RoBERTa late fusion, two stages
python -m neural_scheduler.backchannel.train --use_paper_fusion --text_only ... --output_dir runs/paper_s1
python -m neural_scheduler.backchannel.train --use_paper_fusion --paper_fusion_option opt2 \
    --text_model_checkpoint runs/paper_s1/best_model ... --output_dir runs/paper_s2

# GPT-2 late fusion (same two stages)
python -m neural_scheduler.backchannel.train_gpt2_baseline \
    --config configs/backchannel/gpt2_paper_fusion.yaml --text_only ... --output_dir runs/gpt2_s1

# low-rank multimodal fusion
python -m neural_scheduler.backchannel.train --use_mmf2f --phase binary ... --output_dir runs/mmf2f

# text-only RoBERTa (optionally a binary warm-up passed as --init_checkpoint)
python -m neural_scheduler.backchannel.train_text_only --phase merged ... --output_dir runs/text_merged
```

`eval` reads the architecture from `config.json`, so it works for all of them
except the text-only model, whose trainer reports its own validation metrics.
Slurm templates: `scripts/slurm/backchannel_{label,prep,train,eval}.sh`
(configured through environment variables; give `sbatch --output` a path
outside your home directory).

The only metrics that survive from the research runs are for the GPT-2
late-fusion baseline (binary, 15 s audio, batch 32, lr 5e-5): stage 1
(text only) best validation macro-F1 0.651 (accuracy 0.772, macro-AUC 0.724);
stage 2 (frozen text + audio) 0.650 before the job hit its time limit. Adding
audio did not help that baseline.

## Use in the inference pipeline

```python
from neural_scheduler.backchannel.predictor import BackchannelClassifier, MERGED_TO_CANONICAL_BC_TAG

clf = BackchannelClassifier("runs/joint_merged/best_model", device="cuda", confidence_threshold=0.5)
for i, turn in enumerate(segments):          # [{"speaker": 1|2, "text": ..., "audio_path": ...}]
    text = clf.build_input_text(segments, i, context_window_turns=10, context_window_future_turns=5)
    label, confidence, probs = clf.predict(text, audio=turn["audio_path"])
    turn["bc_class"] = MERGED_TO_CANONICAL_BC_TAG[label]   # feeds later turns' context
```

- Text: the target turn, up to N past and M future *turns*, one line each;
  the target's speaker is `A`. Past turns that received a backchannel carry
  `[BC:Neutral]` (after an `Acknowledgment` decision) or `[BC:Emotional]`
  (after `Reactive`).
- Audio: the target turn's voice line, mono 16 kHz, first 15 s. A path, an
  in-memory waveform with its sample rate, or `None`.
- Decision: `p_bc = P(Acknowledgment) + P(Reactive)`. If `p_bc ≥ threshold`
  the more probable of the two is returned, otherwise `NoBackchannel`. The
  research runs used thresholds between 0.2 and 0.5 with the joint model.
- The tokenizer is looked up in `tokenizer_dir`, `<model_dir>/tokenizer`, then
  `<model_dir>/../tokenizer`. Checkpoints written by the research training
  script load unchanged.
- `TextOnlyBackchannelClassifier` is the same interface for the text-only
  model (`predict(input_text)`).

`predictor.py` does not import scikit-learn or tensorboard.

## Migration from the research tree

| Research file (`BackchannelPrediction/`) | Release |
|---|---|
| `preprocess_scripts/extract_backchannel_dataset_sharded_v2.py` | `labeling/extract.py`, `discover.py`, `candidates.py`, `turns.py`, `prompt.py` |
| `preprocess_scripts/qwen_process_one_example_test_batch.py` | `labeling/qwen_labeler.py` |
| `preprocess_scripts/bc_validator.py` | `labeling/validator.py` |
| `preprocess_scripts/extract_backchannel_dataset_sharded_v2_diagnostic.py` | `labeling/diagnose.py` |
| `preprocess_scripts/extract_backchannel_dataset_sharded_v2.md` | this page |
| `preprocess_scripts/extract_backchannel_dataset_sharded_v2_submit.sh` | `scripts/slurm/backchannel_label.sh` |
| `naive_roberta_classifier/prep_classification_data.py` | `data/prep.py` |
| `naive_roberta_classifier/prep_classification_data_duration_based.py` | `data/prep.py --label_scheme duration` |
| `naive_roberta_classifier/train_classifier_3_class_joint_with_audio.py` | `models/joint.py`, `models/baselines.py`, `models/factory.py`, `data/dataset.py`, `data/audio.py`, `metrics.py`, `training.py`, `train.py`, `eval.py` |
| `naive_roberta_classifier/train_classifier_3_class_joint_with_audio_with_GPT2.py` | `train_gpt2_baseline.py`, `models.GPT2PaperFusionClassifier` |
| `naive_roberta_classifier/train_classifier_3_class.py`, `train_classifier.py` | `train_text_only.py` |
| `naive_roberta_classifier/ablation.py` | `ablation.py` |
| `naive_roberta_classifier/analyze_bc_stats.py` | `stats.py` |
| `naive_roberta_classifier/*.sh` | `scripts/slurm/backchannel_{prep,train,eval}.sh`, `configs/backchannel/*.yaml` |
| `BackchannelClassifier` in `step_1_no_tts_classifier_voiceline_given_with_audio.py` | `predictor.BackchannelClassifier` |
| `BackchannelClassifier` in `step_1_no_tts_classifier_voiceline_given.py` | `predictor.TextOnlyBackchannelClassifier` |

Not carried over: `extract_backchannel_dataset.py` and
`extract_backchannel_dataset_sharded.py` (earlier extraction formats),
`visualize_backchannel_dataset.py`, `visualize_BC_stats.py` (hard-coded
numbers), `submit_sharded.sh`, `run_classify_backchannel_dignostic.sh`.

Differences from the research code:

- `--dataset_root`, `--output_dir` and `--qwen_model_path` of the extraction
  have no default and must be given.
- `config.json` gains an `architecture` key and checkpoints also contain their
  tokenizer. Old checkpoints still load; a GPT-2 late-fusion config (which has
  no architecture flag) is now recognised by `eval`, `ablation` and the
  predictor.
- The predictor zero-pads audio shorter than 25 ms (including `audio=None`)
  to one HuBERT frame instead of failing inside HuBERT.
- File discovery no longer needs pandas.
- New options, all off by default: `--relative_paths` (extraction),
  `--audio_root` and `--strict` (training / evaluation), `--config`,
  `match_by_position` in `generate_prompt`.

Everything else computes the same thing: `tests/test_backchannel_*.py` compares
the labelling pipeline, both prep schemes (byte-identical files), datasets,
metrics, every architecture (state-dict keys, seeded initialisation, logits,
gradients) and full seeded training runs with the research scripts
(`NTPP_ORIG=/path/to/NTPP pytest -m equivalence tests/test_backchannel_*`).

## Known issues

These are properties of the method as it was run. They are kept so that the
release reproduces the research code.

**Data**

- *Validation is the test file.* The research runs used the same `test.jsonl`
  for early stopping and for the reported evaluation. Use separate files.
- *The test split is down-sampled too.* `prep` keeps 4 negatives per positive
  in every split, so accuracy and precision do not reflect the natural prior
  (most turn groups have no backchannel).
- *One file per `split` value.* Preparing `train_naturalistic` writes only
  `train.jsonl`; a held-out file needs a separate run on test interactions.
  How the research `test.jsonl` was produced is not recorded.
- *`interaction_id` is only unique within one (split, dataset_type)*, so ids
  collide when subsets are pooled.
- *Duration scheme.* Target labels reuse the names `Neutral` / `Emotional` for
  short / long, while context tags keep the semantic class.
  `label_id` then only takes the values 0, 1, 3. The two schemes must be
  written to different output directories.
- *Future backchannel tags in training text.* Context turn groups after the
  target carry their `[BC:*]` tag; at inference only past turns can.
- *Prompt marking.* `generate_prompt` marks every context line whose text
  equals the candidate's (for example a second "yeah"), not only the candidate.
  `match_by_position=True` marks only the candidate but changes the labelling.
- *Validator.* A `NotBackchannel` answer with an empty word is counted as
  invalid (harmless: both outcomes mean "not a backchannel"). The quote-repair
  fallback breaks words that contain an apostrophe. The JSON search is greedy
  and fails on two objects in one answer.
- *Prompt wording.* The user message says "Context Phrase / Classification
  target phrase", which the system prompt does not describe.
- *Stored paths are absolute* unless `--relative_paths` is used.

**Model and training**

- *Train / inference mismatch.* Training context is a ±10 s window of merged
  turn groups with all six tags; inference context is a number of script lines
  with only `Neutral` / `Emotional` tags on past turns. Training audio is
  natural speech; inference audio is usually synthesised.
- *One learning rate* for RoBERTa and the randomly initialised fusion layers
  (2e-4 in the provided config, ten times the argparse default).
- *Checkpoints include the frozen HuBERT weights* (about 95 M parameters).
- *Non-strict loading.* Missing and unexpected checkpoint keys are reported,
  not fatal, unless `--strict` is given.
- *Padding affects audio features.* HuBERT-base normalises with group norm over
  the padded waveform, so an example's logits depend slightly on how much it
  was padded in its batch.
- *`no_audio` ablation.* It zeroes the waveform and sets the length to one
  sample; HuBERT's own attention mask is not well defined for that length, so
  the ablation measures "silent audio", not "audio branch removed".
- *GPT-2 baseline run.* The reference configuration combines balanced sampling
  with loss weighting, which corrects for imbalance twice (the trainer warns).
- *Inference class names are fixed.* The decision rule reads the
  `Acknowledgment` and `Reactive` probabilities, so only `merged` models are
  usable in the pipeline.

**Artifacts**

- The research classifier checkpoints and the derived labelling and
  classification data were stored on scratch storage that has since been
  purged. No metrics of the final joint classifier survive. To use the
  classifier in the pipeline it has to be retrained: run labelling, prep and
  training as described above.
- It is not recorded whether the checkpoint used for the reported results was
  trained on the semantic or the duration label scheme; the checkpoint path
  used by the pipeline scripts corresponds to the semantic one.
