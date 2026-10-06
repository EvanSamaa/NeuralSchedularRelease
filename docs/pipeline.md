# Inference pipeline

The pipeline takes a two-speaker dialogue whose content is fixed (text to be
synthesised, or recorded voice lines) and produces dual-channel audio in which
the scheduler model has decided when every turn and every listener backchannel
happens.

```
dialogue (text JSONL, or JSON with voice lines)
 └─ step 1  synthesize     backchannel decisions, turn audio, backchannel audio
 │                         → <out>/segment_cache/<dialogue_id>/
 └─ step 2  tokenize       speech tokens + VAD → script tokens and a source map
 │                         → <cache>/source_map/, tokenized_result.jsonl
 └─ step 3  schedule       three-pass search under the scheduler model, then
                           reconstruction from the original audio
                           → channel_a.wav, channel_b.wav, channel_sum.wav, timeline.json
```

Steps 1 and 2 are documented here; step 3 (the search and its options) is
documented in `docs/scheduler.md`. This page also covers the runner that
chains the three steps from one YAML file.

## Requirements

| What | Needed by | How it is found |
|---|---|---|
| Chatterbox TTS (`chatterbox-tts`) | step 1 | Python package |
| Backchannel classifier checkpoint | step 1, unless `classifier.type: none` | `classifier.dir` / `--bc_classifier_dir` |
| Two reference voice recordings | step 1 | `tts.ref_s1`, `tts.ref_s2` |
| GLM-4-Voice repository (speech tokenizer code) | step 2 | `GLM4VOICE_REPO` environment variable |
| `THUDM/glm-4-voice-tokenizer`, `THUDM/glm-4-voice-9b` (tokenizer files) | step 2 | Hugging Face cache (`HF_HOME`) |
| Silero VAD | step 2; step 1 naive baseline | torch hub; release pinned in `neural_scheduler/data/vad.py`, override with `SILERO_VAD_REPO` |
| Scheduler checkpoint | step 3 | `scheduler.checkpoint` |
| `ffmpeg` | step 3 video | `PATH` |

**Two Python environments.** Chatterbox pins different package versions
(notably `numpy`) than the rest of the code, so step 1 normally runs in its own
environment. The runner supports that per step (see [Environments](#environments)).
The `neural_scheduler` package must be importable in both (`pip install -e .`,
or add the repository to `PYTHONPATH` — append, do not replace an existing
value).

## Quick start

```bash
# 1. copy a config and fill in the REQUIRED paths (two reference voices, scheduler checkpoint)
cp configs/pipeline/tts_no_classifier.yaml my_run.yaml

# 2. look at the commands it would run
python -m neural_scheduler.pipeline.run --config my_run.yaml --dry_run

# 3. run all three steps
python -m neural_scheduler.pipeline.run --config my_run.yaml
```

`tts_no_classifier.yaml` schedules turn-taking only and needs no backchannel
classifier. The other configs add a classifier, voice lines or the benchmark
set-up; each lists the paths it requires. Single values can be changed on the
command line:

```bash
python -m neural_scheduler.pipeline.run --config my_run.yaml \
    --set io.input_jsonl=examples/scenes/everyday_scenes.jsonl io.out_dir=outputs/everyday
```

Results for dialogue `N` are in
`<out_dir>[/<variant>]/<checkpoint parent dir>_<checkpoint dir>/sample_N/segment_search_v2/`.

### Shipped configs

| Config | Scenario | Mirrors (research tree, `experiments/`) |
|---|---|---|
| `tts_no_classifier.yaml` | text → TTS, turn-taking only, no classifier needed | – (scheduler settings of the next one) |
| `tts_turn_taking.yaml` | text → TTS, turn-taking only (classifier loaded, threshold 1.0) | `inference_qualitative_examples/run_pipeline_standard_run.sh` |
| `tts_with_backchannel.yaml` | text → TTS, classifier backchannels, long turn split into sub-turns | `inference_pipeline_10_examples/tts_psychiatrist/run_pipeline_standard_run_with_bc_3_seg.sh` |
| `voice_lines.yaml` | recorded voice lines, turn-taking only | `inference_pipeline_10_examples/napster/run_pipeline.sh` (parameters) |
| `strong_overlap.yaml` | every speaker change overlaps the previous turn | `inference_pipeline_10_examples/…/run_pipeline_extreme_overlap.sh` (parameters) |
| `fdb_backchannel.yaml` | Full-Duplex-Bench backchannel task (`bc_only` mode) | `inference_pipeline_bc_only_full_duplex_bench/run_step{1_classifier_with_audio,2,3}_longer.sh` |

## Input formats

### Text dialogues (`--input_jsonl`)

One dialogue per line:

```json
{"text": "[S1] Is the new model trained yet\n[S2] Yes but the loss is still high [S2] I will look at it now"}
```

- `[S1]` / `[S2]` start a turn. Consecutive turns of one speaker are kept as
  separate turns: each is scheduled on its own and can host its own
  backchannel. This is how a long utterance is split into "sub-turns".
- A newline inside a turn separates sentences; they are synthesised one by one
  and joined with `intra_turn_pause_ms` of silence.
- Other keys on the line are ignored. Dialogue `i` gets the id `example_000i`.

### Text dialogues with their own backchannel words (`--bc_vocab_jsonl`)

```json
{"text": "[S1] ...", "bc_vocab": ["uhhmm", "for sure"]}
```

Used without a classifier: every turn receives a backchannel with probability
`--bc_probability`, drawn from the line's `bc_vocab` (or `--bc_vocab_subset`).

### Voice lines (`--input_dialogue_json`)

One JSON object:

```json
{
  "dialogue_id": "bike_shop",
  "turns": [
    {"turn_id": 0, "speaker": 1, "text": "Hi, I dropped my bike off on Monday.", "audio_path": "audio/turn0_s1.wav"},
    {"turn_id": 1, "speaker": 2, "text": "Let me have a look.", "audio_path": "audio/turn1_s2.wav"}
  ]
}
```

- `turn_id` (int), `speaker` (1 or 2), `text` and `audio_path` are required.
  `dialogue_id` defaults to the file name.
- A relative `audio_path` is resolved against the JSON file.
- Voice lines may have any sample rate and channel count; they are resampled to
  the TTS rate and mixed to mono.
- `text` is only used by the backchannel classifier.

### Benchmark records (`--input_bc_gt_jsonl`)

One JSON object per line, written by the Full-Duplex-Bench preparation script:

```json
{"dialogue_id": "12",
 "turns": [{"turn_id": 0, "speaker": 1, "text": "...", "audio_path": "...",
            "slice_start_sec": 0.0, "slice_end_sec": 6.4}],
 "tags_removed": 3,
 "gt_bc_per_turn": [{"turn_id": 0, "bc_count": 1, "bc_events": [{"tag": "...", "timestamp": 4.1}]}]}
```

`dialogue_id` must be an integer string (it is added to the seed).
`slice_*_sec` locate the voice line inside the full recording; step 2 in
benchmark mode needs them.

### Backchannel vocabulary (`--bc_vocab_per_class`)

```json
{"Acknowledgment": ["uh-huh", "yeah", "right"], "Reactive": ["wow", "nice"]}
```

The classifier predicts a class; the word is drawn from that class's list.

## Step 1 — `python -m neural_scheduler.pipeline.synthesize`

For each dialogue:

1. **Backchannel decisions.** For every turn the classifier sees the text of
   the surrounding turns (the target turn's speaker is `A`, the listener `B`,
   earlier turns carry the tag of the backchannel they received, the target
   line ends in `<bc_slot>`) and, for the joint model, the turn's audio. A
   backchannel is placed when `P(Acknowledgment) + P(Reactive) >=
   --bc_confidence_threshold`; its class is the more probable of the two.
2. **Word choice.** A word of that class is drawn at random, avoiding the word
   used for the previous backchannel when the class has another
   (`--bc_word_selection avoid_repeat`).
3. **Voice-line input only:** a backchannel on a turn whose same-speaker
   predecessor already has one is dropped (`--no_suppress_consecutive_bcs`
   keeps it), and if no turn ended up with a backchannel one Acknowledgment is
   forced on the most likely turn (`--no-ensure_at_least_one_bc` disables this).
4. **Audio.** Voice lines are loaded and padded with `voice_line_padding_ms`
   of silence on both sides; text turns are synthesised and padded with
   `bc_silence_padding_ms`. Each backchannel is synthesised in the *listener's*
   voice and padded with `bc_silence_padding_ms`. The padding keeps the speech
   tokenizer away from clip boundaries; step 2 removes it again with VAD.

### Cache layout

```
<out_dir>/segment_cache/<dialogue_id>/
  manifest.json
  bc_decisions.json          (not written without a classifier on text input)
  turn_metadata.json
  audio_segments/seg_0000_s1_turn0_speech.wav, seg_0003_s1_turn1_bc_yeah.wav, ...
  combined_mono.wav, combined_dual.wav      turns back to back (naive layout)
  naive/                     with --run_naive_only: the same layout without backchannels
  bc_eval.json               benchmark input only
<out_dir>/bc_eval_summary.json               benchmark input only
```

`manifest.json`:

```json
{"speaker_1": [
   {"filename": "seg_0000_s1_turn0_speech.wav", "turn_id": 0, "type": "speech",
    "source_audio_path": "../../../audio/turn0_s1.wav", "sample_rate": 24000,
    "input_wav_slice_start_sec": 0.0, "input_wav_slice_end_sec": 6.4},
   {"filename": "seg_0003_s1_turn1_bc_yeah.wav", "turn_id": 1, "type": "backchannel",
    "bc_text": "yeah", "bc_class": "Acknowledgment", "bc_confidence": 0.71, "sample_rate": 24000}],
 "speaker_2": [...]}
```

A backchannel on speaker X's turn `N` is listed under the **other** speaker
with the same `turn_id` `N`. A turn id present on both channels is how the
scheduler recognises a backchannel and its host turn. `source_audio_path`
(voice lines only) is relative to the cache directory unless
`--absolute_paths` is given; `input_wav_slice_*` appear only when the input
provided `slice_*_sec`.

`bc_decisions.json`: one entry per turn —
`{turn_id, list_idx, pred_label, confidence, all_probs: {NoBackchannel,
Acknowledgment, Reactive}, bc_word, suppressed}` for voice lines,
`{turn_idx, pred_label, confidence, all_probs, bc_word}` for text input.

`turn_metadata.json`: one entry per turn — `{turn_id, speaker_id, text,
start_time, end_time, start_frame, end_frame, duration}` in the naive layout;
voice lines add `audio_path` and `input_wav_slice_{start,end}_sec`.

`bc_eval.json` / `bc_eval_summary.json`: per-turn rows (`predicted_bc`,
`gt_bc`, ...) and `TP, FP, FN, TN, precision, recall, f1`; the summary holds
`per_example` and the pooled `aggregate`.

### Options

Input (exactly one) and output:

| Flag | Default | |
|---|---|---|
| `--input_dialogue_json` | – | voice-line dialogue |
| `--input_jsonl` | – | text dialogues |
| `--bc_vocab_jsonl` | – | text dialogues with per-dialogue backchannel words |
| `--input_bc_gt_jsonl` | – | benchmark records; also evaluates the decisions |
| `--out_dir` | required | caches go to `<out_dir>/segment_cache/<dialogue_id>/` |

Backchannel classifier:

| Flag | Default | |
|---|---|---|
| `--bc_classifier_type` | `joint` | `joint` text+audio model, `text` RoBERTa text-only model, `none` |
| `--no_classifier` | off | same as `--bc_classifier_type none` |
| `--bc_classifier_dir` | – | checkpoint directory |
| `--bc_tokenizer_dir` | – | defaults to `<dir>/tokenizer`, then `<dir>/../tokenizer` |
| `--bc_confidence_threshold` | 0.0 | 1.0 never backchannels |
| `--bc_context_window_turns` | 5 | earlier turns in the text context |
| `--bc_context_window_future_turns` | 5 | later turns in the text context |
| `--no_bc_context` | off | do not tag earlier turns with their backchannel class |
| `--no_suppress_consecutive_bcs` | off | voice lines: keep backchannels on consecutive same-speaker turns |
| `--bc_max_audio_seconds` | 15.0 | joint model: cap on the target turn's audio |
| `--bc_vocab_per_class` | – | required with a classifier |
| `--bc_vocab` | `hmmm uhhmm` | additional words made available |
| `--bc_word_selection` | `avoid_repeat` | or `random` |
| `--ensure_at_least_one_bc` / `--no-…` | on | voice lines: force one backchannel if none was predicted |
| `--bc_probability` | 1.0 | without a classifier: backchannel probability per turn |
| `--bc_vocab_subset` | `hmmm uhhmm` | without a classifier: words to draw from |

With `--bc_classifier_type none`, text input gets random backchannels
(`--bc_probability`, 0 for none) and voice-line input gets no backchannels.
Benchmark input requires a classifier.

Speech synthesis:

| Flag | Default | |
|---|---|---|
| `--tts_backend` | `chatterbox` | or `chatterbox_turbo` (faster; ignores the next two) |
| `--exaggeration`, `--cfg_weight` | 0.5, 0.5 | Chatterbox controls |
| `--ref_s1`, `--ref_s2` | required | reference recordings for the two voices |
| `--seed` | 42 | benchmark input uses `seed + int(dialogue_id)` per dialogue |
| `--device` | `cuda` | |

Silence and padding (milliseconds):

| Flag | Default |
|---|---|
| `--inter_turn_silence_ms` | 1000 |
| `--intra_turn_pause_ms` | 1000 |
| `--bc_silence_padding_ms` | 5000 |
| `--voice_line_padding_ms` | 5000 |

Naive baseline (turns back to back, no backchannels) and cache handling:

| Flag | |
|---|---|
| `--run_naive_only` | voice-line / benchmark input: after the normal run, also write the baseline to `<cache>/naive/`. No effect on text input. (The name is historical: it does not skip anything.) |
| `--naive_only` | skip classifier and backchannels and write only the baseline into the cache directory; works for every input kind |
| `--naive_vad_trim` / `--no-naive_vad_trim` | trim each turn to its speech with Silero VAD before concatenating (default on, threshold 0.6) |
| `--absolute_paths` | store absolute voice-line paths in the cache |
| `--clean_cache` | delete an existing `audio_segments/` before writing |

## Step 2 — `python -m neural_scheduler.pipeline.tokenize`

Per speaker, the cached segments are sorted by turn id, resampled to 16 kHz and
concatenated with 2 s of silence between them. The concatenation is tokenised
once (GLM-4-Voice speech tokenizer, one code per 80 ms) and run through Silero
VAD. For each turn, the VAD regions inside its segment(s) become the turn's
speech blocks:

```
SIL <|start-turn-N|> block [SIL block ...] <|end-turn-N|> SIL <|start-turn-M|> ...
```

One silence token separates the blocks of a turn. The record's `input_ids` are

```
<|start-of-script-A|> script A <|end-of-script-A|>
<|start-of-script-B|> script B <|end-of-script-B|>
<|start-of-dialogue|> placeholder <|end-of-dialogue|>
```

where the placeholder dialogue is the two concatenations side by side (the
scheduler replaces it).

Outputs:

```
<cache>/source_map/
  concatenated_audio_{a,b}.wav    16 kHz; raw token i covers samples [1280 i, 1280 (i+1))
  raw_tokens_{a,b}.json           speech-tokenizer codes of the whole concatenation
  script_tokens_{a,b}.json        vocabulary ids of the script
  script_source_map_{a,b}.json    per script token: index into raw_tokens, or null for markers/silence
  script_offsets.json             {script_a_start, script_a_end, script_b_start, script_b_end} in input_ids
<cache_base_dir>/tokenized_result.jsonl
  {"input_ids": [...], "generation_start_index": int, "generation_end_index": int,
   "remaining_durations": [...],
   "turn_boundaries": [{"turn_id": 0, "channel": "A", "type": "speech"}, ...],
   "source_map_dir": "example_0000/source_map", "source_cache": "example_0000", "example_idx": 0}
```

`source_cache` and `source_map_dir` are relative to the JSONL file, so an
output directory can be moved as a whole. A cache that fails (for example a
missing segment file) is logged and skipped; its `example_idx` is not reused.

| Flag | Default | |
|---|---|---|
| `--cache_base_dir` | required | the `segment_cache/` folder of step 1 |
| `--output_path` | `<cache_base_dir>/tokenized_result.jsonl` | |
| `--audio_tokenizer_path` | `THUDM/glm-4-voice-tokenizer` | |
| `--model_id` | `THUDM/glm-4-voice-9b` | text tokenizer |
| `--device` | `cuda` | |
| `--vad_threshold` | 0.2 | lower keeps quieter speech; the released runs used 0.05–0.25 |
| `--vad_min_speech_ms` | 50 | |
| `--vad_min_silence_ms` | 50 | shorter pauses do not split a turn into blocks |
| `--max_turns` | 20 | turns with an id ≥ this are dropped from the scripts (the vocabulary has 50) |
| `--absolute_paths` | off | write absolute paths into the JSONL |

### Benchmark mode — `python -m neural_scheduler.pipeline.tokenize_icc`

For backchannel-only scheduling: speaker 1 is a real recording whose timing is
kept.

- Channel A is tokenised from the complete recording
  `<icc_base_dir>/<id>/input.wav` (`<id>` = cache directory name). Each turn's
  span is the manifest's `input_wav_slice_{start,end}_sec`, intersected with
  VAD, so raw-token indices are positions in the dialogue.
- Channel B is tokenised from the backchannel clips in the cache, as above.
- Speaker-1 backchannels and speaker-2 speech entries in the manifest are
  ignored.

The record additionally has `example_id` and

```json
"s1_turn_timing": [{"turn_id": 0, "start_token": 6, "end_token": 71,
                    "sub_segments": [{"start_token": 6, "end_token": 30}, {"start_token": 36, "end_token": 71}]}]
```

(12.5 Hz token steps), which the scheduler uses to pin speaker 1. The source
map also holds `silenced_tokens_{a,b}.json`. Options are those of script mode
with `--segment_cache_dir` (required) instead of `--cache_base_dir`, plus
`--icc_base_dir` (required).

## The runner — `python -m neural_scheduler.pipeline.run`

```
python -m neural_scheduler.pipeline.run --config CONFIG.yaml
       [--steps 1 2 3] [--set KEY=VALUE [KEY=VALUE ...]] [--dry_run]
```

It builds the three command lines from the config, prints them, and runs them
in order, stopping at the first failure. `--dry_run` only prints. `--set`
overrides config keys by dotted path (`--set tts.seed=46
scheduler.pass1.search_window=15`); values are parsed as YAML, `null` removes a
key. Only keys present in the config are passed on — anything else keeps the
default of the step's own command line. Unknown keys are an error. Relative
paths are relative to the working directory.

### Config reference

Top level:

| Key | |
|---|---|
| `mode` | `three_pass` (default) or `bc_only` (benchmark: step 2 uses `tokenize_icc`, step 3 gets `--mode bc_only`) |
| `steps` | subset of `[1, 2, 3]`, default all |
| `variants` | optional list; each variant runs all steps into `<io.out_dir>/<variant>` |
| `variant_overrides` | optional `{variant: {dotted.key: value}}` |
| `environment` | optional per-step interpreter and environment, see below |

The variant `bc_never` has a built-in override
`classifier.confidence_threshold: 1.0` (the classifier runs but never
backchannels). Any other variant name is just a sub-directory unless
`variant_overrides` gives it settings.

`io`:

| Key | Step flag |
|---|---|
| `out_dir` (required) | `--out_dir`, and the default locations below |
| `input_jsonl` / `input_dialogue_json` / `bc_vocab_jsonl` / `input_bc_gt_jsonl` | same-named flags; exactly one when step 1 runs |
| `icc_base_dir` | `--icc_base_dir` (required for step 2 in `bc_only` mode) |
| `tokenized_jsonl` | where step 2 writes / step 3 reads; default `{out_dir}/segment_cache/tokenized_result.jsonl` |
| `output_root` | step 3 `--output_root`; default `{out_dir}` |
| `absolute_paths`, `clean_cache` | `--absolute_paths`, `--clean_cache` |

`{out_dir}` in `tokenized_jsonl` / `output_root` is replaced by the variant's
output directory.

`classifier` → step 1:

| Key | Flag |
|---|---|
| `type` | `--bc_classifier_type` |
| `dir`, `tokenizer_dir` | `--bc_classifier_dir`, `--bc_tokenizer_dir` |
| `confidence_threshold` | `--bc_confidence_threshold` |
| `context_window_turns`, `context_window_future_turns` | `--bc_context_window_turns`, `--bc_context_window_future_turns` |
| `bc_context: false` | `--no_bc_context` |
| `suppress_consecutive_bcs: false` | `--no_suppress_consecutive_bcs` |
| `max_audio_seconds` | `--bc_max_audio_seconds` |
| `vocab_per_class` | `--bc_vocab_per_class` |
| `vocab` (list) | `--bc_vocab` |
| `word_selection` | `--bc_word_selection` |
| `ensure_at_least_one_bc` | `--ensure_at_least_one_bc` / `--no-ensure_at_least_one_bc` |
| `probability`, `vocab_subset` (list) | `--bc_probability`, `--bc_vocab_subset` |

`tts` → step 1: `backend`, `ref_s1`, `ref_s2`, `exaggeration`, `cfg_weight`,
`seed`, `device`, `inter_turn_silence_ms`, `intra_turn_pause_ms`,
`bc_silence_padding_ms`, `voice_line_padding_ms`, `run_naive_only`,
`naive_only`, `naive_vad_trim` (flags of the same name; `backend` is
`--tts_backend`).

`tokenizer` → step 2: `audio_tokenizer_path`, `model_id`, `device`, `max_turns`.

`vad` → step 2: `threshold`, `min_speech_ms`, `min_silence_ms`
(`--vad_threshold`, ...).

`scheduler` → step 3: `checkpoint` (required), `device`, `output_sr`,
`output_sub_dir`, `save_video`, `save_input_scripts`, `num_examples`,
`randomize`, and

| Section | Keys → flags |
|---|---|
| `pass1` | `search_window`, `search_stride`, `scoring_context`, `inter_turn_gap` → `--pass1_<key>`; `schedule_subturns`, `speaker_{1,2}_subturn_{min,max}_pause`, `subturn_search_stride`, `use_speaker_gap_constraints`, `speaker_{1,2}_{min,max}_gap`, `use_strong_overlap_constraints`, `strong_overlap_{min,max}` → `--pass_1_<key>` |
| `pass2` | `search_stride`, `scoring_context`, `bc_position_limit`, `bc_effect`, `max_bc_overlap_ratio`, `score_only_pauses` → `--pass2_<key>` |
| `pass3` | `scoring_context`, `max_extra_gap` → `--pass3_<key>` |

### Environments

```yaml
environment:
  step1:
    python: /envs/tts/bin/python                    # interpreter for this step
    env: {NEURAL_SCHEDULER_STUB_TORCHVISION_META: "1"}
  step2:
    env: {GLM4VOICE_REPO: /path/to/GLM-4-Voice}
  step3:
    env_setup: "source /path/to/activate_main_env.sh"   # shell snippet run before the step
```

- `python`: interpreter (default: the one running the runner).
- `env_setup`: the step is run as `bash -c "<env_setup> && exec python -m ..."`.
  Use it when an environment needs more than a different interpreter (module
  loads, activation scripts).
- `env`: extra environment variables.

`NEURAL_SCHEDULER_STUB_TORCHVISION_META=1` is for Chatterbox environments in
which importing `transformers` fails with `operator torchvision::nms does not
exist` (a torchvision build that does not match torch). It replaces
`torchvision._meta_registrations` with an empty module before anything is
imported. Leave it unset otherwise.

## Migration from the research tree

Paths are relative to `ntpp/model/chatGLM_dual_timing_only/experiments/`.

| Research tree | Release |
|---|---|
| `inference_pipeline_bc_only_full_duplex_bench/step_1_no_tts_classifier_voiceline_given_with_audio.py` | `neural_scheduler/pipeline/synthesize.py` (default options) |
| `…/step_1_no_tts_classifier_voiceline_given_with_audio_turbo.py` | `synthesize.py --tts_backend chatterbox_turbo`; its `--run_naive_only` is `--naive_only --no-naive_vad_trim` |
| `inference_pipeline_10_examples/step_1_no_tts_classifier_voiceline_given.py`, `…bc_only…/step_1_no_tts_classifier_voiceline_given.py` (text-only classifier) | `synthesize.py --bc_classifier_type text --bc_word_selection random --no-ensure_at_least_one_bc` |
| `BackchannelClassifier` classes inside the step-1 scripts, `--bc_train_script_dir`, `BC_TRAIN_SCRIPT_DIR` | `neural_scheduler/backchannel/predictor.py`; the flag and variable are gone |
| `ChatterboxDialogueGenerator` | `synthesize.DialogueSynthesizer` + `pipeline/tts.py` |
| `inference_pipeline_10_examples/step_2_tokenization_with_source_map.py` | `neural_scheduler/pipeline/tokenize.py` |
| `inference_pipeline_bc_only_full_duplex_bench/step_2_tokenization_with_source_map.py` | `neural_scheduler/pipeline/tokenize_icc.py` |
| `step_2_tokenization_util.py` (four helpers) | `neural_scheduler/data/{speech_tokenizer,vad,sequence,durations}.py`; its `TTSToTokenProcessor` class is not carried over |
| `step_3_inference_reconstruct.py`, `step_4_reconstruct_audio_with_schedule_result.py`, `segment_level_search_v6*.py` | `neural_scheduler/pipeline/{schedule,reconstruct}.py`, `neural_scheduler/scheduler/` |
| every `run_pipeline*.sh`, `pipeline_*.sh`, `run_step*.sh` | `neural_scheduler/pipeline/run.py` + one YAML |
| `VARIANTS=("bc_never")` / `("classifier")` | `variants: [bc_never]` / `[classifier]` |
| `PASS2_vad_*` shell variables | `vad.*` (they were always step-2 settings) |
| `PASS1_MIN_GAP`, `PASS3_SEARCH_WINDOW` | removed (had no effect) |
| `inference_pipeline_Feb16`, `inference_pipelin_March31`, `inference_pipeline_April_10` | superseded; not carried over |

Which config replaces which script family:

| Scripts | Config |
|---|---|
| `inference_qualitative_examples/run_pipeline_standard_run.sh` | `tts_turn_taking.yaml` |
| `inference_qualitative_examples/run_pipeline_standard_run_naive_only.sh` | `tts_turn_taking.yaml --steps 1` (that script ran step 1 only; the naive baseline is the cache's `combined_dual.wav`) |
| `inference_qualitative_examples/run_pipeline_standard_run_with_bc.sh` | `tts_with_backchannel.yaml` with `classifier.type: joint`, threshold 0.2, `pass2.bc_position_limit: 0.7`, input `examples/scenes/storytelling_scenes.jsonl` |
| `inference_pipeline_10_examples/tts_psychiatrist/run_pipeline_standard_run*.sh` | `tts_with_backchannel.yaml` (`--set io.input_jsonl=…` for the 1–5 segment scenes) |
| `inference_pipeline_10_examples/{napster,royal_with_cheese,what_cant_i_see_through,cunk_on_earth,heat}/…` | `voice_lines.yaml` with the script's parameters; the scenes themselves are not distributed |
| `…/heat/pipeline_with_bc_gen_audio*.sh` + `pipeline_with_bc_schedule*.sh` | any config with `--steps 1`, then `--steps 2 3` |
| `inference_pipeline_10_examples/…/run_pipeline_extreme_overlap.sh` | `strong_overlap.yaml` |
| `inference_pipeline_bc_only_full_duplex_bench/run_step*.sh`, `best_threshold_check/*.sh` | `fdb_backchannel.yaml` (`--set classifier.confidence_threshold=0.7` for the sweep) |

### Deliberate differences from the research scripts

- Output and reference-voice defaults that pointed into a dated experiment
  folder are now required arguments; exactly one input must be given (the
  built-in sample dialogue is gone).
- Voice-line paths in `manifest.json` / `turn_metadata.json` and the cache
  paths in `tokenized_result.jsonl` are relative (`--absolute_paths` restores
  the old content byte for byte).
- Step 2 tokenises and VADs each speaker once. The research script did it
  twice with identical results; the second pass only fed the placeholder
  dialogue.
- `--bc_classifier_type none` on voice-line input schedules the dialogue
  without backchannels (previously an assertion error).
- A missing `--seed` no longer crashes voice-line input (`seed += 1` on `None`).
- `--bc_vocab_jsonl` without a classifier now makes the per-dialogue words
  available before synthesis (previously rejected as "not in cache").
- New, default-off options: `--clean_cache`, `--naive_only`,
  `--bc_word_selection random`, `--no-ensure_at_least_one_bc`, `--max_turns`.

With the same inputs, seeds and models everything else is unchanged; this is
checked against the research scripts in `tests/test_pipeline_synthesize.py`
and `tests/test_pipeline_tokenize.py`.

## Known issues (steps 1 and 2)

These are properties of the released results and are kept as they are.

1. **Seeds of backchannel synthesis.** Voice-line input: speaker 1's
   backchannels use `seed, seed+1, …`, speaker 2's all reuse the final value.
   Text input: speaker 1's backchannels all use `seed`, speaker 2's each draw a
   fresh random seed. Turn synthesis itself is never seeded. Identical words in
   one dialogue can therefore sound identical or different depending on the
   speaker.
2. **Stale files in `audio_segments/`.** Re-running step 1 into an existing
   cache with other backchannel decisions leaves the old clips behind
   (harmless — only files in the manifest are used — unless you inspect the
   folder). `--clean_cache` removes them.
3. **Backchannel file names contain the raw word** (spaces become `_` with a
   classifier, punctuation is kept).
4. **`start_frame` / `end_frame` in `turn_metadata.json` assume 16 kHz**
   whatever the audio rate is; `start_time` / `end_time` are correct.
5. **"No backchannel" with a classifier still loads it** (`bc_never` raises the
   threshold). Use `classifier.type: none` to avoid loading it.
6. **`ensure_at_least_one_bc` applies to voice-line input only**; text input
   may end up without any backchannel.
7. **Turn limit.** Step 2 drops turns with id ≥ 20 (a warning is logged) while
   the vocabulary and the scheduler support 50. Raise `--max_turns` for longer
   dialogues; the released runs used 20.
8. **Classifier context differs from training.** At inference the text window
   is counted in turns and only earlier turns carry backchannel tags (always
   `Neutral` or `Emotional`); training used a ±10 s window with tags on both
   sides. See `docs/backchannel.md`.
9. **Silence code inside speech.** A speech frame whose code happens to be the
   silence code (8520) is indistinguishable from a block boundary downstream.
10. **30 s tokenizer chunks** fall at arbitrary points of the concatenated
    audio.
11. **VAD release matters.** Segment boundaries move by a frame between Silero
    releases, which changes script tokens. The loader pins the release the
    experiments used.
12. **Benchmark mode** needs integer `dialogue_id`s and manifests with slice
    times; a cache without them fails and is skipped.
