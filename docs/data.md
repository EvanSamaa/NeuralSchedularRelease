# Data preprocessing

`neural_scheduler.data` turns two-channel conversations (one speaker per
channel) into the JSONL files the scheduler is trained on. Each line is one
20.48 s window: the *script* of each speaker (what will be said, timing
removed) followed by the time-aligned *dialogue* (when it is said).

## Requirements

- `pip install -e ".[data]"`
- A checkout of [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice); the
  speech tokenizer class `WhisperVQEncoder` is imported from it:
  `export GLM4VOICE_REPO=/path/to/GLM-4-Voice`
- Hugging Face models `THUDM/glm-4-voice-tokenizer` (WhisperVQ) and
  `THUDM/glm-4-voice-9b` (tokenizer files only), cached under `HF_HOME`.
- Silero VAD, fetched through `torch.hub` into `TORCH_HOME`. The release is
  pinned to **`snakers4/silero-vad:v6.0`**, the version the published data was
  produced with. Later Silero releases move segment boundaries by tens of
  milliseconds, which changes turn boundaries and therefore the tokens. Override
  with `--vad_repo` or `SILERO_VAD_REPO` (a hub spec or a local checkout).
- One GPU per job (`--device cpu` works and gives identical output, at roughly
  15 s per window instead of a fraction of a second).

## Pipeline

### Fisher

Fisher English is distributed as two-channel 8 kHz telephone audio. Two
equivalent inputs are supported; both resample to 16 kHz with
`torchaudio.transforms.Resample` and index conversations in sorted file-name
order.

```bash
# (optional) pack a directory of stereo files into the LMDB format
python -m neural_scheduler.data.sources.fisher \
    --audio_dir /data/fisher/stereo --lmdb_path fisher_audio.lmdb --keys_path fisher_audio_keys.txt

# 1. tokenise, one job per shard (GPU)
python -m neural_scheduler.data.preprocess --dataset fisher \
    --lmdb_path fisher_audio.lmdb --keys_path fisher_audio_keys.txt \
    --output_dir $OUT/fisher --num_shards 21 --shard_id $I
#    or, without an LMDB:   --audio_dir /data/fisher/stereo

# 2. merge into train / val / test
python -m neural_scheduler.data.merge --dataset fisher --input_dir $OUT/fisher
```

The published Fisher data was produced from the LMDB (5,850 conversations,
keys `fe_03_00001.wav` ...; audio stored as float32 at 8 kHz), with 21 shards.
The tool that originally built that LMDB is not part of this repository;
`neural_scheduler.data.sources.fisher` writes the same record format, and the
directory source produces the same audio tensors as the LMDB source (tested).
Raw Fisher `.sph` files must be readable by libsndfile or converted to WAV
first (e.g. with `sph2pipe`).

Step 1 writes `fisher_audio_tokens_v2_shard_{i}_of_{N}.jsonl`; step 2 writes
`fisher_audio_tokens_v2_{train,val,test}.jsonl`, the names the training
scripts read. Conversations are assigned to splits 98 / 1 / 1 with seed 42, so
no conversation is split across files.

### Seamless Interaction

One mono WAV per participant, named
`V{vendor}_S{session}_I{interaction}_P{participant}.wav` under
`{data_root}/{improvised,naturalistic}/{train,dev,test}/`.

```bash
# 1. shard manifests, once per data type and split (CPU)
python -m neural_scheduler.data.preprocess --dataset seamless --create_manifests \
    --data_root $SEAMLESS --data_type improvised --split train \
    --output_dir $OUT/seamless --num_shards 25

# 2. tokenise, one job per shard (GPU)
python -m neural_scheduler.data.preprocess --dataset seamless \
    --data_type improvised --split train --output_dir $OUT/seamless --shard_id $I

# 3. merge improvised + naturalistic per split
python -m neural_scheduler.data.merge --dataset seamless --input_dir $OUT/seamless --trainer_names
```

The published data used improvised/train 25 shards, naturalistic/train 25,
improvised/test 2, naturalistic/test 2 (the dev split was not processed).

Step 1 groups files by (vendor, session, interaction), sorts each group by
participant suffix, takes the first file as channel A (left) and the second as
channel B (right), drops groups with fewer than two existing files and writes
`shard_manifests_{type}_{split}/shard_{id:05d}.json` plus `metadata.json`.
Step 2 writes
`seamless_interaction_{type}_{split}_audio_tokens_shard_{id:05d}_of_{N}.jsonl`.
Step 3 concatenates shards per split, improvised before naturalistic, without
shuffling (the trainer shuffles). By default it writes
`seamless_interaction_{split}_audio_tokens.jsonl`; with `--trainer_names` it
writes the names the released training launchers read:

| Split | File |
|---|---|
| train | `audio_tokens_train_with_durations_and_turnID_as_tokens.jsonl` |
| test | `audio_tokens_val_with_durations_and_turnID_as_tokens.jsonl` (the test split is the validation set) |

`--shard_id -1` processes every interaction sequentially without manifests and
writes `..._audio_tokens_debug.jsonl`.

### Slurm

`scripts/slurm/preprocess_fisher.sh`, `preprocess_seamless_manifests.sh` and
`preprocess_seamless.sh` are array-job templates configured through environment
variables; usage is in each file's header. Pass `--array=0-$((NUM_SHARDS-1))`
on the `sbatch` command line.

### Inspecting the output

```bash
python -m neural_scheduler.data.inspect show --input_file train.jsonl --num_samples 2
python -m neural_scheduler.data.inspect diagnose --input_file train.jsonl --sample_idx 0
```

`show` prints both script regions token by token, the interleaved dialogue with
its duration labels, a consistency check of those labels and the turn
structure. `diagnose` recomputes the turn assignment from the stored VAD
timestamps step by step and compares it with the stored turn boundaries.
`--visualize N` on the preprocessing command prints the same breakdown for the
first N records as they are produced.

## Record format

```
input_ids               List[int]   full sequence in vocabulary ids
generation_start_index  int         index of the first dialogue token
generation_end_index    int         index of <|end-of-dialogue|>
remaining_durations     List[int]   one label per dialogue token (len = end - start)
turn_boundaries         List[{turn_id, channel: "A"|"B", start_sec, end_sec}]   token-quantised
source_file             int         conversation index (Fisher: position in the key list;
                                    Seamless: example_id within its data type and split)
window_idx              int
vad_timestamps_left     List[{start, end}]   raw Silero output in seconds, relative to the window
vad_timestamps_right    List[{start, end}]
window_start_sec        float
window_end_sec          float
audio_path_left         str         Seamless only
audio_path_right        str         Seamless only
```

The trainer uses `input_ids`, `generation_start_index`, `generation_end_index`
and `remaining_durations`; the rest is provenance.

## Sequence layout

```
<|start-of-script-A|> script_A <|end-of-script-A|>
<|start-of-script-B|> script_B <|end-of-script-B|>
<|start-of-dialogue|> A0 B0 A1 B1 ... A255 B255 <|end-of-dialogue|>
```

- **Windows.** 327,680 samples per channel at 16 kHz (20.48 s) = 256 speech
  tokens per channel = 512 interleaved dialogue tokens. Windows do not overlap;
  the last one is shorter. A window in which VAD finds no speech on either
  channel produces no record. Each window is tokenised and VAD-processed on its
  own.
- **Speech tokens.** GLM-4-Voice WhisperVQ, 12.5 Hz (one token per 80 ms),
  codec ids 0..16383. Codec id `k` is the vocabulary token `<|audio_k|>` with id
  `152353 + k`.
- **VAD.** Silero, `threshold=0.2`, `min_speech_duration_ms=50`,
  `min_silence_duration_ms=50`, per channel. Sample positions become token
  indices with `int(sample / sr * 12.5)` (floor at both ends).
- **Dialogue.** Every frame outside a VAD segment is replaced by the silence
  token `<|audio_8520|>` (id 160873); speech frames keep their codec ids.
- **Script.** Per channel:
  `SIL <|start-turn-N|> seg1 [SIL seg2 ...] <|end-turn-N|> SIL <|start-turn-M|> ...`.
  It contains exactly the speech tokens of the channel's VAD segments, taken
  from the same tokenisation as the dialogue, so the script is a subsequence of
  the dialogue's speech tokens. One `SIL` leads, one separates the VAD segments
  inside a turn, one follows every turn. Turn markers appear only in the script.
- **Loss region.** `generation_start_index` marks where the dialogue starts; the
  model is trained on the dialogue only.

### Turn ids

1. Pool the VAD segments of both channels (token units) and sort by
   `(start, channel)`.
2. Merge consecutive segments of the same channel in that order into one turn;
   the originals are kept as its sub-segments.
3. In order, a turn of which at least 50 % is covered by an earlier turn takes
   that turn's id (first match wins). This makes a backchannel share the id of
   the turn it overlaps. Otherwise the turn gets a new id.
4. Renumber ids contiguously from 0.

Ids are shared between the speakers. There are marker tokens for 50 turns
(`<|start-turn-i|>` = 168742 + 2i, `<|end-turn-i|>` = 168743 + 2i).

### Duration labels

The dialogue is split back into its two channels. Within a channel, a run of
`r` identical tokens is labelled `r-1, r-2, ..., 0`: how many more times the
token repeats. The labels are interleaved like the tokens. In practice only
silence forms runs, so the label is a per-channel countdown of the silence that
remains (0..255). The model clamps it to its 32 duration classes.

## Sharding and resuming

- Fisher shards are contiguous, balanced ranges of the conversation index.
  Seamless shards are the manifests.
- `--resume` appends to the shard file and skips conversations that already
  have at least one record. A conversation that was interrupted half-way is
  therefore treated as finished.
- `--resume_per_window` skips only conversations recorded in the
  `<shard>.jsonl.completed` side file and, for the others, only the windows
  already written. Use it to complete a shard whose job was killed.
- Each job logs to `processing_{dataset}_shard_{i}_of_{N}.log` in the output
  directory. The summary line reports conversations processed / skipped /
  failed to load and windows written / failed, with windows dropped for an
  unknown audio code counted separately.

## Size and throughput

| | Fisher | Seamless Interaction |
|---|---|---|
| Conversations | 5,850 | improvised + naturalistic, train and test |
| Shards | 21 | 25 + 25 + 2 + 2 |
| Records written | 173,804 | 716,839 |
| Windows dropped | 2,949 (1.7 %, see below) | 0 |
| Merged train file | 1.6 GB | 6.4 GB |
| Validation records | 1,707 | 9,700 |
| GPU time | about 21 s per 12-minute conversation (3 h job limit per shard) | 12 h job limit per shard |

## Licensing

- **Fisher English** is distributed by the LDC under licence. Neither the audio
  nor anything derived from it may be redistributed; that includes the token
  files, because GLM-4-Voice speech tokens can be decoded back to intelligible
  speech.
- **Seamless Interaction** has its own licence; check it before sharing derived
  tokens. Records contain the paths of the source files (use
  `--relative_audio_paths` to avoid embedding absolute paths).
- This repository contains code only.

## Options that change the data

These are off by default; the defaults reproduce the published datasets.

| Flag | Effect |
|---|---|
| `--add_audio_16383_token` | Appends `<|audio_16383|>` to the vocabulary *after* the turn tokens (id 168842, inside GLM-4-Voice's padded vocabulary, no other id moves) so windows containing codec id 16383 are kept. The model must be trained with a tokenizer extended the same way; the released checkpoints were not. |
| `--source_id_style name` | Stores a string in `source_file` (Fisher: the file name; Seamless: `{type}_{split}_V.._S.._I..`) that is unique across data types and splits. |
| `--relative_audio_paths` | Seamless: stores audio paths relative to `--data_root`. |
| `--vad_repo` | A different Silero VAD version. |

## Known issues

1. **Windows with codec id 16383 are dropped.** Stock GLM-4-Voice has tokens
   `<|audio_0|>` .. `<|audio_16382|>` only. When WhisperVQ emits 16383 inside a
   speech segment the window cannot be encoded and is dropped. This removed
   2,949 of 176,753 Fisher windows (1.7 %) and none of Seamless. The default
   keeps this behaviour, and reports the count; `--add_audio_16383_token` is the
   opt-in alternative for new training runs.
2. **Seamless `source_file` is not globally unique.** It restarts at 0 for each
   (data type, split), so values collide after merging. Nothing in training
   depends on it; use `--source_id_style name` if you need conversation-level
   bookkeeping.
3. **More than 50 turns in a window.** Turns with id 50 or higher have no
   marker tokens and are left out of the script, while their speech stays in
   the dialogue.
4. **Very short VAD segments.** A segment shorter than one token can floor to an
   empty span; it then contributes an empty `<|start-turn-N|><|end-turn-N|>`
   pair to the script and consumes a turn id.
5. **`turn_boundaries` vs `vad_timestamps_*`.** The former are quantised to
   tokens, the latter are raw VAD seconds.
6. **Seamless channel assignment** is by lexicographic participant suffix;
   interactions with more than two participants use the first two.
7. **`--resume` granularity**, see above.
8. **Seamless validation set.** The released training setup validates on the
   Seamless *test* split; there is no separate held-out test file in that
   layout.

## Reproducibility checks

- The functions for turn ids, silencing, script construction, duration labels,
  timestamp conversion, shard arithmetic, manifests and merging are tested
  against the original research scripts (`tests/test_data_equivalence.py`,
  needs `NTPP_ORIG`).
- Run on CPU with the real WhisperVQ and Silero v6.0, this code reproduces the
  published records exactly for the windows compared (first three windows of
  Fisher conversation 0 and of the first Seamless validation conversation).

## Migration from the research tree

| Research tree (`NTPP/`) | Here |
|---|---|
| `scripts/processing_for_timing_only_with_duration_prediction/preprocess_fisher_sharded_DTW_with_duration_and_turnID_tokens.py` | `data/preprocess.py --dataset fisher`; core in `data/{speech_tokenizer,vad,turns,durations,sequence,sharding}.py`; audio in `data/sources/fisher.py` |
| `.../preprocess_seamless_sharded_DTW_with_duration_and_turnID_tokens.py` | `data/preprocess.py --dataset seamless` (`--split_dataset` is now `--create_manifests`, `--shard_count` is `--num_shards`; both old spellings still work); `data/sources/seamless.py` |
| `.../merge_shards_fisher.py` | `data/merge.py --dataset fisher` |
| `.../merge_shards_seamless.py` | `data/merge.py --dataset seamless` |
| `.../visualize_duration_and_turn_id.py`, the `visualize_sample` copies inside both preprocessing scripts | `data/inspect.py show` |
| `.../validate_duration_and_turn_id.py` | `data/inspect.py diagnose` |
| `.../submit_job_fisher.sh`, `submit_job_seamless.sh`, `split_seamless.sh` | `scripts/slurm/preprocess_fisher.sh`, `preprocess_seamless.sh`, `preprocess_seamless_manifests.sh` |
| `WavLMDBDataset` (external `interactive_avatar` repository) | `data/sources/fisher.py::FisherLMDBSource` |
| manual rename of the merged Seamless files | `data/merge.py --trainer_names` |

Not carried over:

| Research tree | Why |
|---|---|
| `scripts/preprocessing_pipeline/` | First generation: full-content dual-channel text files for the abandoned `chatGLM_dual` model; also the unrelated OpenMOSS tokeniser. |
| `scripts/preprocessing_for_timing_only/` | Second generation (no turn tokens, no durations); superseded step by step by the scripts above. |
| `.../add_duration.py`, `add_duration_and_turn_id.py`, `run_duration_and_turn_id.sh` | Post-processing of second-generation data into the `turn_ids`-column format of the turn-embedding model, which is not released. Its vocabulary puts `<|audio_16383|>` at id 168742, where this release has `<|start-turn-0|>`; the two formats are not interchangeable. |
| `.../validate_dataset_with_old_ones.py`, `ntpp/train/check_data.py`, `ntpp/train/test_load_dataset.py`, `scripts/continue_pretrain/` | Unfinished, broken or unrelated. |
| logs, test outputs, `dataset/Dual-channel/*.wav` | Outputs and licensed Fisher content. |

Behaviour that differs from the research scripts (none of it changes records):

- Silero VAD is pinned to v6.0 instead of following `master`.
- A window dropped for codec id 16383 is logged as one warning line and counted
  separately, instead of an error with a traceback.
- The empty trailing window of a conversation whose length is an exact multiple
  of the window is not attempted.
- Per-shard log files carry the dataset (and Seamless data type and split) in
  their name, so jobs sharing an output directory no longer write to the same
  log.
- A `<shard>.jsonl.completed` side file is written next to each shard file.
- The speech tokenizer device is an argument (`--device`) instead of a
  hard-coded `"cuda"`.
