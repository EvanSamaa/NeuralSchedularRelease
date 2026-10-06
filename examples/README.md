# Examples

Text-only inputs for the inference pipeline. There is **no audio** in this
folder: you supply the reference voices (and, in voice-line mode, the recorded
lines) yourself. Use recordings you have the right to use and, for voice
cloning, the speaker's consent.

## `scenes/` — text dialogues (`--input_jsonl` / `io.input_jsonl`)

One dialogue per line, `{"text": "[S1] ... [S2] ..."}`. Each `[S1]` / `[S2]`
tag starts a turn. Two consecutive turns of the same speaker are scheduled
separately (and can each receive a backchannel); a newline inside a turn
separates sentences that are synthesised one by one.

| File | Content |
|---|---|
| `plain_turntaking.jsonl` | Five four-line toy dialogues; the fastest smoke test. |
| `everyday_scenes.jsonl` | 20 short customer-service and small-talk dialogues with frequent speaker changes (turn-taking). |
| `everyday_scenes_selections.txt` | Indices (0-based, i.e. cache `example_000N`) of the `everyday_scenes.jsonl` dialogues used in the turn-taking listening test. |
| `storytelling_scenes.jsonl` | 20 dialogues with longer turns (anecdotes, job descriptions) that invite listener backchannels. |
| `psychiatrist.jsonl` | One scene with two long turns. |
| `psychiatrist_3_segments.jsonl` | The same scene with the last long turn split into three consecutive `[S2]` turns, so that it can host up to three backchannels. |
| `monologue_single_backchannel.jsonl` | `--bc_vocab_jsonl` format: `{"text": ..., "bc_vocab": [...]}`; a one-speaker monologue, each line with a different backchannel phrase (no classifier). |

## `bc_vocab/` — backchannel words per class (`--bc_vocab_per_class`)

`{"Acknowledgment": [...], "Reactive": [...]}`: the words the pipeline may
synthesise for each class the classifier predicts.

| File | Used with |
|---|---|
| `everyday.json` | `everyday_scenes.jsonl`, `storytelling_scenes.jsonl` |
| `psychiatrist.json`, `psychiatrist_alt.json` | the psychiatrist scenes |
| `benchmark_short.json`, `benchmark_short_friendly.json`, `benchmark_long.json`, `benchmark_longer.json` | Full-Duplex-Bench backchannel runs with progressively longer phrases |
| `benchmark_longer_reported.json` | variant of `benchmark_longer.json` kept from the benchmark experiments |

## `voice_lines/` — recorded lines (`--input_dialogue_json` / `io.input_dialogue_json`)

`dialogue_template.json` shows the format: one entry per turn with `turn_id`,
`speaker` (1 or 2), `text` and `audio_path`. A relative `audio_path` is resolved
against the JSON file, so put your recordings in `voice_lines/audio/` or edit
the paths. Consecutive turns of one speaker are allowed.

## Running one

    python -m neural_scheduler.pipeline.run --config configs/pipeline/tts_turn_taking.yaml \
        --set io.input_jsonl=examples/scenes/plain_turntaking.jsonl \
              tts.ref_s1=/path/to/voice_a.wav tts.ref_s2=/path/to/voice_b.wav \
              scheduler.checkpoint=/path/to/checkpoint io.out_dir=outputs/plain

See `docs/pipeline.md`.
