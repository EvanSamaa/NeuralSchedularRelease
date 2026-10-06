# Neural Scheduler

Neural Scheduler decides the *timing* of a two-speaker dialogue whose
*content* is already fixed. Given a script — text to synthesise, or recorded
voice lines — it chooses when each utterance starts, how long the gaps and
overlaps are, where a speaker pauses, and where the listener backchannels,
and writes dual-channel audio with that timing.

```
dialogue script ──► 1. synthesize   voice lines / TTS, backchannel decisions and audio
                └─► 2. tokenize     12.5 Hz speech tokens + VAD → script tokens
                └─► 3. schedule     three-pass search scored by the scheduler model,
                                    then reconstruction from the original audio
                                    → channel_a.wav, channel_b.wav, timeline.json
```

The scheduler model is GLM-4-Voice (9B) with a script / speaker-A /
speaker-B channel embedding and a duration head, fine-tuned on Seamless
Interaction to continue `script A, script B` with a time-aligned,
frame-interleaved dialogue token stream. At inference it is not sampled
from: candidate schedules are assembled from the script tokens and the one
with the lowest negative log-likelihood is kept.

## Repository layout

| Path | Content | Documentation |
|---|---|---|
| `neural_scheduler/model/` | scheduler model, config, checkpoint loading | [docs/model.md](docs/model.md) |
| `neural_scheduler/train/` | fine-tuning entry point | [docs/training.md](docs/training.md) |
| `neural_scheduler/data/` | Fisher / Seamless Interaction preprocessing, collator | [docs/data.md](docs/data.md) |
| `neural_scheduler/backchannel/` | backchannel labelling, classifier, predictor | [docs/backchannel.md](docs/backchannel.md) |
| `neural_scheduler/scheduler/` | the three-pass search | [docs/scheduler.md](docs/scheduler.md) |
| `neural_scheduler/pipeline/` | steps 1–4 and the YAML runner | [docs/pipeline.md](docs/pipeline.md) |
| `neural_scheduler/evaluation/` | timing distributions, perturbation tests, NLL diagnostics, Full-Duplex-Bench adapter | [docs/evaluation.md](docs/evaluation.md) |
| `neural_scheduler/viz/`, `neural_scheduler/analysis/` | timelines, figures, listening-test stimuli and analysis | [docs/analysis.md](docs/analysis.md) |
| `configs/` | pipeline scenarios, classifier settings, DeepSpeed config | |
| `scripts/slurm/` | job templates driven by environment variables | |
| `examples/` | text-only scenes and backchannel vocabularies | [examples/README.md](examples/README.md) |
| `analysis_data/` | anonymised listening-test responses | [analysis_data/README.md](analysis_data/README.md) |
| `tests/` | unit tests and equivalence tests against the research code | [Tests](#tests) |

[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) lists behaviour that is kept on
purpose because the released checkpoint and results depend on it.
[docs/MIGRATION.md](docs/MIGRATION.md) maps the research tree to this one.

## Installation

```bash
git clone https://github.com/EvanSamaa/NeuralSchedularRelease
cd NeuralSchedularRelease
pip install -e ".[pipeline]"        # extras: train, data, backchannel, pipeline, eval, analysis, tts, dev
```

Python 3.10+; developed with Python 3.11, torch 2.6 and
**transformers 4.46** (the model code relies on that series).

Also needed, depending on what you run:

| What | For | How it is found |
|---|---|---|
| A checkout of [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) | speech tokenizer code (pipeline step 2, preprocessing) | `export GLM4VOICE_REPO=/path/to/GLM-4-Voice` |
| `THUDM/glm-4-voice-9b`, `THUDM/glm-4-voice-tokenizer` | base model, tokenizers | Hugging Face cache (`HF_HOME`) |
| Scheduler checkpoint | pipeline step 3, evaluations | a path in the config / `--checkpoint` |
| Backchannel classifier checkpoint | pipeline step 1 with backchannels | a path in the config |
| Chatterbox TTS | pipeline step 1 | `pip install -e ".[tts]"`, usually in its own environment (it pins a different numpy) |
| Silero VAD | step 2, preprocessing, figures | torch hub; the release is pinned because newer ones change segment boundaries |
| `ffmpeg` | videos | `PATH` |

No model weights, datasets or audio are included in this repository.

## Quick start

```bash
cp configs/pipeline/tts_no_classifier.yaml my_run.yaml     # fill in the paths marked REQUIRED
python -m neural_scheduler.pipeline.run --config my_run.yaml --dry_run   # print the three commands
python -m neural_scheduler.pipeline.run --config my_run.yaml
```

`tts_no_classifier.yaml` schedules turn-taking for the text scenes in
`examples/` and needs two reference voice recordings of your own and a
scheduler checkpoint. Other scenarios in `configs/pipeline/`:

| Config | Scenario |
|---|---|
| `tts_turn_taking.yaml` | text → TTS → turn-taking, classifier loaded but disabled |
| `tts_with_backchannel.yaml` | text → TTS, classifier-driven backchannels, sub-turn scheduling |
| `voice_lines.yaml` | recorded voice lines |
| `strong_overlap.yaml` | forced overlap between speakers |
| `fdb_backchannel.yaml` | backchannel-only mode for Full-Duplex-Bench |

Each step is also a command of its own
(`python -m neural_scheduler.pipeline.{synthesize,tokenize,schedule} --help`).

## Other entry points

| Task | Command |
|---|---|
| Preprocess training data | `python -m neural_scheduler.data.preprocess --dataset {fisher,seamless} …` then `python -m neural_scheduler.data.merge …` |
| Fine-tune the scheduler | `python -m neural_scheduler.train.train …` (`scripts/slurm/train_multinode.sh`) |
| Label backchannels / prepare data / train the classifier | `python -m neural_scheduler.backchannel.labeling.extract`, `…backchannel.data.prep`, `…backchannel.train` |
| Gap / overlap distributions | `python -m neural_scheduler.evaluation.timing_distribution …` |
| Turn-order and silence perturbation | `python -m neural_scheduler.evaluation.turn_order …`, `…evaluation.silence_perturbation …` |
| Full-Duplex-Bench backchannel metrics | `python -m neural_scheduler.evaluation.fdb.prepare_icc …`, `…evaluation.fdb.eval_backchannel …` |
| Listening-test analysis | `python -m neural_scheduler.analysis.user_study main --csv analysis_data/user_study/responses_main.csv …` |
| Compare schedules | `python -m neural_scheduler.viz.timeline …` |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest                                   # unit tests
NTPP_ORIG=/path/to/research/tree python -m pytest  # + equivalence tests
```

The equivalence tests import the original research implementation and
require identical results from this package: model outputs on small random
configurations, preprocessing records, the search's decisions under a
deterministic stand-in model, pipeline caches, evaluation statistics and
figures. They are skipped when `NTPP_ORIG` is not set. All tests run on CPU.

## What is not here

- The scheduler and classifier checkpoints.
- Fisher (LDC) and Seamless Interaction data or anything derived from them.
- Full-Duplex-Bench and its data (CC BY-NC 4.0); only an adapter for this
  project's output layout is included.
- Reference voices and any generated audio or video.

## Licence and attribution

Apache License 2.0. The model code is derived from GLM-4-Voice and the
project began as a fork of NTPP; see [NOTICE](NOTICE).
