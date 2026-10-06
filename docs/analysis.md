# Visualisation and analysis

Two subpackages turn pipeline outputs into figures, listening-test stimuli
and the user-study numbers:

| Module | Purpose |
|---|---|
| `neural_scheduler.viz.timeline` | `timeline.json` export of a schedule; side-by-side comparison of several schedules (interactive HTML + compact PNG) |
| `neural_scheduler.viz.three_pass` | seven diagnostic plots of one three-pass search |
| `neural_scheduler.viz.waveform_video` | rolling dual-waveform video of a two-channel dialogue |
| `neural_scheduler.viz.vad_plot` | Silero VAD wrapper and voice-activity bar plots |
| `neural_scheduler.viz.ffmpeg` | loudness matching, mixing, labelled A/B clips |
| `neural_scheduler.analysis.figures` | the two voice-activity figures |
| `neural_scheduler.analysis.stimuli` | A/B comparison videos and mixed audio |
| `neural_scheduler.analysis.user_study` | forced-choice study analysis |
| `neural_scheduler.analysis.webquestions_tts` | synthetic question/answer dialogues for the turn-order evaluation |

Requirements beyond the core package: `matplotlib`, `pandas`, `scipy`,
`statsmodels`, `torchaudio`, `librosa`, `opencv-python`, `soundfile`;
optionally `krippendorff`. `ffmpeg` and `ffprobe` must be on `PATH` (a build
with `libx264`, `aac`, and the `ebur128` and `drawtext` filters) for videos
and loudness matching. Silero VAD is loaded with
`torch.hub.load("snakers4/silero-vad", "silero_vad")`, which needs network
access or a populated torch-hub cache. `webquestions_tts` needs the `piper`
command-line tool.

All tools take explicit output locations; nothing is written to the current
directory by default.

## Inputs the tools expect

The tools read the directory layout written by the inference pipeline.

* **Scheduler output of one dialogue** — `<run>/<checkpoint>/sample_<N>/<subdir>/`
  with `channel_a.wav`, `channel_b.wav`, `channel_sum.wav` and
  `timeline.json`. `<subdir>` is `segment_search_v2` (or
  `segment_search_hp`; the backchannel tools take whichever exists).
* **Naive baseline** — `<run>/segment_cache/example_<XXXX>/combined_dual.wav`,
  the stereo back-to-back layout written by pipeline step 1 (speaker 1 left,
  speaker 2 right).
* **PersonaPlex baseline** (backchannel comparison only) —
  `<personaplex_dir>/<N>/output.wav`, the listener channel produced by
  PersonaPlex for Full-Duplex-Bench ICC sample `N`. These are not part of
  this repository; produce them by running PersonaPlex on the benchmark
  inputs. The first 4 s of each file are muted before use, in the figures,
  the videos and the audio export alike.
* **Sample lists** — plain text. For backchannel tools: one `.../sample_<N>`
  directory per line (`N` selects the PersonaPlex file). For turn-taking
  tools: one integer example id per line (`example_<XXXX>` / `sample_<N>`).
  For the paired tools: one `<wav_a>, <wav_b>` pair per line, where `wav_a`
  is the proposed system's mix.

## Schedule timelines

Pipeline step 3 calls `export_scheduling_state_json(state, path)` to write
`timeline.json`. To compare schedules:

```bash
# every timeline.json under a directory, labelled by its first sub-directory
python -m neural_scheduler.viz.timeline --scan_dir outputs/my_scene

# explicit files and labels
python -m neural_scheduler.viz.timeline \
    outputs/no_bc/.../timeline.json:"no backchannel" \
    outputs/with_bc/.../timeline.json:"with backchannel" \
    -o compare.html
```

This writes `compare.html` (interactive: zoom, seconds/steps) and
`compare.png` (300 dpi, single-column width; `--no-png` skips it,
`--fig-width`, `--dpi`, `--max-label-chars` adjust it). Bars are labelled
with the turn text and backchannel word when `turn_metadata.json` and
`bc_decisions.json` are found in a `segment_cache/example_*/` directory up
to three levels above the timeline; pass
`path:name:turn_metadata.json:bc_decisions.json` to name them explicitly.
With `--scan_dir` the output defaults to
`<scan_dir>/timeline_comparison.html` and the figure width to 3.5 in.

`visualize_scheduling(state, save_dir)` in `viz.three_pass` prints a text
summary and writes `timeline.png`, `pass1_nll.png`, `pass1_shifts.png`,
`pass2_nll.png`, `pass2_strategies.png`, `pass3_nll.png` and
`pass3_summary.png`; plots whose pass produced no diagnostics are skipped.

`create_dual_waveform_video(audio1, audio2, output_name)` in
`viz.waveform_video` writes the video next to `audio1`.

## Voice-activity figures

```bash
# backchannel figure: speaker turn / scheduler backchannel / PersonaPlex backchannel
python -m neural_scheduler.analysis.figures bc \
    --sample_list good_examples.txt \
    --personaplex_dir personaplex/icc_output \
    --out_dir figures/vad_bc
# → vad_sample_01.png, ...   (--threshold 0.4 --min_speech_ms 100 --min_silence_ms 80 --silence_first_n 4.0)

# turn-taking figure: naive A/B above, scheduled A/B below
python -m neural_scheduler.analysis.figures turn-take \
    --selections selections.txt \
    --naive_dir   outputs/tts_scenes/no_bc/bc_never/segment_cache \
    --model_dir   outputs/tts_scenes/no_bc/bc_never/<checkpoint> \
    --out_dir figures/vad_turn_take --save_pdf
# → vad_example_01.png/.pdf, ...   (--threshold 0.2 --min_speech_ms 100 --min_silence_ms 80)
```

Figures are 3.5 in wide at 300 dpi with 7 pt text. The turn-taking figure
labels its rows 3–0 (naive A, naive B, proposed A, proposed B) rather than by
name.

## Listening-test stimuli

```bash
# scheduler vs PersonaPlex backchannels → bc_video_NN.mp4 + ab_mapping.json
python -m neural_scheduler.analysis.stimuli videos-bc \
    --sample_list good_examples.txt --personaplex_dir personaplex/icc_output \
    --out_dir study/backchannel --seed 0

# naive vs scheduled turn-taking → turntake_video_NN.mp4 + ab_mapping.json
python -m neural_scheduler.analysis.stimuli videos-turn-take \
    --selections selections.txt --naive_dir <run>/segment_cache --model_dir <run>/<checkpoint> \
    --out_dir study/turn_taking --seed 0

# arbitrary pairs of mixes → video_NN.mp4 + ab_mapping.json
python -m neural_scheduler.analysis.stimuli videos-paired \
    --sample_list pairs.txt --out_dir study/paired --seed 0
```

Each video is clip "A", a silent pause (`--pause_sec`, default 2 s) and clip
"B" on a black 1280×720 background. Which system is A is drawn at random per
item and recorded in `ab_mapping.json`, which `user_study` accepts as a key
file. Audio is 22.05 kHz mono: each speaker channel is gained to −23 LUFS
independently and the two are summed (`videos-paired` normalises each
already-mixed file as a whole).

The same comparisons as plain wavs, for a demo page or supplementary
material:

```bash
python -m neural_scheduler.analysis.stimuli audio-bc        --sample_list ... --personaplex_dir ... --out_dir audio/bc
python -m neural_scheduler.analysis.stimuli audio-turn-take --selections ... --naive_dir ... --model_dir ... --out_dir audio/tt
python -m neural_scheduler.analysis.stimuli audio-paired    --sample_list pairs.txt --out_dir audio/paired
```

`audio-bc` and `audio-turn-take` average the two channels (no loudness
matching) and write 16-bit wavs named `sample_<id>_{proposed,personaplex}.wav`
and `example_<XXXX>_{naive,proposed}.wav`; `audio-turn-take` exports the
first `--max_samples` (5) ids only. `audio-paired` writes loudness-normalised
`sample_NN_{proposed,naive}.wav`, numbered from `--number_offset` (5) so the
two turn-taking sets can share a folder.

## User studies

Two forced-choice studies were run with a web form. Each item was one A/B
video; respondents chose A or B.

| Study | Items | Comparison | Question | Respondents |
|---|---|---|---|---|
| main, items 1–10 | 10 | scheduler backchannels vs PersonaPlex, same speaker audio (Full-Duplex-Bench ICC samples) | "Which clip had a listener that felt more natural, well-timed, and engaged?" | 20 |
| main, items 11–20 | 10 | scheduled turn-taking vs naive concatenation of the same synthesised lines | "In which clip did the conversation flow feel more natural and less robotic?" | 20 |
| dramatic | 5 | scheduled turn-taking vs naive concatenation on dramatic scenes | "In which clip did the conversation feel more genuine where the turn-taking feel more intentional?" | 14 |

Anonymised responses and A/B keys are in `analysis_data/user_study/` (see
`analysis_data/README.md`). The stimulus videos are not distributed.

```bash
python -m neural_scheduler.analysis.user_study main \
    --csv analysis_data/user_study/responses_main.csv \
    --bc  analysis_data/user_study/ab_key_backchannel.json \
    --tt  analysis_data/user_study/ab_key_turn_taking.json

python -m neural_scheduler.analysis.user_study dramatic \
    --csv analysis_data/user_study/responses_dramatic.csv \
    --key analysis_data/user_study/ab_key_dramatic.json
```

With the shipped data these print:

| Block | Proposed preferred | Wilson 95% CI | Pooled binomial p |
|---|---|---|---|
| Backchannel (vs PersonaPlex) | 135 / 200 = 67.5% | [60.7%, 73.6%] | 8.3e-07 |
| Turn-taking (vs naive), all | 102 / 200 = 51.0% | [44.1%, 57.8%] | 0.83 |
| Turn-taking, items 11–15 | 53 / 100 = 53.0% | [43.3%, 62.5%] | 0.62 |
| Turn-taking, items 16–20 | 49 / 100 = 49.0% | [39.4%, 58.7%] | 0.92 |
| Dramatic turn-taking (vs naive) | 45 / 70 = 64.3% | [52.6%, 74.5%] | 0.022 |

How trials are treated:

* **Per item** — the number of respondents who chose the proposed system,
  with an exact two-sided binomial test against 0.5 (`scipy.stats.binomtest`).
* **Per block** — all respondent × item responses are pooled into one
  proportion; the interval is a Wilson score interval
  (`statsmodels.stats.proportion.proportion_confint`) and the p-value an
  exact binomial test on the pooled count. Responses of one respondent, and
  responses to one item, are treated as independent trials (see Known
  issues).
* **Agreement** — Krippendorff's alpha (nominal) over a respondents × items
  matrix of "chose proposed" indicators, printed only if the `krippendorff`
  package is installed.

The main study reads answers as they are; the `dramatic` analysis strips
surrounding whitespace first.

Key files are JSON lists with one entry per item. The side of the proposed
system is taken from the first field present among `proposed_is_A`,
`model_is_A`, `naive_is_A` (negated) and `first_is_wav_a`, so both the
sanitised keys and the `ab_mapping.json` files written by `stimuli` work.
For the `dramatic` sub-command the key can instead be given inline:
`--model_is_A A B A A A`.

## Synthetic question/answer dialogues

```bash
python -m neural_scheduler.analysis.webquestions_tts \
    --input webquestions/trainmodel.json --output data/voiced_webquestions \
    --max-items 2000 --silence 0.5
```

writes one stereo 22.05 kHz wav per WebQuestions item (`<qId>.wav`): the
question on the left channel (`en_US-lessac-medium`), 0.5 s of silence, the
answer on the right channel (`en_US-amy-medium`). Voices are Piper model
names or `.onnx` paths (`--question_voice`, `--answer_voice`);
`--piper_data_dir` tells Piper where to find or download them. Tokenise the
result with the preprocessing pipeline to obtain the input of the turn-order
perturbation evaluation.

## Migration

Old paths are relative to
`ntpp/model/chatGLM_dual_timing_only/experiments/` in the research tree.

| Research file | Release module |
|---|---|
| `inference_pipeline_10_examples/visualize_multiple_schedule.py` | `viz/timeline.py` |
| `inference_pipeline_10_examples/visualize_timelines.sh` | `python -m neural_scheduler.viz.timeline --scan_dir <dir>` |
| `inference_pipeline_10_examples/visualize_3_pass_v4.py` | `viz/three_pass.py` |
| `render_visualiazation_and_audio.py` | `viz/waveform_video.py` |
| `inference_qualitative_examples/generate_figure_bc.py` | `analysis/figures.py bc` |
| `inference_qualitative_examples/generate_figure_turn_take.py` | `analysis/figures.py turn-take` |
| `inference_qualitative_examples/generate_videos_bc.py` | `analysis/stimuli.py videos-bc` |
| `inference_qualitative_examples/generate_videos_turn_take.py` | `analysis/stimuli.py videos-turn-take` |
| `inference_qualitative_examples/generate_videos_turn_take_dramatic.py` | `analysis/stimuli.py videos-paired` |
| `inference_qualitative_examples/generate_folder_bc.py` | `analysis/stimuli.py audio-bc` |
| `inference_qualitative_examples/generate_folder_turn_take.py` | `analysis/stimuli.py audio-turn-take` |
| `inference_qualitative_examples/generate_folder_turn_take_dramatic.py` | `analysis/stimuli.py audio-paired` |
| `inference_qualitative_examples/user_study_analysis.py` | `analysis/user_study.py main` |
| `inference_qualitative_examples/user_study_analysis_dramatic.py` | `analysis/user_study.py dramatic` |
| `experiment_turnunderstanding/generate_turnbased_speech.py` | `analysis/webquestions_tts.py` |
| Silero loading, wav loading and bar plotting duplicated in the two figure scripts | `viz/vad_plot.py` |
| ffmpeg helpers duplicated in the video and folder scripts | `viz/ffmpeg.py` |
| `inference_qualitative_examples/user_20.csv`, `dramatic_14.csv`, `user_study_*/ab_mapping.json` | `analysis_data/user_study/` (timestamps and paths removed) |

Not carried over: `examine_personaplex_videos.py` (browsing aid for baseline
outputs), the `run_gen_*.sh` one-line wrappers (replaced by the commands
above), `user_19.csv` (an earlier export, a prefix of `user_20.csv`), the
pickled-state command line of `visualize_3_pass_v4.py` (nothing writes that
pickle), and all stimulus audio and video.

Differences from the research scripts:

* Output locations are required arguments; the research scripts defaulted to
  directories under the current directory (and, for the user study and the
  WebQuestions script, to absolute paths on the development machine).
* `videos-turn-take` writes one set of videos. The research file contained
  the script twice, so running it produced `video_NN.mp4` and then
  `turntake_video_NN.mp4` with a new random assignment, and the saved mapping
  described only the second set. `--video_prefix` selects the name
  (default `turntake_video_`, the set that was used in the study).
* `user_study dramatic` requires a key (`--key` or `--model_is_A`); the
  research script defaulted to `A B A A A`, the key of the one study it was
  written for.
* `webquestions_tts --silence` now sets the gap; it was parsed and ignored.
  The default (0.5 s) is what the research script always used.
  The research script also tried to pre-download voices into
  `~/.local/share/piper/voices`; voice lookup is now left to Piper or
  `--piper_data_dir`.
* `viz.timeline` takes `-o` as a required argument unless `--scan_dir` is
  used.
* The waveform video no longer tries `module load ffmpeg` when `ffmpeg` is
  missing (that call could never succeed from Python); it still falls back
  to `moviepy`.

Everything else is unchanged and is covered by equivalence tests against the
research scripts (`tests/test_viz_*.py`, `tests/test_analysis_*.py`): same
`timeline.json` text, same HTML, pixel-identical figures, identical A/B
mappings and audio for a given seed, and character-identical study reports.

## Known issues

* **A/B assignment of the studies cannot be regenerated.** The stimuli were
  produced without `--seed`; the saved mappings (shipped as the sanitised
  keys) are the only record. New stimuli should always be made with
  `--seed`.
* **Pooled tests treat trials as independent.** The block-level interval
  and p-value pool every respondent × item response. Responses from the same
  respondent, and to the same item, are correlated, so the intervals are
  narrower than a mixed-effects or per-respondent analysis would give.
* **No multiple-comparison correction.** Per-item p-values are reported as
  they are.
* **Answer normalisation differs between the two analyses.** `dramatic`
  strips whitespace from answers, `main` does not; an answer such as `"A "`
  would count as a vote for clip B in `main`. The shipped data contain only
  `A` and `B`.
* **The original help text of the dramatic key was wrong.** It described the
  default as "all A" while the default was `A B A A A`; the release has no
  default.
* **Row labels of the turn-taking figure are numeric** (3–0), because the
  named labels were disabled in the script that produced the figure.
* **PersonaPlex muting is a fixed 4 s**, applied to every baseline file
  regardless of its content (`--silence_first_n` on `figures bc` and
  `stimuli audio-bc`; fixed in `stimuli videos-bc`).
* **Loudness measurement needs enough audio.** `ebur128` reports no
  integrated loudness for clips shorter than about 0.4 s, and the tools then
  fail with "Could not parse LUFS".
* **`drawtext` asks for the font "DejaVu Sans"**; where fontconfig cannot
  find it, ffmpeg substitutes another font and the A/B labels look
  different.
* **The waveform video is written next to the first input file**, and
  `create_dual_waveform_video` returns the silent temporary video if neither
  `ffmpeg` nor `moviepy` is available.
* **Timeline labels are found by directory convention** (first
  `segment_cache/example_*` with metadata, up to three levels up). A run
  with several dialogues in one cache labels every timeline with the first
  example's text unless metadata paths are given explicitly.
