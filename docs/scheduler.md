# Scheduler search

`neural_scheduler.scheduler` decides *when* each turn and each backchannel of a
scripted two-speaker dialogue happens. It does not generate anything: the
content of every turn is fixed, and the scheduler LM is only used to score
candidate timings.

| Module | Contents |
|---|---|
| `scheduler/pipeline.py` | `generate_dialogue_three_pass`, `generate_dialogue_bc_only` |
| `scheduler/pass1.py` | onsets of primary turns |
| `scheduler/pass1_anchored.py` | primary turns at known positions (backchannel-only mode) |
| `scheduler/pass2.py` | backchannel onsets and host pauses |
| `scheduler/pass3.py` | re-timing after backchannels |
| `scheduler/scoring.py` | NLL of a candidate under the model |
| `scheduler/search.py` | "score every candidate, keep the first best" |
| `scheduler/script_parse.py` | reading turns out of the script |
| `scheduler/subsegments.py`, `dialogue.py` | splitting / joining speech blocks, building token sequences |
| `scheduler/config.py` | the defaults of every hyperparameter |
| `scheduler/types.py` | `TurnInfo`, `PlacedTurn`, `PlacedSubSegment`, `SchedulingState` |
| `pipeline/schedule.py` | step-3 command line |
| `pipeline/reconstruct.py` | step 4: schedule → audio |

## 1. Input

The search works on the sequence written by the tokenisation step:

```
<|start-of-script-A|> SIL <|start-turn-0|> speech [SIL speech ...] <|end-turn-0|> SIL ... <|end-of-script-A|>
<|start-of-script-B|> SIL <|start-turn-1|> speech ...              <|end-turn-1|> SIL ... <|end-of-script-B|>
<|start-of-dialogue|>
```

* Speech tokens are GLM-4-Voice codec tokens at 12.5 Hz (one token = one
  *step* = 80 ms). Everything below is measured in steps.
* Turn ids give the conversational order. Inside a turn, a silence token
  separates *speech blocks* (called sub-segments in the code); these come from
  VAD pauses inside the voice line.
* A turn id that appears in only one script is a **primary turn**. A turn id
  that appears in both scripts is a primary turn on one channel and a
  **backchannel** on the other; the primary turn is its *host*.
* Anything after `<|start-of-dialogue|>` is ignored.

## 2. What is searched and how it is scored

A candidate schedule is an assignment of an onset step to every turn placed so
far. It is rendered as an interleaved dialogue `A0 B0 A1 B1 ...` in which each
turn's tokens are written at its onset on its channel and every other position
is the silence token.

The **score** of a candidate is the sum of per-token negative log-likelihoods
under the scheduler LM of

```
script + <|start-of-dialogue|> + candidate dialogue + <|end-of-dialogue|>
```

restricted to a window of the dialogue around what is being placed. Channel
ids are 0 for the script and 1 / 2 alternating over the dialogue. Each
candidate costs one full forward pass (`use_cache=False`, batch size 1).

There is no beam. Every decision is a one-dimensional scan: enumerate the
integer candidates, score each, commit the lowest (the earliest on ties), move
on. Turns that have not been placed yet are absent (silence) while earlier
ones are scored.

## 3. The three passes

### Pass 1 – primary turns (`pass_1_primary_turns`)

The model sees a script from which the backchannel turns have been removed.
Primary turns are placed in turn-id order. For each turn:

1. `anchor = end of the previously placed turn + inter_turn_gap`
   (`inter_turn_gap` for the first turn); window = `anchor ± search_window`,
   clipped at 0.
2. With `use_speaker_gap_constraints`, the window is intersected with
   `[prev_end + min, prev_end + max]`. For a speaker switch the range is the
   incoming speaker's `speaker_N_{min,max}_gap`; when the previous turn
   belongs to the same speaker it is `speaker_N_subturn_{min,max}_pause`. If
   the intersection is empty the gap range alone is used.
3. With `strong_overlap_mode`, a speaker switch *replaces* the window by
   `[prev_end − ceil(f_max·L), prev_end − int(f_min·L)]`, where `L` is the
   incoming turn's length and the fractions are capped so the overlap cannot
   exceed the previous turn.
4. The onset can never be earlier than the end of an already placed turn of
   the same speaker.
5. Each onset in the window (stride `search_stride`) is scored on
   `[onset − scoring_context, onset + duration + scoring_context]`.

The turn's speech tokens are placed back to back: pauses that were in the voice
line are dropped. With `schedule_subturns`, a turn with several speech blocks is
instead scheduled block by block. The first block is searched as above (later
blocks provisionally follow at the minimum pause), then for each later block
the pause before it is searched over the speaker's
`[subturn_min_pause, subturn_max_pause]`. The committed turn is one token list
with silence filling the chosen pauses.

### Pass 2 – backchannels (`pass_2_backchannel_placement`)

The model sees the full script. Backchannels are processed in script order.
For a backchannel of length `d` on a host spanning `[host_start, host_end)`:

* earliest onset: `host_start + int(host_len · (1 − bc_position_limit))`, and
  not before the end of an earlier backchannel on the same host;
* latest onset: `host_end − round(d · max_bc_overlap_ratio)`;
* strategies (for the first backchannel on a host only): `no_pause` – the host
  keeps talking; `full_pause` – `d` steps of silence are inserted into the host
  at the backchannel onset; `half_pause` – `d // 2` steps. `bc_effect`
  restricts the set (`np`, `fp`, `hp`, joined by `+`).

Every (onset, strategy) pair is scored as `NLL(with backchannel) − NLL(without)`
on one fixed window covering the host, the latest possible backchannel and a
full pause, plus `scoring_context`. The pair with the lowest difference wins.

If the winner includes a pause, the host is rebuilt with the silence inserted
and **everything that started at or after the pause, on either channel, is
shifted** by its length. With `score_only_pauses` and a `bc_effect` that
excludes `np`, the pause is used for scoring only and the host is left as it
was.

### Pass 3 – re-timing (`pass_3_refinement`)

Pass 2 leaves the rest of the host and the next speaker's turn where pass 1
put them. For every host with backchannels, pass 3:

1. splits the host into its speech blocks;
2. builds *anchors*: each block after the first (`subseg`), and the next
   primary turn by onset (`next_turn`);
3. assigns each backchannel to an anchor – a `subseg` anchor if the backchannel
   lies in the gap before that block, straddles the end of the block before it,
   or lies inside the block before it; the `next_turn` anchor if it lies in the
   host's last block or after the host;
4. for each anchor with backchannels, searches its onset from the end of
   whatever precedes it on its channel up to `current onset + total length of
   the backchannels before it + max_extra_gap`, moving rigidly either the block
   and everything of the host after it, or the next turn and its own
   backchannels;
5. rebuilds the host and propagates the resulting shifts to later turns and
   their backchannels.

## 4. Backchannel-only mode

`generate_dialogue_bc_only` is for the benchmark setting where speaker 1 is a
recording and only speaker 2's backchannels are generated. Pass 1 is replaced
by `pass_1_anchor_primary_turns`: every channel-A turn listed in
`s1_turn_timing` is laid out exactly as given (blocks at the given start steps,
silence between them), with no model call. Pass 2 runs unchanged. Pass 3 is
skipped, because it would move speaker 1.

`s1_turn_timing` entries are
`{"turn_id", "start_token", "end_token", "sub_segments": [{"start_token", "end_token"}, ...]}`
in dialogue steps; `neural_scheduler.pipeline.tokenize_icc` writes them.

A primary turn without an entry is placed at its `start_token`, which is a
position inside the script, not in the dialogue. Benchmark inputs contain no
such turns; a warning is logged if one appears.

## 5. Python API

```python
from neural_scheduler import vocab
from neural_scheduler.model import load_scheduler
from neural_scheduler.scheduler import generate_dialogue_three_pass

model, tokenizer = load_scheduler(checkpoint_dir, device="cuda:0")
channel_a, channel_b, state = generate_dialogue_three_pass(
    model, tokenizer, input_ids, vocab.get_special_token_ids(tokenizer),
    pass1_search_window=10, pass2_bc_effect="hp", device="cuda:0",
)
```

`channel_a` / `channel_b` are equal-length lists of codec ids (8520 = silence),
one per step. `state.refined_placements` is the final schedule.

### Parameters

Defaults live in `scheduler/config.py`. Steps are 80 ms.

| Parameter (`generate_dialogue_three_pass`) | Default | Meaning |
|---|---|---|
| `pass1_search_window` | 15 | half-width of the onset window around the anchor |
| `pass1_search_stride` | 1 | step between candidate onsets |
| `pass1_scoring_context` | 15 | steps scored on each side of the candidate turn |
| `pass1_inter_turn_gap` | 5 | anchor = previous end + this |
| `pass_1_schedule_subturns` | False | search pauses between a turn's speech blocks |
| `pass_1_speaker_{1,2}_subturn_min_pause` | 0 | smallest such pause (negative values act as 0); also the gap range lower bound between consecutive turns of the same speaker when gap constraints are on |
| `pass_1_speaker_{1,2}_subturn_max_pause` | 10 | largest such pause / upper bound |
| `pass_1_subturn_search_stride` | 1 | step between candidate pauses |
| `pass_1_use_speaker_gap_constraints` | False | restrict onsets to a gap range after the previous turn |
| `pass_1_speaker_{1,2}_min_gap` | −5 | smallest gap when that speaker takes the floor; negative = overlap |
| `pass_1_speaker_{1,2}_max_gap` | 20 | largest gap |
| `pass_1_strong_overlap_mode` | False | force speaker switches to overlap |
| `pass_1_strong_overlap_min` / `_max` | 0.4 / 0.6 | overlap as a fraction of the incoming turn |
| `pass2_search_stride` | 1 | step between candidate backchannel onsets |
| `pass2_scoring_context` | 10 | steps scored beyond the host on each side |
| `pass2_bc_position_limit` | 1.0 | fraction of the host's tail in which a backchannel may start |
| `pass2_bc_effect` | None (all) | allowed strategies: `np`, `fp`, `hp` joined by `+` |
| `pass2_max_bc_overlap_ratio` | 0.5 | fraction of the backchannel that must overlap its host |
| `pass2_score_only_pauses` | False | with `np` excluded, score with pauses but do not insert them |
| `pass3_scoring_context` | 10 | steps scored on each side of the moved material |
| `pass3_max_extra_gap` | 10 | silence budget beyond the preceding backchannels' length |
| `device`, `verbose` | `"cuda"`, True | |

`generate_dialogue_bc_only` takes `s1_turn_timing` and the `pass2_*`
parameters only.

**Two defaults differ between the function and the command line**, as they
always have: `pass2_scoring_context` is 10 in Python and 0 on the command
line, and `pass2_bc_effect` is all strategies (`None`) in Python and
`"hp+fp+np"` on the command line (the same set). Both are recorded in
`config.CLI_DEFAULT_OVERRIDES`. The values used for the published examples are
in `configs/pipeline/`.

## 6. Command line (step 3)

```
python -m neural_scheduler.pipeline.schedule \
    --tokenized_jsonl OUT/tokenized_result.jsonl --cache_base_dir OUT/segment_cache \
    --output_root OUT/scheduled --checkpoint CKPT [--mode three_pass|bc_only] [options]
```

| Flag | Default | |
|---|---|---|
| `--tokenized_jsonl` | required | JSONL from the tokenisation step |
| `--cache_base_dir` | – | only for samples without `source_cache` (`<dir>/example_XXXX`) |
| `--output_root` | required | |
| `--output_sub_dir` | `segment_search_v2` | folder that receives the scheduled dialogue |
| `--checkpoint` | required | scheduler checkpoint directory |
| `--device` | `cuda:0` | |
| `--mode` | `three_pass` | or `bc_only` |
| `--num_examples`, `--randomize` | all | subset of the JSONL |
| `--output_sr` | 22050 | sample rate of written WAVs |
| `--save_video` / `--no-save_video` | on for `three_pass`, off for `bc_only` | waveform video (needs ffmpeg, OpenCV) |
| `--save_input_scripts` / `--no-save_input_scripts` | on | also write the unscheduled voice lines |
| `--pass1_search_window`, `--pass1_search_stride`, `--pass1_scoring_context`, `--pass1_inter_turn_gap` | 15, 1, 15, 5 | |
| `--pass_1_schedule_subturns` | off | |
| `--pass_1_speaker_{1,2}_subturn_{min,max}_pause`, `--pass_1_subturn_search_stride` | 0, 10, 1 | |
| `--pass_1_use_speaker_gap_constraints` | off | |
| `--pass_1_speaker_{1,2}_{min,max}_gap` | −5, 20 | |
| `--pass_1_use_strong_overlap_constraints` | off | enables `pass_1_strong_overlap_mode` |
| `--pass_1_strong_overlap_min`, `--pass_1_strong_overlap_max` | 0.4, 0.6 | |
| `--pass2_search_stride`, `--pass2_scoring_context`, `--pass2_bc_position_limit` | 1, **0**, 1.0 | |
| `--pass2_bc_effect`, `--pass2_max_bc_overlap_ratio` | `hp+fp+np`, 0.5 | |
| `--pass2_score_only_pauses` | off | |
| `--pass3_scoring_context`, `--pass3_max_extra_gap` | 10, 10 | |

A sample's `source_cache` may be absolute or relative to the JSONL file, so a
run directory can be moved as a whole.

### Output

```
<output_root>/<checkpoint parent>_<checkpoint name>/sample_<id>/
    input_script/channel_{a,b,sum}.wav        voice lines before scheduling
    <output_sub_dir>/
        channel_a.wav channel_b.wav channel_sum.wav
        timeline.json
        timeline.png pass1_nll.png pass1_shifts.png pass2_nll.png pass2_strategies.png
        pass3_nll.png pass3_summary.png
        dual_channel_waveform.mp4             with --save_video
```

`<id>` is the line index in `three_pass` mode and the cache directory name in
`bc_only` mode (the benchmark sample id).

### `timeline.json`

Written by `neural_scheduler.viz.timeline.export_scheduling_state_json`.

* `meta.token_rate_hz`
* `primary_turns[]`, `backchannel_turns[]`, `backchannel_units[]` – turns as
  read from the script: `turn_id, channel, start_token, end_token, duration,
  sub_segments[{start, end}], is_backchannel, host_turn_id, bc_group_index`
* `placed_primary[]`, `placed_backchannels[]`, `refined_placements[]` – the
  same plus `onset_step, gap_to_previous, bc_offset_from_host_start, end_step`.
  `refined_placements` is the final schedule.
* `pass1_nll_scores{turn_id: [...]}`, `pass1_subturn_nll_scores`,
  `pass2_nll_scores{"(turn_id, group)": [...]}`
* `pass3_adjustments{turn_id: {original_turn_end, new_turn_end, total_shift}}`,
  `pass3_sub_segment_details{turn_id: [breakpoint results]}`
* `diagnostics{pass1, pass2, pass3}` – candidates, scores and choices of
  every decision.

## 7. Reconstruction (step 4)

`pipeline/reconstruct.py` turns the two lists of codec ids into audio without a
vocoder. The tokenisation step stored, for each speech token of a speaker's
script, the index of the 80 ms slice of that speaker's audio it came from. For
each scheduled token the next script token with the same codec id (at or after
a running pointer) is found and its slice (1280 samples at 16 kHz) is copied;
silence becomes zeros. A token with no match ahead reuses the first match
behind the pointer; a token with no match at all is silent. The log line
`N matched, N silence, N out of order, N unmatched` is the check that the
schedule reproduced the script.

Outputs are 16 kHz internally and resampled to `--output_sr` when written.

## 8. Performance

* One forward pass per candidate, over the whole sequence. A dialogue with
  `T` primary turns and `B` backchannels costs roughly
  `T·(2·search_window + 1) + Σ_B (host length × strategies) + pass 3` forward
  passes. Pass 2 dominates when `bc_position_limit` is large.
* The script prefix and the dialogue before the scoring window are identical
  across the candidates of one decision; they are not reused.
* In pass 1 every candidate dialogue is padded to an upper bound of the whole
  conversation's length, also for the first turn.
* `search_stride` / `pass2_search_stride` reduce the cost linearly.

## 9. Known issues

These are properties of the algorithm as it was used for the published
results. They are kept so that the release reproduces those results.

1. **Pass 3 mis-splits a host that already contains silence.** Pass 3 divides
   the host's token list among its speech blocks in proportion to their
   durations and then re-inserts the gaps recorded in the block metadata. If
   pass 2 inserted a pause (or pass 1 scheduled sub-turn pauses), the token
   list already contains that silence, so filler tokens are handed to the
   neighbouring blocks *and* the gap is added again. Example: a 20-token host
   at onset 50 receives a 6-step pause at step 60 and ends at 76 after pass 2.
   Pass 3 starts from block 0 = 10 speech + 3 silence and block 1 = 3 silence +
   10 speech at onset 69, ending at 82; speech resumes at step 72 instead of
   66. The pass-3 search can remove the re-added gap but not the embedded
   filler.
2. **Roles follow duration.** When a turn id is on both channels, the channel
   with more speech is the primary turn. A backchannel longer than its host
   (a short host such as "Nothing." with a long backchannel phrase) becomes
   the primary turn and the host becomes the backchannel. The
   `type: speech | backchannel` written by the tokenisation step is not used.
   (`tests/test_scheduler_unit.py::test_known_issue_longer_backchannel_becomes_the_primary_turn`.)
3. **Pauses inside a voice line are dropped** unless
   `pass_1_schedule_subturns` is on, because a turn's tokens are read from the
   script without its silence. Multi-word backchannels lose their internal
   pause for the same reason.
4. **Negative sub-turn minimum pause.** `speaker_N_subturn_min_pause < 0` is
   treated as 0 when searching pauses inside a turn, but is used as given for
   the gap range between consecutive turns of one speaker (where the
   same-speaker floor then applies).
5. **Start-of-dialogue bias.** The scoring window is clipped at step 0, so the
   earliest candidates of the first turn are scored on fewer tokens and tend
   to win.
6. **Backchannel-only fallback.** A primary turn without known timing is not
   searched (section 4).
7. **Silence code inside speech.** Codec id 8520 is treated as silence
   everywhere; a speech frame that the tokenizer maps to 8520 acts as a block
   boundary and is reconstructed as silence.

## 10. Migration from the research code

Source: `ntpp/model/chatGLM_dual_timing_only/experiments/` in the NTPP tree.

| Research code | Release |
|---|---|
| `inference_pipeline_10_examples/segment_level_search_v6.py` | `neural_scheduler/scheduler/` |
| `TurnInfo`, `PlacedTurn`, `PlacedSubSegment`, `SchedulingState` | `scheduler/types.py` (same fields) |
| `compute_turn_boundaries_from_script_only`, `parse_turn_structure`, `_extract_script_turn_content`, `strip_backchannels_from_script` | `scheduler/script_parse.py` (no `tokenizer` argument) |
| `build_dialogue_from_placements`, `build_dialogue_from_sub_segments_and_turns` | `scheduler/dialogue.py` |
| `compute_nll_for_placement` | `scheduler/scoring.py` |
| `_assemble_turn_from_sub_segments`, `_split_turn_tokens_by_sub_segments`, `_build_host_turn_with_pause`, `split_turn_into_sub_segments`, `compute_original_sub_segment_onsets` | `scheduler/subsegments.py` |
| `pass_1_primary_turns` | `scheduler/pass1.py` |
| `parse_bc_effect`, `pass_2_backchannel_placement` | `scheduler/pass2.py` |
| `pass_3_refinement` | `scheduler/pass3.py` |
| `_build_s1_placed_turn_from_gt`, `pass_1_anchor_s1_search_s2` | `scheduler/pass1_anchored.py` (`pass_1_anchor_primary_turns`) |
| `generate_dialogue_three_pass`, `generate_dialogue_bc_only` | `scheduler/pipeline.py` |
| `inference_pipeline_bc_only_full_duplex_bench/segment_level_search_v6_bc_only.py` | same package; its pass-2 difference is `pass2_score_only_pauses` |
| `step_3_inference_reconstruct.py` (both copies) | `pipeline/schedule.py` (`--mode`) |
| `load_corrupted_checkpoint` | `neural_scheduler.model.load_scheduler` |
| `get_special_tokens` (four copies) | `neural_scheduler.vocab.get_special_token_ids` |
| `step_4_reconstruct_audio_with_schedule_result.py` | `pipeline/reconstruct.py` |

Removed:

* functions that were never called: `_allocate_tokens_to_sub_segments`,
  `compute_turn_boundaries_from_input_ids`, `compute_nll_batch`,
  `_extract_script_turn_content_segmented`, `split_backchannel_segments`, the
  two identity `rebuild_script_with_backchannels` helpers;
* parameters that had no effect: `pass1_min_gap` / `--pass1_min_gap`,
  `pass3_search_window` / `--pass3_search_window`,
  `pass2_bc_gap_threshold_steps`, the `tokenizer` argument of the passes and
  helpers, and the `pass1_*` arguments of `generate_dialogue_bc_only`;
* from step 3: constrained beam search and its helpers, `SearchVisualizer`,
  token-to-waveform decoding through the GLM-4-Voice decoder.

Changes that are visible in outputs (none affects a schedule):

* `pass3_sub_segment_details` in `SchedulingState` / `timeline.json` now holds
  the breakpoint results of each host. It was always empty because a key was
  misspelled.
* `gap_constraint_applied` in the pass-1 diagnostics of single-block turns now
  says whether a speaker-switch gap constraint was applied. It was always
  False.
* Turns anchored in backchannel-only mode report `nll_scores: [0.0]` (as the
  benchmark copy did), not `[]`.
* `pass_1_primary_turns` called directly defaults to `inter_turn_gap=5`, the
  value both entry points always passed (its own default used to be 4).
* `--save_video` and `--save_input_scripts` can now be switched off.
* In `bc_only` mode the benchmark copy applied "score-only pauses" whenever
  `--pass2_bc_effect` excluded `np`; it is now the explicit
  `--pass2_score_only_pauses`. Benchmark runs used `np`, where the two are the
  same.

## 11. Tests

```
NTPP_ORIG=/path/to/NTPP pytest tests/test_scheduler_*.py tests/test_reconstruct_*.py
```

`test_scheduler_equivalence.py` and `test_scheduler_passes_equivalence.py`
import the research implementation from `NTPP_ORIG` and run it next to this
package with the same deterministic stub model over random scripts and random
hyperparameters; schedules, output tokens, the number of model calls and all
recorded scores must be identical (apart from the three diagnostic changes
above). `test_reconstruct_equivalence.py` does the same for audio
reconstruction. Without `NTPP_ORIG` these are skipped and the unit and
command-line tests still run.
