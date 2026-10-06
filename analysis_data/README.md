# Analysis data

## `user_study/`

Responses of the two forced-choice listening studies, and the keys needed to
score them. Everything here is anonymous: the response form collected no
names, e-mail addresses or free text, and the submission timestamps of the
original export have been replaced by a row index.

| File | Content |
|---|---|
| `responses_main.csv` | 20 respondents × 20 items |
| `responses_dramatic.csv` | 14 respondents × 5 items |
| `ab_key_backchannel.json` | key for items 1–10 of the main study |
| `ab_key_turn_taking.json` | key for items 11–20 of the main study |
| `ab_key_dramatic.json` | key for the 5 items of the second study |

### Response files

One row per respondent, in submission order. The first column, `respondent`,
is an index with no meaning beyond the row. Every other column is one item;
its header is the question shown for that item and its value is the clip the
respondent chose, `A` or `B`. Each item was a video playing clip A, a
two-second pause, then clip B.

Questions:

* Main study, items 1–10 (backchannels; the proposed scheduler against
  PersonaPlex, both listening to the same speaker recording):
  "Which clip had a listener that felt more natural, well-timed, and
  engaged?"
* Main study, items 11–20 (turn-taking; the proposed scheduler against naive
  back-to-back concatenation of the same synthesised lines):
  "In which clip did the conversation flow feel more natural and less
  robotic?"
* Second study, items 1–5 (turn-taking on dramatic scenes; scheduler against
  naive concatenation):
  "In which clip did the conversation feel more genuine where the turn-taking
  feel more intentional?"

### Key files

A JSON list with one entry per item, in column order:

```json
{"item": 1, "video_file": "bc_video_01.mp4", "proposed_is_A": true}
```

`proposed_is_A` says whether the proposed scheduler was clip A (otherwise it
was clip B and the baseline was clip A). `video_file` is the name the item's
video had in the study and is only used as a label in the report. The
stimulus videos and audio are not distributed.

### Reproducing the reported numbers

```bash
python -m neural_scheduler.analysis.user_study main \
    --csv analysis_data/user_study/responses_main.csv \
    --bc  analysis_data/user_study/ab_key_backchannel.json \
    --tt  analysis_data/user_study/ab_key_turn_taking.json

python -m neural_scheduler.analysis.user_study dramatic \
    --csv analysis_data/user_study/responses_dramatic.csv \
    --key analysis_data/user_study/ab_key_dramatic.json
```
