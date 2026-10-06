# From the research tree to this repository

The research code lived in a fork of NTPP (`NTPP/` below). This page maps its
main files to their new homes. Every area page ends with a complete table for
that area.

## Where things went

| Research tree | Here | Details |
|---|---|---|
| `ntpp/model/chatGLM_dual_timing_only/modeling_chatglm_timing*.py` (4 files), `configuration_chatglm_timing.py` | `neural_scheduler/model/modeling.py`, `configuration.py` | [model.md](model.md#migration-from-the-research-tree) |
| `load_corrupted_checkpoint` (16 copies) | `neural_scheduler.model.load_scheduler` | [model.md](model.md) |
| `ntpp/train/sft_glm_4_voice_pred_timing_only*.py` (5 files), `custom_collator.py` | `neural_scheduler/train/`, `neural_scheduler/data/{collator,features}.py` | [training.md](training.md#migration-from-the-research-tree) |
| `ntpp/train/launch_training_*.sh` (28 files), `ntpp/config/ds_sft_glm4_voice.json` | `scripts/slurm/train_*.sh`, `configs/deepspeed_zero3.json` | [training.md](training.md) |
| `scripts/processing_for_timing_only_with_duration_prediction/preprocess_{fisher,seamless}_…py`, `merge_shards_*.py`, `visualize_…py`, `validate_…py` | `neural_scheduler/data/` (`preprocess`, `merge`, `inspect` commands) | [data.md](data.md#migration-from-the-research-tree) |
| `BackchannelPrediction/preprocess_scripts/` | `neural_scheduler/backchannel/labeling/` | [backchannel.md](backchannel.md#migration-from-the-research-tree) |
| `BackchannelPrediction/naive_roberta_classifier/` | `neural_scheduler/backchannel/{data,models}/`, `train.py`, `eval.py`, `ablation.py`, `stats.py` | [backchannel.md](backchannel.md) |
| `BackchannelClassifier` classes inside the step-1 scripts | `neural_scheduler/backchannel/predictor.py` | [backchannel.md](backchannel.md) |
| `…/inference_pipeline_bc_only_full_duplex_bench/step_1_…_with_audio{,_turbo}.py`, `…/inference_pipeline_10_examples/step_1_….py` | `neural_scheduler/pipeline/synthesize.py`, `tts.py` | [pipeline.md](pipeline.md#migration-from-the-research-tree) |
| `step_2_tokenization_with_source_map.py` (both folders), `step_2_tokenization_util.py` | `neural_scheduler/pipeline/tokenize.py`, `tokenize_icc.py` | [pipeline.md](pipeline.md) |
| `step_3_inference_reconstruct.py` (both folders) | `neural_scheduler/pipeline/schedule.py` | [scheduler.md](scheduler.md#10-migration-from-the-research-code) |
| `segment_level_search_v6.py`, `segment_level_search_v6_bc_only.py` | `neural_scheduler/scheduler/` | [scheduler.md](scheduler.md) |
| `step_4_reconstruct_audio_with_schedule_result.py` | `neural_scheduler/pipeline/reconstruct.py` | [scheduler.md](scheduler.md) |
| ~45 `run_pipeline*.sh` / `pipeline_*.sh` | `neural_scheduler/pipeline/run.py` + `configs/pipeline/*.yaml` | [pipeline.md](pipeline.md) |
| `visualize_multiple_schedule.py`, `visualize_3_pass_v4.py`, `render_visualiazation_and_audio.py`, `visualize_timelines.sh` | `neural_scheduler/viz/{timeline,three_pass,waveform_video}.py` | [analysis.md](analysis.md#migration) |
| `experiment_with_mutual_silence.py` | `neural_scheduler/evaluation/timing_distribution.py` | [evaluation.md](evaluation.md#migration-from-the-research-tree) |
| `varying_order_helpers.py`, `experiment_turnunderstanding/run_turn_perturbation_experiment.py` | `neural_scheduler/evaluation/turn_order.py` | [evaluation.md](evaluation.md) |
| `varying_silence_helpers.py`, the silence blocks of `experiment_with_perplexity.py` | `neural_scheduler/evaluation/silence_perturbation.py` | [evaluation.md](evaluation.md) |
| `nll_diagnostics.py`, `nll_diagnostics_2.py`, `experiment_with_nll.py` | `neural_scheduler/evaluation/{nll_diagnostics,onset_benchmark}.py` | [evaluation.md](evaluation.md) |
| `eval_full_duplex_bench_prep_data.py`, the patched `Full-Duplex-Bench/…/eval_backchannel.py` | `neural_scheduler/evaluation/fdb/` | [evaluation.md](evaluation.md) |
| `inference_qualitative_examples/user_study_analysis*.py`, `generate_figure_*.py`, `generate_videos_*.py`, `generate_folder_*.py` | `neural_scheduler/analysis/{user_study,figures,stimuli}.py` | [analysis.md](analysis.md#migration) |
| `experiment_turnunderstanding/generate_turnbased_speech.py` | `neural_scheduler/analysis/webquestions_tts.py` | [analysis.md](analysis.md) |
| special-token lists and magic numbers (8+ copies) | `neural_scheduler/vocab.py`, `neural_scheduler/constants.py` | |

## Not carried over

| Research tree | Why |
|---|---|
| `inference_pipeline_Feb16/`, `inference_pipelin_March31/`, `inference_pipeline_April_10/` | earlier snapshots of the pipeline |
| `modeling_chatglm_timing_duration_pred_logic_clock.py` and its trainers | turn-id embedding variant; superseded by turn ids as tokens, and its vocabulary is incompatible |
| `scheduled_sampling.py`, `duration_weighted_loss.py` | experiments not used by the released checkpoint |
| `ntpp/model/chatGLM_dual/`, `sft_glm_4_voice.py`, `sft_openMOSS.py`, `ntpp/inference.py` | predecessors and unrelated experiments |
| `dual_llama.py`, `dual_phi.py`, `pretrain.py`, `sft.py`, `ntpp/utils/`, `higgs_audio_inference/` | upstream NTPP code; see <https://github.com/Chaos96/NTPP> |
| `scripts/preprocessing_pipeline/`, `scripts/preprocessing_for_timing_only/`, `add_duration*.py` | earlier generations of the data pipeline |
| `dtw_module.py`, `chatGLM_dual_timing_only/inference.py`, `examine_data_for_silence.py`, `experiment_with_inferencing.py`, `experiment_with_perplexity.py`, `run_inferencing_naive*.py` | superseded or scratch; the functions the evaluations need are in `evaluation/common.py` |
| `train_classifier.py` (7-class text-only) and the first two `extract_backchannel_dataset*.py` | superseded |
| `Full-Duplex-Bench/` | third-party benchmark (CC BY-NC 4.0) and its data |
| scene folders built from film / TV audio, reference voices, generated audio and video, datasets, checkpoints, logs | third-party or generated content |

## Behaviour changes

The restructuring is behaviour-preserving by default. The deliberate
exceptions — crash fixes, options that are now required instead of pointing
at a fixed folder, relative paths in caches — are listed per area in the
pages linked above; behaviour that was kept although it is questionable is in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).
