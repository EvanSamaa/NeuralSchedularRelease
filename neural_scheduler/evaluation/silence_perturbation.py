"""Silence perturbation: does the scheduler notice an unnatural pause length?

A speech segment is picked in one channel of a ground-truth dialogue, the
silence before (or after) it is lengthened or shortened by a range of deltas,
the dialogue is truncated just after the segment, and each variant is scored.
A model that has learned timing should score the recorded pause (delta 0) best.

Scoring compatibility. The curves produced during the research were scored with
two quirks, both reproduced by ``legacy_scoring=True`` (the default):

* channel ids were 0 for the script *and* dialogue A, and 1 for dialogue B,
  instead of the 0 / 1 / 2 layout the model is trained with;
* the two return values of the scorer were swapped on unpacking, so the field
  named ``perplexities`` (and ``original_perplexity``) holds the summed NLL of
  the truncated dialogue and ``nlls`` holds its perplexity.

``legacy_scoring=False`` (``--corrected_scoring`` on the command line) uses the
trained channel layout and stores each quantity under its own name.

    python -m neural_scheduler.evaluation.silence_perturbation \\
        --checkpoint CKPT --data_jsonl VAL.jsonl --out_dir OUT --seed 0
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from ..constants import SILENCE_CODE
from .common import (
    compute_full_nll,
    extract_script_contents,
    load_jsonl_examples,
    prepare_inference_input,
    remove_duplicated_silence_tokens,
    to_serializable,
)
from .segments import SpeechSegment, find_speech_segments


def _score(model, context_ids, context_length, dialogue_a, dialogue_b, device, legacy_scoring: bool):
    """Returns the pair callers unpack as ``(nll, ppl)``; see the module docstring."""
    perplexity, total_nll = compute_full_nll(
        model, context_ids, context_length, dialogue_a, dialogue_b,
        device=device, legacy_channel_ids=legacy_scoring,
    )
    return (perplexity, total_nll) if legacy_scoring else (total_nll, perplexity)


def perturb_silence_before_segment(
    token_sequence: List[int],
    segment: SpeechSegment,
    new_silence_len: int,
    silence_token: int = SILENCE_CODE,
    truncate_after_segment: bool = True,
    include_post_silence_tokens: int = 0,
) -> List[int]:
    """
    Create a new token sequence with modified silence before the given segment.

    Args:
        token_sequence: Original token sequence
        segment: The speech segment to modify silence before
        new_silence_len: New number of silence tokens to insert
        silence_token: The silence token value
        truncate_after_segment: If True, truncate sequence after segment ends
        include_post_silence_tokens: Number of silence tokens to include after segment
                                     (only used if truncate_after_segment=True)

    Returns:
        Modified token sequence
    """
    # Parts: before the pre-silence, new silence, the segment
    pre_silence_start = segment.start_idx - segment.pre_silence_len

    before_part = token_sequence[:pre_silence_start]
    new_silence = [silence_token] * new_silence_len
    segment_part = token_sequence[segment.start_idx:segment.end_idx]

    if truncate_after_segment:
        # Optionally include some post-silence tokens for context
        post_silence = [silence_token] * include_post_silence_tokens
        return before_part + new_silence + segment_part + post_silence
    else:
        after_part = token_sequence[segment.end_idx:]
        return before_part + new_silence + segment_part + after_part


def perturb_silence_after_segment(
    token_sequence: List[int],
    segment: SpeechSegment,
    new_silence_len: int,
    silence_token: int = SILENCE_CODE,
    truncate_after_silence: bool = True,
) -> List[int]:
    """
    Create a new token sequence with modified silence after the given segment.
    Truncates after the new silence by default.

    Args:
        token_sequence: Original token sequence
        segment: The speech segment to modify silence after
        new_silence_len: New number of silence tokens to insert
        silence_token: The silence token value
        truncate_after_silence: If True, truncate sequence after the new silence

    Returns:
        Modified token sequence
    """
    before_part = token_sequence[:segment.end_idx]
    new_silence = [silence_token] * new_silence_len

    if truncate_after_silence:
        return before_part + new_silence
    else:
        post_silence_end = segment.end_idx + segment.post_silence_len
        after_part = token_sequence[post_silence_end:]
        return before_part + new_silence + after_part


def truncate_to_segment(
    token_sequence: List[int],
    segment: SpeechSegment,
    include_post_silence_tokens: int = 0,
    silence_token: int = SILENCE_CODE,
) -> List[int]:
    """
    Truncate token sequence to end right after the given segment.

    Args:
        token_sequence: Original token sequence
        segment: The speech segment to truncate after
        include_post_silence_tokens: Number of silence tokens to include after segment
        silence_token: The silence token value

    Returns:
        Truncated token sequence
    """
    truncated = token_sequence[:segment.end_idx]
    if include_post_silence_tokens > 0:
        truncated += [silence_token] * include_post_silence_tokens
    return truncated


def run_silence_perturbation_experiment(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    context_length: int,
    dialogue_ids_a: List[int],
    dialogue_ids_b: List[int],
    offset: int,
    device: str = "cuda",
    perturbation_range: range = range(-10, 21),
    num_segments_to_test: int = 5,
    random_seed: int = 42,
    perturb_location: str = "before",  # "before" or "after"
    min_segment_length: int = 1,  # Minimum number of tokens in segment
    max_segment_length: int = None,  # Maximum number of tokens in segment (None = no limit)
    min_silence_length: int = 1,  # Minimum original silence length to consider
    max_silence_length: int = None,  # Maximum original silence length (None = no limit)
    legacy_scoring: bool = True,
) -> Dict:
    """
    Run the silence perturbation experiment.

    Args:
        model: The LLM model
        tokenizer: The tokenizer
        context_ids: Context token tensor
        context_length: Length of context
        dialogue_ids_a: Dialogue tokens for channel A (with offset applied)
        dialogue_ids_b: Dialogue tokens for channel B (with offset applied)
        offset: Token offset (to convert between audio tokens and token IDs)
        device: CUDA device
        perturbation_range: Range of silence length deltas to test
        num_segments_to_test: Number of random segments to test
        random_seed: Random seed for reproducibility
        perturb_location: Whether to perturb silence "before" or "after" segment
        min_segment_length: Minimum number of tokens in the speech segment
        max_segment_length: Maximum number of tokens in the speech segment (None = no limit)
        min_silence_length: Minimum original silence length to consider for perturbation
        max_silence_length: Maximum original silence length (None = no limit)
        legacy_scoring: Score as the research code did (see module docstring)

    Returns:
        Dictionary with experiment results
    """
    random.seed(random_seed)

    # Convert to audio token space (remove offset)
    dialogue_a_audio = [x - offset for x in dialogue_ids_a]
    dialogue_b_audio = [x - offset for x in dialogue_ids_b]

    silence_token = SILENCE_CODE

    # Find speech segments in both channels
    segments_a = find_speech_segments(dialogue_a_audio, silence_token)
    segments_b = find_speech_segments(dialogue_b_audio, silence_token)

    print(f"Found {len(segments_a)} segments in channel A")
    print(f"Found {len(segments_b)} segments in channel B")

    # Filter segments based on criteria
    def filter_segment(segment: SpeechSegment) -> bool:
        # Check segment length
        segment_length = segment.end_idx - segment.start_idx
        if segment_length < min_segment_length:
            return False
        if max_segment_length is not None and segment_length > max_segment_length:
            return False

        # Check silence length based on perturb_location
        if perturb_location == "before":
            silence_len = segment.pre_silence_len
        else:
            silence_len = segment.post_silence_len

        if silence_len < min_silence_length:
            return False
        if max_silence_length is not None and silence_len > max_silence_length:
            return False

        return True

    valid_segments_a = [s for s in segments_a if filter_segment(s)]
    valid_segments_b = [s for s in segments_b if filter_segment(s)]

    print(f"Valid segments after filtering:")
    print(f"  Channel A: {len(valid_segments_a)} (segment len: [{min_segment_length}, {max_segment_length or 'inf'}], "
          f"silence len: [{min_silence_length}, {max_silence_length or 'inf'}])")
    print(f"  Channel B: {len(valid_segments_b)}")

    results = {
        'experiments': [],
        'original_perplexity': None,
        'filter_params': {
            'min_segment_length': min_segment_length,
            'max_segment_length': max_segment_length,
            'min_silence_length': min_silence_length,
            'max_silence_length': max_silence_length,
            'perturb_location': perturb_location,
        },
    }

    # Compute original perplexity
    dialogue_a_torch = torch.tensor(dialogue_ids_a, device=device).unsqueeze(0)
    dialogue_b_torch = torch.tensor(dialogue_ids_b, device=device).unsqueeze(0)

    orig_nll, orig_ppl = _score(
        model, context_ids, context_length, dialogue_a_torch, dialogue_b_torch, device, legacy_scoring
    )
    results['original_perplexity'] = orig_ppl
    print(f"Original perplexity: {orig_ppl:.4f}")

    # Randomly select segments to test
    all_valid_segments = (
        [(s, 'A') for s in valid_segments_a] +
        [(s, 'B') for s in valid_segments_b]
    )

    if len(all_valid_segments) == 0:
        print("No valid segments found matching the filter criteria!")
        return results

    selected = random.sample(
        all_valid_segments,
        min(num_segments_to_test, len(all_valid_segments))
    )

    for seg_idx, (segment, channel) in enumerate(selected):
        segment_length = segment.end_idx - segment.start_idx
        print(f"\n--- Testing segment {seg_idx + 1}/{len(selected)} from channel {channel} ---")
        print(f"  Segment: tokens [{segment.start_idx}:{segment.end_idx}] (length: {segment_length})")
        print(f"  Pre-silence: {segment.pre_silence_len}, Post-silence: {segment.post_silence_len}")

        if perturb_location == "before":
            original_silence = segment.pre_silence_len
        else:
            original_silence = segment.post_silence_len

        if original_silence == 0:
            print(f"  Skipping: no {perturb_location} silence to perturb")
            continue

        experiment_result = {
            'channel': channel,
            'segment_start': segment.start_idx,
            'segment_end': segment.end_idx,
            'segment_length': segment_length,  # Store segment length for analysis
            'original_silence_len': original_silence,
            'perturb_location': perturb_location,
            'truncation_idx': segment.end_idx,  # Where we truncate
            'perturbations': [],
            'perplexities': [],
            'nlls': [],
            'sequence_lengths': [],  # Track truncated sequence lengths
        }

        # For truncation: we need to figure out the corresponding position in the OTHER channel
        # Since dialogue is interleaved (A[0], B[0], A[1], B[1], ...),
        # the segment indices in channel A/B correspond to time steps
        # We truncate both channels to the same time step (after the segment ends)

        # The segment end index in one channel corresponds to the same time step in the other
        truncation_time_idx = segment.end_idx

        print(f"  Truncating sequences at time index {truncation_time_idx}")
        print(f"  (Original full lengths: A={len(dialogue_a_audio)}, B={len(dialogue_b_audio)})")

        # Test each perturbation
        for delta in perturbation_range:
            new_silence_len = max(0, original_silence + delta)

            # Create perturbed dialogue with truncation
            if channel == 'A':
                if perturb_location == "before":
                    perturbed_a = perturb_silence_before_segment(
                        dialogue_a_audio, segment, new_silence_len, silence_token,
                        truncate_after_segment=True,
                        include_post_silence_tokens=1,  # Include 1 silence token for context
                    )
                else:
                    perturbed_a = perturb_silence_after_segment(
                        dialogue_a_audio, segment, new_silence_len, silence_token,
                        truncate_after_silence=True,
                    )
                # Truncate channel B to match the same time extent
                # Use the length of the perturbed_a as reference
                perturbed_b = dialogue_b_audio[:len(perturbed_a)]
            else:
                if perturb_location == "before":
                    perturbed_b = perturb_silence_before_segment(
                        dialogue_b_audio, segment, new_silence_len, silence_token,
                        truncate_after_segment=True,
                        include_post_silence_tokens=1,
                    )
                else:
                    perturbed_b = perturb_silence_after_segment(
                        dialogue_b_audio, segment, new_silence_len, silence_token,
                        truncate_after_silence=True,
                    )
                # Truncate channel A to match
                perturbed_a = dialogue_a_audio[:len(perturbed_b)]

            # Convert back to token IDs
            perturbed_a_ids = [x + offset for x in perturbed_a]
            perturbed_b_ids = [x + offset for x in perturbed_b]

            # Ensure equal length by padding the shorter one with silence
            max_len = max(len(perturbed_a_ids), len(perturbed_b_ids))
            if len(perturbed_a_ids) < max_len:
                perturbed_a_ids += [silence_token + offset] * (max_len - len(perturbed_a_ids))
            if len(perturbed_b_ids) < max_len:
                perturbed_b_ids += [silence_token + offset] * (max_len - len(perturbed_b_ids))

            perturbed_a_torch = torch.tensor(perturbed_a_ids, device=device).unsqueeze(0)
            perturbed_b_torch = torch.tensor(perturbed_b_ids, device=device).unsqueeze(0)

            try:
                nll, ppl = _score(
                    model, context_ids, context_length, perturbed_a_torch, perturbed_b_torch, device, legacy_scoring
                )

                experiment_result['perturbations'].append(delta)
                experiment_result['perplexities'].append(ppl)
                experiment_result['nlls'].append(nll)
                experiment_result['sequence_lengths'].append(max_len)

            except Exception as e:
                print(f"  Error at delta={delta}: {e}")
                continue

        results['experiments'].append(experiment_result)
        print(f"  Tested {len(experiment_result['perturbations'])} perturbations")

    return results


def plot_perturbation_results(results: Dict, save_path: str = None, show_nll: bool = False, normalize: bool = True):
    """
    Create visualizations for the perturbation experiment results.

    Args:
        results: Dictionary from run_silence_perturbation_experiment
        save_path: Optional path to save the figure
        show_nll: If True, show NLL instead of perplexity
        normalize: If True, normalize values relative to delta=0 (original silence)
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    n_experiments = len(results['experiments'])

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_experiments, 2,
        figsize=(14, 4 * n_experiments),
        squeeze=False
    )

    colors = {'A': 'blue', 'B': 'red'}
    metric_name = 'NLL' if show_nll else 'Perplexity'
    if normalize:
        metric_name = f'Normalized {metric_name}'

    for idx, exp in enumerate(results['experiments']):
        channel = exp['channel']
        original_silence = exp['original_silence_len']
        perturbations = np.array(exp['perturbations'])

        if show_nll:
            metric_values = np.array(exp['nlls'])
        else:
            metric_values = np.array(exp['perplexities'])

        # Normalize relative to delta=0
        if normalize:
            zero_idx = np.where(perturbations == 0)[0]
            if len(zero_idx) > 0:
                baseline = metric_values[zero_idx[0]]
            else:
                # Fallback: use the value closest to delta=0
                closest_idx = np.argmin(np.abs(perturbations))
                baseline = metric_values[closest_idx]
            metric_values = metric_values / baseline

        # Left plot: Metric vs perturbation delta
        ax1 = axes[idx, 0]
        ax1.plot(perturbations, metric_values,
                 color=colors[channel], linewidth=2, marker='o', markersize=4)
        ax1.axvline(x=0, color='green', linestyle='--', linewidth=2,
                    label='Original silence')
        if normalize:
            ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1,
                        label='Baseline (δ=0)')

        # Mark the minimum point
        min_idx = np.argmin(metric_values)
        ax1.scatter([perturbations[min_idx]], [metric_values[min_idx]],
                    color='gold', s=150, zorder=5, edgecolor='black',
                    label=f'Min (Δ={perturbations[min_idx]}, val={metric_values[min_idx]:.3f})')

        ax1.set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
        ax1.set_ylabel(metric_name, fontsize=11)

        # Add truncation info to title
        truncation_info = f"Truncated at idx {exp.get('truncation_idx', 'N/A')}"
        ax1.set_title(
            f'Channel {channel} - Segment [{exp["segment_start"]}:{exp["segment_end"]}]\n'
            f'Original {exp["perturb_location"]} silence: {original_silence} tokens | {truncation_info}',
            fontsize=11
        )
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Right plot: Metric vs absolute silence length
        ax2 = axes[idx, 1]
        absolute_silence = original_silence + perturbations
        ax2.plot(absolute_silence, metric_values,
                 color=colors[channel], linewidth=2, marker='o', markersize=4)
        ax2.axvline(x=original_silence, color='green', linestyle='--',
                    linewidth=2, label=f'Original ({original_silence})')
        if normalize:
            ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)

        # Mark minimum
        ax2.scatter([absolute_silence[min_idx]], [metric_values[min_idx]],
                    color='gold', s=150, zorder=5, edgecolor='black',
                    label=f'Min (len={absolute_silence[min_idx]}, val={metric_values[min_idx]:.3f})')

        ax2.set_xlabel('Absolute Silence Length (tokens)', fontsize=11)
        ax2.set_ylabel(metric_name, fontsize=11)
        ax2.set_title(
            f'Absolute Silence Length Effect\n'
            f'Min at silence length = {absolute_silence[min_idx]}',
            fontsize=12
        )
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


def plot_aggregate_results(results: Dict, save_path: str = None):
    """
    Create aggregate visualization across all tested segments.
    All segments normalized to the same scale (relative to their delta=0 baseline).
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Collect all normalized curves
    all_curves_a = []
    all_curves_b = []

    for exp in results['experiments']:
        perturbations = np.array(exp['perturbations'])
        perplexities = np.array(exp['perplexities'])

        # Normalize relative to delta=0
        zero_idx = np.where(perturbations == 0)[0]
        if len(zero_idx) > 0:
            baseline_ppl = perplexities[zero_idx[0]]
        else:
            closest_idx = np.argmin(np.abs(perturbations))
            baseline_ppl = perplexities[closest_idx]

        normalized_ppl = perplexities / baseline_ppl

        channel = exp['channel']
        color = 'blue' if channel == 'A' else 'red'
        alpha = 0.6

        # Plot individual curves
        axes[0].plot(perturbations, normalized_ppl,
                     color=color, alpha=alpha, linewidth=1.5)

        # Store for aggregation
        if channel == 'A':
            all_curves_a.append((perturbations, normalized_ppl))
        else:
            all_curves_b.append((perturbations, normalized_ppl))

    axes[0].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axes[0].axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    axes[0].set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
    axes[0].set_ylabel('Normalized Perplexity (rel. to δ=0)', fontsize=11)
    axes[0].set_title('All Segments Overlaid', fontsize=12)

    # Legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Channel A'),
        Patch(facecolor='red', alpha=0.6, label='Channel B'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='Original (δ=0)'),
        plt.Line2D([0], [0], color='gray', linestyle=':', label='Baseline (1.0)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle plot: Mean ± std across all segments
    # First, interpolate all curves to common x-axis
    all_perturbations = set()
    for exp in results['experiments']:
        all_perturbations.update(exp['perturbations'])
    common_x = np.array(sorted(all_perturbations))

    # Collect interpolated values at each perturbation level
    perturbation_to_values = {p: [] for p in common_x}

    for exp in results['experiments']:
        perturbations = np.array(exp['perturbations'])
        perplexities = np.array(exp['perplexities'])

        # Normalize
        zero_idx = np.where(perturbations == 0)[0]
        if len(zero_idx) > 0:
            baseline = perplexities[zero_idx[0]]
        else:
            closest_idx = np.argmin(np.abs(perturbations))
            baseline = perplexities[closest_idx]
        normalized = perplexities / baseline

        # Add to collection
        for p, v in zip(perturbations, normalized):
            if p in perturbation_to_values:
                perturbation_to_values[p].append(v)

    # Compute mean and std
    means = []
    stds = []
    valid_x = []
    for p in common_x:
        values = perturbation_to_values[p]
        if len(values) > 0:
            means.append(np.mean(values))
            stds.append(np.std(values))
            valid_x.append(p)

    valid_x = np.array(valid_x)
    means = np.array(means)
    stds = np.array(stds)

    axes[1].plot(valid_x, means, color='purple', linewidth=2, marker='o', markersize=4)
    axes[1].fill_between(valid_x, means - stds, means + stds,
                          color='purple', alpha=0.2, label='±1 std')
    axes[1].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axes[1].axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
    axes[1].set_ylabel('Normalized Perplexity', fontsize=11)
    axes[1].set_title('Mean ± Std Across All Segments', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Mark minimum of mean curve
    min_idx = np.argmin(means)
    axes[1].scatter([valid_x[min_idx]], [means[min_idx]],
                    color='gold', s=150, zorder=5, edgecolor='black',
                    label=f'Min mean (δ={valid_x[min_idx]})')

    # Right plot: Box plot of normalized perplexity per perturbation level
    box_data = [perturbation_to_values[p] for p in valid_x if len(perturbation_to_values[p]) > 0]
    box_positions = [p for p in valid_x if len(perturbation_to_values[p]) > 0]

    bp = axes[2].boxplot(box_data, positions=box_positions, widths=0.6,
                          patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    axes[2].axvline(x=0, color='green', linestyle='--', linewidth=2)
    axes[2].axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    axes[2].set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
    axes[2].set_ylabel('Normalized Perplexity', fontsize=11)
    axes[2].set_title('Distribution per Perturbation Level', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


def merge_experiment_results(results_list: List[Dict]) -> Dict:
    """
    Merge results from multiple samples into a single results dictionary.

    Args:
        results_list: List of result dictionaries from run_silence_perturbation_experiment

    Returns:
        Merged results dictionary with all experiments combined
    """
    merged = {
        'experiments': [],
        'original_perplexities': [],  # Track per-sample
        'sample_ids': [],
        'num_samples': len(results_list),
    }

    for sample_idx, results in enumerate(results_list):
        if results is None:
            continue

        # Add sample_id to each experiment for tracking
        for exp in results.get('experiments', []):
            exp_with_sample = exp.copy()
            exp_with_sample['sample_id'] = sample_idx
            merged['experiments'].append(exp_with_sample)

        if results.get('original_perplexity') is not None:
            merged['original_perplexities'].append(results['original_perplexity'])
            merged['sample_ids'].append(sample_idx)

    # Compute aggregate original perplexity
    if merged['original_perplexities']:
        merged['original_perplexity'] = np.mean(merged['original_perplexities'])
    else:
        merged['original_perplexity'] = None

    return merged


def run_multi_sample_experiment(
    model,
    tokenizer,
    samples: List[Dict],
    offset: int,
    device: str = "cuda",
    perturbation_range: range = range(-10, 21),
    num_segments_per_sample: int = 5,
    perturb_location: str = "before",
    random_seed: int = 42,
    verbose: bool = True,
    min_segment_length: int = 1,
    max_segment_length: int = None,
    min_silence_length: int = 1,
    max_silence_length: int = None,
    legacy_scoring: bool = True,
) -> Dict:
    """
    Run silence perturbation experiment across multiple samples.

    Args:
        model: The LLM model
        tokenizer: The tokenizer
        samples: List of sample dictionaries with 'input_ids' key
        offset: Token offset for audio tokens
        device: CUDA device
        perturbation_range: Range of silence deltas to test
        num_segments_per_sample: Number of segments to test per sample
        perturb_location: "before" or "after"
        random_seed: Base random seed (incremented per sample)
        verbose: Print progress
        min_segment_length: Minimum number of tokens in the speech segment
        max_segment_length: Maximum number of tokens in the speech segment (None = no limit)
        min_silence_length: Minimum original silence length to consider for perturbation
        max_silence_length: Maximum original silence length (None = no limit)

    Returns:
        Merged results dictionary
    """
    all_results = []

    if verbose:
        print(f"Segment filters: length=[{min_segment_length}, {max_segment_length or 'inf'}], "
              f"silence=[{min_silence_length}, {max_silence_length or 'inf'}]")

    for sample_idx, sample in enumerate(samples):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{len(samples)}")
            print(f"{'='*60}")

        try:
            # Extract script and dialogue from sample
            input_ids = sample['input_ids']
            script_a_ids, script_b_ids, dialogue_ids = extract_script_contents(
                tokenizer, input_ids, return_dialogue=True
            )

            if len(dialogue_ids) < 10:
                if verbose:
                    print(f"  Skipping sample {sample_idx}: dialogue too short ({len(dialogue_ids)} tokens)")
                all_results.append(None)
                continue

            # Prepare dialogue channels
            dialogue_ids_a = dialogue_ids[::2]
            dialogue_ids_b = dialogue_ids[1::2]

            # Prepare context
            script_a_audio = [x - offset for x in script_a_ids]
            script_b_audio = [x - offset for x in script_b_ids]

            # Remove duplicated silence for context
            script_a_no_silence = remove_duplicated_silence_tokens(script_a_audio, target_val=SILENCE_CODE)
            script_b_no_silence = remove_duplicated_silence_tokens(script_b_audio, target_val=SILENCE_CODE)

            context_ids, context_length = prepare_inference_input(
                script_a_no_silence, script_b_no_silence,
                tokenizer, device, already_tokenized=False
            )

            # Run experiment for this sample
            sample_results = run_silence_perturbation_experiment(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                context_length=context_length,
                dialogue_ids_a=dialogue_ids_a,
                dialogue_ids_b=dialogue_ids_b,
                offset=offset,
                device=device,
                perturbation_range=perturbation_range,
                num_segments_to_test=num_segments_per_sample,
                random_seed=random_seed + sample_idx,  # Different seed per sample
                perturb_location=perturb_location,
                min_segment_length=min_segment_length,
                max_segment_length=max_segment_length,
                min_silence_length=min_silence_length,
                max_silence_length=max_silence_length,
                legacy_scoring=legacy_scoring,
            )

            all_results.append(sample_results)

            if verbose:
                n_exp = len(sample_results.get('experiments', []))
                print(f"  Completed: {n_exp} segments tested")

        except Exception as e:
            if verbose:
                print(f"  Error processing sample {sample_idx}: {e}")
            all_results.append(None)
            continue

    # Merge all results
    merged_results = merge_experiment_results(all_results)

    # Store filter params in merged results
    merged_results['filter_params'] = {
        'min_segment_length': min_segment_length,
        'max_segment_length': max_segment_length,
        'min_silence_length': min_silence_length,
        'max_silence_length': max_silence_length,
        'perturb_location': perturb_location,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-sample experiment complete!")
        print(f"  Total samples processed: {len([r for r in all_results if r is not None])}/{len(samples)}")
        print(f"  Total segments tested: {len(merged_results['experiments'])}")
        print(f"{'='*60}")

    return merged_results


def plot_multi_sample_aggregate(
    results: Dict,
    save_path: str = None,
    show_sample_breakdown: bool = True,
):
    """
    Create aggregate visualization for multi-sample experiment results.

    Args:
        results: Merged results from run_multi_sample_experiment
        save_path: Optional path to save figure
        show_sample_breakdown: If True, color-code by sample
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    n_cols = 3 if show_sample_breakdown else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # Add filter params to title if available
    filter_info = ""
    if 'filter_params' in results:
        fp = results['filter_params']
        filter_info = f"\nFilters: seg_len=[{fp.get('min_segment_length', 1)}, {fp.get('max_segment_length') or 'inf'}]"

    # Get unique sample IDs for coloring
    sample_ids = list(set(exp.get('sample_id', 0) for exp in results['experiments']))
    n_samples = len(sample_ids)
    cmap = plt.get_cmap('tab10', max(n_samples, 10))

    # Collect data for aggregation
    perturbation_to_values = {}

    for exp in results['experiments']:
        perturbations = np.array(exp['perturbations'])
        perplexities = np.array(exp['perplexities'])
        sample_id = exp.get('sample_id', 0)
        channel = exp['channel']

        # Normalize relative to delta=0
        zero_idx = np.where(perturbations == 0)[0]
        if len(zero_idx) > 0:
            baseline_ppl = perplexities[zero_idx[0]]
        else:
            closest_idx = np.argmin(np.abs(perturbations))
            baseline_ppl = perplexities[closest_idx]

        if baseline_ppl == 0:
            continue

        normalized_ppl = perplexities / baseline_ppl

        # Store for aggregation
        for p, v in zip(perturbations, normalized_ppl):
            if p not in perturbation_to_values:
                perturbation_to_values[p] = []
            perturbation_to_values[p].append(v)

        # Plot individual curves (left panel)
        if show_sample_breakdown:
            color = cmap(sample_id % 10)
            alpha = 0.4
        else:
            color = 'blue' if channel == 'A' else 'red'
            alpha = 0.3

        axes[0].plot(perturbations, normalized_ppl,
                     color=color, alpha=alpha, linewidth=1)

    axes[0].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Original (δ=0)')
    axes[0].axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Baseline')
    axes[0].set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
    axes[0].set_ylabel('Normalized Perplexity', fontsize=11)
    axes[0].set_title(f'All Segments ({len(results["experiments"])} total)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle panel: Mean ± std
    sorted_perturbations = sorted(perturbation_to_values.keys())
    means = []
    stds = []
    counts = []

    for p in sorted_perturbations:
        values = perturbation_to_values[p]
        means.append(np.mean(values))
        stds.append(np.std(values))
        counts.append(len(values))

    means = np.array(means)
    stds = np.array(stds)
    x = np.array(sorted_perturbations)

    ax_mean = axes[1]
    ax_mean.plot(x, means, color='purple', linewidth=2, marker='o', markersize=4)
    ax_mean.fill_between(x, means - stds, means + stds,
                          color='purple', alpha=0.2, label='±1 std')
    ax_mean.axvline(x=0, color='green', linestyle='--', linewidth=2)
    ax_mean.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)

    # Mark minimum
    min_idx = np.argmin(means)
    ax_mean.scatter([x[min_idx]], [means[min_idx]],
                    color='gold', s=150, zorder=5, edgecolor='black')
    ax_mean.annotate(f'Min: δ={x[min_idx]}\nPPL={means[min_idx]:.3f}',
                     xy=(x[min_idx], means[min_idx]),
                     xytext=(x[min_idx] + 2, means[min_idx] + 0.05),
                     fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    ax_mean.set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
    ax_mean.set_ylabel('Normalized Perplexity', fontsize=11)
    ax_mean.set_title(f'Mean ± Std (n={len(results["experiments"])} segments)', fontsize=12)
    ax_mean.legend(loc='upper right')
    ax_mean.grid(True, alpha=0.3)

    # Right panel: Per-sample breakdown or box plot
    if show_sample_breakdown and n_samples > 1:
        ax_sample = axes[2]

        # Compute per-sample mean curves
        sample_means = {sid: {} for sid in sample_ids}

        for exp in results['experiments']:
            sample_id = exp.get('sample_id', 0)
            perturbations = np.array(exp['perturbations'])
            perplexities = np.array(exp['perplexities'])

            zero_idx = np.where(perturbations == 0)[0]
            if len(zero_idx) > 0:
                baseline = perplexities[zero_idx[0]]
            else:
                baseline = perplexities[np.argmin(np.abs(perturbations))]

            if baseline == 0:
                continue

            normalized = perplexities / baseline

            for p, v in zip(perturbations, normalized):
                if p not in sample_means[sample_id]:
                    sample_means[sample_id][p] = []
                sample_means[sample_id][p].append(v)

        # Plot mean curve per sample
        for sid in sample_ids:
            if not sample_means[sid]:
                continue
            sorted_p = sorted(sample_means[sid].keys())
            sample_curve = [np.mean(sample_means[sid][p]) for p in sorted_p]
            ax_sample.plot(sorted_p, sample_curve,
                          color=cmap(sid % 10), linewidth=2,
                          label=f'Sample {sid}', alpha=0.8)

        ax_sample.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax_sample.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
        ax_sample.set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
        ax_sample.set_ylabel('Normalized Perplexity', fontsize=11)
        ax_sample.set_title('Per-Sample Mean Curves', fontsize=12)
        ax_sample.legend(loc='upper right', fontsize=8, ncol=2)
        ax_sample.grid(True, alpha=0.3)

    else:
        # Box plot
        ax_box = axes[2] if n_cols == 3 else axes[1]
        box_data = [perturbation_to_values[p] for p in sorted_perturbations]

        bp = ax_box.boxplot(box_data, positions=sorted_perturbations, widths=0.6,
                            patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax_box.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax_box.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
        ax_box.set_xlabel('Silence Duration Delta (tokens)', fontsize=11)
        ax_box.set_ylabel('Normalized Perplexity', fontsize=11)
        ax_box.set_title('Distribution per Perturbation Level', fontsize=12)
        ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


def plot_by_segment_length(
    results: Dict,
    length_bins: List[Tuple[int, int]] = None,
    save_path: str = None,
):
    """
    Plot normalized perplexity curves grouped by segment length.

    Args:
        results: Results dictionary with 'experiments' containing 'segment_length'
        length_bins: List of (min, max) tuples defining length bins.
                     None = auto-create bins based on data
        save_path: Optional path to save figure
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    # Check if segment_length is available
    if 'segment_length' not in results['experiments'][0]:
        print("Error: segment_length not found in experiment results. Re-run experiment with updated code.")
        return

    # Get all segment lengths
    all_lengths = [exp['segment_length'] for exp in results['experiments']]

    if length_bins is None:
        # Auto-create bins based on quartiles
        q1, q2, q3 = np.percentile(all_lengths, [25, 50, 75])
        length_bins = [
            (1, int(q1)),
            (int(q1) + 1, int(q2)),
            (int(q2) + 1, int(q3)),
            (int(q3) + 1, max(all_lengths) + 1),
        ]
        print(f"Auto-generated bins based on quartiles: {length_bins}")

    # Group experiments by length bin
    binned_experiments = {i: [] for i in range(len(length_bins))}

    for exp in results['experiments']:
        seg_len = exp['segment_length']
        for bin_idx, (min_len, max_len) in enumerate(length_bins):
            if min_len <= seg_len <= max_len:
                binned_experiments[bin_idx].append(exp)
                break

    # Create figure
    n_bins = len(length_bins)
    fig, axes = plt.subplots(1, n_bins + 1, figsize=(5 * (n_bins + 1), 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins))

    # Plot each bin
    bin_means = {}

    for bin_idx, (min_len, max_len) in enumerate(length_bins):
        ax = axes[bin_idx]
        experiments = binned_experiments[bin_idx]

        if not experiments:
            ax.set_title(f'Seg len [{min_len}, {max_len}]\n(no data)', fontsize=11)
            ax.grid(True, alpha=0.3)
            continue

        # Collect normalized curves
        perturbation_to_values = {}

        for exp in experiments:
            perturbations = np.array(exp['perturbations'])
            perplexities = np.array(exp['perplexities'])

            # Normalize
            zero_idx = np.where(perturbations == 0)[0]
            if len(zero_idx) > 0:
                baseline = perplexities[zero_idx[0]]
            else:
                baseline = perplexities[np.argmin(np.abs(perturbations))]

            if baseline == 0:
                continue

            normalized = perplexities / baseline

            # Plot individual curve
            ax.plot(perturbations, normalized, color=colors[bin_idx], alpha=0.3, linewidth=1)

            # Store for mean calculation
            for p, v in zip(perturbations, normalized):
                if p not in perturbation_to_values:
                    perturbation_to_values[p] = []
                perturbation_to_values[p].append(v)

        # Compute and plot mean
        sorted_p = sorted(perturbation_to_values.keys())
        means = [np.mean(perturbation_to_values[p]) for p in sorted_p]
        stds = [np.std(perturbation_to_values[p]) for p in sorted_p]

        ax.plot(sorted_p, means, color=colors[bin_idx], linewidth=3, label='Mean')
        ax.fill_between(sorted_p,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=colors[bin_idx], alpha=0.2)

        # Store for comparison plot
        bin_means[bin_idx] = (sorted_p, means, stds)

        ax.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Silence Delta (tokens)', fontsize=10)
        ax.set_ylabel('Normalized PPL', fontsize=10)
        ax.set_title(f'Seg len [{min_len}, {max_len}]\n(n={len(experiments)})', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Mark minimum
        if means:
            min_idx = np.argmin(means)
            ax.scatter([sorted_p[min_idx]], [means[min_idx]],
                      color='gold', s=100, zorder=5, edgecolor='black')

    # Comparison plot (rightmost)
    ax_comp = axes[-1]

    for bin_idx, (min_len, max_len) in enumerate(length_bins):
        if bin_idx not in bin_means:
            continue
        sorted_p, means, stds = bin_means[bin_idx]
        ax_comp.plot(sorted_p, means, color=colors[bin_idx], linewidth=2,
                     label=f'[{min_len}, {max_len}]')

    ax_comp.axvline(x=0, color='green', linestyle='--', linewidth=2)
    ax_comp.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax_comp.set_xlabel('Silence Delta (tokens)', fontsize=10)
    ax_comp.set_ylabel('Normalized PPL', fontsize=10)
    ax_comp.set_title('Comparison by Segment Length', fontsize=11)
    ax_comp.legend(title='Seg Length', loc='upper right')
    ax_comp.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


def run_length_comparison_experiment(
    model,
    tokenizer,
    samples: List[Dict],
    offset: int,
    device: str = "cuda",
    perturbation_range: range = range(-10, 11),
    num_segments_per_sample: int = 10,
    perturb_location: str = "before",
    random_seed: int = 42,
    length_bins: List[Tuple[int, int]] = None,
    verbose: bool = True,
    legacy_scoring: bool = True,
) -> Dict:
    """
    Convenience function to run experiment and automatically analyze by segment length.

    Args:
        model, tokenizer, samples, offset, device: Standard experiment params
        perturbation_range: Range of silence deltas to test
        num_segments_per_sample: Number of segments per sample
        perturb_location: "before" or "after"
        random_seed: Random seed
        length_bins: Optional list of (min, max) tuples for segment length bins
        verbose: Print progress

    Returns:
        Dictionary with results and statistics by segment length
    """
    # First, run without length filters to get all segments
    results = run_multi_sample_experiment(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        offset=offset,
        device=device,
        perturbation_range=perturbation_range,
        num_segments_per_sample=num_segments_per_sample,
        perturb_location=perturb_location,
        random_seed=random_seed,
        verbose=verbose,
        min_segment_length=1,  # Get all segments
        max_segment_length=None,
        legacy_scoring=legacy_scoring,
    )

    if not results['experiments']:
        print("No experiments completed!")
        return results

    # Compute statistics by segment length
    all_lengths = [exp['segment_length'] for exp in results['experiments']]

    if length_bins is None:
        q1, q2, q3 = np.percentile(all_lengths, [25, 50, 75])
        length_bins = [
            (1, int(q1)),
            (int(q1) + 1, int(q2)),
            (int(q2) + 1, int(q3)),
            (int(q3) + 1, max(all_lengths) + 1),
        ]

    # Compute per-bin statistics
    bin_stats = {}

    for bin_idx, (min_len, max_len) in enumerate(length_bins):
        bin_experiments = [exp for exp in results['experiments']
                          if min_len <= exp['segment_length'] <= max_len]

        if not bin_experiments:
            bin_stats[f"bin_{bin_idx}"] = {
                'range': (min_len, max_len),
                'n_segments': 0,
            }
            continue

        # Compute mean normalized perplexity at each perturbation level
        perturbation_to_values = {}
        for exp in bin_experiments:
            perturbations = np.array(exp['perturbations'])
            perplexities = np.array(exp['perplexities'])

            zero_idx = np.where(perturbations == 0)[0]
            if len(zero_idx) > 0:
                baseline = perplexities[zero_idx[0]]
            else:
                baseline = perplexities[np.argmin(np.abs(perturbations))]

            if baseline == 0:
                continue

            normalized = perplexities / baseline
            for p, v in zip(perturbations, normalized):
                if p not in perturbation_to_values:
                    perturbation_to_values[p] = []
                perturbation_to_values[p].append(v)

        sorted_p = sorted(perturbation_to_values.keys())
        means = [np.mean(perturbation_to_values[p]) for p in sorted_p]

        # Find optimal delta (minimum mean perplexity)
        min_idx = np.argmin(means)
        optimal_delta = sorted_p[min_idx]

        bin_stats[f"bin_{bin_idx}"] = {
            'range': (min_len, max_len),
            'n_segments': len(bin_experiments),
            'mean_segment_length': np.mean([e['segment_length'] for e in bin_experiments]),
            'optimal_delta': optimal_delta,
            'min_mean_ppl': means[min_idx],
            'ppl_at_zero': perturbation_to_values.get(0, [1.0])[0] if 0 in perturbation_to_values else 1.0,
        }

    results['length_bins'] = length_bins
    results['bin_stats'] = bin_stats

    if verbose:
        print("\n" + "="*60)
        print("Statistics by segment length:")
        print("="*60)
        for bin_name, stats in bin_stats.items():
            if stats['n_segments'] == 0:
                print(f"  {bin_name} {stats['range']}: no data")
            else:
                print(f"  {bin_name} {stats['range']}: n={stats['n_segments']}, "
                      f"mean_len={stats['mean_segment_length']:.1f}, "
                      f"optimal_δ={stats['optimal_delta']}, "
                      f"min_ppl={stats['min_mean_ppl']:.4f}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Scheduler checkpoint directory.")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Tokenized validation JSONL.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200, help="Records drawn at random from --data_jsonl.")
    parser.add_argument("--seed", type=int, default=None, help="Seeds the record draw (1986 reproduces the draw of the published runs). "
                             "Omit for an unseeded draw.")
    parser.add_argument("--segment_seed", type=int, default=42,
                        help="Base seed for choosing segments (sample i uses segment_seed + i).")
    parser.add_argument("--num_segments_per_sample", type=int, default=8)
    parser.add_argument("--perturb_location", choices=["before", "after"], default="before")
    parser.add_argument("--delta_min", type=int, default=-15, help="Smallest silence delta (tokens) for the 'all' run.")
    parser.add_argument("--delta_max", type=int, default=15, help="Exclusive upper bound of the delta for the 'all' run.")
    parser.add_argument("--long_min_length", type=int, default=30, help="Minimum segment length of the 'long' run.")
    parser.add_argument("--short_max_length", type=int, default=15, help="Maximum segment length of the 'short' run.")
    parser.add_argument("--runs", nargs="+", choices=["all", "long", "short"], default=["all", "long", "short"])
    parser.add_argument("--corrected_scoring", action="store_true",
                        help="Use the trained channel-id layout and correctly named metrics.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    from ..vocab import audio_token
    from .common import load_model_and_tokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    offset = tokenizer.convert_tokens_to_ids(audio_token(0))
    samples = load_jsonl_examples(args.data_jsonl, num_examples=args.num_samples, seed=args.seed)

    shared = dict(
        model=model, tokenizer=tokenizer, samples=samples, offset=offset, device=args.device,
        num_segments_per_sample=args.num_segments_per_sample,
        perturb_location=args.perturb_location,
        random_seed=args.segment_seed,
        legacy_scoring=not args.corrected_scoring,
    )
    # The 'long' and 'short' runs keep the library's default delta range, as the research runs did.
    run_settings = {
        "all": dict(perturbation_range=range(args.delta_min, args.delta_max)),
        "long": dict(min_segment_length=args.long_min_length, max_segment_length=None),
        "short": dict(min_segment_length=0, max_segment_length=args.short_max_length),
    }
    file_stems = {
        "all": "multi_sample_aggregate",
        "long": "multi_sample_aggregate_long_segments",
        "short": "multi_sample_aggregate_short_segments",
    }
    for run in args.runs:
        results = run_multi_sample_experiment(**shared, **run_settings[run])
        with open(out_dir / f"{file_stems[run]}.json", "w") as f:
            json.dump(to_serializable(results), f, indent=2)
        plot_multi_sample_aggregate(results, save_path=str(out_dir / f"{file_stems[run]}.png"))


if __name__ == "__main__":
    main()
