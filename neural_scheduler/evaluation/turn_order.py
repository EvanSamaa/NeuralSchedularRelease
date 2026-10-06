"""Turn-order perturbation: does the scheduler prefer the true speaking order?

Pairs of speech segments from different channels, one following the other, are
located in a ground-truth dialogue. For each pair the dialogue is truncated just
after the pair and scored three ways: as recorded, with the two segments
swapped in time, and with both starting together. If the model has learned
turn-taking, the recorded order should have the lowest perplexity.

    python -m neural_scheduler.evaluation.turn_order \\
        --checkpoint CKPT --data_jsonl DATA.jsonl --out_dir OUT --num_samples 1500 --seed 0
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..constants import SILENCE_CODE
from .common import (
    compute_dialogue_nll,
    extract_script_contents,
    interleave,
    load_jsonl_examples,
    prepare_inference_input,
    to_serializable,
)
from .segments import SpeechSegment, find_speech_segments_with_channel


@dataclass
class SegmentPair:
    """A pair of segments from different channels where one follows the other."""
    first_segment: SpeechSegment   # The segment that starts first
    second_segment: SpeechSegment  # The segment that starts second
    gap_duration: int              # Silence tokens between end of first and start of second
    overlap_duration: int          # If negative gap, this is the overlap amount

    @property
    def first_channel(self) -> str:
        return self.first_segment.channel

    @property
    def second_channel(self) -> str:
        return self.second_segment.channel

    @property
    def is_turn_taking(self) -> bool:
        """True if segments don't overlap (gap >= 0)."""
        return self.gap_duration >= 0


def find_sequential_segment_pairs(
    segments_a: List[SpeechSegment],
    segments_b: List[SpeechSegment],
    max_gap: int = 20,
    allow_overlap: bool = False,
) -> List[SegmentPair]:
    """
    Find pairs of segments where one clearly follows the other.

    A pair is identified when:
    - One segment ends and another begins within max_gap tokens
    - The segments are from different channels

    Args:
        segments_a: Speech segments from channel A
        segments_b: Speech segments from channel B
        max_gap: Maximum gap (in tokens) between segments to consider them a pair
        allow_overlap: If True, also include pairs with slight overlap

    Returns:
        List of SegmentPair objects
    """
    pairs = []

    # Combine all segments with their timing info
    all_segments = segments_a + segments_b

    # Sort by start index
    all_segments_sorted = sorted(all_segments, key=lambda s: s.start_idx)

    # Find consecutive pairs from different channels
    for i in range(len(all_segments_sorted) - 1):
        seg1 = all_segments_sorted[i]
        seg2 = all_segments_sorted[i + 1]

        # Must be from different channels
        if seg1.channel == seg2.channel:
            continue

        # Calculate gap (positive) or overlap (negative)
        gap = seg2.start_idx - seg1.end_idx

        # Check if within acceptable range
        if gap > max_gap:
            continue

        if gap < 0 and not allow_overlap:
            continue

        overlap = max(0, -gap)
        gap_duration = max(0, gap)

        pairs.append(SegmentPair(
            first_segment=seg1,
            second_segment=seg2,
            gap_duration=gap_duration,
            overlap_duration=overlap
        ))

    return pairs


def create_original_sequence(
    dialogue_a: List[int],
    dialogue_b: List[int],
    pair: SegmentPair,
    truncate_after: bool = True,
    post_tokens: int = 1,
    silence_token: int = SILENCE_CODE,
) -> Tuple[List[int], List[int]]:
    """
    Create the original (unmodified) sequence, optionally truncated after the pair.

    Args:
        dialogue_a: Original channel A tokens
        dialogue_b: Original channel B tokens
        pair: The segment pair
        truncate_after: If True, truncate after the second segment ends
        post_tokens: Number of tokens to include after truncation point
        silence_token: Silence token value

    Returns:
        (truncated_a, truncated_b)
    """
    if truncate_after:
        # Truncate after the second segment ends
        truncate_idx = pair.second_segment.end_idx + post_tokens
        truncated_a = dialogue_a[:truncate_idx]
        truncated_b = dialogue_b[:truncate_idx]

        # Pad to equal length if needed
        max_len = max(len(truncated_a), len(truncated_b))
        truncated_a = truncated_a + [silence_token] * (max_len - len(truncated_a))
        truncated_b = truncated_b + [silence_token] * (max_len - len(truncated_b))

        return truncated_a, truncated_b
    else:
        return dialogue_a.copy(), dialogue_b.copy()


def create_reversed_sequence(
    dialogue_a: List[int],
    dialogue_b: List[int],
    pair: SegmentPair,
    truncate_after: bool = True,
    post_tokens: int = 1,
    silence_token: int = SILENCE_CODE,
) -> Tuple[List[int], List[int]]:
    """
    Create a sequence where the turn order is reversed.

    The second segment now starts where the first segment originally started,
    and the first segment is moved to start after the (now-first) second segment ends,
    maintaining the same gap duration.

    Args:
        dialogue_a: Original channel A tokens
        dialogue_b: Original channel B tokens
        pair: The segment pair to reverse
        truncate_after: If True, truncate after the reversed pair ends
        post_tokens: Number of tokens to include after truncation
        silence_token: Silence token value

    Returns:
        (modified_a, modified_b)
    """
    first_seg = pair.first_segment
    second_seg = pair.second_segment
    gap = pair.gap_duration

    # Get the token content of each segment
    if first_seg.channel == 'A':
        first_content = dialogue_a[first_seg.start_idx:first_seg.end_idx]
        second_content = dialogue_b[second_seg.start_idx:second_seg.end_idx]
    else:
        first_content = dialogue_b[first_seg.start_idx:first_seg.end_idx]
        second_content = dialogue_a[second_seg.start_idx:second_seg.end_idx]

    # New timing: second_segment starts at first_segment's original start
    # first_segment starts after second_segment ends + gap
    new_second_start = first_seg.start_idx
    new_second_end = new_second_start + len(second_content)
    new_first_start = new_second_end + gap
    new_first_end = new_first_start + len(first_content)

    # Create modified sequences
    # Start with everything up to the first segment's start
    modified_a = dialogue_a[:first_seg.start_idx]
    modified_b = dialogue_b[:first_seg.start_idx]

    # Determine truncation point
    truncate_idx = new_first_end + post_tokens if truncate_after else max(len(dialogue_a), len(dialogue_b))

    # Build the modified region
    # We need to place:
    # - second_content in its channel starting at new_second_start
    # - first_content in its channel starting at new_first_start
    # - silence everywhere else

    for t in range(first_seg.start_idx, truncate_idx):
        # Channel A token at time t
        if first_seg.channel == 'A':
            # First segment was in A, now moved later
            if new_first_start <= t < new_first_end:
                token_a = first_content[t - new_first_start]
            else:
                token_a = silence_token
        else:
            # Second segment was in A, now moved earlier
            if new_second_start <= t < new_second_end:
                token_a = second_content[t - new_second_start]
            else:
                token_a = silence_token

        # Channel B token at time t
        if first_seg.channel == 'B':
            # First segment was in B, now moved later
            if new_first_start <= t < new_first_end:
                token_b = first_content[t - new_first_start]
            else:
                token_b = silence_token
        else:
            # Second segment was in B, now moved earlier
            if new_second_start <= t < new_second_end:
                token_b = second_content[t - new_second_start]
            else:
                token_b = silence_token

        modified_a.append(token_a)
        modified_b.append(token_b)

    return modified_a, modified_b


def create_simultaneous_sequence(
    dialogue_a: List[int],
    dialogue_b: List[int],
    pair: SegmentPair,
    truncate_after: bool = True,
    post_tokens: int = 1,
    silence_token: int = SILENCE_CODE,
) -> Tuple[List[int], List[int]]:
    """
    Create a sequence where both segments start at the same time.

    Both segments start at the first segment's original start time.
    The shorter segment is padded with silence at the end.

    Args:
        dialogue_a: Original channel A tokens
        dialogue_b: Original channel B tokens
        pair: The segment pair
        truncate_after: If True, truncate after both segments end
        post_tokens: Number of tokens to include after truncation
        silence_token: Silence token value

    Returns:
        (modified_a, modified_b)
    """
    first_seg = pair.first_segment
    second_seg = pair.second_segment

    # Get the token content of each segment
    if first_seg.channel == 'A':
        first_content = dialogue_a[first_seg.start_idx:first_seg.end_idx]
        second_content = dialogue_b[second_seg.start_idx:second_seg.end_idx]
    else:
        first_content = dialogue_b[first_seg.start_idx:first_seg.end_idx]
        second_content = dialogue_a[second_seg.start_idx:second_seg.end_idx]

    # Both start at first segment's original start
    simultaneous_start = first_seg.start_idx
    max_length = max(len(first_content), len(second_content))
    simultaneous_end = simultaneous_start + max_length

    # Create modified sequences
    modified_a = dialogue_a[:first_seg.start_idx]
    modified_b = dialogue_b[:first_seg.start_idx]

    # Truncation point
    truncate_idx = simultaneous_end + post_tokens if truncate_after else max(len(dialogue_a), len(dialogue_b))

    # Build the modified region
    for t in range(first_seg.start_idx, truncate_idx):
        relative_t = t - simultaneous_start

        # Channel A token
        if first_seg.channel == 'A':
            if 0 <= relative_t < len(first_content):
                token_a = first_content[relative_t]
            else:
                token_a = silence_token
        else:
            if 0 <= relative_t < len(second_content):
                token_a = second_content[relative_t]
            else:
                token_a = silence_token

        # Channel B token
        if first_seg.channel == 'B':
            if 0 <= relative_t < len(first_content):
                token_b = first_content[relative_t]
            else:
                token_b = silence_token
        else:
            if 0 <= relative_t < len(second_content):
                token_b = second_content[relative_t]
            else:
                token_b = silence_token

        modified_a.append(token_a)
        modified_b.append(token_b)

    return modified_a, modified_b


@torch.no_grad()
def compute_truncated_perplexity(
    model,
    context_ids: torch.Tensor,
    context_length: int,
    dialogue_a: List[int],
    dialogue_b: List[int],
    offset: int,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Perplexity and summed NLL of a dialogue given as two codec-id channels.

    ``offset`` converts codec ids to vocab ids. Returns ``(perplexity, nll)``.
    """
    dialogue_ids = interleave([x + offset for x in dialogue_a], [x + offset for x in dialogue_b])
    return compute_dialogue_nll(model, context_ids, context_length, dialogue_ids, device=device)


def run_turn_order_experiment(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    context_length: int,
    dialogue_ids_a: List[int],
    dialogue_ids_b: List[int],
    offset: int,
    device: str = "cuda",
    max_gap: int = 20,
    num_pairs_to_test: int = 10,
    random_seed: int = 42,
    min_segment_length: int = 5,
    max_segment_length: int = None,
) -> Dict:
    """
    Run the turn order perturbation experiment.

    Args:
        model: The LLM model
        tokenizer: Tokenizer
        context_ids: Context token tensor
        context_length: Context length
        dialogue_ids_a: Channel A dialogue tokens (with offset)
        dialogue_ids_b: Channel B dialogue tokens (with offset)
        offset: Audio token offset
        device: CUDA device
        max_gap: Maximum gap between segments to consider a pair
        num_pairs_to_test: Number of segment pairs to test
        random_seed: Random seed
        min_segment_length: Minimum segment length to consider
        max_segment_length: Maximum segment length (None = no limit)

    Returns:
        Dictionary with experiment results
    """
    random.seed(random_seed)

    # Convert to audio token space
    dialogue_a = [x - offset for x in dialogue_ids_a]
    dialogue_b = [x - offset for x in dialogue_ids_b]

    silence_token = SILENCE_CODE

    # Find segments
    segments_a, segments_b = find_speech_segments_with_channel(
        dialogue_a, dialogue_b, silence_token
    )

    print(f"Found {len(segments_a)} segments in channel A")
    print(f"Found {len(segments_b)} segments in channel B")

    # Find sequential pairs
    pairs = find_sequential_segment_pairs(
        segments_a, segments_b,
        max_gap=max_gap,
        allow_overlap=True
    )

    print(f"Found {len(pairs)} sequential segment pairs")

    # Filter pairs by segment length
    def filter_pair(pair: SegmentPair) -> bool:
        for seg in [pair.first_segment, pair.second_segment]:
            if seg.length < min_segment_length:
                return False
            if max_segment_length is not None and seg.length > max_segment_length:
                return False
        return True

    valid_pairs = [p for p in pairs if filter_pair(p)]
    print(f"Valid pairs after filtering: {len(valid_pairs)}")

    if not valid_pairs:
        print("No valid pairs found!")
        return {'experiments': [], 'summary': {}}

    # Select pairs to test
    selected_pairs = random.sample(
        valid_pairs,
        min(num_pairs_to_test, len(valid_pairs))
    )

    results = {
        'experiments': [],
        'summary': {},
    }

    for pair_idx, pair in enumerate(selected_pairs):
        print(f"\n--- Testing pair {pair_idx + 1}/{len(selected_pairs)} ---")
        print(f"  First: {pair.first_channel} [{pair.first_segment.start_idx}:{pair.first_segment.end_idx}] "
              f"(len={pair.first_segment.length})")
        print(f"  Second: {pair.second_channel} [{pair.second_segment.start_idx}:{pair.second_segment.end_idx}] "
              f"(len={pair.second_segment.length})")
        print(f"  Gap: {pair.gap_duration} tokens")

        experiment = {
            'pair_idx': pair_idx,
            'first_channel': pair.first_channel,
            'second_channel': pair.second_channel,
            'first_segment_length': pair.first_segment.length,
            'second_segment_length': pair.second_segment.length,
            'gap_duration': pair.gap_duration,
            'first_start': pair.first_segment.start_idx,
            'first_end': pair.first_segment.end_idx,
            'second_start': pair.second_segment.start_idx,
            'second_end': pair.second_segment.end_idx,
        }

        try:
            # 1. Original order
            orig_a, orig_b = create_original_sequence(
                dialogue_a, dialogue_b, pair,
                truncate_after=True, post_tokens=1, silence_token=silence_token
            )
            orig_ppl, orig_nll = compute_truncated_perplexity(
                model, context_ids, context_length,
                orig_a, orig_b, offset, device
            )
            experiment['original_ppl'] = orig_ppl
            experiment['original_nll'] = orig_nll
            experiment['original_length'] = len(orig_a)
            print(f"  Original: PPL={orig_ppl:.4f}, NLL={orig_nll:.4f}")

            # 2. Reversed order
            rev_a, rev_b = create_reversed_sequence(
                dialogue_a, dialogue_b, pair,
                truncate_after=True, post_tokens=1, silence_token=silence_token
            )
            rev_ppl, rev_nll = compute_truncated_perplexity(
                model, context_ids, context_length,
                rev_a, rev_b, offset, device
            )
            experiment['reversed_ppl'] = rev_ppl
            experiment['reversed_nll'] = rev_nll
            experiment['reversed_length'] = len(rev_a)
            print(f"  Reversed: PPL={rev_ppl:.4f}, NLL={rev_nll:.4f}")

            # 3. Simultaneous
            sim_a, sim_b = create_simultaneous_sequence(
                dialogue_a, dialogue_b, pair,
                truncate_after=True, post_tokens=1, silence_token=silence_token
            )
            sim_ppl, sim_nll = compute_truncated_perplexity(
                model, context_ids, context_length,
                sim_a, sim_b, offset, device
            )
            experiment['simultaneous_ppl'] = sim_ppl
            experiment['simultaneous_nll'] = sim_nll
            experiment['simultaneous_length'] = len(sim_a)
            print(f"  Simultaneous: PPL={sim_ppl:.4f}, NLL={sim_nll:.4f}")

            # Compute relative changes
            experiment['reversed_vs_original'] = rev_ppl / orig_ppl if orig_ppl > 0 else float('inf')
            experiment['simultaneous_vs_original'] = sim_ppl / orig_ppl if orig_ppl > 0 else float('inf')

            # Which order has lowest perplexity?
            min_ppl = min(orig_ppl, rev_ppl, sim_ppl)
            if min_ppl == orig_ppl:
                experiment['best_order'] = 'original'
            elif min_ppl == rev_ppl:
                experiment['best_order'] = 'reversed'
            else:
                experiment['best_order'] = 'simultaneous'

            results['experiments'].append(experiment)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Compute summary statistics
    if results['experiments']:
        exps = results['experiments']

        # Count which order is best
        best_counts = {'original': 0, 'reversed': 0, 'simultaneous': 0}
        for exp in exps:
            best_counts[exp['best_order']] += 1

        results['summary'] = {
            'n_pairs_tested': len(exps),
            'best_order_counts': best_counts,
            'mean_original_ppl': np.mean([e['original_ppl'] for e in exps]),
            'mean_reversed_ppl': np.mean([e['reversed_ppl'] for e in exps]),
            'mean_simultaneous_ppl': np.mean([e['simultaneous_ppl'] for e in exps]),
            'mean_reversed_vs_original': np.mean([e['reversed_vs_original'] for e in exps]),
            'mean_simultaneous_vs_original': np.mean([e['simultaneous_vs_original'] for e in exps]),
        }

        print("\n" + "="*60)
        print("Summary:")
        print(f"  Pairs tested: {results['summary']['n_pairs_tested']}")
        print(f"  Best order counts: {results['summary']['best_order_counts']}")
        print(f"  Mean PPL - Original: {results['summary']['mean_original_ppl']:.4f}")
        print(f"  Mean PPL - Reversed: {results['summary']['mean_reversed_ppl']:.4f}")
        print(f"  Mean PPL - Simultaneous: {results['summary']['mean_simultaneous_ppl']:.4f}")
        print(f"  Mean Reversed/Original ratio: {results['summary']['mean_reversed_vs_original']:.4f}")
        print(f"  Mean Simultaneous/Original ratio: {results['summary']['mean_simultaneous_vs_original']:.4f}")

    return results


def run_multi_sample_turn_order_experiment(
    model,
    tokenizer,
    samples: List[Dict],
    offset: int,
    device: str = "cuda",
    max_gap: int = 20,
    num_pairs_per_sample: int = 5,
    random_seed: int = 42,
    min_segment_length: int = 5,
    max_segment_length: int = None,
    verbose: bool = True,
) -> Dict:
    """
    Run turn order experiment across multiple samples.
    """
    all_experiments = []

    for sample_idx, sample in enumerate(samples):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{len(samples)}")
            print(f"{'='*60}")

        try:
            input_ids = sample['input_ids']
            script_a_ids, script_b_ids, dialogue_ids = extract_script_contents(
                tokenizer, input_ids, return_dialogue=True
            )

            if len(dialogue_ids) < 20:
                if verbose:
                    print(f"  Skipping: dialogue too short ({len(dialogue_ids)} tokens)")
                continue

            dialogue_ids_a = dialogue_ids[::2]
            dialogue_ids_b = dialogue_ids[1::2]

            context_ids, context_length = prepare_inference_input(
                script_a_ids, script_b_ids,
                tokenizer, device, already_tokenized=True
            )

            sample_results = run_turn_order_experiment(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                context_length=context_length,
                dialogue_ids_a=dialogue_ids_a,
                dialogue_ids_b=dialogue_ids_b,
                offset=offset,
                device=device,
                max_gap=max_gap,
                num_pairs_to_test=num_pairs_per_sample,
                random_seed=random_seed + sample_idx,
                min_segment_length=min_segment_length,
                max_segment_length=max_segment_length,
            )

            # Add sample_id to experiments
            for exp in sample_results.get('experiments', []):
                exp['sample_id'] = sample_idx
                all_experiments.append(exp)

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    # Compute overall summary
    merged_results = {
        'experiments': all_experiments,
        'summary': {},
    }

    if all_experiments:
        best_counts = {'original': 0, 'reversed': 0, 'simultaneous': 0}
        for exp in all_experiments:
            best_counts[exp['best_order']] += 1

        merged_results['summary'] = {
            'n_pairs_tested': len(all_experiments),
            'n_samples': len(set(e['sample_id'] for e in all_experiments)),
            'best_order_counts': best_counts,
            'mean_original_ppl': np.mean([e['original_ppl'] for e in all_experiments]),
            'mean_reversed_ppl': np.mean([e['reversed_ppl'] for e in all_experiments]),
            'mean_simultaneous_ppl': np.mean([e['simultaneous_ppl'] for e in all_experiments]),
            'mean_reversed_vs_original': np.mean([e['reversed_vs_original'] for e in all_experiments]),
            'mean_simultaneous_vs_original': np.mean([e['simultaneous_vs_original'] for e in all_experiments]),
            'std_reversed_vs_original': np.std([e['reversed_vs_original'] for e in all_experiments]),
            'std_simultaneous_vs_original': np.std([e['simultaneous_vs_original'] for e in all_experiments]),
        }

        if verbose:
            print("\n" + "="*60)
            print("OVERALL SUMMARY")
            print("="*60)
            s = merged_results['summary']
            print(f"  Total pairs tested: {s['n_pairs_tested']} across {s['n_samples']} samples")
            print(f"  Best order counts: {s['best_order_counts']}")
            print(f"  Mean PPL - Original: {s['mean_original_ppl']:.4f}")
            print(f"  Mean PPL - Reversed: {s['mean_reversed_ppl']:.4f}")
            print(f"  Mean PPL - Simultaneous: {s['mean_simultaneous_ppl']:.4f}")
            print(f"  Reversed/Original: {s['mean_reversed_vs_original']:.4f} ± {s['std_reversed_vs_original']:.4f}")
            print(f"  Simultaneous/Original: {s['mean_simultaneous_vs_original']:.4f} ± {s['std_simultaneous_vs_original']:.4f}")

    return merged_results


def plot_turn_order_results(results: Dict, save_path: str = None):
    """
    Create visualizations for turn order experiment results.
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    exps = results['experiments']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Bar chart of best order counts
    ax1 = axes[0, 0]
    counts = results['summary']['best_order_counts']
    bars = ax1.bar(['Original', 'Reversed', 'Simultaneous'],
                   [counts['original'], counts['reversed'], counts['simultaneous']],
                   color=['green', 'red', 'blue'], alpha=0.7)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Which Order Has Lowest Perplexity?', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for bar, count in zip(bars, [counts['original'], counts['reversed'], counts['simultaneous']]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', fontsize=11)

    # 2. Box plot of perplexity ratios
    ax2 = axes[0, 1]
    reversed_ratios = [e['reversed_vs_original'] for e in exps]
    simultaneous_ratios = [e['simultaneous_vs_original'] for e in exps]

    bp = ax2.boxplot([reversed_ratios, simultaneous_ratios], patch_artist=True)
    ax2.set_xticklabels(['Reversed/Original', 'Simultaneous/Original'])
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('blue')
    bp['boxes'][1].set_alpha(0.5)

    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Original baseline')
    ax2.set_ylabel('Perplexity Ratio', fontsize=11)
    ax2.set_title('Perplexity Ratios vs Original', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: Original vs Reversed perplexity
    ax3 = axes[0, 2]
    orig_ppls = [e['original_ppl'] for e in exps]
    rev_ppls = [e['reversed_ppl'] for e in exps]

    ax3.scatter(orig_ppls, rev_ppls, alpha=0.6, s=50)

    # Add diagonal line (y = x)
    max_val = max(max(orig_ppls), max(rev_ppls))
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x')

    ax3.set_xlabel('Original PPL', fontsize=11)
    ax3.set_ylabel('Reversed PPL', fontsize=11)
    ax3.set_title('Original vs Reversed Perplexity', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Histogram of reversed/original ratios
    ax4 = axes[1, 0]
    ax4.hist(reversed_ratios, bins=20, color='red', alpha=0.6, edgecolor='black')
    ax4.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Original')
    ax4.axvline(x=np.mean(reversed_ratios), color='red', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(reversed_ratios):.3f}')
    ax4.set_xlabel('Reversed / Original Ratio', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Distribution of Reversed/Original Ratios', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Histogram of simultaneous/original ratios
    ax5 = axes[1, 1]
    ax5.hist(simultaneous_ratios, bins=20, color='blue', alpha=0.6, edgecolor='black')
    ax5.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Original')
    ax5.axvline(x=np.mean(simultaneous_ratios), color='blue', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(simultaneous_ratios):.3f}')
    ax5.set_xlabel('Simultaneous / Original Ratio', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Distribution of Simultaneous/Original Ratios', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Effect of gap duration
    ax6 = axes[1, 2]
    gaps = [e['gap_duration'] for e in exps]

    ax6.scatter(gaps, reversed_ratios, alpha=0.6, s=50, color='red', label='Reversed')
    ax6.scatter(gaps, simultaneous_ratios, alpha=0.6, s=50, color='blue', label='Simultaneous')
    ax6.axhline(y=1.0, color='green', linestyle='--', linewidth=2)
    ax6.set_xlabel('Original Gap Duration (tokens)', fontsize=11)
    ax6.set_ylabel('Perplexity Ratio', fontsize=11)
    ax6.set_title('Effect of Gap Duration on Perplexity Ratio', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


def plot_by_first_speaker(results: Dict, save_path: str = None):
    """
    Plot results separated by which channel speaks first (A->B vs B->A).
    """
    if not results['experiments']:
        print("No experiment results to plot!")
        return

    # Separate by first speaker
    a_first = [e for e in results['experiments'] if e['first_channel'] == 'A']
    b_first = [e for e in results['experiments'] if e['first_channel'] == 'B']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Compare reversed ratios
    ax1 = axes[0]
    data = []
    labels = []

    if a_first:
        data.append([e['reversed_vs_original'] for e in a_first])
        labels.append(f'A→B (n={len(a_first)})')
    if b_first:
        data.append([e['reversed_vs_original'] for e in b_first])
        labels.append(f'B→A (n={len(b_first)})')

    if data:
        bp = ax1.boxplot(data, patch_artist=True)
        ax1.set_xticklabels(labels)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)

    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2)
    ax1.set_ylabel('Reversed / Original Ratio', fontsize=11)
    ax1.set_title('Reversed Order Effect by First Speaker', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Compare simultaneous ratios
    ax2 = axes[1]
    data = []
    labels = []

    if a_first:
        data.append([e['simultaneous_vs_original'] for e in a_first])
        labels.append(f'A→B (n={len(a_first)})')
    if b_first:
        data.append([e['simultaneous_vs_original'] for e in b_first])
        labels.append(f'B→A (n={len(b_first)})')

    if data:
        bp = ax2.boxplot(data, patch_artist=True)
        ax2.set_xticklabels(labels)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)

    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2)
    ax2.set_ylabel('Simultaneous / Original Ratio', fontsize=11)
    ax2.set_title('Simultaneous Effect by First Speaker', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Best order distribution by first speaker
    ax3 = axes[2]

    categories = ['Original', 'Reversed', 'Simultaneous']
    x = np.arange(len(categories))
    width = 0.35

    if a_first:
        a_counts = [sum(1 for e in a_first if e['best_order'] == o) for o in ['original', 'reversed', 'simultaneous']]
        ax3.bar(x - width/2, a_counts, width, label='A→B', color='lightblue', edgecolor='black')

    if b_first:
        b_counts = [sum(1 for e in b_first if e['best_order'] == o) for o in ['original', 'reversed', 'simultaneous']]
        ax3.bar(x + width/2, b_counts, width, label='B→A', color='lightcoral', edgecolor='black')

    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Best Order by First Speaker', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()
    return fig


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Scheduler checkpoint directory.")
    parser.add_argument("--data_jsonl", type=str, required=True,
                        help="Tokenized JSONL: the synthetic question/answer set or a validation set.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="turn_order", help="Prefix of the output files.")
    parser.add_argument("--num_samples", type=int, default=1500, help="Records drawn at random from --data_jsonl.")
    parser.add_argument("--seed", type=int, default=None, help="Seeds the record draw (1986 reproduces the draw of the published runs). "
                             "Omit for an unseeded draw.")
    parser.add_argument("--pair_seed", type=int, default=42,
                        help="Base seed for choosing segment pairs (sample i uses pair_seed + i).")
    parser.add_argument("--max_gap", type=int, default=20, help="Largest gap (tokens) between paired segments.")
    parser.add_argument("--num_pairs_per_sample", type=int, default=10)
    parser.add_argument("--min_segment_length", type=int, default=0)
    parser.add_argument("--max_segment_length", type=int, default=None)
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

    results = run_multi_sample_turn_order_experiment(
        model, tokenizer, samples, offset, args.device,
        max_gap=args.max_gap,
        num_pairs_per_sample=args.num_pairs_per_sample,
        random_seed=args.pair_seed,
        min_segment_length=args.min_segment_length,
        max_segment_length=args.max_segment_length,
    )

    with open(out_dir / f"{args.name}_results.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    plot_turn_order_results(results, save_path=str(out_dir / f"{args.name}.png"))
    plot_by_first_speaker(results, save_path=str(out_dir / f"{args.name}_by_speaker.png"))


if __name__ == "__main__":
    main()
