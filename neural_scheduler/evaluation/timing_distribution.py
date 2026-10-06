"""Gap and overlap duration distributions: generated dialogue vs. ground truth.

For each validation window the scheduler samples a dialogue from the two
scripts. Mutual-silence runs (gaps) and simultaneous-speech runs (overlaps) are
extracted from the generated and the ground-truth dialogue and their length
distributions are compared (two-sample KS on pooled run lengths, t-test on
per-dialogue means, share of gap / overlap / single-speaker frames).

Run lengths are in tokens (80 ms each).

    python -m neural_scheduler.evaluation.timing_distribution \\
        --checkpoint CKPT --data_jsonl VAL.jsonl --out_dir OUT --name my_run --seed 0
    python -m neural_scheduler.evaluation.timing_distribution \\
        --plot_only --out_dir OUT --name my_run
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..constants import MS_PER_TOKEN, SILENCE_CODE
from .common import extract_script_contents, generate_dialogue, load_jsonl_examples, to_serializable

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
}

@dataclass
class MutualSilenceStats:
    """Statistics about mutual silence in a dialogue."""
    mutual_silence_lengths: List[int]
    mutual_silence_positions: List[int]
    total_mutual_silence_tokens: int
    total_dialogue_tokens: int
    mutual_silence_ratio: float
    mean_length: float
    median_length: float
    std_length: float
    max_length: int


@dataclass
class SimultaneousSpeechStats:
    """Statistics about simultaneous speech (overlap) in a dialogue."""
    simultaneous_speech_lengths: List[int]
    simultaneous_speech_positions: List[int]
    total_simultaneous_speech_tokens: int
    total_dialogue_tokens: int
    simultaneous_speech_ratio: float
    mean_length: float
    median_length: float
    std_length: float
    max_length: int


@dataclass
class DialogueTimingStats:
    """Combined timing statistics for a dialogue."""
    mutual_silence: MutualSilenceStats
    simultaneous_speech: SimultaneousSpeechStats
    # Derived: tokens where exactly one speaker is active
    single_speaker_tokens: int
    single_speaker_ratio: float


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_mutual_silence_segments(
    dialogue_ids_a: List[int],
    dialogue_ids_b: List[int],
    silence_token: int = SILENCE_CODE,
) -> MutualSilenceStats:
    """
    Extract all mutual silence segments from parallel dialogue channels.
    """
    assert len(dialogue_ids_a) == len(dialogue_ids_b), "Channels must be same length"

    mutual_silence_lengths = []
    mutual_silence_positions = []

    current_silence_length = 0
    current_silence_start = None

    for i, (tok_a, tok_b) in enumerate(zip(dialogue_ids_a, dialogue_ids_b)):
        if tok_a == silence_token and tok_b == silence_token:
            if current_silence_start is None:
                current_silence_start = i
            current_silence_length += 1
        else:
            if current_silence_length > 0:
                mutual_silence_lengths.append(current_silence_length)
                mutual_silence_positions.append(current_silence_start)
                current_silence_length = 0
                current_silence_start = None

    if current_silence_length > 0:
        mutual_silence_lengths.append(current_silence_length)
        mutual_silence_positions.append(current_silence_start)

    total_mutual_silence = sum(mutual_silence_lengths)
    total_tokens = len(dialogue_ids_a)

    if mutual_silence_lengths:
        mean_length = np.mean(mutual_silence_lengths)
        median_length = np.median(mutual_silence_lengths)
        std_length = np.std(mutual_silence_lengths)
        max_length = max(mutual_silence_lengths)
    else:
        mean_length = median_length = std_length = max_length = 0

    return MutualSilenceStats(
        mutual_silence_lengths=mutual_silence_lengths,
        mutual_silence_positions=mutual_silence_positions,
        total_mutual_silence_tokens=total_mutual_silence,
        total_dialogue_tokens=total_tokens,
        mutual_silence_ratio=total_mutual_silence / total_tokens if total_tokens > 0 else 0,
        mean_length=mean_length,
        median_length=median_length,
        std_length=std_length,
        max_length=max_length,
    )


def extract_simultaneous_speech_segments(
    dialogue_ids_a: List[int],
    dialogue_ids_b: List[int],
    silence_token: int = SILENCE_CODE,
) -> SimultaneousSpeechStats:
    """
    Extract all simultaneous speech (overlap) segments from parallel dialogue channels.

    Simultaneous speech occurs when BOTH channels have non-silence tokens at the same position.

    Args:
        dialogue_ids_a: Channel A tokens (in audio token space, not offset)
        dialogue_ids_b: Channel B tokens (in audio token space, not offset)
        silence_token: The silence codec id

    Returns:
        SimultaneousSpeechStats object with detailed statistics
    """
    assert len(dialogue_ids_a) == len(dialogue_ids_b), "Channels must be same length"

    simultaneous_speech_lengths = []
    simultaneous_speech_positions = []

    current_overlap_length = 0
    current_overlap_start = None

    for i, (tok_a, tok_b) in enumerate(zip(dialogue_ids_a, dialogue_ids_b)):
        if tok_a != silence_token and tok_b != silence_token:
            # Both channels are speaking
            if current_overlap_start is None:
                current_overlap_start = i
            current_overlap_length += 1
        else:
            # At least one channel is silent
            if current_overlap_length > 0:
                simultaneous_speech_lengths.append(current_overlap_length)
                simultaneous_speech_positions.append(current_overlap_start)
                current_overlap_length = 0
                current_overlap_start = None

    # Handle case where dialogue ends in overlap
    if current_overlap_length > 0:
        simultaneous_speech_lengths.append(current_overlap_length)
        simultaneous_speech_positions.append(current_overlap_start)

    total_simultaneous_speech = sum(simultaneous_speech_lengths)
    total_tokens = len(dialogue_ids_a)

    if simultaneous_speech_lengths:
        mean_length = np.mean(simultaneous_speech_lengths)
        median_length = np.median(simultaneous_speech_lengths)
        std_length = np.std(simultaneous_speech_lengths)
        max_length = max(simultaneous_speech_lengths)
    else:
        mean_length = median_length = std_length = max_length = 0

    return SimultaneousSpeechStats(
        simultaneous_speech_lengths=simultaneous_speech_lengths,
        simultaneous_speech_positions=simultaneous_speech_positions,
        total_simultaneous_speech_tokens=total_simultaneous_speech,
        total_dialogue_tokens=total_tokens,
        simultaneous_speech_ratio=total_simultaneous_speech / total_tokens if total_tokens > 0 else 0,
        mean_length=mean_length,
        median_length=median_length,
        std_length=std_length,
        max_length=max_length,
    )


def extract_dialogue_timing_stats(
    dialogue_ids_a: List[int],
    dialogue_ids_b: List[int],
    silence_token: int = SILENCE_CODE,
) -> DialogueTimingStats:
    """
    Extract combined mutual silence AND simultaneous speech stats for a dialogue.
    """
    ms_stats = extract_mutual_silence_segments(dialogue_ids_a, dialogue_ids_b, silence_token)
    ss_stats = extract_simultaneous_speech_segments(dialogue_ids_a, dialogue_ids_b, silence_token)

    total_tokens = len(dialogue_ids_a)
    single_speaker_tokens = total_tokens - ms_stats.total_mutual_silence_tokens - ss_stats.total_simultaneous_speech_tokens
    single_speaker_ratio = single_speaker_tokens / total_tokens if total_tokens > 0 else 0

    return DialogueTimingStats(
        mutual_silence=ms_stats,
        simultaneous_speech=ss_stats,
        single_speaker_tokens=single_speaker_tokens,
        single_speaker_ratio=single_speaker_ratio,
    )


# =============================================================================
# COMPARISON / VISUALIZATION
# =============================================================================

def _compare_segment_distributions(
    gt_all_lengths: List[int],
    gen_all_lengths: List[int],
    gt_means: List[float],
    gen_means: List[float],
    gt_ratios: List[float],
    gen_ratios: List[float],
    label: str,
    axes_row,
):
    """
    Helper to plot a row of 3 comparison plots for either mutual silence or simultaneous speech.
    axes_row should be a list/array of 3 axes.
    """
    gt_length_dist = Counter(gt_all_lengths)
    gen_length_dist = Counter(gen_all_lengths)

    # KS test
    if gt_all_lengths and gen_all_lengths:
        ks_statistic, ks_pvalue = stats.ks_2samp(gt_all_lengths, gen_all_lengths)
    else:
        ks_statistic = ks_pvalue = None

    # t-test on per-sample means
    if gt_means and gen_means:
        t_statistic, t_pvalue = stats.ttest_ind(gt_means, gen_means)
    else:
        t_statistic = t_pvalue = None

    # Plot 1: Histogram
    ax1 = axes_row[0]
    max_length = max(max(gt_all_lengths or [0]), max(gen_all_lengths or [0]))
    bins = range(0, min(max_length + 2, 60))  # cap bins for readability
    ax1.hist(gt_all_lengths, bins=bins, alpha=0.5, label='Ground Truth', density=True, color='blue')
    ax1.hist(gen_all_lengths, bins=bins, alpha=0.5, label='Generated', density=True, color='red')
    ax1.set_xlabel(f'{label} Length (tokens)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{label} Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: CDF
    ax2 = axes_row[1]
    if gt_all_lengths:
        sorted_gt = np.sort(gt_all_lengths)
        cdf_gt = np.arange(1, len(sorted_gt) + 1) / len(sorted_gt)
        ax2.plot(sorted_gt, cdf_gt, label='Ground Truth', linewidth=2, color='blue')
    if gen_all_lengths:
        sorted_gen = np.sort(gen_all_lengths)
        cdf_gen = np.arange(1, len(sorted_gen) + 1) / len(sorted_gen)
        ax2.plot(sorted_gen, cdf_gen, label='Generated', linewidth=2, color='red')
    ks_text = f'KS={ks_statistic:.4f}, p={ks_pvalue:.4f}' if ks_statistic is not None else ''
    ax2.set_xlabel(f'{label} Length (tokens)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'{label} CDF ({ks_text})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot of ratios
    ax3 = axes_row[2]
    ax3.boxplot([gt_ratios, gen_ratios])
    ax3.set_xticklabels(['Ground Truth', 'Generated'])
    ax3.set_ylabel(f'{label} Ratio')
    t_text = f't={t_statistic:.3f}, p={t_pvalue:.4f}' if t_statistic is not None else ''
    ax3.set_title(f'{label} Ratio ({t_text})')
    ax3.grid(True, alpha=0.3)

    return {
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        't_statistic': t_statistic,
        't_pvalue': t_pvalue,
        'gt_mean_length': np.mean(gt_all_lengths) if gt_all_lengths else 0,
        'gen_mean_length': np.mean(gen_all_lengths) if gen_all_lengths else 0,
        'gt_median_length': np.median(gt_all_lengths) if gt_all_lengths else 0,
        'gen_median_length': np.median(gen_all_lengths) if gen_all_lengths else 0,
        'gt_mean_ratio': np.mean(gt_ratios) if gt_ratios else 0,
        'gen_mean_ratio': np.mean(gen_ratios) if gen_ratios else 0,
        'gt_length_distribution': dict(gt_length_dist),
        'gen_length_distribution': dict(gen_length_dist),
    }


def compare_dialogue_timing_distributions(
    ground_truth_stats: List[DialogueTimingStats],
    generated_stats: List[DialogueTimingStats],
    save_path: str = None,
) -> Dict:
    """
    Compare BOTH mutual silence AND simultaneous speech distributions
    between ground truth and generated dialogues.

    Returns:
        Dictionary with comparison metrics for both phenomena.
    """
    # --- Aggregate mutual silence ---
    gt_ms_lengths = []
    gen_ms_lengths = []
    gt_ms_means = []
    gen_ms_means = []
    gt_ms_ratios = []
    gen_ms_ratios = []

    for s in ground_truth_stats:
        gt_ms_lengths.extend(s.mutual_silence.mutual_silence_lengths)
        if s.mutual_silence.mutual_silence_lengths:
            gt_ms_means.append(s.mutual_silence.mean_length)
        gt_ms_ratios.append(s.mutual_silence.mutual_silence_ratio)

    for s in generated_stats:
        gen_ms_lengths.extend(s.mutual_silence.mutual_silence_lengths)
        if s.mutual_silence.mutual_silence_lengths:
            gen_ms_means.append(s.mutual_silence.mean_length)
        gen_ms_ratios.append(s.mutual_silence.mutual_silence_ratio)

    # --- Aggregate simultaneous speech ---
    gt_ss_lengths = []
    gen_ss_lengths = []
    gt_ss_means = []
    gen_ss_means = []
    gt_ss_ratios = []
    gen_ss_ratios = []

    for s in ground_truth_stats:
        gt_ss_lengths.extend(s.simultaneous_speech.simultaneous_speech_lengths)
        if s.simultaneous_speech.simultaneous_speech_lengths:
            gt_ss_means.append(s.simultaneous_speech.mean_length)
        gt_ss_ratios.append(s.simultaneous_speech.simultaneous_speech_ratio)

    for s in generated_stats:
        gen_ss_lengths.extend(s.simultaneous_speech.simultaneous_speech_lengths)
        if s.simultaneous_speech.simultaneous_speech_lengths:
            gen_ss_means.append(s.simultaneous_speech.mean_length)
        gen_ss_ratios.append(s.simultaneous_speech.simultaneous_speech_ratio)

    # --- Create figure: 3 rows x 3 cols ---
    # Row 0: Mutual silence (histogram, CDF, ratio boxplot)
    # Row 1: Simultaneous speech (histogram, CDF, ratio boxplot)
    # Row 2: Summary panel
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    ms_results = _compare_segment_distributions(
        gt_ms_lengths, gen_ms_lengths,
        gt_ms_means, gen_ms_means,
        gt_ms_ratios, gen_ms_ratios,
        label="Mutual Silence",
        axes_row=axes[0],
    )

    ss_results = _compare_segment_distributions(
        gt_ss_lengths, gen_ss_lengths,
        gt_ss_means, gen_ss_means,
        gt_ss_ratios, gen_ss_ratios,
        label="Simultaneous Speech",
        axes_row=axes[1],
    )

    # --- Row 2: Summary / breakdown ---
    # Plot: stacked bar of dialogue composition (mutual silence, simultaneous speech, single speaker)
    ax_comp = axes[2, 0]
    gt_ms_ratio_mean = np.mean(gt_ms_ratios) if gt_ms_ratios else 0
    gt_ss_ratio_mean = np.mean(gt_ss_ratios) if gt_ss_ratios else 0
    gt_single_ratio_mean = np.mean([s.single_speaker_ratio for s in ground_truth_stats])
    gen_ms_ratio_mean = np.mean(gen_ms_ratios) if gen_ms_ratios else 0
    gen_ss_ratio_mean = np.mean(gen_ss_ratios) if gen_ss_ratios else 0
    gen_single_ratio_mean = np.mean([s.single_speaker_ratio for s in generated_stats])

    categories = ['Ground Truth', 'Generated']
    ms_vals = [gt_ms_ratio_mean, gen_ms_ratio_mean]
    ss_vals = [gt_ss_ratio_mean, gen_ss_ratio_mean]
    single_vals = [gt_single_ratio_mean, gen_single_ratio_mean]

    x = np.arange(len(categories))
    width = 0.5
    ax_comp.bar(x, ms_vals, width, label='Mutual Silence', color='#4a86c8')
    ax_comp.bar(x, ss_vals, width, bottom=ms_vals, label='Simultaneous Speech', color='#e8524a')
    ax_comp.bar(x, single_vals, width,
                bottom=[ms_vals[i] + ss_vals[i] for i in range(len(categories))],
                label='Single Speaker', color='#7fc97f')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(categories)
    ax_comp.set_ylabel('Proportion of Dialogue')
    ax_comp.set_title('Dialogue Composition Breakdown')
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3, axis='y')

    # Mean length comparison bar chart
    ax_means = axes[2, 1]
    labels = ['Mutual Silence', 'Simultaneous Speech']
    gt_mean_lengths = [ms_results['gt_mean_length'], ss_results['gt_mean_length']]
    gen_mean_lengths = [ms_results['gen_mean_length'], ss_results['gen_mean_length']]
    x2 = np.arange(len(labels))
    bar_w = 0.35
    ax_means.bar(x2 - bar_w/2, gt_mean_lengths, bar_w, label='Ground Truth', color='blue', alpha=0.7)
    ax_means.bar(x2 + bar_w/2, gen_mean_lengths, bar_w, label='Generated', color='red', alpha=0.7)
    ax_means.set_xticks(x2)
    ax_means.set_xticklabels(labels)
    ax_means.set_ylabel('Mean Segment Length (tokens)')
    ax_means.set_title('Mean Segment Lengths')
    ax_means.legend()
    ax_means.grid(True, alpha=0.3, axis='y')

    # Summary text
    ax_summary = axes[2, 2]
    ax_summary.axis('off')

    def fmt_test(label, res):
        ks = f"D={res['ks_statistic']:.4f}, p={res['ks_pvalue']:.4f}" if res['ks_statistic'] is not None else "N/A"
        tt = f"t={res['t_statistic']:.3f}, p={res['t_pvalue']:.4f}" if res['t_statistic'] is not None else "N/A"
        return (
            f"  {label}:\n"
            f"    GT mean len: {res['gt_mean_length']:.2f}, ratio: {res['gt_mean_ratio']:.3f}\n"
            f"    Gen mean len: {res['gen_mean_length']:.2f}, ratio: {res['gen_mean_ratio']:.3f}\n"
            f"    KS: {ks}\n"
            f"    t-test: {tt}\n"
        )

    summary_text = (
        "Summary\n"
        "-------\n"
        f"Samples: GT={len(ground_truth_stats)}, Gen={len(generated_stats)}\n\n"
        + fmt_test("Mutual Silence", ms_results)
        + "\n"
        + fmt_test("Simultaneous Speech", ss_results)
        + "\nInterpretation:\n"
        "  p < 0.05 → distributions differ significantly"
    )
    ax_summary.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center', family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.close(fig)

    return {
        'mutual_silence': ms_results,
        'simultaneous_speech': ss_results,
        'gt_single_speaker_ratio': gt_single_ratio_mean,
        'gen_single_speaker_ratio': gen_single_ratio_mean,
    }

def plot_distribution_only(
    ground_truth_stats: List[DialogueTimingStats],
    generated_stats: List[DialogueTimingStats],
    save_path: str = None,
    tokens_to_ms: float = float(MS_PER_TOKEN),
    x_limit_ms: float = 2000.0,  # how far out on each side to show; tails beyond are clipped.
    num_bins: int = 25,          # number of bins per side (overlap / gap)
    title: str = None,           # optional custom title; None = default
) -> None:
    """
    FTO-style figure on a single panel: signed x-axis where overlap durations
    are plotted as negative values and gap (mutual silence) durations as positive
    values. Ground Truth shown as filled bars; Generated shown as a step outline
    layered on top, so overlap between the two distributions stays legible.

    Sized to fit side-by-side in a single column of a two-column paper
    (≈3.3 in wide × 2.6 in tall — half a ~6.875″ column).
    """
    # Aggregate lengths (in tokens), then convert to signed ms (overlap negative, gap positive)
    gt_ms_lengths, gen_ms_lengths = [], []
    gt_ss_lengths, gen_ss_lengths = [], []
    for s in ground_truth_stats:
        gt_ms_lengths.extend(s.mutual_silence.mutual_silence_lengths)
        gt_ss_lengths.extend(s.simultaneous_speech.simultaneous_speech_lengths)
    for s in generated_stats:
        gen_ms_lengths.extend(s.mutual_silence.mutual_silence_lengths)
        gen_ss_lengths.extend(s.simultaneous_speech.simultaneous_speech_lengths)

    gt_signed = np.array(
        [-l * tokens_to_ms for l in gt_ss_lengths] +
        [ l * tokens_to_ms for l in gt_ms_lengths]
    )
    gen_signed = np.array(
        [-l * tokens_to_ms for l in gen_ss_lengths] +
        [ l * tokens_to_ms for l in gen_ms_lengths]
    )

    # Shared bin edges — use num_bins to keep bars readable at small widths.
    x_limit = float(x_limit_ms)
    edges = np.linspace(-x_limit, x_limit, 2 * num_bins + 1)

    # Colors: a muted neutral for GT (the "reference"), a strong contrasting hue for Generated.
    gt_face   = '#bdbdbd'   # light gray fill
    gt_edge   = '#525252'   # darker gray outline
    gen_color = '#d62728'   # matplotlib red — strong against gray

    # --- Narrow figure for half of a single column ---
    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    # Ground truth: filled translucent bars
    ax.hist(
        gt_signed, bins=edges, density=True,
        color=gt_face, edgecolor=gt_edge, linewidth=0.5,
        alpha=0.85, label=f'GT (n={len(gt_signed)})',
        zorder=1,
    )

    # Generated: step outline on top — readable wherever it overlaps GT
    ax.hist(
        gen_signed, bins=edges, density=True,
        histtype='step', color=gen_color, linewidth=1.5,
        label=f'Gen (n={len(gen_signed)})',
        zorder=3,
    )

    # Center line + light section shading to reinforce the overlap/gap split
    ax.axvspan(-x_limit, 0, color='#f0f0f0', alpha=0.4, zorder=0)
    ax.axvline(0, color='black', linewidth=1.0, zorder=2)

    # Section labels just inside the top of the plot
    ymax = ax.get_ylim()[1]
    ax.text(-x_limit * 0.5, ymax * 0.95, 'Overlap',
            ha='center', va='top', fontsize=8, fontweight='bold', color='#333333')
    ax.text( x_limit * 0.5, ymax * 0.95, 'Gap',
            ha='center', va='top', fontsize=8, fontweight='bold', color='#333333')

    ax.set_xlim(-x_limit, x_limit)
    ax.set_xlabel('Between-speaker interval (ms)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Density', fontsize=8, fontweight='bold')
    if title is not None:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    else:
        ax.set_title('Between-speaker intervals', fontsize=9, fontweight='bold', pad=6)
    ax.grid(True, alpha=0.25, axis='y', zorder=0)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=6,
              handlelength=1.2, handletextpad=0.4, borderpad=0.3)

    # Bold tick labels
    ax.tick_params(axis='both', labelsize=7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Tidy spines
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution-only plot to {save_path}")
    plt.close(fig)


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def _stats_to_dict(s: DialogueTimingStats) -> Dict:
    """Convert a DialogueTimingStats dataclass to a JSON-safe dict."""
    ms = s.mutual_silence
    ss = s.simultaneous_speech
    return {
        'mutual_silence': {
            'mutual_silence_lengths': list(ms.mutual_silence_lengths),
            'mutual_silence_positions': list(ms.mutual_silence_positions),
            'total_mutual_silence_tokens': int(ms.total_mutual_silence_tokens),
            'total_dialogue_tokens': int(ms.total_dialogue_tokens),
            'mutual_silence_ratio': float(ms.mutual_silence_ratio),
            'mean_length': float(ms.mean_length),
            'median_length': float(ms.median_length),
            'std_length': float(ms.std_length),
            'max_length': int(ms.max_length),
        },
        'simultaneous_speech': {
            'simultaneous_speech_lengths': list(ss.simultaneous_speech_lengths),
            'simultaneous_speech_positions': list(ss.simultaneous_speech_positions),
            'total_simultaneous_speech_tokens': int(ss.total_simultaneous_speech_tokens),
            'total_dialogue_tokens': int(ss.total_dialogue_tokens),
            'simultaneous_speech_ratio': float(ss.simultaneous_speech_ratio),
            'mean_length': float(ss.mean_length),
            'median_length': float(ss.median_length),
            'std_length': float(ss.std_length),
            'max_length': int(ss.max_length),
        },
        'single_speaker_tokens': int(s.single_speaker_tokens),
        'single_speaker_ratio': float(s.single_speaker_ratio),
    }


def _dict_to_stats(d: Dict) -> DialogueTimingStats:
    """Reconstruct a DialogueTimingStats from a JSON-loaded dict."""
    ms_d = d['mutual_silence']
    ss_d = d['simultaneous_speech']
    ms = MutualSilenceStats(
        mutual_silence_lengths=ms_d['mutual_silence_lengths'],
        mutual_silence_positions=ms_d['mutual_silence_positions'],
        total_mutual_silence_tokens=ms_d['total_mutual_silence_tokens'],
        total_dialogue_tokens=ms_d['total_dialogue_tokens'],
        mutual_silence_ratio=ms_d['mutual_silence_ratio'],
        mean_length=ms_d['mean_length'],
        median_length=ms_d['median_length'],
        std_length=ms_d['std_length'],
        max_length=ms_d['max_length'],
    )
    ss = SimultaneousSpeechStats(
        simultaneous_speech_lengths=ss_d['simultaneous_speech_lengths'],
        simultaneous_speech_positions=ss_d['simultaneous_speech_positions'],
        total_simultaneous_speech_tokens=ss_d['total_simultaneous_speech_tokens'],
        total_dialogue_tokens=ss_d['total_dialogue_tokens'],
        simultaneous_speech_ratio=ss_d['simultaneous_speech_ratio'],
        mean_length=ss_d['mean_length'],
        median_length=ss_d['median_length'],
        std_length=ss_d['std_length'],
        max_length=ss_d['max_length'],
    )
    return DialogueTimingStats(
        mutual_silence=ms,
        simultaneous_speech=ss,
        single_speaker_tokens=d['single_speaker_tokens'],
        single_speaker_ratio=d['single_speaker_ratio'],
    )


def save_experiment_results_json(
    ground_truth_stats: List[DialogueTimingStats],
    generated_stats: List[DialogueTimingStats],
    comparison: Dict,
    generation_params: Dict,
    save_path: str,
    extra_metadata: Dict = None,
) -> None:
    """Dump everything needed to replot later."""
    payload = {
        'metadata': to_serializable(extra_metadata or {}),
        'generation_params': to_serializable(generation_params),
        'comparison': to_serializable(comparison),
        'ground_truth_stats': [_stats_to_dict(s) for s in ground_truth_stats],
        'generated_stats': [_stats_to_dict(s) for s in generated_stats],
    }
    with open(save_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved experiment JSON to {save_path}")


def load_experiment_results_json(load_path: str) -> Dict:
    """
    Reload a saved experiment. Returns a dict with the same shape as
    run_dialogue_timing_experiment's return value, so you can pass
    `ground_truth_stats` and `generated_stats` straight back into any
    of the plotting functions.
    """
    with open(load_path, 'r') as f:
        payload = json.load(f)
    return {
        'ground_truth_stats': [_dict_to_stats(d) for d in payload['ground_truth_stats']],
        'generated_stats':    [_dict_to_stats(d) for d in payload['generated_stats']],
        'comparison':         payload['comparison'],
        'generation_params':  payload['generation_params'],
        'metadata':           payload.get('metadata', {}),
    }


# =============================================================================
# EXPERIMENT
# =============================================================================

def output_paths(out_dir: str, name: str, suffix: str = ".png") -> Dict[str, Path]:
    """Files written for experiment ``name``: ``<out_dir>/<name>/<name>{.png,_distribution_only.png,.json}``."""
    folder = Path(out_dir) / name
    return {
        "dir": folder,
        "main_png": folder / f"{name}{suffix}",
        "distribution_png": folder / f"{name}_distribution_only{suffix}",
        "json": folder / f"{name}.json",
    }


def run_dialogue_timing_experiment(
    model,
    tokenizer,
    samples: List[Dict],
    offset: int,
    device: str = "cuda",
    num_samples: int = 50,
    generation_params: Optional[Dict] = None,
    verbose: bool = True,
    out_dir: Optional[str] = None,
    name: str = "dialogue_timing_comparison",
    extra_metadata: Optional[Dict] = None,
) -> Dict:
    """Generate a dialogue for each of the first ``num_samples`` samples and
    compare gap / overlap statistics with the ground truth.

    ``offset`` is the vocab id of ``<|audio_0|>``. Samples whose dialogue has
    fewer than 10 tokens, whose generation has fewer than 10 tokens in either
    channel, or that raise, are skipped; a skipped generation still leaves the
    sample's ground truth in the reference set. With ``out_dir`` the two
    figures and a JSON holding everything needed to re-plot are written.
    """
    if generation_params is None:
        generation_params = dict(DEFAULT_GENERATION_PARAMS)

    ground_truth_stats = []
    generated_stats = []
    silence_token = SILENCE_CODE

    def describe(tag: str, timing: DialogueTimingStats) -> None:
        ms, ss = timing.mutual_silence, timing.simultaneous_speech
        print(f"  {tag} Mutual Silence: {len(ms.mutual_silence_lengths)} segments, "
              f"mean={ms.mean_length:.2f}, ratio={ms.mutual_silence_ratio:.3f}")
        print(f"  {tag} Simult. Speech: {len(ss.simultaneous_speech_lengths)} segments, "
              f"mean={ss.mean_length:.2f}, ratio={ss.simultaneous_speech_ratio:.3f}")
        print(f"  {tag} Single Speaker ratio: {timing.single_speaker_ratio:.3f}")

    for i, sample in enumerate(samples[:num_samples]):
        if verbose:
            print(f"\nProcessing sample {i + 1}/{num_samples}")
        try:
            script_a_ids, script_b_ids, dialogue_ids = extract_script_contents(
                tokenizer, sample["input_ids"], return_dialogue=True
            )
            if len(dialogue_ids) < 10:
                if verbose:
                    print(f"  Skipping: dialogue too short ({len(dialogue_ids)} tokens)")
                continue

            gt_dialogue_a = [x - offset for x in dialogue_ids[::2]]
            gt_dialogue_b = [x - offset for x in dialogue_ids[1::2]]
            gt_timing = extract_dialogue_timing_stats(gt_dialogue_a, gt_dialogue_b, silence_token)
            ground_truth_stats.append(gt_timing)
            if verbose:
                describe("GT", gt_timing)

            gen_a, gen_b = generate_dialogue(
                model=model,
                tokenizer=tokenizer,
                tokens_channel_a=list(script_a_ids),
                tokens_channel_b=list(script_b_ids),
                device=device,
                return_channel_separated=True,
                verbose=False,
                already_tokenized=True,
                **generation_params,
            )
            if len(gen_a) < 10 or len(gen_b) < 10:
                if verbose:
                    print("  Skipping: generated dialogue too short")
                continue

            min_len = min(len(gen_a), len(gen_b))
            gen_timing = extract_dialogue_timing_stats(gen_a[:min_len], gen_b[:min_len], silence_token)
            generated_stats.append(gen_timing)
            if verbose:
                describe("GEN", gen_timing)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    if verbose:
        print(f"\nGround truth samples: {len(ground_truth_stats)}, generated samples: {len(generated_stats)}")

    paths = output_paths(out_dir, name) if out_dir is not None else None
    if paths is not None:
        paths["dir"].mkdir(parents=True, exist_ok=True)

    comparison_results = compare_dialogue_timing_distributions(
        ground_truth_stats,
        generated_stats,
        save_path=str(paths["main_png"]) if paths else None,
    )

    if paths is not None:
        plot_distribution_only(ground_truth_stats, generated_stats, save_path=str(paths["distribution_png"]))
        metadata = {
            "num_samples_requested": num_samples,
            "num_gt_samples": len(ground_truth_stats),
            "num_gen_samples": len(generated_stats),
            "silence_token": silence_token,
            "offset": int(offset),
        }
        metadata.update(extra_metadata or {})
        save_experiment_results_json(
            ground_truth_stats=ground_truth_stats,
            generated_stats=generated_stats,
            comparison=comparison_results,
            generation_params=generation_params,
            save_path=str(paths["json"]),
            extra_metadata=metadata,
        )

    return {
        "ground_truth_stats": ground_truth_stats,
        "generated_stats": generated_stats,
        "comparison": comparison_results,
        "generation_params": generation_params,
        "output_dir": str(paths["dir"]) if paths else None,
    }


def print_summary(comparison: Dict) -> None:
    for metric_name, metric_key in [("Mutual Silence", "mutual_silence"), ("Simultaneous Speech", "simultaneous_speech")]:
        m = comparison[metric_key]
        print(f"\n--- {metric_name} ---")
        if m["ks_statistic"] is not None:
            print(f"KS test: D={m['ks_statistic']:.4f}, p-value={m['ks_pvalue']:.4f}")
        if m["t_statistic"] is not None:
            print(f"t-test: t={m['t_statistic']:.3f}, p-value={m['t_pvalue']:.4f}")
        print(f"GT mean length: {m['gt_mean_length']:.2f} tokens, ratio: {m['gt_mean_ratio']:.3f}")
        print(f"Gen mean length: {m['gen_mean_length']:.2f} tokens, ratio: {m['gen_mean_ratio']:.3f}")
    print("\n--- Dialogue Composition ---")
    print(f"GT single-speaker ratio: {comparison['gt_single_speaker_ratio']:.3f}")
    print(f"Gen single-speaker ratio: {comparison['gen_single_speaker_ratio']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--model_id", dest="checkpoint", type=str, default=None,
                        help="Scheduler checkpoint directory (not needed with --plot_only).")
    parser.add_argument("--data_jsonl", type=str, default=None,
                        help="Tokenized validation JSONL (records with 'input_ids').")
    parser.add_argument("--out_dir", type=str, required=True, help="Outputs go to <out_dir>/<name>/.")
    parser.add_argument("--name", type=str, default="dialogue_timing_comparison", help="Experiment name.")
    parser.add_argument("--num_samples", type=int, default=100, help="Dialogues to generate.")
    parser.add_argument("--pool_size", type=int, default=200,
                        help="Records drawn at random from --data_jsonl; the first --num_samples are used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seeds the record draw and the sampler (set after the model is loaded). "
                             "1986 reproduces the record draw of the published runs; omit for an unseeded run.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_GENERATION_PARAMS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_GENERATION_PARAMS["temperature"])
    parser.add_argument("--top_k", type=int, default=DEFAULT_GENERATION_PARAMS["top_k"])
    parser.add_argument("--top_p", type=float, default=DEFAULT_GENERATION_PARAMS["top_p"])
    parser.add_argument("--plot_only", action="store_true",
                        help="Re-plot from a saved JSON; no model or data needed.")
    parser.add_argument("--json_path", type=str, default=None,
                        help="JSON to re-plot (default: <out_dir>/<name>/<name>.json).")
    parser.add_argument("--plot_title", type=str, default=None, help="Title of the distribution-only plot.")
    parser.add_argument("--num_bins", type=int, default=25, help="Bins per side of the distribution-only plot.")
    parser.add_argument("--x_limit_ms", type=float, default=2000.0, help="X-axis limit (ms) of that plot.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = output_paths(args.out_dir, args.name)

    if args.plot_only:
        json_path = Path(args.json_path) if args.json_path else paths["json"]
        if not json_path.exists():
            sys.exit(f"--plot_only needs --json_path or a JSON at {json_path}")
        loaded = load_experiment_results_json(str(json_path))
        gt_stats, gen_stats = loaded["ground_truth_stats"], loaded["generated_stats"]
        print(f"Loaded {json_path}: GT samples {len(gt_stats)}, generated samples {len(gen_stats)}")
        paths["dir"].mkdir(parents=True, exist_ok=True)
        compare_dialogue_timing_distributions(gt_stats, gen_stats, save_path=str(paths["main_png"]))
        plot_distribution_only(
            gt_stats, gen_stats,
            save_path=str(paths["distribution_png"]),
            num_bins=args.num_bins,
            x_limit_ms=args.x_limit_ms,
            title=args.plot_title,
        )
        return

    if not args.checkpoint or not args.data_jsonl:
        sys.exit("--checkpoint and --data_jsonl are required unless --plot_only is given")

    import torch

    from ..vocab import audio_token
    from .common import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    offset = tokenizer.convert_tokens_to_ids(audio_token(0))
    samples = load_jsonl_examples(args.data_jsonl, num_examples=args.pool_size, seed=args.seed)

    results = run_dialogue_timing_experiment(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        offset=offset,
        device=args.device,
        num_samples=args.num_samples,
        generation_params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "do_sample": True,
        },
        verbose=True,
        out_dir=args.out_dir,
        name=args.name,
        extra_metadata={"checkpoint": args.checkpoint, "data_jsonl": args.data_jsonl, "seed": args.seed},
    )
    print_summary(results["comparison"])


if __name__ == "__main__":
    main()
