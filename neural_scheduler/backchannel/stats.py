"""Statistics of the labelled backchannel dataset.

    python -m neural_scheduler.backchannel.stats \
        --input_dir /path/to/backchannel_v2 --output bc_stats.json

Reads the ``*_interactions.jsonl`` files written by
:mod:`neural_scheduler.backchannel.labeling.extract` and reports:

1. backchannels per 10-second window;
2. class distribution, overall and per speaker;
3. intervals between consecutive backchannels of the same speaker;
4. backchannel durations, overall and per class;
5. backchannels per turn id;
6. duration of the speech a backchannel accompanies and its relative position in it;
7. per-interaction counts, rates per minute and durations.
"""

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .data.prep import load_all_interactions


# ============================================================================
# Stat 1: BC density per 10-second window
# ============================================================================

def compute_bc_density(interactions: List[Dict], window_sec: float = 10.0) -> Dict:
    """Count BCs per fixed-duration window across all interactions."""
    bc_counts_per_window = []

    for interaction in interactions:
        turns = interaction['turns']
        if not turns:
            continue

        rec_start = min(t['start'] for t in turns)
        rec_end = max(t['end'] for t in turns)

        bc_times = [t['start'] for t in turns if t['label'].startswith('backchannel-')]

        # Slide windows
        window_start = rec_start
        while window_start + window_sec <= rec_end:
            count = sum(1 for bt in bc_times if window_start <= bt < window_start + window_sec)
            bc_counts_per_window.append(count)
            window_start += window_sec

    arr = np.array(bc_counts_per_window)
    hist, bin_edges = np.histogram(arr, bins=range(0, int(arr.max()) + 3))

    return {
        'window_sec': window_sec,
        'mean': float(arr.mean()) if len(arr) > 0 else 0,
        'std': float(arr.std()) if len(arr) > 0 else 0,
        'median': float(np.median(arr)) if len(arr) > 0 else 0,
        'max': int(arr.max()) if len(arr) > 0 else 0,
        'num_windows': len(arr),
        'histogram': {
            'bin_edges': [int(b) for b in bin_edges],
            'counts': [int(c) for c in hist],
        },
        'percentiles': {
            'p25': float(np.percentile(arr, 25)) if len(arr) > 0 else 0,
            'p50': float(np.percentile(arr, 50)) if len(arr) > 0 else 0,
            'p75': float(np.percentile(arr, 75)) if len(arr) > 0 else 0,
            'p90': float(np.percentile(arr, 90)) if len(arr) > 0 else 0,
            'p95': float(np.percentile(arr, 95)) if len(arr) > 0 else 0,
        },
    }


# ============================================================================
# Stat 2: BC class distribution
# ============================================================================

def compute_class_distribution(interactions: List[Dict]) -> Dict:
    """Distribution of backchannel classes overall and per speaker."""
    overall = Counter()
    per_speaker = {0: Counter(), 1: Counter()}

    for interaction in interactions:
        for turn in interaction['turns']:
            label = turn['label']
            if label.startswith('backchannel-'):
                bc_class = label.split('-', 1)[1]
                overall[bc_class] += 1
                per_speaker[turn['speaker_id']][bc_class] += 1

    total = sum(overall.values())
    speech_count = sum(
        1 for inter in interactions for t in inter['turns']
        if t['label'] == 'Speech'
    )

    return {
        'total_backchannels': total,
        'total_speech_segments': speech_count,
        'bc_ratio': total / (total + speech_count) if (total + speech_count) > 0 else 0,
        'overall': {k: v for k, v in sorted(overall.items(), key=lambda x: -x[1])},
        'overall_pct': {
            k: round(v / total * 100, 2) for k, v in sorted(overall.items(), key=lambda x: -x[1])
        } if total > 0 else {},
        'speaker_0': dict(per_speaker[0]),
        'speaker_1': dict(per_speaker[1]),
    }


# ============================================================================
# Stat 3: Inter-BC interval (consecutive BCs from same speaker)
# ============================================================================

def compute_inter_bc_intervals(interactions: List[Dict]) -> Dict:
    """Time between consecutive backchannels from the same speaker."""
    intervals = {0: [], 1: []}

    for interaction in interactions:
        bcs_by_speaker = defaultdict(list)
        for turn in interaction['turns']:
            if turn['label'].startswith('backchannel-'):
                bcs_by_speaker[turn['speaker_id']].append(turn['start'])

        for spk, times in bcs_by_speaker.items():
            times_sorted = sorted(times)
            for i in range(1, len(times_sorted)):
                intervals[spk].append(times_sorted[i] - times_sorted[i - 1])

    all_intervals = intervals[0] + intervals[1]
    def summarize(values):
        if len(values) == 0:
            return {'count': 0}
        a = np.array(values)
        hist, edges = np.histogram(a, bins=50)
        return {
            'count': len(a),
            'mean': float(a.mean()),
            'std': float(a.std()),
            'median': float(np.median(a)),
            'min': float(a.min()),
            'max': float(a.max()),
            'percentiles': {
                'p10': float(np.percentile(a, 10)),
                'p25': float(np.percentile(a, 25)),
                'p50': float(np.percentile(a, 50)),
                'p75': float(np.percentile(a, 75)),
                'p90': float(np.percentile(a, 90)),
            },
            'histogram': {
                'bin_edges': [round(float(e), 3) for e in edges],
                'counts': [int(c) for c in hist],
            },
        }

    return {
        'all': summarize(all_intervals),
        'speaker_0': summarize(intervals[0]),
        'speaker_1': summarize(intervals[1]),
    }


# ============================================================================
# Stat 4: BC duration distribution
# ============================================================================

def compute_bc_durations(interactions: List[Dict]) -> Dict:
    """Distribution of backchannel durations."""
    durations_by_class = defaultdict(list)
    all_durations = []

    for interaction in interactions:
        for turn in interaction['turns']:
            if turn['label'].startswith('backchannel-'):
                dur = turn['end'] - turn['start']
                bc_class = turn['label'].split('-', 1)[1]
                durations_by_class[bc_class].append(dur)
                all_durations.append(dur)

    def summarize(values):
        if not values:
            return {'count': 0}
        a = np.array(values)
        return {
            'count': len(a),
            'mean': round(float(a.mean()), 4),
            'std': round(float(a.std()), 4),
            'median': round(float(np.median(a)), 4),
            'min': round(float(a.min()), 4),
            'max': round(float(a.max()), 4),
        }

    result = {'all': summarize(all_durations)}
    for cls, durs in sorted(durations_by_class.items()):
        result[cls] = summarize(durs)
    return result


# ============================================================================
# Stat 5: BCs per turn ID
# ============================================================================

def compute_bcs_per_turn(interactions: List[Dict]) -> Dict:
    """How many backchannels share each turn ID."""
    bc_counts_per_turn = []

    for interaction in interactions:
        turn_groups = defaultdict(list)
        for t in interaction['turns']:
            turn_groups[t['turn_id']].append(t)

        for turn_id, segments in turn_groups.items():
            n_bc = sum(1 for s in segments if s['label'].startswith('backchannel-'))
            if n_bc > 0:
                bc_counts_per_turn.append(n_bc)

    arr = np.array(bc_counts_per_turn) if bc_counts_per_turn else np.array([0])
    hist = Counter(bc_counts_per_turn)

    return {
        'mean': round(float(arr.mean()), 3),
        'max': int(arr.max()),
        'distribution': {str(k): v for k, v in sorted(hist.items())},
        'total_turns_with_bc': len(bc_counts_per_turn),
    }


# ============================================================================
# Stat 6: Context duration (how long is the speech the BC overlaps)
# ============================================================================

def compute_bc_context_stats(interactions: List[Dict]) -> Dict:
    """For each BC, measure the concurrent main-speaker segment duration."""
    context_durations = []
    bc_relative_positions = []  # where in the main speech does the BC fall (0=start, 1=end)

    for interaction in interactions:
        turn_groups = defaultdict(list)
        for t in interaction['turns']:
            turn_groups[t['turn_id']].append(t)

        for turn_id, segments in turn_groups.items():
            bcs = [s for s in segments if s['label'].startswith('backchannel-')]
            speech = [s for s in segments if s['label'] == 'Speech']

            if not bcs or not speech:
                continue

            # Use the full span of speech segments as context
            speech_start = min(s['start'] for s in speech)
            speech_end = max(s['end'] for s in speech)
            speech_dur = speech_end - speech_start

            for bc in bcs:
                context_durations.append(speech_dur)
                if speech_dur > 0:
                    rel_pos = (bc['start'] - speech_start) / speech_dur
                    bc_relative_positions.append(max(0.0, min(1.0, rel_pos)))

    def summarize(values):
        if not values:
            return {'count': 0}
        a = np.array(values)
        hist, edges = np.histogram(a, bins=50)
        return {
            'count': len(a),
            'mean': round(float(a.mean()), 3),
            'std': round(float(a.std()), 3),
            'median': round(float(np.median(a)), 3),
            'histogram': {
                'bin_edges': [round(float(e), 3) for e in edges],
                'counts': [int(c) for c in hist],
            },
        }

    return {
        'context_duration': summarize(context_durations),
        'relative_position_in_speech': summarize(bc_relative_positions),
    }


# ============================================================================
# Stat 7: Interaction-level summary
# ============================================================================

def compute_interaction_level_stats(interactions: List[Dict]) -> Dict:
    """Per-interaction aggregate stats."""
    bc_counts = []
    bc_rates = []  # BCs per minute
    durations = []

    for interaction in interactions:
        turns = interaction['turns']
        if not turns:
            continue
        n_bc = interaction['num_backchannels']
        rec_start = min(t['start'] for t in turns)
        rec_end = max(t['end'] for t in turns)
        dur_min = (rec_end - rec_start) / 60.0

        bc_counts.append(n_bc)
        durations.append(rec_end - rec_start)
        if dur_min > 0:
            bc_rates.append(n_bc / dur_min)

    return {
        'num_interactions': len(interactions),
        'bcs_per_interaction': {
            'mean': round(float(np.mean(bc_counts)), 2),
            'std': round(float(np.std(bc_counts)), 2),
            'median': float(np.median(bc_counts)),
            'min': int(np.min(bc_counts)),
            'max': int(np.max(bc_counts)),
        },
        'bcs_per_minute': {
            'mean': round(float(np.mean(bc_rates)), 2) if bc_rates else 0,
            'std': round(float(np.std(bc_rates)), 2) if bc_rates else 0,
            'median': round(float(np.median(bc_rates)), 2) if bc_rates else 0,
        },
        'interaction_duration_sec': {
            'mean': round(float(np.mean(durations)), 1),
            'median': round(float(np.median(durations)), 1),
        },
    }



def compute_all_stats(interactions: List[Dict]) -> Dict:
    """All statistics for a list of interaction records."""
    return {
        'bc_density': compute_bc_density(interactions, window_sec=10.0),
        'class_distribution': compute_class_distribution(interactions),
        'inter_bc_intervals': compute_inter_bc_intervals(interactions),
        'bc_durations': compute_bc_durations(interactions),
        'bcs_per_turn': compute_bcs_per_turn(interactions),
        'bc_context': compute_bc_context_stats(interactions),
        'interaction_level': compute_interaction_level_stats(interactions),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Analyze backchannel statistics')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing shard JSONL files')
    parser.add_argument('--output', type=str, default='bc_stats.json',
                        help='Output JSON file for statistics')
    args = parser.parse_args(argv)

    interactions = load_all_interactions(args.input_dir)
    print(f"Loaded {len(interactions)} interactions")
    if not interactions:
        print("No interactions found!")
        return

    print("Computing statistics...")
    stats = compute_all_stats(interactions)

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats written to: {args.output}")

    # Print summary to console
    cd = stats['class_distribution']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Interactions:        {stats['interaction_level']['num_interactions']}")
    print(f"Total BCs:           {cd['total_backchannels']}")
    print(f"Total Speech segs:   {cd['total_speech_segments']}")
    print(f"BC ratio:            {cd['bc_ratio']:.3f}")
    print(f"BCs/min (mean):      {stats['interaction_level']['bcs_per_minute']['mean']}")
    print("\nClass distribution:")
    for cls, pct in cd.get('overall_pct', {}).items():
        print(f"  {cls:20s}: {cd['overall'][cls]:6d} ({pct:.1f}%)")
    print(f"\nBC density per {stats['bc_density']['window_sec']}s window:")
    d = stats['bc_density']
    print(f"  mean={d['mean']:.2f}, median={d['median']:.1f}, "
          f"p90={d['percentiles']['p90']:.1f}, max={d['max']}")
    print("\nInter-BC interval (all, seconds):")
    ibi = stats['inter_bc_intervals']['all']
    if ibi['count'] > 0:
        print(f"  mean={ibi['mean']:.2f}, median={ibi['median']:.2f}, "
              f"p25={ibi['percentiles']['p25']:.2f}, p75={ibi['percentiles']['p75']:.2f}")
    print("\nBCs per turn (turns that have ≥1 BC):")
    bpt = stats['bcs_per_turn']
    print(f"  mean={bpt['mean']}, max={bpt['max']}")
    print(f"  distribution: {bpt['distribution']}")


if __name__ == '__main__':
    main()
