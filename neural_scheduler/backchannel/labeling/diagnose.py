"""Sanity checks for one shard written by :mod:`.extract`.

    python -m neural_scheduler.backchannel.labeling.diagnose \
        --interactions_path /path/to/shard0of8_interactions.jsonl

Reports record integrity (required fields, count consistency), the label
distribution, whether shared turn ids correspond to overlapping segments,
duration / word-count statistics and transcript samples per backchannel class,
and a summary of the failures file (auto-detected beside the interactions file).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

REQUIRED_INTERACTION_FIELDS = {
    'interaction_id', 'identifier_left', 'identifier_right',
    'wav_left', 'wav_right', 'transcript_left', 'transcript_right',
    'split', 'dataset_type', 'sr',
    'num_turns', 'num_sentences', 'num_backchannels', 'turns',
}

REQUIRED_TURN_FIELDS = {'text', 'start', 'end', 'speaker_id', 'turn_id', 'label'}

SPEECH_LABEL = 'Speech'
BC_PREFIX = 'backchannel-'
BC_CLASSES = ['Neutral', 'Positive', 'Emotional', 'Anticipation', 'Confirmation', 'Repetition']
ALL_EXPECTED_LABELS = {SPEECH_LABEL} | {f'{BC_PREFIX}{c}' for c in BC_CLASSES}

SEP = '=' * 70
SEP_MINOR = '-' * 70

# ============================================================================
# Loading
# ============================================================================

def load_interactions(path: str) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """
    Load all interactions from a JSONL file.

    Returns:
        (records, parse_errors) where parse_errors is a list of
        (line_number, error_message) for lines that failed to parse.
    """
    records = []
    parse_errors = []
    with open(path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append((lineno, str(e)))
    return records, parse_errors


def load_failures(path: str) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """Load failure records from a JSONL file."""
    return load_interactions(path)  # same format


# ============================================================================
# Check 1: Record integrity
# ============================================================================

def check_integrity(records: List[Dict]) -> Dict:
    """
    Verify all required fields are present and internal counts are consistent.

    Returns a dict with:
        missing_fields      : list of (interaction_id, missing_field_set)
        count_mismatches    : list of (interaction_id, field, stored, actual)
        unexpected_labels   : list of (interaction_id, turn_idx, label)
        duplicate_ids       : list of interaction_ids that appear more than once
        empty_turns         : list of interaction_ids with 0 turns
    """
    results = {
        'missing_fields': [],
        'count_mismatches': [],
        'unexpected_labels': [],
        'duplicate_ids': [],
        'empty_turns': [],
    }

    seen_ids = Counter()
    for rec in records:
        iid = rec.get('interaction_id', '?')
        seen_ids[iid] += 1

        # Missing top-level fields
        missing = REQUIRED_INTERACTION_FIELDS - set(rec.keys())
        if missing:
            results['missing_fields'].append((iid, missing))
            continue

        turns = rec['turns']

        # Empty turns
        if len(turns) == 0:
            results['empty_turns'].append(iid)

        # Count consistency
        actual_sentences = len(turns)
        actual_turns = len(set(t['turn_id'] for t in turns))
        actual_bcs = sum(1 for t in turns if t['label'] != SPEECH_LABEL)

        if rec['num_sentences'] != actual_sentences:
            results['count_mismatches'].append(
                (iid, 'num_sentences', rec['num_sentences'], actual_sentences)
            )
        if rec['num_turns'] != actual_turns:
            results['count_mismatches'].append(
                (iid, 'num_turns', rec['num_turns'], actual_turns)
            )
        if rec['num_backchannels'] != actual_bcs:
            results['count_mismatches'].append(
                (iid, 'num_backchannels', rec['num_backchannels'], actual_bcs)
            )

        # Per-turn field checks and label validity
        for t_idx, turn in enumerate(turns):
            missing_turn = REQUIRED_TURN_FIELDS - set(turn.keys())
            if missing_turn:
                results['missing_fields'].append((iid, f'turn[{t_idx}] missing {missing_turn}'))
            label = turn.get('label', '')
            if label not in ALL_EXPECTED_LABELS:
                results['unexpected_labels'].append((iid, t_idx, label))

    # Duplicate interaction IDs
    results['duplicate_ids'] = [iid for iid, cnt in seen_ids.items() if cnt > 1]

    return results


def print_integrity(results: Dict, total: int):
    print(SEP)
    print("CHECK 1: RECORD INTEGRITY")
    print(SEP)
    print(f"  Total records loaded:      {total}")

    issues = sum([
        len(results['missing_fields']),
        len(results['count_mismatches']),
        len(results['unexpected_labels']),
        len(results['duplicate_ids']),
        len(results['empty_turns']),
    ])
    status = '✅ PASSED' if issues == 0 else f'❌ {issues} issue(s) found'
    print(f"  Overall status:            {status}")

    if results['missing_fields']:
        print(f"\n  Missing fields ({len(results['missing_fields'])} records):")
        for iid, missing in results['missing_fields'][:5]:
            print(f"    interaction {iid}: {missing}")
        if len(results['missing_fields']) > 5:
            print(f"    ... and {len(results['missing_fields']) - 5} more")

    if results['count_mismatches']:
        print(f"\n  Count mismatches ({len(results['count_mismatches'])}):")
        for iid, field, stored, actual in results['count_mismatches'][:5]:
            print(f"    interaction {iid}: {field} stored={stored} actual={actual}")

    if results['unexpected_labels']:
        print(f"\n  Unexpected labels ({len(results['unexpected_labels'])}):")
        for iid, t_idx, label in results['unexpected_labels'][:5]:
            print(f"    interaction {iid} turn[{t_idx}]: '{label}'")

    if results['duplicate_ids']:
        print(f"\n  Duplicate interaction IDs: {results['duplicate_ids'][:10]}")

    if results['empty_turns']:
        print(f"\n  Interactions with 0 turns: {len(results['empty_turns'])}")
        print(f"    IDs: {results['empty_turns'][:10]}")


# ============================================================================
# Check 2: Label distribution
# ============================================================================

def check_label_distribution(records: List[Dict]) -> Dict:
    label_counts = Counter()
    bc_per_interaction = []
    zero_bc_interactions = 0
    duration_per_label = defaultdict(list)

    for rec in records:
        bc_count = 0
        for turn in rec['turns']:
            label = turn['label']
            label_counts[label] += 1
            dur = turn['end'] - turn['start']
            duration_per_label[label].append(dur)
            if label != SPEECH_LABEL:
                bc_count += 1
        bc_per_interaction.append(bc_count)
        if bc_count == 0:
            zero_bc_interactions += 1

    return {
        'label_counts': label_counts,
        'bc_per_interaction': bc_per_interaction,
        'zero_bc_interactions': zero_bc_interactions,
        'duration_per_label': duration_per_label,
    }


def print_label_distribution(results: Dict, total_interactions: int):
    import statistics
    print(f"\n{SEP}")
    print("CHECK 2: LABEL DISTRIBUTION")
    print(SEP)

    label_counts = results['label_counts']
    total_turns = sum(label_counts.values())
    duration_per_label = results['duration_per_label']

    print(f"\n  {'Label':<28} {'Count':>8} {'%':>8} {'Avg dur':>10} {'Med dur':>10}")
    print(f"  {SEP_MINOR}")

    # Speech first, then BCs sorted by count
    ordered = [SPEECH_LABEL] + sorted(
        [l for l in label_counts if l != SPEECH_LABEL],
        key=lambda x: -label_counts[x]
    )
    for label in ordered:
        if label not in label_counts:
            continue
        count = label_counts[label]
        pct = 100 * count / total_turns if total_turns > 0 else 0
        durs = duration_per_label[label]
        avg_dur = sum(durs) / len(durs) if durs else 0
        med_dur = statistics.median(durs) if durs else 0
        print(f"  {label:<28} {count:>8} {pct:>7.1f}% {avg_dur:>9.2f}s {med_dur:>9.2f}s")

    print(f"\n  Total turns: {total_turns}")
    print(f"  Total interactions: {total_interactions}")

    bc_per = results['bc_per_interaction']
    if bc_per:
        avg_bc = sum(bc_per) / len(bc_per)
        print(f"  Avg BCs per interaction: {avg_bc:.2f}")
        print(f"  Interactions with 0 BCs: {results['zero_bc_interactions']} "
              f"({100*results['zero_bc_interactions']/total_interactions:.1f}%)")
        print(f"  Max BCs in one interaction: {max(bc_per)}")

        # Distribution of BC counts
        bc_dist = Counter(bc_per)
        print("\n  BC count distribution (top 10):")
        for count, freq in sorted(bc_dist.items())[:10]:
            bar = '█' * min(30, freq)
            print(f"    {count:>3} BCs: {freq:>5} interactions  {bar}")


# ============================================================================
# Check 3: Turn ID sanity
# ============================================================================

def check_turn_id_sanity(records: List[Dict], sample_size: int = 20) -> Dict:
    """
    Verify turn ID assignment correctness.

    EXPECTED (not a violation):
      - Same-speaker consecutive segments share a turn ID: this is the result
        of merge_adjacent_same_speaker() and is correct behaviour.
      - Cross-speaker segments share a turn ID because one overlaps >= 50%
        with the other: this is the backchannel-during-speech case.

    TRUE VIOLATION (indicates a bug):
      - Two same-speaker segments share a turn ID but are NOT consecutive in
        the time-sorted order — i.e. a different speaker's segment appears
        between them in time. This would mean the overlap rule incorrectly
        grouped non-adjacent same-speaker segments.
    """
    violations = []
    bc_shared_turns = []
    sampled = 0

    for rec in records:
        turns = rec['turns']  # already sorted by start time
        iid = rec['interaction_id']

        by_turn = defaultdict(list)
        for t in turns:
            by_turn[t['turn_id']].append(t)

        for tid, segs in by_turn.items():
            if len(segs) < 2:
                continue

            labels = [s['label'] for s in segs]
            has_speech = SPEECH_LABEL in labels
            has_bc = any(l.startswith(BC_PREFIX) for l in labels)

            # Collect BC+Speech shared-turn examples (good — overlap rule worked)
            if has_speech and has_bc and sampled < sample_size:
                bc_seg = next(s for s in segs if s['label'].startswith(BC_PREFIX))
                sp_seg = next(s for s in segs if s['label'] == SPEECH_LABEL)
                bc_shared_turns.append({
                    'interaction_id': iid,
                    'turn_id': tid,
                    'bc': {
                        'text': bc_seg['text'],
                        'start': bc_seg['start'],
                        'end': bc_seg['end'],
                        'label': bc_seg['label'],
                    },
                    'speech': {
                        'text': sp_seg['text'][:60],
                        'start': sp_seg['start'],
                        'end': sp_seg['end'],
                    },
                })
                sampled += 1

            # True violation: same-speaker non-consecutive segments sharing a turn ID.
            # Same-speaker consecutive segments sharing a turn ID is CORRECT
            # (result of merging) and should NOT be flagged.
            speakers_in_turn = set(s['speaker_id'] for s in segs)
            if len(speakers_in_turn) == 1:
                speaker_id = segs[0]['speaker_id']

                # Find positions of these segments in the full time-sorted turn list
                seg_keys = {(s['start'], s['end']) for s in segs}
                positions = [
                    i for i, t in enumerate(turns)
                    if (t['start'], t['end']) in seg_keys
                    and t['speaker_id'] == speaker_id
                ]

                if not positions:
                    continue

                # Check whether any other speaker's segment falls between them
                min_pos, max_pos = min(positions), max(positions)
                interleaved = turns[min_pos:max_pos + 1]
                intruders = [
                    t for t in interleaved
                    if t['speaker_id'] != speaker_id
                ]

                if intruders:
                    violations.append({
                        'interaction_id': iid,
                        'turn_id': tid,
                        'segments': [
                            {
                                'text': s['text'][:40],
                                'start': s['start'],
                                'end': s['end'],
                                'speaker_id': s['speaker_id'],
                            }
                            for s in segs
                        ],
                        'intruding_segments': [
                            {
                                'text': t['text'][:40],
                                'start': t['start'],
                                'end': t['end'],
                                'speaker_id': t['speaker_id'],
                            }
                            for t in intruders[:2]
                        ],
                    })

    return {
        'violations': violations,
        'bc_shared_turn_examples': bc_shared_turns,
    }


def print_turn_id_sanity(results: Dict):
    print(f"\n{SEP}")
    print("CHECK 3: TURN ID SANITY")
    print(SEP)

    violations = results['violations']
    examples = results['bc_shared_turn_examples']

    if violations:
        print(f"\n  ❌ {len(violations)} non-consecutive same-speaker turn ID violations:")
        for v in violations[:3]:
            print(f"    interaction {v['interaction_id']}, turn {v['turn_id']}:")
            print("      same-speaker segments:")
            for seg in v['segments']:
                print(f"        spk{seg['speaker_id']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}")
            print("      intruding other-speaker segment(s) between them:")
            for seg in v.get('intruding_segments', []):
                print(f"        spk{seg['speaker_id']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}")
    else:
        print("\n  ✅ No turn ID violations found")

    if examples:
        print(f"\n  ✅ {len(examples)} BC+Speech shared-turn examples (expected behaviour):")
        for ex in examples[:3]:
            bc = ex['bc']
            sp = ex['speech']
            print(f"\n    interaction {ex['interaction_id']}, turn {ex['turn_id']}:")
            print(f"      Speech [{sp['start']:.2f}-{sp['end']:.2f}]: \"{sp['text']}\"")
            print(f"      {bc['label']} [{bc['start']:.2f}-{bc['end']:.2f}]: \"{bc['text']}\"")
    else:
        print("\n  ⚠️  No BC+Speech shared turns found in sampled interactions")
        print("     (This may indicate the BC lookup is not matching — see Check 4)")


# ============================================================================
# Check 4: BC candidate quality
# ============================================================================

def check_bc_quality(records: List[Dict], samples_per_class: int = 3) -> Dict:
    """
    For each confirmed BC, check:
      - Duration is < 2.0s (backchannels should be short)
      - Collect transcript samples per class
    """
    duration_violations = []  # BCs that are suspiciously long
    class_samples = defaultdict(list)
    class_durations = defaultdict(list)

    for rec in records:
        for turn in rec['turns']:
            label = turn['label']
            if not label.startswith(BC_PREFIX):
                continue
            bc_class = label[len(BC_PREFIX):]
            dur = turn['end'] - turn['start']
            class_durations[bc_class].append(dur)

            if dur > 2.0:
                duration_violations.append({
                    'interaction_id': rec['interaction_id'],
                    'label': label,
                    'duration': dur,
                    'text': turn['text'],
                })

            if len(class_samples[bc_class]) < samples_per_class:
                class_samples[bc_class].append({
                    'text': turn['text'],
                    'start': turn['start'],
                    'end': turn['end'],
                    'duration': dur,
                    'speaker_id': turn['speaker_id'],
                    'interaction_id': rec['interaction_id'],
                })

    return {
        'duration_violations': duration_violations,
        'class_samples': class_samples,
        'class_durations': class_durations,
    }


def print_bc_quality(results: Dict):
    print(f"\n{SEP}")
    print("CHECK 4: BC CANDIDATE QUALITY")
    print(SEP)

    violations = results['duration_violations']
    class_samples = results['class_samples']
    class_durations = results['class_durations']

    if not class_samples:
        print("\n  ⚠️  No backchannel turns found in the output.")
        print("     Possible causes:")
        print("     - BC lookup keys (speaker_id, start, end) not matching transcript sentences")
        print("     - Qwen classified all candidates as NotBackchannel")
        print("     - No backchannel candidates passed the rule-based pre-filter")
        print("     - Only a small number of interactions were processed (check --max_interactions)")
        return

    # Duration stats per class
    print(f"\n  {'Class':<20} {'Count':>7} {'Avg dur':>10} {'Min':>8} {'Max':>8}")
    print(f"  {SEP_MINOR}")
    for bc_class in BC_CLASSES:
        durs = class_durations.get(bc_class, [])
        if not durs:
            print(f"  {bc_class:<20} {'0':>7}")
            continue
        avg = sum(durs) / len(durs)
        print(f"  {bc_class:<20} {len(durs):>7} {avg:>9.2f}s {min(durs):>7.2f}s {max(durs):>7.2f}s")

    if violations:
        print(f"\n  ⚠️  {len(violations)} BCs with duration > 2.0s (may be mislabelled):")
        for v in violations[:5]:
            print(f"    [{v['label']}] {v['duration']:.2f}s: \"{v['text']}\"")
    else:
        print("\n  ✅ All BCs have duration <= 2.0s")

    # Transcript samples per class
    print("\n  Transcript samples per class:")
    for bc_class in BC_CLASSES:
        samples = class_samples.get(bc_class, [])
        if not samples:
            print(f"\n  [{bc_class}] — no examples found")
            continue
        print(f"\n  [{bc_class}] ({len(class_durations.get(bc_class, []))} total)")
        for s in samples:
            print(f"    spk{s['speaker_id']} [{s['start']:.2f}-{s['end']:.2f}] "
                  f"({s['duration']:.2f}s): \"{s['text']}\"")


# ============================================================================
# Check 5: Failures summary
# ============================================================================

def check_failures(path: str) -> Dict:
    if not os.path.exists(path):
        return {'found': False}

    records, parse_errors = load_failures(path)
    error_messages = Counter()
    for rec in records:
        # Take just the first line of the traceback as the key
        err = rec.get('error', 'unknown')
        error_messages[err.split('\n')[0][:120]] += 1

    return {
        'found': True,
        'total_failures': len(records),
        'parse_errors': parse_errors,
        'error_messages': error_messages,
        'samples': records[:3],
    }


def print_failures(results: Dict):
    print(f"\n{SEP}")
    print("CHECK 5: FAILURES SUMMARY")
    print(SEP)

    if not results['found']:
        print("\n  No failures file found (or path not provided)")
        return

    total = results['total_failures']
    print(f"\n  Total failed interactions: {total}")

    if total == 0:
        print("  ✅ No failures recorded")
        return

    print("\n  Most common errors:")
    for msg, count in results['error_messages'].most_common(5):
        print(f"    [{count}x] {msg}")

    print("\n  Sample failure records:")
    for rec in results['samples']:
        print(f"    interaction {rec.get('interaction_id', '?')}: "
              f"{rec.get('identifier_left', '?')} / {rec.get('identifier_right', '?')}")
        print(f"      error: {str(rec.get('error', ''))[:100]}")


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Diagnose a shard of labelled interactions.')
    parser.add_argument('--interactions_path', type=str, required=True,
                        help='Path to a *_interactions.jsonl shard file')
    parser.add_argument('--failures_path', type=str, default=None,
                        help='Path to the *_failures.jsonl file (auto-detected if not set)')
    parser.add_argument('--samples_per_class', type=int, default=3,
                        help='Transcript samples to show per backchannel class')
    parser.add_argument('--turn_sanity_samples', type=int, default=20,
                        help='Interactions to sample for the turn-id sanity check')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    failures_path = args.failures_path
    if failures_path is None:
        failures_path = args.interactions_path.replace('_interactions.jsonl', '_failures.jsonl')

    print(SEP)
    print("BACKCHANNEL LABELLING OUTPUT DIAGNOSTICS")
    print(SEP)
    print(f"  Interactions file: {args.interactions_path}")
    print(f"  Failures file:     {failures_path}")

    print("\nLoading interactions...")
    records, parse_errors = load_interactions(args.interactions_path)
    print(f"  Loaded {len(records)} records, {len(parse_errors)} parse errors")
    if parse_errors:
        print("  Parse errors:")
        for lineno, err in parse_errors[:5]:
            print(f"    line {lineno}: {err}")
    if not records:
        print("\n  No records to diagnose. Exiting.")
        sys.exit(1)

    print_integrity(check_integrity(records), len(records))
    print_label_distribution(check_label_distribution(records), len(records))
    print_turn_id_sanity(check_turn_id_sanity(records, sample_size=args.turn_sanity_samples))
    print_bc_quality(check_bc_quality(records, samples_per_class=args.samples_per_class))
    print_failures(check_failures(failures_path))

    print(f"\n{SEP}")
    print("DIAGNOSTICS COMPLETE")
    print(SEP)


if __name__ == '__main__':
    main()
