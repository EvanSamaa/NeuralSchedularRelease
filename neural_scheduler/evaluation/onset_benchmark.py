"""Onset-shift benchmark: is the ground-truth onset the model's NLL minimum?

Selected speech segments of a ground-truth dialogue are moved earlier and later
by up to ``shift_range`` steps with everything else fixed, and each variant is
scored. The output shows, per segment and aggregated, how NLL in a window
around the segment changes with the shift, split into the onset token and its
context. Results are written as interactive HTML (Chart.js from a CDN).

Run it through ``python -m neural_scheduler.evaluation.nll_diagnostics --onset_benchmark``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..constants import SILENCE_CODE
from ..vocab import audio_token
from .common import compute_per_token_nll

AGGREGATE_TEMPLATE = Path(__file__).parent / "templates" / "aggregate_benchmark.html"


# ==============================================================
# Data structures
# ==============================================================

@dataclass
class SegmentInfo:
    """Describes a segment within a dialogue for benchmarking."""
    channel: str                    # 'A' or 'B'
    onset_step: int                 # ground-truth onset step (pair index)
    offset_step: int                # step where segment ends
    length: int                     # number of speech tokens
    context_type: str               # 'turn_taking', 'backchannel', 'post_silence', 'overlap_start'
    other_channel_state: str        # what the other channel is doing at onset: 'silent', 'speaking', 'ending'
    silence_before: int             # how many silence steps preceded this onset


@dataclass
class ShiftResult:
    """NLL results for one onset shift of one segment."""
    shift: int                      # shift in steps (negative=earlier, 0=ground truth, positive=later)
    onset_step: int                 # actual onset step after shift
    total_nll: float                # total NLL over scoring window
    onset_token_nll: float          # NLL of just the first speech token
    context_nll: float              # NLL of non-onset tokens in the window
    silence_nll_before: float       # NLL of silence tokens before onset in window
    speech_nll_after: float         # NLL of speech tokens after onset in window
    per_token_nlls: List[float]     # full per-token NLL in the scoring window


@dataclass
class SegmentBenchmark:
    """Full benchmark results for one segment."""
    segment: SegmentInfo
    shifts: List[ShiftResult]
    seq_label: str
    scoring_window: Tuple[int, int]  # (start_step, end_step) of the scoring window


# ==============================================================
# Identify interesting segments in ground truth
# ==============================================================

def find_benchmark_segments(
    dialogue_ids: List[int],
    silence_token_id: int,
    min_segment_length: int = 5,
    max_segments: int = 10,
) -> List[SegmentInfo]:
    """
    Scan a ground-truth dialogue and identify interesting segments to benchmark.
    Tries to find a mix of turn-taking, backchannel, and post-silence segments.
    """
    num_pairs = len(dialogue_ids) // 2
    segments = []

    # Track per-channel state
    a_speaking = False
    b_speaking = False
    a_silence_run = 0
    b_silence_run = 0
    a_speech_run = 0
    b_speech_run = 0

    for step in range(num_pairs):
        tok_a = dialogue_ids[step * 2]
        tok_b = dialogue_ids[step * 2 + 1]

        is_sil_a = (tok_a == silence_token_id)
        is_sil_b = (tok_b == silence_token_id)

        # Detect onsets
        a_onset = (not is_sil_a) and (not a_speaking)
        b_onset = (not is_sil_b) and (not b_speaking)

        if a_onset:
            # Find segment length
            seg_len = 0
            for future_step in range(step, num_pairs):
                future_tok = dialogue_ids[future_step * 2]
                if future_tok == silence_token_id:
                    break
                seg_len += 1

            if seg_len >= min_segment_length:
                # Classify context
                if b_speaking:
                    if b_speech_run <= 10:
                        ctx = 'overlap_start'
                    else:
                        ctx = 'backchannel'
                    other_state = 'speaking'
                elif b_silence_run > 0 and b_silence_run <= 5:
                    ctx = 'turn_taking'
                    other_state = 'ending'
                else:
                    ctx = 'post_silence'
                    other_state = 'silent'

                segments.append(SegmentInfo(
                    channel='A',
                    onset_step=step,
                    offset_step=step + seg_len,
                    length=seg_len,
                    context_type=ctx,
                    other_channel_state=other_state,
                    silence_before=a_silence_run,
                ))

        if b_onset:
            seg_len = 0
            for future_step in range(step, num_pairs):
                future_tok = dialogue_ids[future_step * 2 + 1]
                if future_tok == silence_token_id:
                    break
                seg_len += 1

            if seg_len >= min_segment_length:
                if a_speaking:
                    if a_speech_run <= 10:
                        ctx = 'overlap_start'
                    else:
                        ctx = 'backchannel'
                    other_state = 'speaking'
                elif a_silence_run > 0 and a_silence_run <= 5:
                    ctx = 'turn_taking'
                    other_state = 'ending'
                else:
                    ctx = 'post_silence'
                    other_state = 'silent'

                segments.append(SegmentInfo(
                    channel='B',
                    onset_step=step,
                    offset_step=step + seg_len,
                    length=seg_len,
                    context_type=ctx,
                    other_channel_state=other_state,
                    silence_before=b_silence_run,
                ))

        # Update state
        if is_sil_a:
            a_silence_run += 1
            a_speech_run = 0
            a_speaking = False
        else:
            a_speech_run += 1
            a_silence_run = 0
            a_speaking = True

        if is_sil_b:
            b_silence_run += 1
            b_speech_run = 0
            b_speaking = False
        else:
            b_speech_run += 1
            b_silence_run = 0
            b_speaking = True

    # Try to get a diverse mix
    by_type = {}
    for seg in segments:
        key = seg.context_type
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(seg)

    selected = []
    for ctx_type, segs in by_type.items():
        # Pick up to 3 per type, preferring different channels
        a_segs = [s for s in segs if s.channel == 'A']
        b_segs = [s for s in segs if s.channel == 'B']
        picked = []
        if a_segs:
            picked.append(a_segs[len(a_segs) // 2])  # pick middle one
        if b_segs:
            picked.append(b_segs[len(b_segs) // 2])
        if len(picked) < 2 and segs:
            picked.append(segs[0])
        selected.extend(picked[:3])

    return selected[:max_segments]


# ==============================================================
# Build shifted dialogue sequences
# ==============================================================

def build_shifted_dialogue(
    dialogue_ids: List[int],
    segment: SegmentInfo,
    shift: int,
    silence_token_id: int,
) -> Optional[List[int]]:
    """
    Create a modified dialogue where the target segment's onset is shifted
    by `shift` steps. Positive = later onset, negative = earlier onset.

    The segment's speech tokens are moved as a block. The vacated positions
    become silence, and the positions now occupied by speech lose whatever
    was there before (which should be silence for valid shifts).

    Returns None if the shift would push the segment out of bounds or
    cause it to overlap with another segment on the same channel.
    """
    num_pairs = len(dialogue_ids) // 2
    ch_offset = 0 if segment.channel == 'A' else 1

    new_onset = segment.onset_step + shift
    new_offset = new_onset + segment.length

    # Bounds check
    if new_onset < 0 or new_offset > num_pairs:
        return None

    # Extract the original speech tokens for this segment
    speech_tokens = []
    for step in range(segment.onset_step, segment.offset_step):
        speech_tokens.append(dialogue_ids[step * 2 + ch_offset])

    # Check that the new positions are available (silence) on this channel,
    # excluding the original segment positions
    original_positions = set(range(segment.onset_step, segment.offset_step))
    new_positions = set(range(new_onset, new_offset))

    # Positions that are new (not in original) must currently be silence
    for step in (new_positions - original_positions):
        if dialogue_ids[step * 2 + ch_offset] != silence_token_id:
            return None  # would collide with another segment

    # Build the modified dialogue
    new_dialogue = list(dialogue_ids)

    # Clear original positions to silence
    for step in range(segment.onset_step, segment.offset_step):
        new_dialogue[step * 2 + ch_offset] = silence_token_id

    # Place speech tokens at new positions
    for i, step in enumerate(range(new_onset, new_offset)):
        new_dialogue[step * 2 + ch_offset] = speech_tokens[i]

    return new_dialogue


# ==============================================================
# Score a shifted dialogue
# ==============================================================

@torch.no_grad()
def score_shifted_dialogue(
    model,
    context_ids: List[int],
    dialogue_ids: List[int],
    segment: SegmentInfo,
    shift: int,
    silence_token_id: int,
    window_padding: int = 15,
    device: torch.device = torch.device('cuda'),
) -> Optional[ShiftResult]:
    """
    Score a shifted version of the dialogue, focusing on a window around
    the segment onset.

    The scoring window is centered on the region of interest:
      [min(original_onset, shifted_onset) - padding,
       max(original_offset, shifted_offset) + padding]

    But we run the full sequence through the model (needed for correct conditioning).
    We just report NLL decomposed within the scoring window.
    """
    shifted_dialogue = build_shifted_dialogue(
        dialogue_ids, segment, shift, silence_token_id
    )
    if shifted_dialogue is None:
        return None

    new_onset = segment.onset_step + shift
    new_offset = new_onset + segment.length
    num_pairs = len(dialogue_ids) // 2

    # Define scoring window (in pair steps)
    window_start = max(0, min(segment.onset_step, new_onset) - window_padding)
    window_end = min(num_pairs, max(segment.offset_step, new_offset) + window_padding)

    all_token_nlls = compute_per_token_nll(model, context_ids, shifted_dialogue, device)

    # Extract NLLs within the scoring window
    ch_offset = 0 if segment.channel == 'A' else 1
    window_nlls = []
    onset_nll = 0.0
    silence_nll_before = 0.0
    speech_nll_after = 0.0
    silence_count_before = 0
    speech_count_after = 0

    for step in range(window_start, window_end):
        # NLL for channel of interest
        idx = step * 2 + ch_offset
        if idx < len(all_token_nlls):
            nll_val = float(all_token_nlls[idx])
            window_nlls.append(nll_val)

            if step == new_onset:
                onset_nll = nll_val
            elif step < new_onset:
                silence_nll_before += nll_val
                silence_count_before += 1
            elif step > new_onset and step < new_offset:
                speech_nll_after += nll_val
                speech_count_after += 1

    # Also collect per-token NLLs for both channels in the window for detailed analysis
    per_token_nlls = []
    for step in range(window_start, window_end):
        for ch in [0, 1]:
            idx = step * 2 + ch
            if idx < len(all_token_nlls):
                per_token_nlls.append(float(all_token_nlls[idx]))

    total_nll = sum(window_nlls)
    context_nll = total_nll - onset_nll

    return ShiftResult(
        shift=shift,
        onset_step=new_onset,
        total_nll=total_nll,
        onset_token_nll=onset_nll,
        context_nll=context_nll,
        silence_nll_before=silence_nll_before / max(1, silence_count_before),
        speech_nll_after=speech_nll_after / max(1, speech_count_after),
        per_token_nlls=per_token_nlls,
    )


# ==============================================================
# Main benchmark runner
# ==============================================================

@torch.no_grad()
def run_onset_benchmark(
    model,
    tokenizer,
    ground_truth_sequences: List[Dict],
    shift_range: int = 20,
    max_segments_per_seq: int = 5,
    window_padding: int = 15,
    device: str = 'cuda',
    verbose: bool = True,
) -> List[SegmentBenchmark]:
    """
    Run the onset timing benchmark.

    For each ground-truth sequence, identify interesting segments,
    then shift each segment's onset by [-shift_range, +shift_range]
    steps and measure NLL.

    Args:
        model: The ChatGLM model
        tokenizer: Tokenizer with audio tokens
        ground_truth_sequences: List of dicts with 'context_ids', 'dialogue_ids', 'label'
        shift_range: Max shift in either direction (steps)
        max_segments_per_seq: Max segments to benchmark per sequence
        window_padding: Steps of context around the segment for scoring
        device: torch device
        verbose: print progress

    Returns:
        List of SegmentBenchmark results
    """
    model.eval()
    silence_token_id = tokenizer.convert_tokens_to_ids(audio_token(SILENCE_CODE))
    dev = torch.device(device)

    all_benchmarks = []

    for seq_idx, seq in enumerate(ground_truth_sequences):
        label = seq.get('label', f'sequence_{seq_idx}')
        context_ids = seq['context_ids']
        dialogue_ids = seq['dialogue_ids']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Sequence: {label}")
            print(f"  Dialogue length: {len(dialogue_ids)//2} steps")

        # Find interesting segments
        segments = find_benchmark_segments(
            dialogue_ids, silence_token_id,
            max_segments=max_segments_per_seq,
        )

        if verbose:
            print(f"  Found {len(segments)} segments to benchmark:")
            for seg in segments:
                print(f"    Ch {seg.channel} @ step {seg.onset_step}, "
                      f"len={seg.length}, type={seg.context_type}, "
                      f"other={seg.other_channel_state}, "
                      f"silence_before={seg.silence_before}")

        for seg_idx, segment in enumerate(segments):
            if verbose:
                print(f"\n  Benchmarking segment {seg_idx+1}/{len(segments)}: "
                      f"Ch {segment.channel} @ step {segment.onset_step} "
                      f"({segment.context_type})")

            shifts = []
            for shift in range(-shift_range, shift_range + 1):
                result = score_shifted_dialogue(
                    model=model,
                    context_ids=context_ids,
                    dialogue_ids=dialogue_ids,
                    segment=segment,
                    shift=shift,
                    silence_token_id=silence_token_id,
                    window_padding=window_padding,
                    device=dev,
                )
                if result is not None:
                    shifts.append(result)

            if verbose and shifts:
                # Find the best shift
                best = min(shifts, key=lambda r: r.total_nll)
                gt = next((r for r in shifts if r.shift == 0), None)
                print(f"    Shifts evaluated: {len(shifts)} "
                      f"(range [{shifts[0].shift}, {shifts[-1].shift}])")
                print(f"    Best shift: {best.shift} (NLL={best.total_nll:.4f})")
                if gt:
                    print(f"    Ground truth (shift=0): NLL={gt.total_nll:.4f}")
                    print(f"    GT onset NLL: {gt.onset_token_nll:.4f}, "
                          f"context NLL: {gt.context_nll:.4f}")

            window_start = max(0, segment.onset_step - window_padding)
            window_end = min(len(dialogue_ids)//2, segment.offset_step + window_padding)

            all_benchmarks.append(SegmentBenchmark(
                segment=segment,
                shifts=shifts,
                seq_label=label,
                scoring_window=(window_start, window_end),
            ))

    return all_benchmarks


# ==============================================================
# Visualization
# ==============================================================

def _benchmarks_to_viz_data(benchmarks: List[SegmentBenchmark]) -> list:
    viz_data = []
    for bm in benchmarks:
        if not bm.shifts:
            continue
        seg = bm.segment
        shift_data = []
        for r in bm.shifts:
            shift_data.append({
                'shift': r.shift,
                'total_nll': round(r.total_nll, 5),
                'onset_nll': round(r.onset_token_nll, 5),
                'context_nll': round(r.context_nll, 5),
                'silence_nll_before': round(r.silence_nll_before, 5),
                'speech_nll_after': round(r.speech_nll_after, 5),
            })
        viz_data.append({
            'label': f"{bm.seq_label} — Ch {seg.channel} @ step {seg.onset_step}",
            'channel': seg.channel,
            'onset_step': seg.onset_step,
            'seg_length': seg.length,
            'context_type': seg.context_type,
            'other_state': seg.other_channel_state,
            'silence_before': seg.silence_before,
            'shifts': shift_data,
        })
    return viz_data


def generate_aggregate_viz(
    benchmarks: List[SegmentBenchmark],
    output_path: str,
    template_path: Optional[str] = None,
) -> str:
    """Write the aggregated view (all segments on one page) by filling the HTML template."""
    template = Path(template_path) if template_path else AGGREGATE_TEMPLATE
    html = template.read_text().replace('__DATA_PLACEHOLDER__', json.dumps(_benchmarks_to_viz_data(benchmarks)))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    return output_path


def generate_benchmark_viz(
    benchmarks: List[SegmentBenchmark],
    output_dir: str,
) -> str:
    """Write the per-segment interactive view to ``<output_dir>/onset_benchmark.html``."""
    html = _build_benchmark_html(_benchmarks_to_viz_data(benchmarks))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / 'onset_benchmark.html')
    with open(output_path, 'w') as f:
        f.write(html)
    return output_path


def _build_benchmark_html(viz_data: list) -> str:
    data_json = json.dumps(viz_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Onset Timing Benchmark</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    padding: 24px;
  }}
  h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 8px; color: #f0f0f0; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 32px; }}
  .segment-container {{
    background: #1a1d27;
    border: 1px solid #2a2d37;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .segment-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .segment-header h2 {{ font-size: 1rem; font-weight: 600; color: #f0f0f0; }}
  .segment-meta {{
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .meta-tag {{
    background: #252833;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    color: #aaa;
  }}
  .meta-tag.turn {{ border-left: 3px solid #5bf6a6; }}
  .meta-tag.backchannel {{ border-left: 3px solid #f6d45b; }}
  .meta-tag.silence {{ border-left: 3px solid #888; }}
  .meta-tag.overlap {{ border-left: 3px solid #b05bf6; }}
  .charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .chart-box {{
    background: #12141c;
    border-radius: 8px;
    padding: 12px;
  }}
  .chart-box h3 {{
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 8px;
  }}
  .chart-container {{ position: relative; height: 280px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }}
  .summary-card {{
    background: #1a1d27;
    border: 1px solid #2a2d37;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }}
  .summary-card .label {{ font-size: 0.75rem; color: #888; }}
  .summary-card .value {{ font-size: 1.3rem; font-weight: 700; margin-top: 4px; color: #5bf6a6; }}
  .summary-card .sub {{ font-size: 0.75rem; color: #666; margin-top: 2px; }}
</style>
</head>
<body>

<h1>Onset Timing Benchmark</h1>
<p class="subtitle">How does NLL change when we shift segment onset times? Shift=0 is ground truth.</p>

<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Segments Tested</div>
    <div class="value" id="stat-count">-</div>
  </div>
  <div class="summary-card">
    <div class="label">GT is Optimal</div>
    <div class="value" id="stat-gt-optimal">-</div>
    <div class="sub">shift=0 has lowest total NLL</div>
  </div>
  <div class="summary-card">
    <div class="label">Avg Best Shift</div>
    <div class="value" id="stat-avg-shift">-</div>
    <div class="sub">mean optimal shift from GT</div>
  </div>
</div>

<div id="segments-container"></div>

<script>
const DATA = {data_json};

Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2d37';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';

const COLORS = {{
  total: '#5b9cf6',
  onset: '#f67b5b',
  context: '#5bf6a6',
  silence_before: '#888',
  speech_after: '#f6d45b',
}};

// Summary stats
let gtOptimalCount = 0;
let bestShifts = [];

DATA.forEach(seg => {{
  const shifts = seg.shifts;
  if (!shifts.length) return;
  const best = shifts.reduce((a, b) => a.total_nll < b.total_nll ? a : b);
  if (best.shift === 0) gtOptimalCount++;
  bestShifts.push(best.shift);
}});

document.getElementById('stat-count').textContent = DATA.length;
document.getElementById('stat-gt-optimal').textContent =
  `${{gtOptimalCount}} / ${{DATA.length}} (${{Math.round(100*gtOptimalCount/Math.max(1,DATA.length))}}%)`;
document.getElementById('stat-avg-shift').textContent =
  (bestShifts.reduce((a,b) => a+b, 0) / Math.max(1, bestShifts.length)).toFixed(1) + ' steps';

// Render each segment
const container = document.getElementById('segments-container');

DATA.forEach((seg, idx) => {{
  const shifts = seg.shifts;
  if (!shifts.length) return;

  const best = shifts.reduce((a, b) => a.total_nll < b.total_nll ? a : b);
  const gt = shifts.find(s => s.shift === 0);

  const ctxClass = seg.context_type === 'turn_taking' ? 'turn' :
                   seg.context_type === 'backchannel' ? 'backchannel' :
                   seg.context_type === 'overlap_start' ? 'overlap' : 'silence';

  const div = document.createElement('div');
  div.className = 'segment-container';
  div.innerHTML = `
    <div class="segment-header">
      <h2>${{seg.label}}</h2>
    </div>
    <div class="segment-meta">
      <span class="meta-tag ${{ctxClass}}">${{seg.context_type}}</span>
      <span class="meta-tag">Channel ${{seg.channel}}</span>
      <span class="meta-tag">Length: ${{seg.seg_length}} tokens</span>
      <span class="meta-tag">Silence before: ${{seg.silence_before}} steps</span>
      <span class="meta-tag">Other channel: ${{seg.other_state}}</span>
      <span class="meta-tag">Best shift: ${{best.shift}} (NLL=${{best.total_nll.toFixed(4)}})</span>
      ${{gt ? `<span class="meta-tag">GT NLL: ${{gt.total_nll.toFixed(4)}}</span>` : ''}}
    </div>
    <div class="charts-row">
      <div class="chart-box">
        <h3>Total NLL vs Onset Shift</h3>
        <div class="chart-container"><canvas id="chart-total-${{idx}}"></canvas></div>
      </div>
      <div class="chart-box">
        <h3>NLL Decomposition vs Shift</h3>
        <div class="chart-container"><canvas id="chart-decomp-${{idx}}"></canvas></div>
      </div>
    </div>
  `;
  container.appendChild(div);

  // Total NLL chart
  const ctxTotal = document.getElementById(`chart-total-${{idx}}`).getContext('2d');
  new Chart(ctxTotal, {{
    type: 'line',
    data: {{
      labels: shifts.map(s => s.shift),
      datasets: [{{
        label: 'Total NLL',
        data: shifts.map(s => s.total_nll),
        borderColor: COLORS.total,
        backgroundColor: COLORS.total + '22',
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.2,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        annotation: {{
          annotations: {{
            gtLine: {{
              type: 'line',
              xMin: shifts.findIndex(s => s.shift === 0),
              xMax: shifts.findIndex(s => s.shift === 0),
              borderColor: '#fff4',
              borderWidth: 1,
              borderDash: [4, 4],
              label: {{ content: 'GT', display: true, color: '#fff8', font: {{ size: 10 }} }}
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Onset shift (steps)' }} }},
        y: {{ title: {{ display: true, text: 'Total NLL (scoring window)' }} }},
      }},
    }},
  }});

  // Decomposition chart
  const ctxDecomp = document.getElementById(`chart-decomp-${{idx}}`).getContext('2d');
  new Chart(ctxDecomp, {{
    type: 'line',
    data: {{
      labels: shifts.map(s => s.shift),
      datasets: [{{
        label: 'Onset token NLL',
        data: shifts.map(s => s.onset_nll),
        borderColor: COLORS.onset,
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.2,
      }}, {{
        label: 'Context NLL (non-onset)',
        data: shifts.map(s => s.context_nll),
        borderColor: COLORS.context,
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.2,
      }}, {{
        label: 'Avg silence NLL before',
        data: shifts.map(s => s.silence_nll_before),
        borderColor: COLORS.silence_before,
        borderWidth: 1.5,
        pointRadius: 1,
        borderDash: [4, 4],
        tension: 0.2,
      }}, {{
        label: 'Avg speech NLL after',
        data: shifts.map(s => s.speech_nll_after),
        borderColor: COLORS.speech_after,
        borderWidth: 1.5,
        pointRadius: 1,
        borderDash: [4, 4],
        tension: 0.2,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ title: {{ display: true, text: 'Onset shift (steps)' }} }},
        y: {{ title: {{ display: true, text: 'NLL' }} }},
      }},
    }},
  }});
}});
</script>
</body>
</html>"""
