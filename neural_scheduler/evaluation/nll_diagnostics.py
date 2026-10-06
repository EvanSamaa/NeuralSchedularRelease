"""Per-token NLL of ground-truth dialogues, broken down by dialogue event.

Scores each dialogue token under the scheduler and annotates it with its
channel, region (mutual silence, single speaker, overlap), whether it is a
speech onset or offset, and how long the preceding silence or current segment
is. The result is written as a self-contained interactive HTML page (the page
loads Chart.js from a CDN) and optionally as CSV.

    python -m neural_scheduler.evaluation.nll_diagnostics \\
        --checkpoint CKPT --data_jsonl VAL.jsonl --out_dir OUT --seed 0 --onset_benchmark

A sequence passed to :func:`run_diagnostics` is a dict with ``context_ids``
(everything up to and including ``<|start-of-dialogue|>``), ``dialogue_ids``
(interleaved ``[a0, b0, a1, b1, ...]`` vocab ids) and an optional ``label``.
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..constants import SILENCE_CODE
from ..vocab import audio_token
from .common import compute_per_token_nll, extract_script_contents, load_jsonl_examples, prepare_inference_input


# ==============================================================
# Data structures
# ==============================================================

@dataclass
class TokenEvent:
    """Annotated token at a specific position."""
    step: int               # dialogue step (pair index)
    channel: str            # 'A' or 'B'
    token_id: int
    nll: float
    is_silence: bool
    region: str             # 'silence', 'single_A', 'single_B', 'overlap'
    silence_run_before: int # how many consecutive silence steps preceded this on same channel
    is_onset: bool          # first non-silence token after silence
    is_offset: bool         # first silence token after speech
    steps_into_segment: int # how far into current speech segment (0 if silence)


@dataclass
class SequenceDiagnostics:
    """Full diagnostic results for one ground-truth sequence."""
    label: str
    events: List[TokenEvent]
    total_steps: int
    # Aggregated stats
    mean_nll: float = 0.0
    mean_nll_a: float = 0.0
    mean_nll_b: float = 0.0
    mean_nll_by_region: Dict[str, float] = field(default_factory=dict)
    onset_nlls: List[Tuple[int, float, str]] = field(default_factory=list)  # (silence_before, nll, channel)
    offset_nlls: List[Tuple[int, float, str]] = field(default_factory=list) # (segment_len, nll, channel)


# ==============================================================
# Event annotation
# ==============================================================

def annotate_dialogue(
    dialogue_ids: List[int],
    token_nlls: np.ndarray,
    silence_token_id: int,
) -> List[TokenEvent]:
    """
    Annotate each token in the dialogue with contextual information.
    """
    num_pairs = len(dialogue_ids) // 2
    events = []

    # Track per-channel state
    a_silence_run = 0
    b_silence_run = 0
    a_in_speech = False
    b_in_speech = False
    a_segment_pos = 0
    b_segment_pos = 0

    for step in range(num_pairs):
        idx_a = step * 2
        idx_b = step * 2 + 1

        tok_a = dialogue_ids[idx_a]
        tok_b = dialogue_ids[idx_b]

        is_sil_a = (tok_a == silence_token_id)
        is_sil_b = (tok_b == silence_token_id)

        # Determine region
        if is_sil_a and is_sil_b:
            region = 'silence'
        elif not is_sil_a and is_sil_b:
            region = 'single_A'
        elif is_sil_a and not is_sil_b:
            region = 'single_B'
        else:
            region = 'overlap'

        # Channel A event
        is_onset_a = (not is_sil_a) and (not a_in_speech)
        is_offset_a = is_sil_a and a_in_speech

        if is_sil_a:
            a_silence_run += 1
            a_segment_pos = 0
            a_in_speech = False
        else:
            if is_onset_a:
                silence_before_a = a_silence_run
            else:
                silence_before_a = 0
            a_silence_run = 0
            a_in_speech = True
            a_segment_pos += 1

        events.append(TokenEvent(
            step=step,
            channel='A',
            token_id=tok_a,
            nll=float(token_nlls[idx_a]),
            is_silence=is_sil_a,
            region=region,
            silence_run_before=a_silence_run if is_sil_a else (silence_before_a if is_onset_a else 0),
            is_onset=is_onset_a,
            is_offset=is_offset_a,
            steps_into_segment=0 if is_sil_a else a_segment_pos,
        ))

        # Channel B event
        is_onset_b = (not is_sil_b) and (not b_in_speech)
        is_offset_b = is_sil_b and b_in_speech

        if is_sil_b:
            b_silence_run += 1
            b_segment_pos = 0
            b_in_speech = False
        else:
            if is_onset_b:
                silence_before_b = b_silence_run
            else:
                silence_before_b = 0
            b_silence_run = 0
            b_in_speech = True
            b_segment_pos += 1

        events.append(TokenEvent(
            step=step,
            channel='B',
            token_id=tok_b,
            nll=float(token_nlls[idx_b]),
            is_silence=is_sil_b,
            region=region,
            silence_run_before=b_silence_run if is_sil_b else (silence_before_b if is_onset_b else 0),
            is_onset=is_onset_b,
            is_offset=is_offset_b,
            steps_into_segment=0 if is_sil_b else b_segment_pos,
        ))

    return events


# ==============================================================
# Aggregation
# ==============================================================

def compute_diagnostics(events: List[TokenEvent], label: str) -> SequenceDiagnostics:
    """Compute aggregate statistics from annotated events."""
    diag = SequenceDiagnostics(
        label=label,
        events=events,
        total_steps=max(e.step for e in events) + 1 if events else 0,
    )

    all_nlls = [e.nll for e in events]
    a_nlls = [e.nll for e in events if e.channel == 'A']
    b_nlls = [e.nll for e in events if e.channel == 'B']

    diag.mean_nll = np.mean(all_nlls) if all_nlls else 0.0
    diag.mean_nll_a = np.mean(a_nlls) if a_nlls else 0.0
    diag.mean_nll_b = np.mean(b_nlls) if b_nlls else 0.0

    # NLL by region
    for region in ['silence', 'single_A', 'single_B', 'overlap']:
        region_nlls = [e.nll for e in events if e.region == region]
        diag.mean_nll_by_region[region] = np.mean(region_nlls) if region_nlls else 0.0

    # Onset NLLs (with preceding silence duration)
    for e in events:
        if e.is_onset:
            diag.onset_nlls.append((e.silence_run_before, e.nll, e.channel))

    # Offset NLLs (with segment length)
    for e in events:
        if e.is_offset:
            diag.offset_nlls.append((e.steps_into_segment, e.nll, e.channel))

    return diag


# ==============================================================
# Main runner
# ==============================================================

@torch.no_grad()
def run_diagnostics(
    model,
    tokenizer,
    ground_truth_sequences: List[Dict],
    device: str = 'cuda',
    verbose: bool = True,
) -> List[SequenceDiagnostics]:
    """
    Run NLL diagnostics on a set of ground-truth dialogue sequences.

    Args:
        model: The ChatGLM model
        tokenizer: Tokenizer with audio tokens
        ground_truth_sequences: List of dicts, each with:
            - 'context_ids': list[int] — tokens before dialogue
            - 'dialogue_ids': list[int] — interleaved [a0, b0, a1, b1, ...]
            - 'label': str (optional)
        device: torch device
        verbose: print progress

    Returns:
        List of SequenceDiagnostics, one per sequence
    """
    model.eval()
    silence_token_id = tokenizer.convert_tokens_to_ids(audio_token(SILENCE_CODE))

    all_results = []

    for i, seq in enumerate(ground_truth_sequences):
        label = seq.get('label', f'sequence_{i}')
        if verbose:
            print(f"\nProcessing {label}...")

        context_ids = seq['context_ids']
        dialogue_ids = seq['dialogue_ids']

        if verbose:
            num_pairs = len(dialogue_ids) // 2
            sil_count = sum(1 for t in dialogue_ids if t == silence_token_id)
            print(f"  {num_pairs} dialogue steps, {sil_count}/{len(dialogue_ids)} silence tokens")

        # Compute per-token NLL
        token_nlls = compute_per_token_nll(model, context_ids, dialogue_ids, torch.device(device))

        # Annotate
        events = annotate_dialogue(dialogue_ids, token_nlls, silence_token_id)

        # Aggregate
        diag = compute_diagnostics(events, label)
        all_results.append(diag)

        if verbose:
            print(f"  Mean NLL: {diag.mean_nll:.4f} (A={diag.mean_nll_a:.4f}, B={diag.mean_nll_b:.4f})")
            for region, nll in diag.mean_nll_by_region.items():
                count = sum(1 for e in events if e.region == region)
                print(f"  Region '{region}': mean NLL={nll:.4f} ({count} tokens)")
            print(f"  Onsets: {len(diag.onset_nlls)}, Offsets: {len(diag.offset_nlls)}")

    return all_results


# ==============================================================
# Visualization
# ==============================================================

def generate_visualizations(
    results: List[SequenceDiagnostics],
    output_dir: str,
) -> str:
    """
    Generate an HTML file with interactive visualizations of NLL diagnostics.

    Returns the path to the generated HTML file.
    """
    # Aggregate data across all sequences
    all_data = {
        'per_sequence': [],
        'onset_silence_vs_nll': [],     # (silence_duration, nll, channel, seq_label)
        'nll_by_region': {},            # region -> [nll values]
        'nll_timeseries': [],           # per-sequence timeseries
        'silence_nll_trajectory': [],   # NLL of silence tokens as function of silence run length
        'speech_nll_trajectory': [],    # NLL of speech tokens as function of position in segment
        'nll_around_onset': [],         # NLL in window around speech onsets
    }

    for diag in results:
        all_data['per_sequence'].append({
            'label': diag.label,
            'mean_nll': diag.mean_nll,
            'mean_nll_a': diag.mean_nll_a,
            'mean_nll_b': diag.mean_nll_b,
            'nll_by_region': diag.mean_nll_by_region,
        })

        # Onset analysis
        for sil_dur, nll, ch in diag.onset_nlls:
            all_data['onset_silence_vs_nll'].append({
                'silence_before': sil_dur,
                'nll': nll,
                'channel': ch,
                'seq': diag.label,
            })

        # Timeseries (subsample for large sequences)
        step_nlls_a = {}
        step_nlls_b = {}
        for e in diag.events:
            if e.channel == 'A':
                step_nlls_a[e.step] = e.nll
            else:
                step_nlls_b[e.step] = e.nll

        # Smoothed timeseries
        window = 20
        steps = sorted(set(step_nlls_a.keys()) | set(step_nlls_b.keys()))
        smoothed_a = []
        smoothed_b = []
        smoothed_steps = []
        for s in range(0, len(steps), max(1, len(steps) // 500)):
            step = steps[s]
            # Window average
            win_a = [step_nlls_a.get(steps[j], 0) for j in range(max(0, s-window), min(len(steps), s+window+1)) if steps[j] in step_nlls_a]
            win_b = [step_nlls_b.get(steps[j], 0) for j in range(max(0, s-window), min(len(steps), s+window+1)) if steps[j] in step_nlls_b]
            smoothed_steps.append(step)
            smoothed_a.append(np.mean(win_a) if win_a else 0)
            smoothed_b.append(np.mean(win_b) if win_b else 0)

        # Region annotations for timeseries
        region_spans = []
        for e in diag.events:
            if e.channel == 'A':  # one per step
                region_spans.append({'step': e.step, 'region': e.region})

        all_data['nll_timeseries'].append({
            'label': diag.label,
            'steps': smoothed_steps,
            'nll_a': [round(v, 4) for v in smoothed_a],
            'nll_b': [round(v, 4) for v in smoothed_b],
            'regions': region_spans[::max(1, len(region_spans)//500)],
        })

        # NLL by region
        for e in diag.events:
            if e.region not in all_data['nll_by_region']:
                all_data['nll_by_region'][e.region] = []
            all_data['nll_by_region'][e.region].append(e.nll)

        # Silence NLL trajectory: NLL of silence tokens vs how long silence has been going
        for e in diag.events:
            if e.is_silence and e.silence_run_before > 0:
                all_data['silence_nll_trajectory'].append({
                    'run_length': e.silence_run_before,
                    'nll': e.nll,
                    'channel': e.channel,
                })

        # Speech NLL trajectory: NLL vs position within speech segment
        for e in diag.events:
            if not e.is_silence and e.steps_into_segment > 0:
                all_data['speech_nll_trajectory'].append({
                    'pos': e.steps_into_segment,
                    'nll': e.nll,
                    'channel': e.channel,
                })

        # NLL around onsets: collect a window of NLL values around each onset
        onset_steps = [e.step for e in diag.events if e.is_onset]
        onset_window = 30  # steps before and after
        for onset_step in onset_steps:
            onset_ch = None
            for e in diag.events:
                if e.step == onset_step and e.is_onset:
                    onset_ch = e.channel
                    break
            if onset_ch is None:
                continue

            window_data = []
            for offset in range(-onset_window, onset_window + 1):
                s = onset_step + offset
                for e in diag.events:
                    if e.step == s and e.channel == onset_ch:
                        window_data.append({'offset': offset, 'nll': e.nll, 'channel': onset_ch})
                        break

            if len(window_data) > onset_window:  # need reasonable coverage
                all_data['nll_around_onset'].append(window_data)

    # Aggregate onset window data by offset
    onset_by_offset = {}
    for window_data in all_data['nll_around_onset']:
        for point in window_data:
            off = point['offset']
            if off not in onset_by_offset:
                onset_by_offset[off] = []
            onset_by_offset[off].append(point['nll'])

    onset_avg = []
    for off in sorted(onset_by_offset.keys()):
        vals = onset_by_offset[off]
        onset_avg.append({
            'offset': off,
            'mean_nll': round(np.mean(vals), 4),
            'std_nll': round(np.std(vals), 4),
            'count': len(vals),
        })

    # Aggregate silence trajectory
    sil_traj_by_len = {}
    for item in all_data['silence_nll_trajectory']:
        rl = item['run_length']
        if rl not in sil_traj_by_len:
            sil_traj_by_len[rl] = []
        sil_traj_by_len[rl].append(item['nll'])

    sil_traj_avg = []
    for rl in sorted(sil_traj_by_len.keys()):
        if rl <= 200:  # cap for readability
            vals = sil_traj_by_len[rl]
            sil_traj_avg.append({
                'run_length': rl,
                'mean_nll': round(np.mean(vals), 4),
                'std_nll': round(np.std(vals), 4),
                'count': len(vals),
            })

    # Aggregate speech trajectory
    speech_traj_by_pos = {}
    for item in all_data['speech_nll_trajectory']:
        p = item['pos']
        if p not in speech_traj_by_pos:
            speech_traj_by_pos[p] = []
        speech_traj_by_pos[p].append(item['nll'])

    speech_traj_avg = []
    for p in sorted(speech_traj_by_pos.keys()):
        if p <= 200:
            vals = speech_traj_by_pos[p]
            speech_traj_avg.append({
                'pos': p,
                'mean_nll': round(np.mean(vals), 4),
                'std_nll': round(np.std(vals), 4),
                'count': len(vals),
            })

    # Region boxplot data
    region_stats = {}
    for region, vals in all_data['nll_by_region'].items():
        arr = np.array(vals)
        region_stats[region] = {
            'mean': round(np.mean(arr), 4),
            'median': round(np.median(arr), 4),
            'std': round(np.std(arr), 4),
            'q25': round(np.percentile(arr, 25), 4),
            'q75': round(np.percentile(arr, 75), 4),
            'count': len(vals),
        }

    # Onset scatter data
    onset_scatter = []
    for item in all_data['onset_silence_vs_nll']:
        onset_scatter.append({
            'x': item['silence_before'],
            'y': round(item['nll'], 4),
            'ch': item['channel'],
        })

    # Build the viz data JSON
    viz_data = {
        'per_sequence': all_data['per_sequence'],
        'timeseries': all_data['nll_timeseries'],
        'region_stats': region_stats,
        'onset_avg': onset_avg,
        'sil_traj_avg': sil_traj_avg,
        'speech_traj_avg': speech_traj_avg,
        'onset_scatter': onset_scatter,
    }

    # Generate HTML
    html = _build_html(viz_data)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / 'nll_diagnostics.html')
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def _build_html(viz_data: dict) -> str:
    """Build a self-contained HTML page with Chart.js visualizations."""

    data_json = json.dumps(viz_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NLL Diagnostics — Dual-Channel Dialogue Model</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    padding: 24px;
  }}
  h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: #f0f0f0;
  }}
  .subtitle {{
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 32px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}
  .card {{
    background: #1a1d27;
    border: 1px solid #2a2d37;
    border-radius: 12px;
    padding: 20px;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: #f0f0f0;
  }}
  .card .desc {{
    font-size: 0.8rem;
    color: #777;
    margin-bottom: 16px;
  }}
  .chart-container {{
    position: relative;
    height: 320px;
  }}
  .card.full .chart-container {{ height: 400px; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: #1a1d27;
    border: 1px solid #2a2d37;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }}
  .stat-card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-card .value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
  .stat-card .sub {{ font-size: 0.75rem; color: #666; margin-top: 2px; }}
  .color-a {{ color: #5b9cf6; }}
  .color-b {{ color: #f67b5b; }}
  .color-sil {{ color: #888; }}
  .color-over {{ color: #b05bf6; }}
  .seq-selector {{
    margin-bottom: 16px;
  }}
  .seq-selector select {{
    background: #1a1d27;
    color: #e0e0e0;
    border: 1px solid #2a2d37;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.85rem;
  }}
</style>
</head>
<body>

<h1>NLL Diagnostics — Dual-Channel Dialogue Model</h1>
<p class="subtitle">Analyzing model behavior on ground-truth dialogue sequences</p>

<div id="stats-container" class="stats-grid"></div>

<div class="grid">
  <!-- 1. NLL Timeseries -->
  <div class="card full">
    <h2>NLL Over Time (Smoothed)</h2>
    <p class="desc">Per-channel NLL trajectory with sliding window average. Background bands show dialogue regions.</p>
    <div class="seq-selector">
      <select id="seq-select" onchange="updateTimeseries()"></select>
    </div>
    <div class="chart-container">
      <canvas id="chart-timeseries"></canvas>
    </div>
  </div>

  <!-- 2. NLL by Region -->
  <div class="card">
    <h2>NLL by Dialogue Region</h2>
    <p class="desc">Mean NLL when both silent, one speaking, or overlapping.</p>
    <div class="chart-container">
      <canvas id="chart-region"></canvas>
    </div>
  </div>

  <!-- 3. NLL Around Speech Onsets -->
  <div class="card">
    <h2>NLL Around Speech Onsets</h2>
    <p class="desc">Average NLL in a window around the moment speech begins. Step 0 = onset.</p>
    <div class="chart-container">
      <canvas id="chart-onset-window"></canvas>
    </div>
  </div>

  <!-- 4. Silence Duration vs Onset NLL -->
  <div class="card">
    <h2>Silence Duration → Onset NLL</h2>
    <p class="desc">Does the model increasingly "expect" speech after longer silence?</p>
    <div class="chart-container">
      <canvas id="chart-onset-scatter"></canvas>
    </div>
  </div>

  <!-- 5. Silence NLL Trajectory -->
  <div class="card">
    <h2>Silence Token NLL vs Run Length</h2>
    <p class="desc">How does the NLL of predicting silence change as silence continues?</p>
    <div class="chart-container">
      <canvas id="chart-silence-traj"></canvas>
    </div>
  </div>

  <!-- 6. Speech NLL Trajectory -->
  <div class="card">
    <h2>Speech Token NLL vs Position in Segment</h2>
    <p class="desc">Does the model get better at predicting speech tokens deeper into a segment?</p>
    <div class="chart-container">
      <canvas id="chart-speech-traj"></canvas>
    </div>
  </div>

  <!-- 7. Per-Sequence Summary -->
  <div class="card">
    <h2>Per-Sequence Summary</h2>
    <p class="desc">Mean NLL broken down by channel for each sequence.</p>
    <div class="chart-container">
      <canvas id="chart-per-seq"></canvas>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2d37';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';

const COLORS = {{
  A: '#5b9cf6',
  B: '#f67b5b',
  silence: '#555',
  single_A: '#5b9cf644',
  single_B: '#f67b5b44',
  overlap: '#b05bf644',
  accent: '#5bf6a6',
}};

// --- Stats cards ---
function renderStats() {{
  const container = document.getElementById('stats-container');
  const regions = DATA.region_stats;
  const cards = [
    {{ label: 'Mutual Silence', value: regions.silence?.mean || 0, sub: `n=${{regions.silence?.count || 0}}`, cls: 'color-sil' }},
    {{ label: 'Single Speaker A', value: regions.single_A?.mean || 0, sub: `n=${{regions.single_A?.count || 0}}`, cls: 'color-a' }},
    {{ label: 'Single Speaker B', value: regions.single_B?.mean || 0, sub: `n=${{regions.single_B?.count || 0}}`, cls: 'color-b' }},
    {{ label: 'Overlap', value: regions.overlap?.mean || 0, sub: `n=${{regions.overlap?.count || 0}}`, cls: 'color-over' }},
  ];
  container.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${{c.label}}</div>
      <div class="value ${{c.cls}}">${{c.value.toFixed(3)}}</div>
      <div class="sub">${{c.sub}}</div>
    </div>
  `).join('');
}}

// --- Timeseries ---
let timeseriesChart = null;
function initTimeseries() {{
  const select = document.getElementById('seq-select');
  DATA.timeseries.forEach((ts, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = ts.label;
    select.appendChild(opt);
  }});
  updateTimeseries();
}}

function updateTimeseries() {{
  const idx = parseInt(document.getElementById('seq-select').value);
  const ts = DATA.timeseries[idx];
  if (!ts) return;

  if (timeseriesChart) timeseriesChart.destroy();

  const ctx = document.getElementById('chart-timeseries').getContext('2d');
  timeseriesChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: ts.steps,
      datasets: [
        {{
          label: 'Channel A',
          data: ts.nll_a,
          borderColor: COLORS.A,
          backgroundColor: COLORS.A + '33',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
        }},
        {{
          label: 'Channel B',
          data: ts.nll_b,
          borderColor: COLORS.B,
          backgroundColor: COLORS.B + '33',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
        }},
      ],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Dialogue Step' }} }},
        y: {{ title: {{ display: true, text: 'NLL (smoothed)' }} }},
      }},
      plugins: {{
        legend: {{ position: 'top' }},
      }},
    }},
  }});
}}

// --- Region bar chart ---
function renderRegionChart() {{
  const regions = DATA.region_stats;
  const labels = ['silence', 'single_A', 'single_B', 'overlap'];
  const displayLabels = ['Mutual Silence', 'Single A', 'Single B', 'Overlap'];
  const colors = [COLORS.silence, COLORS.A, COLORS.B, '#b05bf6'];

  const ctx = document.getElementById('chart-region').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: displayLabels,
      datasets: [{{
        label: 'Mean NLL',
        data: labels.map(l => regions[l]?.mean || 0),
        backgroundColor: colors.map(c => c + '88'),
        borderColor: colors,
        borderWidth: 1.5,
      }}, {{
        label: 'Median NLL',
        data: labels.map(l => regions[l]?.median || 0),
        backgroundColor: colors.map(c => c + '44'),
        borderColor: colors.map(c => c + '88'),
        borderWidth: 1.5,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        y: {{ title: {{ display: true, text: 'NLL' }}, beginAtZero: true }},
      }},
    }},
  }});
}}

// --- Onset window chart ---
function renderOnsetWindow() {{
  const data = DATA.onset_avg;
  if (!data.length) return;

  const ctx = document.getElementById('chart-onset-window').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: data.map(d => d.offset),
      datasets: [{{
        label: 'Mean NLL',
        data: data.map(d => d.mean_nll),
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accent + '33',
        borderWidth: 2,
        pointRadius: 1,
        fill: true,
        tension: 0.3,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ title: {{ display: true, text: 'Steps from onset (0 = speech starts)' }} }},
        y: {{ title: {{ display: true, text: 'Mean NLL' }} }},
      }},
      plugins: {{
        annotation: {{
          annotations: {{
            line1: {{ type: 'line', xMin: 0, xMax: 0, borderColor: '#fff', borderWidth: 1 }}
          }}
        }}
      }}
    }},
  }});
}}

// --- Onset scatter ---
function renderOnsetScatter() {{
  const data = DATA.onset_scatter;
  if (!data.length) return;

  const aData = data.filter(d => d.ch === 'A');
  const bData = data.filter(d => d.ch === 'B');

  const ctx = document.getElementById('chart-onset-scatter').getContext('2d');
  new Chart(ctx, {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Channel A onset',
        data: aData.map(d => ({{ x: d.x, y: d.y }})),
        backgroundColor: COLORS.A + '66',
        borderColor: COLORS.A,
        pointRadius: 3,
      }}, {{
        label: 'Channel B onset',
        data: bData.map(d => ({{ x: d.x, y: d.y }})),
        backgroundColor: COLORS.B + '66',
        borderColor: COLORS.B,
        pointRadius: 3,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ title: {{ display: true, text: 'Silence steps before onset' }} }},
        y: {{ title: {{ display: true, text: 'Onset token NLL' }} }},
      }},
    }},
  }});
}}

// --- Silence trajectory ---
function renderSilenceTraj() {{
  const data = DATA.sil_traj_avg;
  if (!data.length) return;

  const ctx = document.getElementById('chart-silence-traj').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: data.map(d => d.run_length),
      datasets: [{{
        label: 'Mean NLL of silence token',
        data: data.map(d => d.mean_nll),
        borderColor: COLORS.silence,
        backgroundColor: COLORS.silence + '33',
        borderWidth: 2,
        pointRadius: 1,
        fill: true,
        tension: 0.3,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ title: {{ display: true, text: 'Consecutive silence steps' }} }},
        y: {{ title: {{ display: true, text: 'Mean NLL' }} }},
      }},
    }},
  }});
}}

// --- Speech trajectory ---
function renderSpeechTraj() {{
  const data = DATA.speech_traj_avg;
  if (!data.length) return;

  const ctx = document.getElementById('chart-speech-traj').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: data.map(d => d.pos),
      datasets: [{{
        label: 'Mean NLL of speech token',
        data: data.map(d => d.mean_nll),
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accent + '33',
        borderWidth: 2,
        pointRadius: 1,
        fill: true,
        tension: 0.3,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ title: {{ display: true, text: 'Position in speech segment' }} }},
        y: {{ title: {{ display: true, text: 'Mean NLL' }} }},
      }},
    }},
  }});
}}

// --- Per-sequence summary ---
function renderPerSeq() {{
  const data = DATA.per_sequence;
  if (!data.length) return;

  const ctx = document.getElementById('chart-per-seq').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: data.map(d => d.label),
      datasets: [{{
        label: 'Channel A',
        data: data.map(d => d.mean_nll_a),
        backgroundColor: COLORS.A + '88',
        borderColor: COLORS.A,
        borderWidth: 1,
      }}, {{
        label: 'Channel B',
        data: data.map(d => d.mean_nll_b),
        backgroundColor: COLORS.B + '88',
        borderColor: COLORS.B,
        borderWidth: 1,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        y: {{ title: {{ display: true, text: 'Mean NLL' }}, beginAtZero: true }},
      }},
    }},
  }});
}}

// --- Initialize ---
renderStats();
initTimeseries();
renderRegionChart();
renderOnsetWindow();
renderOnsetScatter();
renderSilenceTraj();
renderSpeechTraj();
renderPerSeq();
</script>
</body>
</html>"""


# ==============================================================
# Export raw data for custom analysis
# ==============================================================

def export_events_to_csv(
    results: List[SequenceDiagnostics],
    output_path: str,
):
    """Export all token events to CSV for external analysis."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sequence', 'step', 'channel', 'token_id', 'nll',
            'is_silence', 'region', 'silence_run_before',
            'is_onset', 'is_offset', 'steps_into_segment',
        ])
        for diag in results:
            for e in diag.events:
                writer.writerow([
                    diag.label, e.step, e.channel, e.token_id,
                    round(e.nll, 6), e.is_silence, e.region,
                    e.silence_run_before, e.is_onset, e.is_offset,
                    e.steps_into_segment,
                ])
    return output_path


# ==============================================================
# CLI
# ==============================================================

def build_sequences(tokenizer, samples: List[Dict], device: str = "cpu") -> List[Dict]:
    """Turn dataset records into the sequence dicts :func:`run_diagnostics` takes."""
    sequences = []
    for i, sample in enumerate(samples):
        script_a_ids, script_b_ids, dialogue_ids = extract_script_contents(
            tokenizer, sample["input_ids"], return_dialogue=True
        )
        context_ids, _ = prepare_inference_input(script_a_ids, script_b_ids, tokenizer, device, already_tokenized=True)
        sequences.append({
            "context_ids": context_ids.cpu().tolist()[0],
            "dialogue_ids": dialogue_ids,
            "label": "sample_" + str(sample.get("source_file", i)),
        })
    return sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Scheduler checkpoint directory.")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Tokenized validation JSONL.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pool_size", type=int, default=200, help="Records drawn at random from --data_jsonl.")
    parser.add_argument("--num_sequences", type=int, default=20, help="How many of the drawn records to score.")
    parser.add_argument("--seed", type=int, default=None, help="Seeds the record draw (1986 reproduces the draw of the published runs). "
                             "Omit for an unseeded draw.")
    parser.add_argument("--export_csv", action="store_true", help="Also write nll_events.csv.")
    parser.add_argument("--onset_benchmark", action="store_true",
                        help="Also run the onset-shift benchmark on the same sequences.")
    parser.add_argument("--shift_range", type=int, default=20)
    parser.add_argument("--max_segments_per_seq", type=int, default=5)
    parser.add_argument("--window_padding", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    from .common import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    samples = load_jsonl_examples(args.data_jsonl, num_examples=args.pool_size, seed=args.seed)
    sequences = build_sequences(tokenizer, samples[:args.num_sequences])

    results = run_diagnostics(model, tokenizer, sequences, device=args.device)
    print(f"Wrote {generate_visualizations(results, args.out_dir)}")
    if args.export_csv:
        print(f"Wrote {export_events_to_csv(results, str(Path(args.out_dir) / 'nll_events.csv'))}")

    if args.onset_benchmark:
        from .onset_benchmark import generate_aggregate_viz, generate_benchmark_viz, run_onset_benchmark

        benchmarks = run_onset_benchmark(
            model, tokenizer, sequences,
            shift_range=args.shift_range,
            max_segments_per_seq=args.max_segments_per_seq,
            window_padding=args.window_padding,
            device=args.device,
        )
        print(f"Wrote {generate_benchmark_viz(benchmarks, args.out_dir)}")
        print(f"Wrote {generate_aggregate_viz(benchmarks, str(Path(args.out_dir) / 'aggregate_benchmark.html'))}")


if __name__ == "__main__":
    main()
