"""Scheduling-state export and multi-schedule timeline comparison.

* :func:`export_scheduling_state_json` serialises a ``SchedulingState`` (the
  result of the three-pass search) to the ``timeline.json`` format.
* :func:`visualize_timeline_comparison` writes an interactive HTML page that
  compares the final timelines of several ``timeline.json`` files.
* :func:`render_timeline_comparison_png` renders the same comparison as a
  compact, single-column figure.

Command line::

    python -m neural_scheduler.viz.timeline run_a/timeline.json:baseline \
        run_b/timeline.json:"with pauses" -o compare.html
    python -m neural_scheduler.viz.timeline --scan_dir experiment_dir
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from ..constants import TOKEN_RATE_HZ

# ---------------------------------------------------------------------------
# 1.  Export SchedulingState → JSON
# ---------------------------------------------------------------------------

def _placed_turn_to_dict(pt) -> Dict:
    """Convert a PlacedTurn (with nested TurnInfo) to a plain dict."""
    ti = pt.turn_info
    d = {
        'turn_id': ti.turn_id,
        'channel': ti.channel,
        'start_token': ti.start_token,
        'end_token': ti.end_token,
        'duration': ti.duration,
        'sub_segments': ti.sub_segments,
        'is_backchannel': ti.is_backchannel,
        'host_turn_id': ti.host_turn_id,
        'bc_group_index': ti.bc_group_index,
        'onset_step': pt.onset_step,
        'gap_to_previous': pt.gap_to_previous,
        'bc_offset_from_host_start': pt.bc_offset_from_host_start,
        # derived convenience fields
        'end_step': pt.onset_step + ti.duration,
    }
    return d


def _turn_info_to_dict(ti) -> Dict:
    """Convert a TurnInfo to a plain dict (no placement info)."""
    return {
        'turn_id': ti.turn_id,
        'channel': ti.channel,
        'start_token': ti.start_token,
        'end_token': ti.end_token,
        'duration': ti.duration,
        'sub_segments': ti.sub_segments,
        'is_backchannel': ti.is_backchannel,
        'host_turn_id': ti.host_turn_id,
        'bc_group_index': ti.bc_group_index,
    }


def _make_json_safe(obj):
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            # Convert tuple keys to strings
            key = str(k) if isinstance(k, tuple) else k
            safe[key] = _make_json_safe(v)
        return safe
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def export_scheduling_state_json(state, path: Optional[str] = None) -> str:
    """
    Serialize a SchedulingState object to a JSON string.

    Args:
        state: SchedulingState dataclass instance
        path:  If provided, also write to this file path.

    Returns:
        JSON string
    """
    data = {
        'meta': {
            'token_rate_hz': TOKEN_RATE_HZ,
        },

        # ---- Turn inventories (unplaced) ----
        'primary_turns': [_turn_info_to_dict(t) for t in state.primary_turns],
        'backchannel_turns': [_turn_info_to_dict(t) for t in state.backchannel_turns],
        'backchannel_units': [_turn_info_to_dict(t) for t in state.backchannel_units],

        # ---- Pass 1 placements ----
        'placed_primary': [_placed_turn_to_dict(pt) for pt in state.placed_primary],
        'pass1_nll_scores': _make_json_safe(state.pass1_nll_scores),
        'pass1_subturn_nll_scores': _make_json_safe(state.pass1_subturn_nll_scores),

        # ---- Pass 2 placements ----
        'placed_backchannels': [_placed_turn_to_dict(pt) for pt in state.placed_backchannels],
        'pass2_nll_scores': _make_json_safe(state.pass2_nll_scores),

        # ---- Pass 3 / final ----
        'refined_placements': [_placed_turn_to_dict(pt) for pt in state.refined_placements],
        'pass3_adjustments': _make_json_safe(state.pass3_adjustments),
        'pass3_sub_segment_details': _make_json_safe(state.pass3_sub_segment_details),

        # ---- Full diagnostics ----
        'diagnostics': _make_json_safe(state.diagnostics),
    }

    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(json_str)

    return json_str


# ---------------------------------------------------------------------------
# 2.  Multi-timeline comparison HTML visualizer
# ---------------------------------------------------------------------------

def visualize_timeline_comparison(
    state_paths_and_names: List[Tuple[str, str]],
    output_path: str = "timeline_comparison.html",
    token_rate_hz: float = TOKEN_RATE_HZ,
    max_label_chars: int = 30,
    metadata_overrides: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
):
    """
    Generate an interactive HTML page comparing final timelines from
    multiple SchedulingState JSON files.

    Args:
        state_paths_and_names: List of (json_path, display_name) tuples.
            e.g. [("run_a.json", "baseline"), ("run_b.json", "fp+hp only")]
        output_path: Where to write the HTML file.
        token_rate_hz: Token rate for converting steps → seconds.
        max_label_chars: Maximum characters for turn text labels before
            truncation with "...". Set to 0 to disable text labels and
            fall back to T0/BC0 style.
        metadata_overrides: Optional dict mapping json_path →
            (turn_metadata_path, bc_decisions_path). Any value can be None
            to let auto-discovery handle it. If None, every state uses
            auto-discovery.

    Returns:
        Path to the written HTML file.
    """
    # Load all states
    datasets = []
    for json_path, name in state_paths_and_names:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Resolve metadata paths (explicit override takes precedence)
        if metadata_overrides and json_path in metadata_overrides:
            meta_p, bc_p = metadata_overrides[json_path]
        else:
            meta_p, bc_p = (None, None)

        labels = _load_turn_labels(
            json_path,
            turn_meta_path=meta_p,
            bc_decisions_path=bc_p,
            max_chars=max_label_chars if max_label_chars > 0 else 20,
        ) if max_label_chars > 0 else {'primary': {}, 'bc': {}}

        datasets.append({'name': name, 'data': data, 'labels': labels})

    datasets_json = json.dumps(datasets, ensure_ascii=False)

    html = _COMPARISON_HTML_TEMPLATE.replace('__DATASETS_JSON__', datasets_json)
    html = html.replace('__TOKEN_RATE_HZ__', str(token_rate_hz))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# Shared helper: compute per-sub-segment onset/end in token steps
# ---------------------------------------------------------------------------

def _truncate_label(text: str, max_chars: int = 20) -> str:
    """Collapse whitespace and truncate a long string with '...'."""
    if not text:
        return ""
    # Collapse runs of whitespace (incl. newlines) and strip quotes/braces
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    # Truncate on a word boundary if possible, else hard cut
    cut = cleaned[:max_chars].rstrip()
    # Don't leave a partial word if we can avoid it
    last_space = cut.rfind(" ")
    if last_space >= max_chars - 6:  # keep the cut if it's close to the end
        cut = cut[:last_space]
    return cut.rstrip(",.;: ") + "..."


def _discover_metadata_paths(timeline_json_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a path like
      <root>/<config>/.../sample_0/segment_search_v2/timeline.json
    try to find the corresponding
      <root>/<config>/.../segment_cache/example_0000/turn_metadata.json
      <root>/<config>/.../segment_cache/example_0000/bc_decisions.json

    Returns (turn_meta_path or None, bc_decisions_path or None).
    """
    if not timeline_json_path or not os.path.isfile(timeline_json_path):
        return (None, None)

    timeline_dir = os.path.dirname(os.path.abspath(timeline_json_path))

    # Canonical relative path: go up to the checkpoint dir (parent of sample_N
    # and segment_cache), then descend into segment_cache/example_*.
    # We try a couple of levels up in case the layout varies.
    def _scan_for_cache(base: str) -> Tuple[Optional[str], Optional[str]]:
        segment_cache = os.path.join(base, 'segment_cache')
        if not os.path.isdir(segment_cache):
            return (None, None)
        # Pick the first example_* that contains the files (usually example_0000)
        try:
            examples = sorted(
                d for d in os.listdir(segment_cache)
                if d.startswith('example_') and
                   os.path.isdir(os.path.join(segment_cache, d))
            )
        except OSError:
            return (None, None)
        for ex in examples:
            meta = os.path.join(segment_cache, ex, 'turn_metadata.json')
            bc = os.path.join(segment_cache, ex, 'bc_decisions.json')
            if os.path.isfile(meta) or os.path.isfile(bc):
                return (
                    meta if os.path.isfile(meta) else None,
                    bc if os.path.isfile(bc) else None,
                )
        return (None, None)

    # Walk up to three levels looking for a sibling segment_cache dir
    current = timeline_dir
    for _ in range(4):
        found = _scan_for_cache(current)
        if found != (None, None):
            return found
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return (None, None)


def _load_turn_labels(
    timeline_json_path: str,
    turn_meta_path: Optional[str] = None,
    bc_decisions_path: Optional[str] = None,
    max_chars: int = 20,
) -> Dict[str, Dict[int, str]]:
    """
    Load per-turn text labels from turn_metadata.json and bc_decisions.json.
    Any missing file is silently skipped and the corresponding label map
    will be empty.

    If paths are not provided, tries to auto-discover them as siblings of
    the timeline json (see _discover_metadata_paths).

    Returns:
        {'primary': {turn_id: truncated_text},
         'bc':      {turn_id: truncated_text}}
    """
    if turn_meta_path is None or bc_decisions_path is None:
        disc_meta, disc_bc = _discover_metadata_paths(timeline_json_path)
        turn_meta_path = turn_meta_path or disc_meta
        bc_decisions_path = bc_decisions_path or disc_bc

    labels: Dict[str, Dict[int, str]] = {'primary': {}, 'bc': {}}

    if turn_meta_path and os.path.isfile(turn_meta_path):
        try:
            with open(turn_meta_path, 'r') as f:
                entries = json.load(f)
            for e in entries:
                tid = e.get('turn_id')
                text = e.get('text', '')
                if tid is None or not text:
                    continue
                labels['primary'][int(tid)] = _truncate_label(text, max_chars)
        except (OSError, json.JSONDecodeError):
            pass

    if bc_decisions_path and os.path.isfile(bc_decisions_path):
        try:
            with open(bc_decisions_path, 'r') as f:
                entries = json.load(f)
            for e in entries:
                tid = e.get('turn_id')
                word = e.get('bc_word')
                if tid is None or not word:
                    continue
                labels['bc'][int(tid)] = _truncate_label(word, max_chars)
        except (OSError, json.JSONDecodeError):
            pass

    return labels


# ---------------------------------------------------------------------------
# Shared helper: compute per-sub-segment onset/end in token steps
# ---------------------------------------------------------------------------

def _compute_sub_segment_spans(placement: Dict, token_rate_hz: float) -> List[Tuple[int, int]]:
    """
    For a placed turn with sub_segments metadata, compute one (onset, end)
    pair per sub-segment in absolute token steps, preserving the silence
    gaps between sub-segments as encoded in the ground-truth timing.

    Strategy:
      - First sub-seg starts at placement.onset_step.
      - Each subsequent sub-seg's gap = round((this.start - prev.end) * rate).
      - Each sub-seg's token duration is allocated proportionally by its
        second-span so the sub-seg durations sum exactly to
        placement.duration - sum(gap_steps). This mirrors how the scheduling
        code re-allocates token counts across sub-segments.

    Returns:
      List of (onset_step, end_step) tuples. Empty list when the turn has
      no sub_segments metadata (caller should fall back to the single-block
      representation).
    """
    sub_segs = placement.get('sub_segments') or []
    if len(sub_segs) <= 1:
        return []

    onset = placement['onset_step']
    total_dur = placement['duration']

    # Durations of each sub-segment in seconds (from metadata)
    sec_durs = [max(0.0, ss['end'] - ss['start']) for ss in sub_segs]
    sec_total = sum(sec_durs)

    # Inter-sub-seg gaps in token steps
    gap_steps = []
    for i in range(1, len(sub_segs)):
        gap_sec = sub_segs[i]['start'] - sub_segs[i - 1]['end']
        gap_steps.append(max(0, int(round(gap_sec * token_rate_hz))))

    speech_budget = total_dur - sum(gap_steps)
    if speech_budget < 0 or sec_total <= 0:
        # Degenerate: fall back to equal partitioning of total_dur, no gaps
        n = len(sub_segs)
        base = total_dur // n
        spans = []
        cur = onset
        for _ in range(n):
            spans.append((cur, cur + base))
            cur += base
        return spans

    # Proportional allocation of speech tokens per sub-seg (by seconds)
    raw = [d / sec_total * speech_budget for d in sec_durs]
    int_counts = [int(round(x)) for x in raw]
    diff = speech_budget - sum(int_counts)
    if diff != 0:
        # Adjust the largest-duration sub-seg to fix the rounding remainder
        idx = max(range(len(int_counts)), key=lambda i: sec_durs[i])
        int_counts[idx] += diff

    spans = []
    cur = onset
    for i, ss_dur in enumerate(int_counts):
        spans.append((cur, cur + ss_dur))
        cur += ss_dur
        if i < len(gap_steps):
            cur += gap_steps[i]
    return spans


# ---------------------------------------------------------------------------
# 2b. Static PNG renderer (publication quality, matplotlib)
# ---------------------------------------------------------------------------

def render_timeline_comparison_png(
    state_paths_and_names: List[Tuple[str, str]],
    output_path: str = "timeline_comparison.png",
    token_rate_hz: float = TOKEN_RATE_HZ,
    fig_width: float = 3.4,            # single column in a double-column paper
    row_height: float = 0.3,
    dpi: int = 300,
    font_size: int = 7,                # matches VAD script at ~3.5in width
    show_labels: bool = True,
    show_primary_labels: bool = False,     # labels inside primary turn bars
    show_bc_labels: bool = True,           # labels below BC bars
    show_legend: bool = True,         # off by default (was True)
    time_unit: str = "seconds",
    max_label_chars: int = 20,
    metadata_overrides: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
):
    """
    Render a publication-quality PNG comparing final timelines from
    multiple SchedulingState JSON files.

    Style matches the VAD activity figures: larger text, tight row
    height, minimal decoration (no title, no left spine, suppressed
    y-tick labels, light grid on x only).

    Args:
        state_paths_and_names: List of (json_path, display_name) tuples.
        output_path: Where to write the PNG (or PDF/SVG — matplotlib infers
                     from extension).
        token_rate_hz: Token rate for step → second conversion.
        fig_width: Figure width in inches.
        row_height: Height per channel row in inches.
        dpi: Resolution.
        font_size: Base font size (pt).
        show_labels: Whether to draw turn-ID labels inside blocks.
        show_legend: Whether to draw a legend.
        time_unit: "seconds" or "steps".
        max_label_chars: Maximum characters for turn text labels before
                     truncation with "...". Set to 0 to disable text labels
                     and fall back to T0/BC0 style.
        metadata_overrides: Optional dict mapping json_path →
                     (turn_metadata_path, bc_decisions_path) to override
                     auto-discovery.

    Returns:
        Path to the written image file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.size":        font_size,
        "axes.titlesize":   font_size,
        "axes.labelsize":   font_size,
        "xtick.labelsize":  font_size - 1,
        "ytick.labelsize":  font_size - 1,
        "legend.fontsize":  font_size - 2,
        "font.family":      "sans-serif",
    })

    # ── Load data ──
    datasets = []
    for json_path, name in state_paths_and_names:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if metadata_overrides and json_path in metadata_overrides:
            meta_p, bc_p = metadata_overrides[json_path]
        else:
            meta_p, bc_p = (None, None)

        labels = _load_turn_labels(
            json_path,
            turn_meta_path=meta_p,
            bc_decisions_path=bc_p,
            max_chars=max_label_chars if max_label_chars > 0 else 20,
        ) if max_label_chars > 0 else {'primary': {}, 'bc': {}}

        datasets.append({'name': name, 'data': data, 'labels': labels})

    n_states = len(datasets)

    # ── Conversion helper ──
    def to_time(steps):
        if time_unit == "seconds":
            return steps / token_rate_hz
        return steps

    def time_label():
        return "Time (s)" if time_unit == "seconds" else "Time (steps)"

    # ── Colors (same palette as before) ──
    colors = {
        ('primary', 'A'): ('#4facfe', '#2b7abf', 0.85),
        ('primary', 'B'): ('#f77062', '#b84d42', 0.85),
        ('bc', 'A'):      ('#43e97b', '#2da85a', 0.70),
        ('bc', 'B'):      ('#f6d365', '#b8993a', 0.70),
    }

    # ── Layout ──
    # Each state = 2 channel rows; a narrow separator gap between states.
    rows_per_state = 2
    spacing = 0.5          # gap between states in row units (was 0.6)
    total_rows = n_states * rows_per_state + (n_states - 1) * spacing

    fig_height = max(1.2, total_rows * row_height + 0.7)

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.97, top=0.97, bottom=0.18)

    # Global time extent
    global_max = 0
    for ds in datasets:
        placements = ds['data'].get('refined_placements', [])
        if placements:
            global_max = max(global_max,
                             max(p['end_step'] for p in placements))

    bar_height = 0.7

    # ── Draw each state ──
    # y-tick positions and the state-name annotations are collected separately;
    # we do NOT use ax.set_yticklabels so there is no tick text on the y-axis.
    y_tick_positions = []
    ch_label_positions = []  # (y, ch_letter)

    for si, ds in enumerate(datasets):
        placements = ds['data'].get('refined_placements', [])
        base_y = total_rows - si * (rows_per_state + spacing) - 1

        y_a = base_y          # channel A row centre
        y_b = base_y - 1      # channel B row centre
        mid_y = (y_a + y_b) / 2

        y_tick_positions.append(mid_y)



        # Collect channel sub-labels
        ch_label_positions.append((y_a, 'A'))
        ch_label_positions.append((y_b, 'B'))

        bc_bar_height = bar_height * 0.5      # BC bars are shorter than primary

        for p in placements:
            ch = p['channel']
            is_bc = p.get('is_backchannel', False)
            kind = 'bc' if is_bc else 'primary'
            fill, edge, alpha = colors[(kind, ch)]

            y = y_a if ch == 'A' else y_b
            x_start = to_time(p['onset_step'])
            width = to_time(p['duration'])

            if is_bc:
                # Bar occupies the upper half of the row (above centre y)
                # so the lower half is free for the text label.
                bar_bottom = y + 0.02          # just above row centre
                rect = mpatches.FancyBboxPatch(
                    (x_start, bar_bottom),
                    width, bc_bar_height,
                    boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=0.04),
                    facecolor=fill, edgecolor=edge,
                    alpha=alpha, linewidth=0.5,
                )
                ax.add_patch(rect)

                # Text label in the lower half of the row
                if show_labels and show_bc_labels:
                    tid = p['turn_id']
                    text_label = (
                        ds.get('labels', {}).get('bc', {}).get(tid)
                        if max_label_chars > 0 else None
                    )
                    label = text_label or ''
                    if label:
                        ax.text(
                            x_start + width / 2,
                            y - 0.05,          # below row centre, inside lower half
                            label,
                            ha='center', va='top',
                            fontsize=font_size - 2,
                            color=edge,
                            clip_on=False,
                        )
            else:
                # Primary turn — full-height centered bar
                rect = mpatches.FancyBboxPatch(
                    (x_start, y - bar_height / 2),
                    width, bar_height,
                    boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=0.05),
                    facecolor=fill, edgecolor=edge,
                    alpha=alpha, linewidth=0.5,
                )
                ax.add_patch(rect)

                # Label inside the bar when it fits
                if show_labels and show_primary_labels and width > to_time(2):
                    tid = p['turn_id']
                    text_label = (
                        ds.get('labels', {}).get('primary', {}).get(tid)
                        if max_label_chars > 0 else None
                    )
                    short_label = f"T{tid}"

                    x_range = to_time(global_max + 5)
                    data_per_inch = x_range / max(fig_width, 0.1)
                    char_w_data = (font_size - 3) * 0.55 / 72.0 * data_per_inch
                    max_fit_chars = int(width / max(char_w_data, 1e-6))

                    if text_label and len(text_label) <= max_fit_chars:
                        label = text_label
                    elif len(short_label) <= max_fit_chars:
                        label = short_label
                    else:
                        label = None

                    if label:
                        ax.text(
                            x_start + width / 2, y,
                            label,
                            ha='center', va='center',
                            fontsize=font_size - 2,
                            fontfamily='monospace',
                            color=edge,
                            fontweight='bold',
                            clip_on=True,
                        )

        # Tiny italic A/B channel labels inside the left margin
        for ch_letter, y_pos in [('A', y_a), ('B', y_b)]:
            ax.text(
                to_time(0) - to_time(global_max) * 0.008, y_pos,
                ch_letter,
                ha='right', va='center',
                fontsize=font_size - 2,
                fontfamily='monospace',
                color='#888888',
                fontstyle='italic',
            )

    # ── Axes styling (VAD-script approach) ──
    ax.set_xlim(0, to_time(global_max + 5))
    ax.set_ylim(-1.2, total_rows + 0.3)

    ax.set_xlabel("")

    # Suppress y-axis ticks entirely — names are drawn as text annotations
    ax.set_yticks([])

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=False))
    ax.tick_params(axis='x', labelsize=font_size - 1)

    # Light horizontal separators between states (dashed, like VAD divider)
    for si in range(1, n_states):
        sep_y = total_rows - si * (rows_per_state + spacing) + spacing / 2 - 0.5
        ax.axhline(sep_y, color='#cccccc', linewidth=0.5, linestyle='--')

    # Spine style matching VAD script
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_edgecolor('#aaaaaa')
    ax.spines['bottom'].set_linewidth(0.6)

    # Grid on x only, behind everything
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#dddddd', linewidth=0.4, linestyle='--')

    # ── Legend (optional) ──
    if show_legend:
        legend_patches = [
            mpatches.Patch(facecolor=colors[('primary', 'A')][0],
                           edgecolor=colors[('primary', 'A')][1],
                           alpha=0.85, label='Primary A'),
            mpatches.Patch(facecolor=colors[('primary', 'B')][0],
                           edgecolor=colors[('primary', 'B')][1],
                           alpha=0.85, label='Primary B'),
            mpatches.Patch(facecolor=colors[('bc', 'A')][0],
                           edgecolor=colors[('bc', 'A')][1],
                           alpha=0.70, label='BC A'),
            mpatches.Patch(facecolor=colors[('bc', 'B')][0],
                           edgecolor=colors[('bc', 'B')][1],
                           alpha=0.70, label='BC B'),
        ]
        ax.legend(
            handles=legend_patches,
            loc='upper right',
            fontsize=font_size - 2,
            framealpha=0.8,
            edgecolor='#cccccc',
            ncol=4,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return output_path

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_COMPARISON_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Timeline Comparison</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #0e0f14;
    --surface: #16171e;
    --surface-2: #1e2028;
    --border: #2a2c36;
    --text: #e2e4ec;
    --text-dim: #8b8fa4;
    --accent: #6c8aff;
    --ch-a: #4facfe;
    --ch-a-bg: rgba(79, 172, 254, 0.18);
    --ch-a-border: rgba(79, 172, 254, 0.4);
    --ch-b: #f77062;
    --ch-b-bg: rgba(247, 112, 98, 0.18);
    --ch-b-border: rgba(247, 112, 98, 0.4);
    --bc-a: #43e97b;
    --bc-a-bg: rgba(67, 233, 123, 0.18);
    --bc-a-border: rgba(67, 233, 123, 0.4);
    --bc-b: #f6d365;
    --bc-b-bg: rgba(246, 211, 101, 0.18);
    --bc-b-border: rgba(246, 211, 101, 0.4);
    --silence: rgba(255,255,255,0.04);
    --font: 'DM Sans', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    padding: 24px 32px;
    min-height: 100vh;
  }

  h1 {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
  }

  .subtitle {
    color: var(--text-dim);
    font-size: 13px;
    margin-bottom: 28px;
  }

  /* ── Controls ── */
  .controls {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .controls label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .controls select, .controls input[type=range] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 6px;
    outline: none;
  }
  .controls input[type=range] {
    width: 160px;
    accent-color: var(--accent);
  }
  .zoom-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    min-width: 36px;
  }

  /* ── Legend ── */
  .legend {
    display: flex;
    gap: 20px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-dim);
  }
  .legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }

  /* ── Timeline rows ── */
  .state-group {
    margin-bottom: 28px;
  }
  .state-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 6px;
  }
  .state-name {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  .state-stats {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  .timeline-container {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
  }

  .timeline-canvas {
    position: relative;
    min-height: 100px;
  }

  .channel-row {
    position: relative;
    height: 38px;
    border-bottom: 1px solid var(--border);
  }
  .channel-row:last-child { border-bottom: none; }

  .channel-label {
    position: absolute;
    left: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--text-dim);
    z-index: 5;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  /* ── Turn blocks ── */
  .turn-block {
    position: absolute;
    top: 4px;
    height: 30px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    cursor: pointer;
    transition: filter 0.15s, transform 0.15s;
    border: 1px solid;
    overflow: hidden;
    white-space: nowrap;
    min-width: 3px;
  }
  .turn-block:hover {
    filter: brightness(1.3);
    transform: scaleY(1.12);
    z-index: 20;
  }
  .turn-block.primary.ch-A {
    background: var(--ch-a-bg);
    border-color: var(--ch-a-border);
    color: var(--ch-a);
  }
  .turn-block.primary.ch-B {
    background: var(--ch-b-bg);
    border-color: var(--ch-b-border);
    color: var(--ch-b);
  }
  .turn-block.bc.ch-A {
    background: var(--bc-a-bg);
    border-color: var(--bc-a-border);
    color: var(--bc-a);
  }
  .turn-block.bc.ch-B {
    background: var(--bc-b-bg);
    border-color: var(--bc-b-border);
    color: var(--bc-b);
  }

  /* Faint outline strip spanning the full turn extent, drawn behind
     solid sub-segment blocks so inter-sub-seg silences look like
     translucent connector tissue rather than empty gaps. */
  .turn-outline {
    position: absolute;
    top: 4px;
    height: 30px;
    border-radius: 4px;
    border: 1px dashed;
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
  }
  .turn-outline.primary.ch-A {
    background: var(--ch-a-bg);
    border-color: var(--ch-a-border);
  }
  .turn-outline.primary.ch-B {
    background: var(--ch-b-bg);
    border-color: var(--ch-b-border);
  }
  /* Sub-seg blocks sit on top of the outline */
  .turn-block { z-index: 1; }

  /* ── Time axis ── */
  .time-axis {
    position: relative;
    height: 22px;
    border-top: 1px solid var(--border);
  }
  .time-tick {
    position: absolute;
    top: 0;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-dim);
    transform: translateX(-50%);
    padding-top: 4px;
  }
  .time-tick::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    width: 1px;
    height: 4px;
    background: var(--border);
  }

  /* ── Tooltip ── */
  .tooltip {
    display: none;
    position: fixed;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    z-index: 100;
    pointer-events: none;
    max-width: 320px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .tooltip.visible { display: block; }
  .tooltip-title {
    font-weight: 700;
    margin-bottom: 4px;
    font-size: 13px;
  }
  .tooltip-row {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.6;
  }
  .tooltip-key { color: var(--text-dim); }
  .tooltip-val { color: var(--text); font-weight: 500; }

  /* ── Summary table ── */
  .summary-section { margin-top: 40px; }
  .summary-section h2 {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
  }
  table.summary {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: var(--mono);
  }
  table.summary th, table.summary td {
    padding: 8px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  table.summary th {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: var(--surface);
  }
  table.summary td { color: var(--text); }
  table.summary tr:hover td { background: var(--surface-2); }
</style>
</head>
<body>

<h1>Timeline Comparison</h1>
<p class="subtitle">Final placements (after Pass 3 refinement) across scheduling runs</p>

<div class="controls">
  <label>Zoom</label>
  <input type="range" id="zoomSlider" min="1" max="20" step="0.5" value="6"/>
  <span class="zoom-label" id="zoomLabel">6 px/step</span>

  <label style="margin-left:20px">Unit</label>
  <select id="unitSelect">
    <option value="steps">Steps</option>
    <option value="seconds" selected>Seconds</option>
  </select>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:var(--ch-a-bg);border:1px solid var(--ch-a-border)"></div>Primary A</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--ch-b-bg);border:1px solid var(--ch-b-border)"></div>Primary B</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--bc-a-bg);border:1px solid var(--bc-a-border)"></div>BC A</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--bc-b-bg);border:1px solid var(--bc-b-border)"></div>BC B</div>
</div>

<div id="timelines"></div>
<div id="summarySection" class="summary-section"></div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATASETS = __DATASETS_JSON__;
const TOKEN_RATE = __TOKEN_RATE_HZ__;
const LEFT_PAD = 36;  // px, room for channel labels

let pxPerStep = 6;
let unit = 'seconds';

const $timelines = document.getElementById('timelines');
const $tooltip   = document.getElementById('tooltip');
const $zoom      = document.getElementById('zoomSlider');
const $zoomLabel = document.getElementById('zoomLabel');
const $unitSel   = document.getElementById('unitSelect');

function fmt(val) {
  if (unit === 'seconds') return (val / TOKEN_RATE).toFixed(2) + 's';
  return val + ' st';
}

function buildTimeline(ds, idx) {
  const placements = ds.data.refined_placements || [];
  if (!placements.length) return '';

  const maxEnd = Math.max(...placements.map(p => p.end_step));
  const totalWidth = LEFT_PAD + maxEnd * pxPerStep + 40;

  // separate by channel
  const chA = placements.filter(p => p.channel === 'A');
  const chB = placements.filter(p => p.channel === 'B');

  // stats
  const totalDur = maxEnd;
  const nPrimary = placements.filter(p => !p.is_backchannel).length;
  const nBC = placements.filter(p => p.is_backchannel).length;

  // pass2 diagnostics
  const pass2Diag = ds.data.diagnostics?.pass2?.per_backchannel || [];
  const strategyMap = {};
  pass2Diag.forEach(d => {
    const key = d.turn_id + (d.bc_group_index != null ? '.' + d.bc_group_index : '');
    strategyMap[key] = d.best_strategy || 'unknown';
  });

  function computeSubSegSpans(p) {
    // Mirror the Python _compute_sub_segment_spans helper. Returns a list
    // of {onset, end} token-step spans, one per sub-segment. Returns an
    // empty array when the turn has 0-1 sub-segments (caller falls back
    // to a single block).
    const subs = p.sub_segments || [];
    if (subs.length <= 1) return [];

    const onset = p.onset_step;
    const totalDur = p.duration;

    const secDurs = subs.map(s => Math.max(0, s.end - s.start));
    const secTotal = secDurs.reduce((a, b) => a + b, 0);

    const gapSteps = [];
    for (let i = 1; i < subs.length; i++) {
      const gapSec = subs[i].start - subs[i - 1].end;
      gapSteps.push(Math.max(0, Math.round(gapSec * TOKEN_RATE)));
    }

    const speechBudget = totalDur - gapSteps.reduce((a, b) => a + b, 0);
    if (speechBudget < 0 || secTotal <= 0) {
      // Degenerate: fall back to equal partition, no gaps
      const n = subs.length;
      const base = Math.floor(totalDur / n);
      const out = [];
      let cur = onset;
      for (let i = 0; i < n; i++) {
        out.push({ onset: cur, end: cur + base });
        cur += base;
      }
      return out;
    }

    const raw = secDurs.map(d => d / secTotal * speechBudget);
    const ints = raw.map(x => Math.round(x));
    let diff = speechBudget - ints.reduce((a, b) => a + b, 0);
    if (diff !== 0) {
      let idx = 0, bestDur = -1;
      for (let i = 0; i < secDurs.length; i++) {
        if (secDurs[i] > bestDur) { bestDur = secDurs[i]; idx = i; }
      }
      ints[idx] += diff;
    }

    const spans = [];
    let cur = onset;
    for (let i = 0; i < ints.length; i++) {
      spans.push({ onset: cur, end: cur + ints[i] });
      cur += ints[i];
      if (i < gapSteps.length) cur += gapSteps[i];
    }
    return spans;
  }

  // Text labels loaded from turn_metadata.json / bc_decisions.json.
  // Keys are numeric turn_ids stringified by JSON.
  const primaryLabels = (ds.labels && ds.labels.primary) || {};
  const bcLabels      = (ds.labels && ds.labels.bc) || {};

  function renderBlock(p) {
    const cls = (p.is_backchannel ? 'bc' : 'primary') + ' ch-' + p.channel;

    // Short fallback label (T0 / BC0.0 style)
    const shortLabel = p.is_backchannel
      ? 'BC' + p.turn_id + (p.bc_group_index != null ? '.' + p.bc_group_index : '')
      : 'T' + p.turn_id;

    // Rich text label from metadata (may be undefined)
    const textLabel = p.is_backchannel
      ? bcLabels[p.turn_id]
      : primaryLabels[p.turn_id];

    const bcKey = p.turn_id + (p.bc_group_index != null ? '.' + p.bc_group_index : '');
    const strat = strategyMap[bcKey] || '';

    // Shared data attributes for every sub-segment of this turn
    const dataAttrs = `
      data-tid="${p.turn_id}"
      data-ch="${p.channel}"
      data-onset="${p.onset_step}"
      data-end="${p.end_step}"
      data-dur="${p.duration}"
      data-bc="${p.is_backchannel ? 1 : 0}"
      data-host="${p.host_turn_id || ''}"
      data-gap="${p.gap_to_previous != null ? p.gap_to_previous : ''}"
      data-bcoff="${p.bc_offset_from_host_start != null ? p.bc_offset_from_host_start : ''}"
      data-strat="${strat}"
      data-grp="${p.bc_group_index != null ? p.bc_group_index : ''}"
      data-text="${textLabel ? textLabel.replace(/"/g, '&quot;') : ''}"`;

    // Determine sub-segment spans. Backchannels always render as a single
    // block for clarity; only primary turns get split into sub-segments.
    let spans = p.is_backchannel ? [] : computeSubSegSpans(p);
    const hasSubSegs = spans.length > 0;
    if (!hasSubSegs) {
      spans = [{ onset: p.onset_step, end: p.onset_step + p.duration }];
    }

    // Identify which sub-seg's center is closest to the turn's midpoint;
    // that's where we'll draw the label (avoids duplicate labels).
    const turnMid = p.onset_step + p.duration / 2;
    let labelIdx = 0, bestDist = Infinity;
    for (let i = 0; i < spans.length; i++) {
      const segMid = (spans[i].onset + spans[i].end) / 2;
      const dist = Math.abs(segMid - turnMid);
      if (dist < bestDist) { bestDist = dist; labelIdx = i; }
    }

    // Optional outline strip spanning the full turn (primaries with
    // multiple sub-segments only).
    let outline = '';
    if (hasSubSegs) {
      const ox = LEFT_PAD + p.onset_step * pxPerStep;
      const ow = Math.max(p.duration * pxPerStep, 3);
      outline = `<div class="turn-outline ${cls}"
        style="left:${ox}px;width:${ow}px"></div>`;
    }

    // For label sizing, the "true" label width isn't a single sub-seg's
    // width but the span of the whole turn (since the label is centered
    // over the full extent and can visually cross sub-seg gaps). We pick
    // whichever label fits in that full extent.
    const fullTurnPx = Math.max(p.duration * pxPerStep, 3);
    // ~6.5 px per monospace char at 10px font
    const pxPerChar = 6.5;
    const fitsText = textLabel && (textLabel.length * pxPerChar <= fullTurnPx - 6);
    const fitsShort = shortLabel.length * pxPerChar <= fullTurnPx - 4;
    let chosenLabel = '';
    if (fitsText) chosenLabel = textLabel;
    else if (fitsShort) chosenLabel = shortLabel;

    const blocks = spans.map((s, i) => {
      const left = LEFT_PAD + s.onset * pxPerStep;
      const width = Math.max((s.end - s.onset) * pxPerStep, 3);
      const showLbl = (i === labelIdx) && chosenLabel;
      return `<div class="turn-block ${cls}"
        style="left:${left}px;width:${width}px"
        ${dataAttrs}
      >${showLbl ? chosenLabel : ''}</div>`;
    }).join('');

    return outline + blocks;
  }

  // Time ticks
  const tickInterval = unit === 'seconds'
    ? Math.ceil(TOKEN_RATE)   // ~1 sec worth of steps
    : Math.max(1, Math.round(20 / pxPerStep) * 5);
  let ticks = '';
  for (let s = 0; s <= maxEnd; s += tickInterval) {
    const x = LEFT_PAD + s * pxPerStep;
    ticks += `<div class="time-tick" style="left:${x}px">${fmt(s)}</div>`;
  }

  return `
    <div class="state-group" data-idx="${idx}">
      <div class="state-header">
        <span class="state-name">${ds.name}</span>
        <span class="state-stats">${nPrimary} primary · ${nBC} BC · ${fmt(totalDur)} total</span>
      </div>
      <div class="timeline-container">
        <div class="timeline-canvas" style="width:${totalWidth}px">
          <div class="channel-row">
            <span class="channel-label">A</span>
            ${chA.map(renderBlock).join('')}
          </div>
          <div class="channel-row">
            <span class="channel-label">B</span>
            ${chB.map(renderBlock).join('')}
          </div>
          <div class="time-axis" style="width:${totalWidth}px">
            ${ticks}
          </div>
        </div>
      </div>
    </div>`;
}

function buildSummary() {
  let rows = DATASETS.map(ds => {
    const p = ds.data.refined_placements || [];
    const maxEnd = p.length ? Math.max(...p.map(x => x.end_step)) : 0;
    const nPri = p.filter(x => !x.is_backchannel).length;
    const nBC  = p.filter(x => x.is_backchannel).length;

    // gaps between consecutive same-channel primary turns
    const gaps = [];
    const byChannel = {};
    p.filter(x => !x.is_backchannel).forEach(x => {
      if (!byChannel[x.channel]) byChannel[x.channel] = [];
      byChannel[x.channel].push(x);
    });
    for (const ch of Object.keys(byChannel)) {
      const sorted = byChannel[ch].sort((a,b) => a.onset_step - b.onset_step);
      for (let i = 1; i < sorted.length; i++) {
        gaps.push(sorted[i].onset_step - sorted[i-1].end_step);
      }
    }
    const avgGap = gaps.length ? (gaps.reduce((a,b) => a+b, 0) / gaps.length) : 0;
    const maxGap = gaps.length ? Math.max(...gaps) : 0;
    const minGap = gaps.length ? Math.min(...gaps) : 0;

    // strategy distribution
    const pass2 = ds.data.diagnostics?.pass2?.per_backchannel || [];
    const stratCounts = {};
    pass2.forEach(d => {
      const s = d.best_strategy || 'none';
      stratCounts[s] = (stratCounts[s] || 0) + 1;
    });
    const stratStr = Object.entries(stratCounts).map(([k,v]) => `${k}:${v}`).join(' ');

    return `<tr>
      <td>${ds.name}</td>
      <td>${nPri}</td>
      <td>${nBC}</td>
      <td>${fmt(maxEnd)}</td>
      <td>${fmt(avgGap)}</td>
      <td>${fmt(minGap)}</td>
      <td>${fmt(maxGap)}</td>
      <td>${stratStr || '—'}</td>
    </tr>`;
  }).join('');

  return `<h2>Summary</h2>
    <table class="summary">
      <thead><tr>
        <th>Run</th><th>Primary</th><th>BCs</th><th>Total Dur</th>
        <th>Avg Gap</th><th>Min Gap</th><th>Max Gap</th><th>Strategies</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function render() {
  $timelines.innerHTML = DATASETS.map((ds, i) => buildTimeline(ds, i)).join('');
  document.getElementById('summarySection').innerHTML = buildSummary();
  attachTooltips();
}

function attachTooltips() {
  document.querySelectorAll('.turn-block').forEach(el => {
    el.addEventListener('mouseenter', e => {
      const d = el.dataset;
      const isBC = d.bc === '1';
      let title = isBC
        ? `BC Turn ${d.tid}${d.grp ? '.' + d.grp : ''}`
        : `Turn ${d.tid}`;
      title += ` (Ch ${d.ch})`;

      let rows = `
        <div class="tooltip-row"><span class="tooltip-key">Onset</span><span class="tooltip-val">${fmt(+d.onset)}</span></div>
        <div class="tooltip-row"><span class="tooltip-key">End</span><span class="tooltip-val">${fmt(+d.end)}</span></div>
        <div class="tooltip-row"><span class="tooltip-key">Duration</span><span class="tooltip-val">${fmt(+d.dur)}</span></div>`;

      if (d.text) {
        rows += `<div class="tooltip-row"><span class="tooltip-key">Text</span><span class="tooltip-val">${d.text}</span></div>`;
      }
      if (d.gap !== '') {
        rows += `<div class="tooltip-row"><span class="tooltip-key">Gap to prev</span><span class="tooltip-val">${fmt(+d.gap)}</span></div>`;
      }
      if (isBC && d.host) {
        rows += `<div class="tooltip-row"><span class="tooltip-key">Host turn</span><span class="tooltip-val">T${d.host}</span></div>`;
      }
      if (isBC && d.bcoff !== '') {
        rows += `<div class="tooltip-row"><span class="tooltip-key">BC offset</span><span class="tooltip-val">${fmt(+d.bcoff)}</span></div>`;
      }
      if (isBC && d.strat) {
        rows += `<div class="tooltip-row"><span class="tooltip-key">Strategy</span><span class="tooltip-val">${d.strat}</span></div>`;
      }

      $tooltip.innerHTML = `<div class="tooltip-title">${title}</div>${rows}`;
      $tooltip.classList.add('visible');
    });

    el.addEventListener('mousemove', e => {
      $tooltip.style.left = (e.clientX + 14) + 'px';
      $tooltip.style.top  = (e.clientY + 14) + 'px';
    });

    el.addEventListener('mouseleave', () => {
      $tooltip.classList.remove('visible');
    });
  });
}

// ── Events ──
$zoom.addEventListener('input', () => {
  pxPerStep = parseFloat($zoom.value);
  $zoomLabel.textContent = pxPerStep + ' px/step';
  render();
});

$unitSel.addEventListener('change', () => {
  unit = $unitSel.value;
  render();
});

render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def find_timelines(top_dir: str) -> List[Tuple[str, str]]:
    """Find every ``timeline.json`` under ``top_dir``.

    Returns sorted ``(path, label)`` pairs where the label is the first
    directory component below ``top_dir`` (the configuration name).
    """
    top_dir = top_dir.rstrip(os.sep)
    found = []
    for root, _dirs, files in os.walk(top_dir):
        if 'timeline.json' in files:
            found.append(os.path.join(root, 'timeline.json'))
    pairs = []
    for json_path in sorted(found):
        rel_path = os.path.relpath(json_path, top_dir)
        pairs.append((json_path, rel_path.split(os.sep)[0]))
    return pairs


def parse_input_spec(inp: str) -> Tuple[str, str, Optional[Tuple[Optional[str], Optional[str]]]]:
    """Parse ``path``, ``path:name`` or ``path:name:turn_metadata:bc_decisions``.

    The four-part form is only accepted when parts three and four are existing
    files (or empty), so display names containing colons still parse as the
    two-part form. Returns ``(path, name, metadata_override_or_None)``.
    """
    parts = inp.split(':')
    if len(parts) == 4:
        meta_candidate, bc_candidate = parts[2], parts[3]
        both_empty = (meta_candidate == '' and bc_candidate == '')
        meta_ok = (meta_candidate == '' or os.path.isfile(meta_candidate))
        bc_ok = (bc_candidate == '' or os.path.isfile(bc_candidate))
        if not both_empty and meta_ok and bc_ok:
            path, name, meta_p, bc_p = parts
            return path, name, (meta_p if meta_p else None, bc_p if bc_p else None)
    if ':' in inp:
        path, name = inp.split(':', 1)
    else:
        path = inp
        name = os.path.splitext(os.path.basename(path))[0]
    return path, name, None


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare scheduling timelines (timeline.json files).',
        epilog=(
            'Input format:  path:name  or  path:name:turn_metadata:bc_decisions\n'
            'Examples:\n'
            '  python -m neural_scheduler.viz.timeline '
            'run_a.json:baseline run_b.json:"fp+hp only" -o compare.html\n'
            '  python -m neural_scheduler.viz.timeline '
            'run_a.json:baseline:meta.json:bcs.json -o compare.html\n'
            '  python -m neural_scheduler.viz.timeline --scan_dir experiment_dir\n'
            'If only path:name is given, turn_metadata.json and bc_decisions.json '
            'are auto-discovered by walking up from timeline.json looking for a '
            'sibling segment_cache/example_*/ directory.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'inputs', nargs='*',
        help='JSON files as path:name or path:name:turn_metadata:bc_decisions. '
             'e.g. run_a.json:baseline  run_b.json:"fp only"  '
             'run_c.json:salient:meta.json:bc.json'
    )
    parser.add_argument('--scan_dir', default=None,
                        help='Compare every timeline.json found under this directory, '
                             'labelled by its first sub-directory. Output defaults to '
                             '<scan_dir>/timeline_comparison.html and --fig-width to 3.5.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML path (required unless --scan_dir is given)')
    parser.add_argument('--png', default=None,
                        help='Output PNG path. If omitted, derived from --output '
                             'by replacing extension with .png')
    parser.add_argument('--no-png', action='store_true',
                        help='Skip PNG generation')
    parser.add_argument('--dpi', type=int, default=300,
                        help='PNG resolution (default: 300)')
    parser.add_argument('--fig-width', type=float, default=None,
                        help='PNG figure width in inches (default: 3.6, or 3.5 with --scan_dir)')
    parser.add_argument('--max-label-chars', type=int, default=30,
                        help='Maximum characters for turn text labels before '
                             'truncation (default: 30). 0 disables text labels '
                             'and uses T0/BC0 style.')

    args = parser.parse_args(argv)

    pairs: List[Tuple[str, str]] = []
    metadata_overrides: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for inp in args.inputs:
        path, name, override = parse_input_spec(inp)
        if override is not None:
            metadata_overrides[path] = override
        pairs.append((path, name))

    fig_width = args.fig_width
    output = args.output
    if args.scan_dir:
        scanned = find_timelines(args.scan_dir)
        if not scanned:
            parser.error(f'no timeline.json files found under {args.scan_dir}')
        print(f"Found {len(scanned)} timeline(s):")
        for json_path, label in scanned:
            print(f"  {json_path}  ->  {label}")
        pairs.extend(scanned)
        if output is None:
            output = os.path.join(args.scan_dir.rstrip(os.sep), 'timeline_comparison.html')
        if fig_width is None:
            fig_width = 3.5
    if not pairs:
        parser.error('give at least one input or --scan_dir')
    if output is None:
        parser.error('-o/--output is required unless --scan_dir is given')
    if fig_width is None:
        fig_width = 3.6

    out_html = visualize_timeline_comparison(
        pairs, output,
        max_label_chars=args.max_label_chars,
        metadata_overrides=metadata_overrides or None,
    )
    print(f"HTML written to {out_html}")

    if not args.no_png:
        png_path = args.png
        if png_path is None:
            base, _ = os.path.splitext(output)
            png_path = base + '.png'
        out_png = render_timeline_comparison_png(
            pairs, png_path, dpi=args.dpi, fig_width=fig_width,
            max_label_chars=args.max_label_chars,
            metadata_overrides=metadata_overrides or None,
        )
        print(f"PNG written to {out_png}")


if __name__ == '__main__':
    main()
