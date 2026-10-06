"""Diagnostic plots for a three-pass scheduling result.

All plots are drawn from a ``SchedulingState`` alone (no ground-truth
dialogue is needed)::

    from neural_scheduler.viz.three_pass import visualize_scheduling
    visualize_scheduling(state, save_dir="./plots")

writes ``timeline.png``, ``pass1_nll.png``, ``pass1_shifts.png``,
``pass2_nll.png``, ``pass2_strategies.png``, ``pass3_nll.png`` and
``pass3_summary.png``. The individual ``plot_*`` functions can be called
separately.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple

from ..constants import TOKEN_RATE_HZ

# ── Colour palette ──────────────────────────────────────────────────────────
CH_A_COLOR      = "#4C72B0"   # blue
CH_A_COLOR_LT   = "#A8C4E0"
CH_B_COLOR      = "#DD8452"   # orange
CH_B_COLOR_LT   = "#F2CBB0"
BC_COLOR        = "#55A868"   # green
BC_COLOR_LT     = "#B5DFC3"
ANCHOR_COLOR    = "#8C8C8C"
BEST_COLOR      = "#C44E52"   # red
NEXT_TURN_COLOR = "#9B59B6"   # purple — for next_turn breakpoints
STRATEGY_COLORS = {
    "no_pause":   "#4C72B0",
    "full_pause": "#DD8452",
    "half_pause": "#55A868",
}


# ============================================================================
# Helper: seconds string
# ============================================================================

def _steps_to_sec(steps: int) -> float:
    return steps / TOKEN_RATE_HZ

def _sec_label(steps: int) -> str:
    return f"{_steps_to_sec(steps):.2f}s"


def _bp_label(bp: Dict) -> str:
    """Generate a human-readable label for a breakpoint diagnostic dict."""
    anchor_type = bp.get("anchor_type", "subseg")
    if anchor_type == "subseg":
        return f"SS[{bp['sub_segment_index']}]"
    elif anchor_type == "next_turn":
        return f"→T{bp['next_turn_id']}"
    else:
        return f"BP[{bp.get('breakpoint_order', '?')}]"


# ============================================================================
# 1. Timeline overview  (one row per pass)
# ============================================================================

def plot_timeline(state, save_path: Optional[str] = None, show: bool = True):
    """
    Three-row timeline showing placements after each pass.

    Row 0 — Pass 1: primary turns only
    Row 1 — Pass 2: primary turns + backchannels
    Row 2 — Pass 3: refined placements
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                             gridspec_kw={"hspace": 0.35})

    titles = ["Pass 1 — Primary Turns", "Pass 2 — + Backchannels", "Pass 3 — Refined"]

    datasets = [
        (state.placed_primary, []),
        (state.placed_primary, state.placed_backchannels),
        # Pass 3: refined_placements contains everything
        ([pt for pt in state.refined_placements if not pt.turn_info.is_backchannel],
         [pt for pt in state.refined_placements if pt.turn_info.is_backchannel]),
    ]

    for ax, title, (primaries, bcs) in zip(axes, titles, datasets):
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        _draw_timeline_row(ax, primaries, bcs)

    # shared x-axis label
    axes[-1].set_xlabel("Time (tokens)", fontsize=10)

    # legend
    handles = [
        mpatches.Patch(facecolor=CH_A_COLOR, edgecolor="k", linewidth=0.5, label="Channel A"),
        mpatches.Patch(facecolor=CH_B_COLOR, edgecolor="k", linewidth=0.5, label="Channel B"),
        mpatches.Patch(facecolor=BC_COLOR,   edgecolor="k", linewidth=0.5, label="Backchannel"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Three-Pass Scheduling Timeline", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _draw_timeline_row(ax, primaries, backchannels):
    """Draw rectangles for turns on a single axis."""
    y_map = {"A": 1.0, "B": 0.0}
    bar_height = 0.6

    for pt in primaries:
        ch = pt.turn_info.channel
        y = y_map.get(ch, 0.5)
        color = CH_A_COLOR if ch == "A" else CH_B_COLOR
        start_tok = pt.onset_step
        dur_tok = pt.turn_info.duration
        rect = plt.Rectangle(
            (start_tok, y - bar_height / 2), dur_tok, bar_height,
            facecolor=color, edgecolor="k",
            linewidth=0.5, alpha=0.85,
        )
        ax.add_patch(rect)
        # Turn label
        if dur_tok > 3:
            ax.text(start_tok + dur_tok / 2, y, f"T{pt.turn_info.turn_id}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")

    for pt in backchannels:
        ch = pt.turn_info.channel
        y = y_map.get(ch, 0.5)
        start_tok = pt.onset_step
        dur_tok = pt.turn_info.duration
        rect = plt.Rectangle(
            (start_tok, y - bar_height / 2), dur_tok, bar_height,
            facecolor=BC_COLOR, edgecolor="k",
            linewidth=0.5, alpha=0.85,
        )
        ax.add_patch(rect)
        group_str = ""
        if hasattr(pt.turn_info, "bc_group_index") and pt.turn_info.bc_group_index is not None:
            group_str = f".{pt.turn_info.bc_group_index}"
        if dur_tok > 2:
            ax.text(start_tok + dur_tok / 2, y, f"BC{pt.turn_info.turn_id}{group_str}",
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold")

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Ch B", "Ch A"], fontsize=9)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    # Auto x-limit
    all_pts = list(primaries) + list(backchannels)
    if all_pts:
        max_end = max(pt.onset_step + pt.turn_info.duration for pt in all_pts)
        ax.set_xlim(-2, max_end + 6)


# ============================================================================
# 2. Pass 1 — NLL curves per turn
# ============================================================================

def plot_pass1_nll_curves(state, save_path: Optional[str] = None, show: bool = True):
    """
    One subplot per primary turn showing NLL vs candidate onset.
    Marks the anchor (grey dashed) and winning onset (red dot).
    """
    diag = state.diagnostics.get("pass1", {})
    per_turn = diag.get("per_turn", [])

    if not per_turn:
        print("No Pass 1 diagnostics available.")
        return None

    n = len(per_turn)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

    for idx, td in enumerate(per_turn):
        ax = axes[idx // cols][idx % cols]
        candidates = td["candidates"]
        nll_scores = td["nll_scores"]
        anchor = td["script_onset"]
        best = td["best_onset"]
        ch = td["channel"]

        color = CH_A_COLOR if ch == "A" else CH_B_COLOR

        ax.plot(candidates, nll_scores, color=color, linewidth=1.5, alpha=0.9)
        ax.axvline(anchor, color=ANCHOR_COLOR, linestyle="--", linewidth=1,
                   label=f"Anchor {anchor} tok")

        best_idx = candidates.index(best) if best in candidates else 0
        ax.plot(best, nll_scores[best_idx], "o", color=BEST_COLOR, markersize=8,
                zorder=5, label=f"Best {best} tok")

        ax.set_title(f"Turn {td['turn_id']}  (ch {ch}, dur={td['duration']})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Onset (tokens)", fontsize=9)
        ax.set_ylabel("NLL", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Pass 1 — NLL vs Candidate Onset (Primary Turns)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 3. Pass 1 — Shift summary bar chart
# ============================================================================

def plot_pass1_shifts(state, save_path: Optional[str] = None, show: bool = True):
    """Horizontal bar chart showing how much each primary turn shifted from its anchor."""
    diag = state.diagnostics.get("pass1", {})
    per_turn = diag.get("per_turn", [])

    if not per_turn:
        print("No Pass 1 diagnostics available.")
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(per_turn))))

    labels = []
    shifts = []
    colors = []
    for td in per_turn:
        labels.append(f"T{td['turn_id']} ({td['channel']})")
        shifts.append(td["shift"])
        colors.append(CH_A_COLOR if td["channel"] == "A" else CH_B_COLOR)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, shifts, color=colors, edgecolor="k", linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Shift from anchor (steps)", fontsize=10)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    # Add step values as text
    for i, s in enumerate(shifts):
        ha = "left" if s >= 0 else "right"
        offset = 0.3 if s >= 0 else -0.3
        ax.text(s + offset, i, f"{s:+d}", va="center", ha=ha, fontsize=8)

    ax.set_title("Pass 1 — Onset Shift from Anchor", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 4. Pass 2 — NLL-difference curves per backchannel (multi-strategy)
# ============================================================================

def plot_pass2_nll_curves(state, save_path: Optional[str] = None, show: bool = True):
    """
    One subplot per backchannel unit.
    Shows NLL delta (vs baseline) for each strategy across candidate onsets.
    Marks winning (strategy, onset) with a red star.
    Also draws the host turn span as a shaded background.
    """
    diag = state.diagnostics.get("pass2", {})
    per_bc = diag.get("per_backchannel", [])

    if not per_bc:
        print("No Pass 2 diagnostics available.")
        return None

    n = len(per_bc)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    for idx, bd in enumerate(per_bc):
        ax = axes[idx // cols][idx % cols]

        candidates = bd["candidates"]
        cand_sec = candidates
        host_start_sec = bd["host_start"]
        host_end_sec = bd["host_end"]
        best_onset_sec = bd["best_onset"]

        # Shade host turn
        ax.axvspan(host_start_sec, host_end_sec, alpha=0.10, color=ANCHOR_COLOR,
                   label="Host turn span")

        # Plot each strategy's NLL deltas
        strategy_results = bd.get("strategy_results", {})
        for strat_name, strat_data in strategy_results.items():
            deltas = strat_data.get("nll_deltas", [])
            if not deltas:
                continue
            # deltas may be len(candidates) * num_strategies, but each strategy
            # has exactly len(candidates) entries (one per candidate onset)
            n_cands = len(candidates)
            if len(deltas) == n_cands:
                color = STRATEGY_COLORS.get(strat_name, "#999999")
                label = strat_name.replace("_", " ")
                ax.plot(cand_sec, deltas, color=color, linewidth=1.3,
                        alpha=0.85, label=label)

        # Mark winner
        ax.plot(best_onset_sec, bd["best_delta"], "*", color=BEST_COLOR,
                markersize=14, zorder=5,
                label=f"Best: {bd['best_strategy'].replace('_',' ')}")

        # Zero line
        ax.axhline(0, color="k", linewidth=0.6, linestyle=":")

        group_str = ""
        if bd.get("bc_group_index") is not None:
            group_str = f".{bd['bc_group_index']}"

        ax.set_title(f"BC {bd['turn_id']}{group_str}  (ch {bd['channel']}, "
                     f"host T{bd['host_turn_id']}, dur={bd['duration']})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("BC onset (tokens)", fontsize=9)
        ax.set_ylabel("ΔNLL (vs baseline)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Pass 2 — NLL Difference by Strategy & Onset (Backchannels)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 5. Pass 2 — Strategy comparison bar chart
# ============================================================================

def plot_pass2_strategy_comparison(state, save_path: Optional[str] = None, show: bool = True):
    """
    Grouped bar chart: for each BC unit, show the best ΔNLL achieved by
    each strategy. Highlights which strategy won.
    """
    diag = state.diagnostics.get("pass2", {})
    per_bc = diag.get("per_backchannel", [])

    if not per_bc:
        print("No Pass 2 diagnostics available.")
        return None

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(per_bc)), 5))

    strategies = ["no_pause", "full_pause", "half_pause"]
    x = np.arange(len(per_bc))
    width = 0.25

    for s_idx, strat in enumerate(strategies):
        best_deltas = []
        for bd in per_bc:
            sr = bd.get("strategy_results", {}).get(strat, {})
            deltas = sr.get("nll_deltas", [])
            if deltas:
                best_deltas.append(min(deltas))
            else:
                best_deltas.append(0)

        color = STRATEGY_COLORS.get(strat, "#999999")
        bars = ax.bar(x + s_idx * width, best_deltas, width,
                      label=strat.replace("_", " "),
                      color=color, edgecolor="k", linewidth=0.5, alpha=0.85)

    # Mark winners
    for i, bd in enumerate(per_bc):
        winning_strat = bd["best_strategy"]
        s_idx = strategies.index(winning_strat) if winning_strat in strategies else 0
        ax.plot(i + s_idx * width, bd["best_delta"], "*", color=BEST_COLOR,
                markersize=12, zorder=5)

    # Labels
    labels = []
    for bd in per_bc:
        group_str = ""
        if bd.get("bc_group_index") is not None:
            group_str = f".{bd['bc_group_index']}"
        labels.append(f"BC{bd['turn_id']}{group_str}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best ΔNLL per strategy", fontsize=10)
    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax.set_title("Pass 2 — Best ΔNLL by Strategy (★ = winner)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 6. Pass 3 — NLL curves per breakpoint
# ============================================================================

def plot_pass3_nll_curves(state, save_path: Optional[str] = None, show: bool = True):
    """
    One subplot per breakpoint across all host turns refined in Pass 3.
    Shows NLL vs candidate onset for each breakpoint (subseg or next_turn).
    """
    diag = state.diagnostics.get("pass3", {})
    host_turns = diag.get("host_turns_refined", [])

    if not host_turns:
        print("No Pass 3 diagnostics available.")
        return None

    # Collect all breakpoint results
    all_bp = []
    for ht in host_turns:
        for bp in ht.get("breakpoint_results", []):
            all_bp.append((ht["turn_id"], ht["channel"], bp))

    if not all_bp:
        print("No breakpoint results in Pass 3 diagnostics.")
        return None

    n = len(all_bp)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

    for idx, (turn_id, channel, bp) in enumerate(all_bp):
        ax = axes[idx // cols][idx % cols]

        candidates = bp["candidates"]
        nll_scores = bp["nll_scores"]
        orig_onset = bp["original_onset"]
        best_onset = bp["best_onset"]
        anchor_type = bp.get("anchor_type", "subseg")

        # Color: use purple for next_turn breakpoints, channel color for subseg
        if anchor_type == "next_turn":
            next_ch = bp.get("next_turn_channel", channel)
            color = NEXT_TURN_COLOR
        else:
            color = CH_A_COLOR if channel == "A" else CH_B_COLOR

        ax.plot(candidates, nll_scores, color=color, linewidth=1.5, alpha=0.9)
        ax.axvline(orig_onset, color=ANCHOR_COLOR, linestyle="--", linewidth=1,
                   label=f"Original {orig_onset} tok")

        if best_onset in candidates:
            best_idx = candidates.index(best_onset)
            ax.plot(best_onset, nll_scores[best_idx], "o", color=BEST_COLOR,
                    markersize=8, zorder=5, label=f"Best {best_onset} tok")

        # Shade search range
        lo, hi = bp.get("search_range", (candidates[0], candidates[-1]))
        ax.axvspan(lo, hi, alpha=0.06, color=color)

        # Title varies by breakpoint type
        label = _bp_label(bp)
        if anchor_type == "next_turn":
            next_ch = bp.get("next_turn_channel", "?")
            ax.set_title(f"Host T{turn_id} — BP {label} "
                         f"(ch {next_ch}, shift={bp['shift']:+d})",
                         fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"Host T{turn_id} — BP {label} "
                         f"(shift={bp['shift']:+d})",
                         fontsize=10, fontweight="bold")

        ax.set_xlabel("Onset (tokens)", fontsize=9)
        ax.set_ylabel("NLL", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Pass 3 — NLL vs Breakpoint Onset (Sub-Segment & Next-Turn Retiming)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 7. Pass 3 — Breakpoint summary
# ============================================================================

def plot_pass3_breakpoint_summary(state, save_path: Optional[str] = None, show: bool = True):
    """
    Combined view for each host turn refined in Pass 3:
    - Top: mini timeline showing breakpoint shifts (subseg and next_turn)
    - Bottom: shift propagation to downstream turns
    """
    diag = state.diagnostics.get("pass3", {})
    host_turns = diag.get("host_turns_refined", [])
    propagated = diag.get("propagated_shifts", [])

    if not host_turns:
        print("No Pass 3 diagnostics available.")
        return None

    n_hosts = len(host_turns)
    has_propagation = len(propagated) > 0

    fig_height = 3 * n_hosts + (3 if has_propagation else 0)
    n_rows = n_hosts + (1 if has_propagation else 0)

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, fig_height), squeeze=False)

    for h_idx, ht in enumerate(host_turns):
        ax = axes[h_idx][0]
        turn_id = ht["turn_id"]
        channel = ht["channel"]
        next_turn_shift = ht.get("next_turn_shift", 0)

        title_parts = [
            f"Host Turn {turn_id} (ch {channel})",
            f"{ht['num_sub_segments']} sub-segs",
            f"{ht['num_bcs']} BC(s)",
            f"host shift: {ht.get('total_shift', 0):+d}",
        ]
        if next_turn_shift != 0:
            title_parts.append(f"next-turn shift: {next_turn_shift:+d}")

        ax.set_title(" — ".join(title_parts),
                     fontsize=10, fontweight="bold", loc="left")

        # Draw breakpoint shifts as arrows
        for bp in ht.get("breakpoint_results", []):
            orig = bp["original_onset"]
            best = bp["best_onset"]
            anchor_type = bp.get("anchor_type", "subseg")
            label = _bp_label(bp)

            # Use different y-positions and colors for the two types
            if anchor_type == "next_turn":
                y_pos = 0.2
                arrow_color = NEXT_TURN_COLOR
                text_color = NEXT_TURN_COLOR
            else:
                y_pos = 0.5
                arrow_color = BEST_COLOR
                text_color = BEST_COLOR

            if bp["shift"] != 0:
                ax.annotate(
                    "", xy=(best, y_pos), xytext=(orig, y_pos),
                    arrowprops=dict(arrowstyle="->", color=arrow_color,
                                   linewidth=2, alpha=0.8),
                )
                ax.text((orig + best) / 2, y_pos + 0.15,
                        f"{label} {bp['shift']:+d}",
                        ha="center", fontsize=8, color=text_color,
                        fontweight="bold")
            else:
                ax.plot(orig, y_pos, "o", color=ANCHOR_COLOR, markersize=6)
                ax.text(orig, y_pos + 0.15, f"{label} ±0",
                        ha="center", fontsize=8, color=ANCHOR_COLOR)

            # Original position marker
            ax.axvline(orig, color=ANCHOR_COLOR, linestyle=":", linewidth=0.7, alpha=0.5)

        # Draw original and new turn end
        orig_end = ht.get("original_turn_end", 0)
        new_end = ht.get("new_turn_end", 0)

        ax.axvline(orig_end, color=ANCHOR_COLOR, linestyle="--", linewidth=1,
                   label=f"Original end {orig_end} tok")
        if new_end != orig_end:
            ax.axvline(new_end, color=BEST_COLOR, linestyle="--", linewidth=1,
                       label=f"New end {new_end} tok")

        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel("", fontsize=9)
        ax.set_yticks([])

        # Build legend with anchor type indicators
        legend_handles = [
            Line2D([0], [0], color=ANCHOR_COLOR, linestyle="--", linewidth=1,
                   label="Original position"),
        ]
        has_subseg = any(bp.get("anchor_type", "subseg") == "subseg"
                        for bp in ht.get("breakpoint_results", []))
        has_next = any(bp.get("anchor_type") == "next_turn"
                       for bp in ht.get("breakpoint_results", []))
        if has_subseg:
            legend_handles.append(
                Line2D([0], [0], color=BEST_COLOR, linewidth=2,
                       label="Sub-seg breakpoint"))
        if has_next:
            legend_handles.append(
                Line2D([0], [0], color=NEXT_TURN_COLOR, linewidth=2,
                       label="Next-turn breakpoint"))

        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
        ax.set_xlabel("Time (tokens)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    # Propagation subplot
    if has_propagation:
        ax = axes[-1][0]
        ax.set_title("Shift Propagation to Downstream Turns",
                     fontsize=10, fontweight="bold", loc="left")

        for p_idx, prop in enumerate(propagated):
            source = prop["source_turn"]
            shift = prop["shift"]
            affected = prop["affected_turns"]
            shift_type = prop.get("shift_type", "intra_host")

            y = len(propagated) - p_idx - 1

            if shift_type == "next_turn":
                bar_color = NEXT_TURN_COLOR
                next_tid = prop.get("next_turn_id", "?")
                source_label = f"T{source}→T{next_tid}"
            else:
                bar_color = BEST_COLOR if shift > 0 else CH_A_COLOR
                source_label = f"T{source}"

            ax.barh(y, shift, color=bar_color,
                    edgecolor="k", linewidth=0.5, height=0.6, alpha=0.8)
            affected_str = ", ".join(f"T{t}" for t in affected[:5])
            if len(affected) > 5:
                affected_str += f" +{len(affected) - 5} more"

            type_tag = f" [{shift_type}]" if shift_type != "intra_host" else ""
            ax.text(shift + (0.3 if shift >= 0 else -0.3), y,
                    f"{source_label}{type_tag} → [{affected_str}]: {shift:+d} steps",
                    va="center", ha="left" if shift >= 0 else "right",
                    fontsize=8)

        ax.set_yticks([])
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Shift (steps)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Pass 3 — Breakpoint Retiming Summary", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 8. Summary stats (printed, not plotted)
# ============================================================================

def print_summary(state):
    """Print a text summary of the scheduling results."""
    print("=" * 60)
    print("THREE-PASS SCHEDULING SUMMARY")
    print("=" * 60)

    # Pass 1
    diag1 = state.diagnostics.get("pass1", {})
    per_turn = diag1.get("per_turn", [])
    if per_turn:
        shifts = [td["shift"] for td in per_turn]
        print(f"\nPass 1 — {len(per_turn)} primary turns")
        print(f"  Shift range: [{min(shifts):+d}, {max(shifts):+d}] steps")
        print(f"  Mean |shift|: {np.mean(np.abs(shifts)):.1f} steps "
              f"({np.mean(np.abs(shifts)) / TOKEN_RATE_HZ:.3f}s)")
        for td in per_turn:
            print(f"    T{td['turn_id']} (ch {td['channel']}): "
                  f"anchor={td['script_onset']} → onset={td['best_onset']} "
                  f"(shift={td['shift']:+d}), NLL={td['best_nll']:.4f}")

    # Pass 2
    diag2 = state.diagnostics.get("pass2", {})
    per_bc = diag2.get("per_backchannel", [])
    if per_bc:
        print(f"\nPass 2 — {len(per_bc)} backchannel unit(s)")
        strategies_used = [bd["best_strategy"] for bd in per_bc]
        from collections import Counter
        strat_counts = Counter(strategies_used)
        print(f"  Strategies chosen: {dict(strat_counts)}")
        for bd in per_bc:
            group_str = ""
            if bd.get("bc_group_index") is not None:
                group_str = f".{bd['bc_group_index']}"
            print(f"    BC{bd['turn_id']}{group_str} (ch {bd['channel']}, "
                  f"host T{bd['host_turn_id']}): "
                  f"onset={bd['best_onset']} (offset={bd['bc_offset']}), "
                  f"strategy={bd['best_strategy']}, "
                  f"ΔNLL={bd['best_delta']:+.4f}")

    # Pass 3
    diag3 = state.diagnostics.get("pass3", {})
    host_turns = diag3.get("host_turns_refined", [])
    if host_turns:
        print(f"\nPass 3 — {len(host_turns)} host turn(s) refined")
        for ht in host_turns:
            total_shift = ht.get("total_shift", 0)
            next_turn_shift = ht.get("next_turn_shift", 0)
            n_bp = ht.get("num_breakpoints", 0)
            extra = ""
            if next_turn_shift != 0:
                extra = f", next-turn shift={next_turn_shift:+d}"
            print(f"    T{ht['turn_id']} (ch {ht['channel']}): "
                  f"{n_bp} breakpoint(s), "
                  f"host shift={total_shift:+d} steps "
                  f"({total_shift / TOKEN_RATE_HZ:+.3f}s){extra}")
            for bp in ht.get("breakpoint_results", []):
                anchor_type = bp.get("anchor_type", "subseg")
                label = _bp_label(bp)
                type_tag = f" [{anchor_type}]"
                print(f"      BP {label}{type_tag}: "
                      f"orig={bp['original_onset']} → {bp['best_onset']} "
                      f"(shift={bp['shift']:+d}), NLL={bp['best_nll']:.4f}")

        propagated = diag3.get("propagated_shifts", [])
        if propagated:
            print(f"\n  Shift propagation:")
            for prop in propagated:
                shift_type = prop.get("shift_type", "intra_host")
                type_tag = f" [{shift_type}]"
                extra = ""
                if shift_type == "next_turn":
                    extra = f" (via T{prop.get('next_turn_id', '?')})"
                print(f"    T{prop['source_turn']}{type_tag}{extra} "
                      f"shift={prop['shift']:+d} → "
                      f"affected: {prop['affected_turns']}")

    # Pass 3 skipped turns
    skipped = diag3.get("host_turns_skipped", [])
    if skipped:
        print(f"\n  Pass 3 — {len(skipped)} host turn(s) SKIPPED:")
        for sk in skipped:
            print(f"    T{sk['turn_id']} (ch {sk['channel']}): "
                  f"reason={sk['reason']}, "
                  f"{sk['num_sub_segments']} sub-seg(s), "
                  f"{sk['num_bcs']} BC(s)")
            print(f"      Sub-seg onsets: {sk.get('sub_segment_onsets', [])}")
            print(f"      Sub-seg ends:   {sk.get('sub_segment_ends', [])}")
            print(f"      BC onsets:      {sk.get('bc_onsets', [])}")
            print(f"      BC ends:        {sk.get('bc_ends', [])}")
    elif not host_turns:
        print(f"\nPass 3 — No host turns refined or skipped "
              f"(no backchannels placed on any primary turn)")

    print("\n" + "=" * 60)


# ============================================================================
# Master entry point
# ============================================================================

def visualize_scheduling(
    state,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Generate all visualizations from a SchedulingState.

    Args:
        state: SchedulingState from generate_dialogue_three_pass
        save_dir: Directory to save PNGs (None = don't save)
        show: Whether to call plt.show()
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _path(name):
        return os.path.join(save_dir, name) if save_dir else None

    print_summary(state)

    plot_timeline(state, save_path=_path("timeline.png"), show=show)
    plot_pass1_nll_curves(state, save_path=_path("pass1_nll.png"), show=show)
    plot_pass1_shifts(state, save_path=_path("pass1_shifts.png"), show=show)
    plot_pass2_nll_curves(state, save_path=_path("pass2_nll.png"), show=show)
    plot_pass2_strategy_comparison(state, save_path=_path("pass2_strategies.png"), show=show)
    plot_pass3_nll_curves(state, save_path=_path("pass3_nll.png"), show=show)
    plot_pass3_breakpoint_summary(state, save_path=_path("pass3_summary.png"), show=show)
