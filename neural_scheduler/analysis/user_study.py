"""Analysis of the forced-choice listening studies.

Respondents saw pairs of clips labelled A and B and picked one. Each response
CSV has one row per respondent: a first column that is ignored (an anonymous
respondent index, or the form's timestamp) followed by one ``A``/``B`` column
per item.

``main``
    The 20-item study. Items 1-10 compare the scheduler's backchannels with
    PersonaPlex, items 11-20 compare scheduled turn-taking with naive
    concatenation (also reported for items 11-15 and 16-20 separately)::

        python -m neural_scheduler.analysis.user_study main \\
            --csv analysis_data/user_study/responses_main.csv \\
            --bc analysis_data/user_study/ab_key_backchannel.json \\
            --tt analysis_data/user_study/ab_key_turn_taking.json

``dramatic``
    A turn-taking-only study with any number of items; the key is given with
    ``--key`` or inline with ``--model_is_A A B A ...``.

For every item the tool reports how many respondents chose the proposed
system and an exact two-sided binomial p-value against chance; for each
block it reports the pooled win rate with a Wilson 95% interval and a pooled
binomial p-value (every respondent-item response counted as one independent
trial). Krippendorff's alpha is added when the ``krippendorff`` package is
installed.

A/B key files are JSON lists with one entry per item, in item order. The side
of the proposed system is read from the first of these fields present:
``proposed_is_A``, ``model_is_A``, ``naive_is_A`` (negated) or
``first_is_wav_a`` (paired stimuli, where ``wav_a`` is the proposed system),
so the ``ab_mapping.json`` files written by
:mod:`neural_scheduler.analysis.stimuli` can be used directly.
"""

import argparse
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

BC_ITEMS = slice(0, 10)
TT_ITEMS = slice(10, 20)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_responses(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Read a response CSV. Returns the frame and its item columns (all but the first)."""
    df = pd.read_csv(csv_path, header=0)
    return df, list(df.columns[1:])


def proposed_is_A(entry: Dict) -> bool:
    """Whether the proposed system was shown as clip A for a key entry."""
    if "proposed_is_A" in entry:
        return bool(entry["proposed_is_A"])
    if "model_is_A" in entry:
        return bool(entry["model_is_A"])
    if "naive_is_A" in entry:
        return not entry["naive_is_A"]
    if "first_is_wav_a" in entry:
        return bool(entry["first_is_wav_a"])
    raise KeyError(
        "A/B key entry needs one of proposed_is_A, model_is_A, naive_is_A, first_is_wav_a; "
        f"got {sorted(entry)}"
    )


def load_ab_key(path: str) -> List[Dict]:
    """Read an A/B key file into ``[{'video_file': str, 'proposed_is_A': bool}, ...]``."""
    with open(path) as f:
        entries = json.load(f)
    return [
        {
            "video_file": e.get("video_file", f"item_{i:02d}"),
            "proposed_is_A": proposed_is_A(e),
        }
        for i, e in enumerate(entries, start=1)
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def binomial_test(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value."""
    return stats.binomtest(k, n, p=p, alternative="two-sided").pvalue


def ci_95(k: int, n: int) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return lo, hi


def vote_matrix(df: pd.DataFrame, first_item: int, key: List[Dict], strip: bool = False) -> np.ndarray:
    """``[respondents, items]`` array, 1 where the respondent chose the proposed system.

    The key's items are the CSV's item columns starting at ``first_item``
    (0-based); columns are addressed by position, so duplicated question
    headers are fine. With ``strip`` answers are whitespace-stripped first.
    """
    matrix = np.zeros((len(df), len(key)), dtype=int)
    for i, entry in enumerate(key):
        column = df.iloc[:, 1 + first_item + i]
        if strip:
            column = column.str.strip()
        matrix[:, i] = [1 if (r == "A") == entry["proposed_is_A"] else 0 for r in column.values]
    return matrix


def item_wins(matrix: np.ndarray) -> List[int]:
    """Number of respondents choosing the proposed system, per item."""
    return [int(w) for w in matrix.sum(axis=0)]


def aggregate(wins: Sequence[int], n_respondents: int) -> Dict:
    """Pooled win rate over items: ``total``, ``trials``, ``rate``, ``ci_low``, ``ci_high``, ``p``."""
    total = int(sum(wins))
    trials = n_respondents * len(wins)
    lo, hi = ci_95(total, trials)
    return {
        "total": total,
        "trials": trials,
        "rate": total / trials,
        "ci_low": lo,
        "ci_high": hi,
        "p": binomial_test(total, trials),
    }


def krippendorff_alpha(matrix: np.ndarray) -> Optional[float]:
    """Nominal Krippendorff's alpha, or ``None`` if ``krippendorff`` is not installed.

    The ``[respondents, items]`` matrix is passed transposed, as
    ``reliability_data``.
    """
    try:
        import krippendorff
    except ImportError:
        return None
    return krippendorff.alpha(reliability_data=matrix.T, level_of_measurement="nominal")


def _stars(p: float) -> str:
    return "  ***" if p < 0.001 else ("  *" if p < 0.05 else "")


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main study (backchannel + turn-taking)
# ---------------------------------------------------------------------------

def analyze_main(csv_path: str, bc_key_path: str, tt_key_path: str) -> Dict:
    """Analyse the 20-item study, print the report and return the numbers.

    The result has ``n_respondents`` and, for ``backchannel`` and
    ``turn_taking``: ``items`` (``video_file``, ``wins``, ``p`` each),
    ``aggregate`` and ``alpha``; ``turn_taking`` also has ``first_half`` and
    ``second_half`` aggregates.
    """
    df, cols = load_responses(csv_path)
    n = len(df)
    bc_key = load_ab_key(bc_key_path)
    tt_key = load_ab_key(tt_key_path)

    assert len(cols) >= TT_ITEMS.start + len(tt_key), "CSV has fewer item columns than key entries"
    bc_mat = vote_matrix(df, BC_ITEMS.start, bc_key)
    tt_mat = vote_matrix(df, TT_ITEMS.start, tt_key)

    print(f"\nN respondents : {n}")

    results = {"n_respondents": n}
    for name, title, width, key, mat in [
        ("backchannel", "BACKCHANNEL  (model vs. personaplex)", (20, 58, 18), bc_key, bc_mat),
        ("turn_taking", "TURN-TAKING  (model vs. naive)", (22, 60, 20), tt_key, tt_mat),
    ]:
        print_section(title)
        wins = item_wins(mat)
        items = [
            {"video_file": e["video_file"], "wins": w, "p": binomial_test(w, n)}
            for e, w in zip(key, wins)
        ]
        print(f"\n{'Video':<{width[0]}} {'Model wins':<14} {'%':<8} {'p (exact binom)'}")
        print("-" * width[1])
        for it in items:
            sig = " *" if it["p"] < 0.05 else ""
            print(f"  {it['video_file']:<{width[2]}} {it['wins']}/{n:<10}    "
                  f"{it['wins']/n*100:5.1f}%   {it['p']:.4f}{sig}")
        agg = aggregate(wins, n)
        print(f"\n  Aggregate  : {agg['total']}/{agg['trials']}  =  {agg['rate']:.3f}  "
              f"(95% CI [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}])")
        print(f"  Binomial p : {agg['p']:.4e}{_stars(agg['p'])}")
        results[name] = {"items": items, "aggregate": agg}

    tt_items = results["turn_taking"]["items"]
    for half, label, subset in [
        ("first_half", "Videos 01–05", tt_items[:5]),
        ("second_half", "Videos 06–10", tt_items[5:]),
    ]:
        agg = aggregate([it["wins"] for it in subset], n)
        print(f"\n  {label}")
        for it in subset:
            sig = " *" if it["p"] < 0.05 else ""
            print(f"    {it['video_file']:<22} {it['wins']}/{n:<6}  {it['wins']/n*100:5.1f}%   p={it['p']:.4f}{sig}")
        print(f"    Aggregate : {agg['total']}/{agg['trials']}  =  {agg['rate']:.3f}  "
              f"(95% CI [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}])  p={agg['p']:.4e}"
              f"{_stars(agg['p'])}")
        results["turn_taking"][half] = agg

    alpha_bc = krippendorff_alpha(bc_mat)
    alpha_tt = krippendorff_alpha(tt_mat)
    results["backchannel"]["alpha"] = alpha_bc
    results["turn_taking"]["alpha"] = alpha_tt
    if alpha_bc is None:
        print("\n  [skipped] Install krippendorff (pip install krippendorff) for inter-rater agreement.")
    else:
        print_section("INTER-RATER AGREEMENT (Krippendorff's alpha, nominal)")
        print(f"\n  BC task      alpha = {alpha_bc:.3f}")
        print(f"  TurnTake     alpha = {alpha_tt:.3f}")
        print("  (alpha > 0.8 = good; 0.67–0.8 = acceptable; < 0.67 = low)")

    bc, tt = results["backchannel"]["aggregate"], results["turn_taking"]["aggregate"]
    print_section("SUMMARY")
    print(f"""
  BC task   (model vs. personaplex)
    Preference for model : {bc['rate']*100:.1f}%  (95% CI [{bc['ci_low']*100:.1f}%, {bc['ci_high']*100:.1f}%])
    Aggregate p-value    : {bc['p']:.4e}

  TurnTake  (model vs. naive)
    Preference for model : {tt['rate']*100:.1f}%  (95% CI [{tt['ci_low']*100:.1f}%, {tt['ci_high']*100:.1f}%])
    Aggregate p-value    : {tt['p']:.4e}
""")
    return results


# ---------------------------------------------------------------------------
# Turn-taking-only study
# ---------------------------------------------------------------------------

def analyze_turn_taking(csv_path: str, model_is_A_flags: Sequence[bool]) -> Dict:
    """Analyse a turn-taking-only study, print the report and return the numbers.

    ``model_is_A_flags`` has one boolean per item column. Answers are
    whitespace-stripped. The result has ``n_respondents``, ``items``
    (``wins`` and ``p`` each), ``aggregate`` and ``alpha``.
    """
    df, cols = load_responses(csv_path)
    n = len(df)
    n_clips = len(cols)
    assert len(model_is_A_flags) == n_clips, (
        f"the key must have {n_clips} values, got {len(model_is_A_flags)}"
    )

    print(f"\nN respondents : {n}")
    print(f"N clips       : {n_clips}")

    print_section("TURN-TAKING  (model vs. naive)")
    print(f"\n{'Clip':<10} {'Model is':<10} {'Model wins':<14} {'%':<8} {'p (exact binom)'}")
    print("-" * 56)

    key = [{"proposed_is_A": bool(f)} for f in model_is_A_flags]
    mat = vote_matrix(df, 0, key, strip=True)
    wins = item_wins(mat)
    items = []
    for i, w in enumerate(wins):
        p = binomial_test(w, n)
        sig = " *" if p < 0.05 else ""
        print(f"  Clip {i+1:<4}  {'A' if model_is_A_flags[i] else 'B':<10} "
              f"{w}/{n:<10}    {w/n*100:5.1f}%   {p:.4f}{sig}")
        items.append({"wins": w, "p": p})

    agg = aggregate(wins, n)
    print(f"\n  Aggregate  : {agg['total']}/{agg['trials']}  =  {agg['rate']:.3f}  "
          f"(95% CI [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}])")
    print(f"  Binomial p : {agg['p']:.4e}{_stars(agg['p'])}")

    alpha = krippendorff_alpha(mat)
    if alpha is None:
        print("\n  [skipped] Install krippendorff for inter-rater agreement.")
    else:
        print_section("INTER-RATER AGREEMENT (Krippendorff's alpha, nominal)")
        print(f"\n  TurnTake alpha = {alpha:.3f}")
        print("  (alpha > 0.8 = good; 0.67–0.8 = acceptable; < 0.67 = low)")

    print_section("SUMMARY")
    print(f"""
  TurnTake  (model vs. naive)
    Preference for model : {agg['rate']*100:.1f}%  (95% CI [{agg['ci_low']*100:.1f}%, {agg['ci_high']*100:.1f}%])
    Aggregate p-value    : {agg['p']:.4e}
""")
    return {"n_respondents": n, "items": items, "aggregate": agg, "alpha": alpha}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forced-choice listening study analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("main", help="20-item study: 10 backchannel items, then 10 turn-taking items.")
    m.add_argument("--csv", required=True, help="Response CSV.")
    m.add_argument("--bc", required=True, help="A/B key of the backchannel items (JSON).")
    m.add_argument("--tt", required=True, help="A/B key of the turn-taking items (JSON).")

    d = sub.add_parser("dramatic", help="Turn-taking-only study.")
    d.add_argument("--csv", required=True, help="Response CSV.")
    key = d.add_mutually_exclusive_group(required=True)
    key.add_argument("--key", help="A/B key (JSON).")
    key.add_argument("--model_is_A", nargs="+", choices=["A", "B"],
                     help="Which clip (A or B) is the proposed system, one value per item, "
                          "e.g. --model_is_A A B A A A")
    return parser


def main(argv=None) -> Dict:
    args = build_parser().parse_args(argv)
    if args.command == "main":
        return analyze_main(args.csv, args.bc, args.tt)
    if args.key:
        flags = [e["proposed_is_A"] for e in load_ab_key(args.key)]
    else:
        flags = [v == "A" for v in args.model_is_A]
    return analyze_turn_taking(args.csv, flags)


if __name__ == "__main__":
    main()
