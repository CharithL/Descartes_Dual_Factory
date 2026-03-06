"""Visualization: 3-level Ridge dR2 results tables."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path


def _extract_var_dict(raw_json):
    """Convert ridge probe JSON to dict[var_name -> stats].

    The ridge probe saves: {"results": [{"var_name": ..., "delta_R2": ...}, ...]}
    Visualization functions expect: {var_name: {"delta_R2": ...}}
    """
    if isinstance(raw_json, dict) and 'results' in raw_json:
        result_list = raw_json['results']
        if isinstance(result_list, list):
            return {r['var_name']: r for r in result_list if 'var_name' in r}
    # Already in the right format or fallback
    return raw_json


def format_results_table(results, level_name, hidden_size):
    """Format a single level's results as a text table.

    Args:
        results: dict of var_name -> {R2_trained, R2_untrained, delta_R2, category}
        level_name: 'A', 'B', or 'C'
        hidden_size: LSTM hidden size
    """
    level_labels = {
        'A': 'Individual Gates (expected zombie)',
        'B': 'Effective Conductances and Currents (THE KEY TABLE)',
        'C': 'Emergent Properties (gamma_amp candidates)',
    }

    header = f"\nLEVEL {level_name} — {level_labels.get(level_name, '')} (h={hidden_size})"
    sep = "=" * 90
    col_header = f"{'Variable':<30} | {'R2_trained':>10} | {'R2_untrained':>12} | {'dR2':>8} | {'Category':<20}"
    col_sep = "-" * 90

    lines = [sep, header, sep, col_header, col_sep]

    # Sort by dR2 descending
    sorted_vars = sorted(results.items(), key=lambda x: x[1].get('delta_R2', 0), reverse=True)

    for var_name, r in sorted_vars:
        r2_t = r.get('R2_trained', 0)
        r2_u = r.get('R2_untrained', 0)
        delta = r.get('delta_R2', 0)
        cat = r.get('category', 'UNKNOWN')

        # Highlight non-zombie
        marker = " ***" if delta > 0.2 else " *" if delta > 0.1 else ""

        lines.append(
            f"{var_name:<30} | {r2_t:>10.3f} | {r2_u:>12.3f} | {delta:>8.3f} | {cat:<20}{marker}"
        )

    lines.append(sep)

    # Summary
    n_total = len(results)
    n_learned = sum(1 for r in results.values() if r.get('delta_R2', 0) > 0.1)
    n_strong = sum(1 for r in results.values() if r.get('delta_R2', 0) > 0.2)
    lines.append(f"Summary: {n_learned}/{n_total} non-zombie (dR2 > 0.1), {n_strong} strong (dR2 > 0.2)")
    lines.append("")

    return "\n".join(lines)


def print_all_tables(results_dir, hidden_size=128):
    """Print all 3 level tables for a given hidden size."""
    results_dir = Path(results_dir)

    for level in ['A', 'B', 'C']:
        result_path = results_dir / f'ridge_level{level}_h{hidden_size}.json'
        if result_path.exists():
            with open(result_path) as f:
                raw = json.load(f)
            results = _extract_var_dict(raw)
            print(format_results_table(results, level, hidden_size))
        else:
            print(f"\n[Level {level} results not found at {result_path}]")


def plot_delta_r2_bar(results, level_name, hidden_size, save_path=None):
    """Bar chart of dR2 values for one level."""
    if not results:
        return

    # Sort by dR2
    sorted_items = sorted(results.items(), key=lambda x: x[1].get('delta_R2', 0), reverse=True)
    names = [item[0] for item in sorted_items]
    deltas = [item[1].get('delta_R2', 0) for item in sorted_items]

    # Color by category
    colors = []
    for d in deltas:
        if d > 0.2:
            colors.append('#2ecc71')   # Green: strong non-zombie
        elif d > 0.1:
            colors.append('#f39c12')   # Orange: moderate
        else:
            colors.append('#95a5a6')   # Grey: zombie

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 6))
    bars = ax.bar(range(len(names)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Threshold lines
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='dR2 = 0.1 (learned)')
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='dR2 = 0.2 (strong)')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('dR2 (trained − untrained)')
    ax.set_title(f'Level {level_name} Ridge dR2 — LSTM h={hidden_size}')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=min(0, min(deltas) - 0.05))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_levels(results_dir, hidden_size=128, save_dir=None):
    """Generate bar charts for all 3 levels."""
    results_dir = Path(results_dir)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for level in ['A', 'B', 'C']:
        result_path = results_dir / f'ridge_level{level}_h{hidden_size}.json'
        if result_path.exists():
            with open(result_path) as f:
                raw = json.load(f)
            results = _extract_var_dict(raw)
            save_path = save_dir / f'delta_r2_level{level}_h{hidden_size}.png' if save_dir else None
            plot_delta_r2_bar(results, level, hidden_size, save_path)


def plot_cross_hidden_comparison(results_dir, level='B', save_path=None):
    """Compare dR2 across hidden sizes for one level."""
    results_dir = Path(results_dir)
    from l5pc.config import HIDDEN_SIZES

    all_results = {}
    for h in HIDDEN_SIZES:
        rpath = results_dir / f'ridge_level{level}_h{h}.json'
        if rpath.exists():
            with open(rpath) as f:
                raw = json.load(f)
            all_results[h] = _extract_var_dict(raw)

    if not all_results:
        print(f"No results found for Level {level}")
        return

    # Get union of all variable names
    all_vars = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))

    fig, ax = plt.subplots(figsize=(max(10, len(all_vars) * 0.5), 6))
    width = 0.25
    x = np.arange(len(all_vars))

    for i, (h, results) in enumerate(sorted(all_results.items())):
        deltas = [results.get(v, {}).get('delta_R2', 0) for v in all_vars]
        ax.bar(x + i * width, deltas, width, label=f'h={h}', alpha=0.8)

    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_vars, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('dR2')
    ax.set_title(f'Level {level} dR2 across hidden sizes')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
