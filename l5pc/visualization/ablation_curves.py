"""Visualization: Progressive ablation curves."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path


def plot_ablation_curve(ablation_results, var_name, save_path=None):
    """Plot progressive ablation curve for a single variable.

    Shows target-correlated clamp vs random clamp at each k fraction.
    The gap between them reveals whether the variable is causally used.

    Args:
        ablation_results: list of dicts with keys:
            k_frac, target_cc, random_cc_mean, random_cc_std, z_score, verdict
        var_name: Name of the variable being ablated
    """
    k_fracs = [r['k_frac'] for r in ablation_results]
    target_ccs = [r['target_cc'] for r in ablation_results]
    random_means = [r['random_cc_mean'] for r in ablation_results]
    random_stds = [r['random_cc_std'] for r in ablation_results]
    z_scores = [r['z_score'] for r in ablation_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])

    # Top: CC vs k fraction
    ax1.plot(k_fracs, target_ccs, 'o-', color='#e74c3c', linewidth=2,
             markersize=8, label='Target-correlated clamp', zorder=3)
    ax1.plot(k_fracs, random_means, 's--', color='#3498db', linewidth=2,
             markersize=6, label='Random clamp (mean)', zorder=2)
    ax1.fill_between(
        k_fracs,
        [m - 2 * s for m, s in zip(random_means, random_stds)],
        [m + 2 * s for m, s in zip(random_means, random_stds)],
        alpha=0.2, color='#3498db', label='Random ±2σ'
    )

    # Find breaking point (first k where z < -2)
    breaking_k = None
    for r in ablation_results:
        if r['z_score'] < -2.0:
            breaking_k = r['k_frac']
            break

    if breaking_k is not None:
        ax1.axvline(x=breaking_k, color='red', linestyle=':', alpha=0.7)
        ax1.annotate(f'Breaking point\nk={breaking_k:.0%}',
                     xy=(breaking_k, min(target_ccs)), fontsize=9,
                     ha='center', color='red')

    ax1.set_ylabel('Cross-condition correlation')
    ax1.set_title(f'Progressive Ablation: {var_name}')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.02, max(k_fracs) + 0.02)

    # Bottom: z-scores
    colors = ['#e74c3c' if z < -2 else '#95a5a6' for z in z_scores]
    ax2.bar(k_fracs, z_scores, width=0.03, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7, label='z = -2 (causal threshold)')
    ax2.set_xlabel('Fraction of hidden dims clamped (k)')
    ax2.set_ylabel('z-score')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_ablations(ablation_path, save_dir=None):
    """Plot ablation curves for all tested variables."""
    with open(ablation_path) as f:
        all_results = json.load(f)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for var_name, results in all_results.items():
        save_path = save_dir / f'ablation_{var_name}.png' if save_dir else None
        plot_ablation_curve(results, var_name, save_path)


def plot_mandatory_summary(classification_path, save_path=None):
    """Summary plot showing all mandatory variables with their types.

    Color-coded by redundancy type:
    - Red: Concentrated (breaks < 10%)
    - Orange: Distributed (breaks 40-60%)
    - Yellow: Redundant (breaks > 70%)
    """
    with open(classification_path) as f:
        classifications = json.load(f)

    mandatory = {k: v for k, v in classifications.items()
                 if v.get('category', '').startswith('MANDATORY')}

    if not mandatory:
        print("No mandatory variables found.")
        return

    names = list(mandatory.keys())
    breaking_points = [v.get('breaking_k_frac', 0.5) for v in mandatory.values()]
    categories = [v.get('category', '') for v in mandatory.values()]

    color_map = {
        'MANDATORY_CONCENTRATED': '#e74c3c',
        'MANDATORY_DISTRIBUTED': '#f39c12',
        'MANDATORY_REDUNDANT': '#f1c40f',
    }
    colors = [color_map.get(c, '#95a5a6') for c in categories]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), breaking_points, color=colors,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Breaking point (fraction of dims clamped)')
    ax.set_title('Mandatory Variables by Redundancy Type')

    # Add threshold lines
    ax.axvline(x=0.10, color='red', linestyle=':', alpha=0.5, label='Concentrated (<10%)')
    ax.axvline(x=0.60, color='orange', linestyle=':', alpha=0.5, label='Distributed (40-60%)')

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
