"""Visualization: Phase 3 graded replacement degradation curves."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path


def plot_replacement_curve(metrics_by_fraction, metric_name, save_path=None):
    """Plot a single metric vs replacement fraction.

    Args:
        metrics_by_fraction: dict of fraction -> list of metric values (replicates)
        metric_name: e.g., 'cross_condition_cc', 'gamma_power', 'phi'
    """
    fractions = sorted(metrics_by_fraction.keys())
    means = [np.mean(metrics_by_fraction[f]) for f in fractions]
    stds = [np.std(metrics_by_fraction[f]) for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(fractions, means, yerr=stds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, color='#2c3e50')
    ax.fill_between(fractions,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='#3498db')

    ax.set_xlabel('Fraction of L5PCs replaced by surrogate')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Graded Replacement: {metric_name}')
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_four_level_dashboard(all_metrics, save_path=None):
    """4-panel dashboard showing all validation levels vs replacement fraction.

    Args:
        all_metrics: dict with keys 'level1', 'level2', 'level3', 'level4',
            each containing {metric_name: {fraction: [values]}}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    level_configs = {
        'level1': {
            'title': 'Level 1: Output Fidelity',
            'metrics': ['cross_condition_cc', 'spike_rate_ratio', 'vp_distance'],
            'ax': axes[0, 0],
        },
        'level2': {
            'title': 'Level 2: Circuit Integration',
            'metrics': ['gamma_power', 'beta_power', 'mean_pairwise_cc'],
            'ax': axes[0, 1],
        },
        'level3': {
            'title': 'Level 3: DESCARTES',
            'metrics': ['mean_delta_r2_B', 'mean_delta_r2_C', 'n_mandatory_preserved'],
            'ax': axes[1, 0],
        },
        'level4': {
            'title': 'Level 4: Consciousness-Relevant',
            'metrics': ['bac_index', 'pci', 'phi', 'transfer_entropy'],
            'ax': axes[1, 1],
        },
    }

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for level_key, config in level_configs.items():
        ax = config['ax']
        level_data = all_metrics.get(level_key, {})

        for i, metric_name in enumerate(config['metrics']):
            if metric_name not in level_data:
                continue
            metric_data = level_data[metric_name]
            fractions = sorted(metric_data.keys(), key=float)
            means = [np.mean(metric_data[f]) for f in fractions]
            stds = [np.std(metric_data[f]) for f in fractions]
            frac_vals = [float(f) for f in fractions]

            color = colors[i % len(colors)]
            ax.errorbar(frac_vals, means, yerr=stds, fmt='o-', capsize=3,
                        linewidth=1.5, markersize=5, color=color, label=metric_name)

        ax.set_xlabel('Replacement fraction')
        ax.set_title(config['title'])
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(-0.05, 1.05)

    plt.suptitle('Four-Level Validation: Graded Surrogate Replacement', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_test(all_metrics, save_path=None):
    """Plot showing whether the key prediction from guide Section 6.4 holds.

    Prediction:
    - Level 1: PASS (correct spikes)
    - Level 2: PARTIAL PASS (gamma OK, beta may degrade)
    - Level 3: MIXED (some mandatory preserved, some lost)
    - Level 4: SELECTIVE FAIL (Φ drops, PCI may pass, BAC may fail)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    level_names = ['Level 1\nOutput', 'Level 2\nCircuit', 'Level 3\nDESCARTES', 'Level 4\nConsciousness']
    predicted = ['PASS', 'PARTIAL', 'MIXED', 'FAIL']
    pred_colors = ['#2ecc71', '#f39c12', '#f39c12', '#e74c3c']

    # Compute actual pass/fail at 100% replacement
    actual_scores = []
    for level_key in ['level1', 'level2', 'level3', 'level4']:
        level_data = all_metrics.get(level_key, {})
        if not level_data:
            actual_scores.append(0.5)
            continue
        # Average normalized metric at 100% replacement / 0% baseline
        scores = []
        for metric_name, metric_data in level_data.items():
            baseline = np.mean(metric_data.get('0.0', metric_data.get(0.0, [1.0])))
            full_replace = np.mean(metric_data.get('1.0', metric_data.get(1.0, [0.5])))
            if baseline > 0:
                scores.append(full_replace / baseline)
            else:
                scores.append(0.5)
        actual_scores.append(np.mean(scores) if scores else 0.5)

    x = np.arange(len(level_names))
    bars = ax.bar(x, actual_scores, color=pred_colors, edgecolor='black',
                  linewidth=1, alpha=0.8)

    # Annotate predictions
    for i, (pred, score) in enumerate(zip(predicted, actual_scores)):
        ax.annotate(f'Predicted: {pred}\nActual: {score:.2f}',
                    xy=(i, score), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.set_ylabel('Preservation ratio (replaced / baseline)')
    ax.set_title('Key Prediction Test: Functional ≠ Computational ≠ Phenomenal Equivalence')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect preservation')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
