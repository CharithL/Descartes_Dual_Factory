"""Visualization: Phase 2 spatial analysis of ΔR² across dendritic tree."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path


def plot_distance_gradient(results_by_distance, var_name, save_path=None):
    """Plot ΔR² as a function of distance from soma.

    Tests whether biological encoding decreases with distance
    (proximal = non-zombie, distal = zombie).

    Args:
        results_by_distance: list of (distance_um, delta_r2) tuples
        var_name: e.g., 'G_CaHVA', 'I_Ca'
    """
    if not results_by_distance:
        return

    distances, deltas = zip(*sorted(results_by_distance))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(distances, deltas, c=deltas, cmap='RdYlGn', s=60,
               edgecolors='black', linewidth=0.5, vmin=0, vmax=max(0.3, max(deltas)))
    ax.plot(distances, deltas, '-', alpha=0.3, color='gray')

    # Mark hot zone
    from l5pc.config import CA_HOTZONE_START_UM, CA_HOTZONE_END_UM
    ax.axvspan(CA_HOTZONE_START_UM, CA_HOTZONE_END_UM,
               alpha=0.15, color='red', label='Ca²⁺ hot zone (685-885 μm)')

    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='ΔR² = 0.1')
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='ΔR² = 0.2')

    ax.set_xlabel('Distance from soma (μm)')
    ax.set_ylabel('ΔR²')
    ax.set_title(f'Distance-dependent zombie gradient: {var_name}')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hotzone_comparison(hotzone_results, non_hotzone_results, save_path=None):
    """Compare ΔR² for hot zone vs non-hot-zone compartments.

    Tests whether G_CaHVA is mandatory specifically in the 685-885 μm region.
    """
    all_vars = sorted(set(list(hotzone_results.keys()) + list(non_hotzone_results.keys())))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_vars))
    width = 0.35

    hz_deltas = [hotzone_results.get(v, {}).get('delta_R2', 0) for v in all_vars]
    nhz_deltas = [non_hotzone_results.get(v, {}).get('delta_R2', 0) for v in all_vars]

    ax.bar(x - width / 2, hz_deltas, width, label='Hot zone (685-885 μm)',
           color='#e74c3c', alpha=0.8)
    ax.bar(x + width / 2, nhz_deltas, width, label='Non-hot-zone',
           color='#3498db', alpha=0.8)

    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_vars, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ΔR²')
    ax.set_title('Hot Zone vs Non-Hot-Zone ΔR² Comparison')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ih_gradient(ih_results_by_distance, save_path=None):
    """Plot Ih ΔR² vs distance, overlaid with expected exponential gradient.

    Tests whether the network discovers the Ih exponential gradient
    or just encodes a flat average.
    """
    if not ih_results_by_distance:
        return

    distances, deltas = zip(*sorted(ih_results_by_distance))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(distances, deltas, c='#9b59b6', s=60, edgecolors='black',
               linewidth=0.5, zorder=3, label='Network ΔR² for Ih')

    # Overlay expected biological Ih gradient (exponential increase)
    d_arr = np.array(distances)
    # Ih density increases ~10x from soma to distal apical
    ih_biological = 0.01 * np.exp(d_arr / 300)  # Approximate
    ih_biological = ih_biological / ih_biological.max() * max(deltas) if max(deltas) > 0 else ih_biological
    ax.plot(d_arr, ih_biological, '--', color='purple', alpha=0.5,
            label='Biological Ih density (normalized)')

    ax.set_xlabel('Distance from soma (μm)')
    ax.set_ylabel('ΔR²')
    ax.set_title('Ih Gradient Encoding: Does the network discover the gradient?')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dendritic_morphology_overlay(morphology_coords, delta_r2_values,
                                      var_name, save_path=None):
    """Overlay ΔR² values on 2D dendritic morphology.

    Args:
        morphology_coords: list of (x, y, distance_from_soma) per compartment
        delta_r2_values: ΔR² per compartment (same order)
    """
    if not morphology_coords or not delta_r2_values:
        return

    xs, ys, dists = zip(*morphology_coords)

    fig, ax = plt.subplots(figsize=(6, 12))
    sc = ax.scatter(xs, ys, c=delta_r2_values, cmap='RdYlGn',
                    s=30, edgecolors='gray', linewidth=0.3,
                    vmin=0, vmax=max(0.3, max(delta_r2_values)))

    plt.colorbar(sc, ax=ax, label='ΔR²', shrink=0.6)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'ΔR² overlaid on dendrite: {var_name}')
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
