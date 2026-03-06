"""
L5PC DESCARTES -- Final Variable Classification

Combines evidence from all three probing stages:
  1. Ridge DeltaR2 (ridge_probe.py)     -- is it decodable beyond chance?
  2. Voltage baselines (baselines.py)    -- is it more than voltage reencoding?
  3. Causal ablation (ablation.py)       -- does the network actually use it?

Final categories (ordered by scientific importance):

  ZOMBIE:
    DeltaR2 < 0.1.  The network carries no more information about this
    variable than random projections.  Most gate variables fall here.

  VOLTAGE_REENCODING:
    DeltaR2 > 0.1 but R2_above_voltage < 0.05.  The hidden state merely
    re-encodes compartment voltage; the conductance can be trivially
    recovered from V via the steady-state relation G_inf(V).

  LEARNED_BYPRODUCT:
    DeltaR2 > 0.1, above voltage baseline, but ablation shows z > -2.
    The network has learned this information as a byproduct of training,
    but does not causally use it to compute the output.

  MANDATORY_CONCENTRATED:
    Causally necessary (z < -2), model breaks at < 10% clamp.
    Representation is concentrated in a few hidden dimensions.
    Example: g_NMDA_SC at 3.9% in hippocampal experiment.

  MANDATORY_DISTRIBUTED:
    Causally necessary, breaks at 40-60% clamp.
    Information is spread across many dimensions.
    Example: gamma_amp at h=128.

  MANDATORY_REDUNDANT:
    Causally necessary, breaks at > 70% clamp.
    Highly redundant representation requiring massive ablation to disrupt.
    Example: gamma_amp at h=512 (larger network = more redundancy).
"""

import logging
from pathlib import Path
from collections import Counter

from l5pc.config import (
    DELTA_THRESHOLD_LEARNED,
    VOLTAGE_ABOVE_THRESHOLD,
    VOLTAGE_ABOVE_MODERATE,
    CAUSAL_Z_THRESHOLD,
    HIDDEN_SIZES,
)
from l5pc.utils.io import load_results_json, save_results_json

logger = logging.getLogger(__name__)

# All possible categories in order of the classification cascade
CATEGORIES = [
    'ZOMBIE',
    'VOLTAGE_REENCODING',
    'LEARNED_BYPRODUCT',
    'MANDATORY_CONCENTRATED',
    'MANDATORY_DISTRIBUTED',
    'MANDATORY_REDUNDANT',
]


# ---------------------------------------------------------------------------
# Single variable classification
# ---------------------------------------------------------------------------

def classify_variable(ridge_result, baseline_result=None,
                      ablation_result=None):
    """Classify a single variable through the three-stage cascade.

    Parameters
    ----------
    ridge_result : dict
        From ridge_probe.py: must contain 'delta_R2', 'R2_trained',
        'R2_untrained', 'p_value'.
    baseline_result : dict, optional
        From baselines.py: must contain 'R2_above_voltage',
        'baseline_category'.  Only relevant for Level B variables.
    ablation_result : dict, optional
        From ablation.py: must contain 'classification',
        'breaking_point', 'ablation_steps'.

    Returns
    -------
    classification : dict
        Keys: final_category, evidence (sub-dict with all supporting data),
        stage_reached (which stage determined the category).
    """
    delta_r2 = ridge_result.get('delta_R2', 0.0)
    r2_trained = ridge_result.get('R2_trained', 0.0)
    r2_untrained = ridge_result.get('R2_untrained', 0.0)
    p_value = ridge_result.get('p_value', 1.0)
    var_name = ridge_result.get('var_name', 'unknown')

    evidence = {
        'var_name': var_name,
        'delta_R2': delta_r2,
        'R2_trained': r2_trained,
        'R2_untrained': r2_untrained,
        'p_value': p_value,
    }

    # --- Stage 1: Zombie test ---
    if delta_r2 < DELTA_THRESHOLD_LEARNED:
        evidence['stage_reached'] = 'ridge'
        return {
            'final_category': 'ZOMBIE',
            'evidence': evidence,
            'stage_reached': 'ridge',
        }

    # --- Stage 2: Voltage baseline (Level B only) ---
    if baseline_result is not None:
        r2_above_voltage = baseline_result.get('R2_above_voltage', None)
        baseline_cat = baseline_result.get('baseline_category', None)

        evidence['R2_voltage_only'] = baseline_result.get(
            'R2_voltage_only', None
        )
        evidence['R2_temporal'] = baseline_result.get('R2_temporal', None)
        evidence['R2_above_voltage'] = r2_above_voltage
        evidence['baseline_category'] = baseline_cat

        if (r2_above_voltage is not None
                and r2_above_voltage < VOLTAGE_ABOVE_MODERATE):
            evidence['stage_reached'] = 'baseline'
            return {
                'final_category': 'VOLTAGE_REENCODING',
                'evidence': evidence,
                'stage_reached': 'baseline',
            }

    # --- Stage 3: Causal ablation ---
    if ablation_result is not None:
        abl_class = ablation_result.get('classification', 'NON_CAUSAL')
        breaking_point = ablation_result.get('breaking_point', None)

        evidence['ablation_classification'] = abl_class
        evidence['breaking_point'] = breaking_point
        evidence['baseline_cc'] = ablation_result.get('baseline_cc', None)

        # Extract minimum z-score across all k values
        ablation_steps = ablation_result.get('ablation_steps', [])
        if ablation_steps:
            min_z = min(s.get('z_score', 0.0) for s in ablation_steps)
            evidence['min_z_score'] = min_z
        else:
            min_z = 0.0
            evidence['min_z_score'] = None

        evidence['stage_reached'] = 'ablation'

        if abl_class == 'NON_CAUSAL':
            return {
                'final_category': 'LEARNED_BYPRODUCT',
                'evidence': evidence,
                'stage_reached': 'ablation',
            }

        # Causal -- classify by redundancy type
        # (already done in ablation.py, but verify here)
        if abl_class in ('MANDATORY_CONCENTRATED', 'MANDATORY_DISTRIBUTED',
                         'MANDATORY_REDUNDANT'):
            return {
                'final_category': abl_class,
                'evidence': evidence,
                'stage_reached': 'ablation',
            }

    # If ablation was not run, classify as LEARNED (pending ablation)
    evidence['stage_reached'] = 'ridge_only'
    return {
        'final_category': 'LEARNED_BYPRODUCT',
        'evidence': evidence,
        'stage_reached': 'ridge_only',
        'note': 'ablation_not_run',
    }


# ---------------------------------------------------------------------------
# Classify all variables
# ---------------------------------------------------------------------------

def classify_all(ridge_dir, baseline_path=None, ablation_path=None,
                 save_path=None):
    """Classify all variables across all levels and hidden sizes.

    Parameters
    ----------
    ridge_dir : str or Path
        Directory containing ridge result JSONs
        (ridge_levelA_h64.json, ridge_levelB_h128.json, etc.).
    baseline_path : str or Path, optional
        Path to voltage baseline results JSON (Level B only).
    ablation_path : str or Path, optional
        Path to ablation results JSON.
    save_path : str or Path, optional
        Output path for classification results JSON.

    Returns
    -------
    summary : dict
        Full classification results with per-variable details and summary
        statistics.
    """
    ridge_dir = Path(ridge_dir)

    # Load baselines and ablation results if available
    baseline_data = {}
    if baseline_path is not None:
        bp = Path(baseline_path)
        if bp.exists():
            raw = load_results_json(bp)
            for entry in raw.get('results', []):
                baseline_data[entry['var_name']] = entry

    ablation_data = {}
    if ablation_path is not None:
        ap = Path(ablation_path)
        if ap.exists():
            raw = load_results_json(ap)
            for key, entry in raw.get('results', {}).items():
                # Key format: "{level}_{var_name}_h{hs}"
                vname = entry.get('var_name', key)
                hs = entry.get('hidden_size', 0)
                ablation_data[(vname, hs)] = entry

    # Process all ridge results
    all_classifications = []

    for level in ['A', 'B', 'C']:
        for hs in HIDDEN_SIZES:
            ridge_path = ridge_dir / f'ridge_level{level}_h{hs}.json'
            if not ridge_path.exists():
                continue

            ridge_data = load_results_json(ridge_path)

            for r in ridge_data.get('results', []):
                var_name = r['var_name']

                # Get baseline (Level B only)
                bl = baseline_data.get(var_name) if level == 'B' else None

                # Get ablation
                abl = ablation_data.get((var_name, hs))

                classification = classify_variable(r, bl, abl)
                classification['level'] = level
                classification['hidden_size'] = hs
                classification['var_name'] = var_name

                all_classifications.append(classification)

    # Build summary statistics
    category_counts = Counter(c['final_category'] for c in all_classifications)
    per_level = {}
    for level in ['A', 'B', 'C']:
        level_entries = [c for c in all_classifications if c['level'] == level]
        per_level[level] = {
            'total': len(level_entries),
            'categories': dict(Counter(
                c['final_category'] for c in level_entries
            )),
        }

    per_hidden = {}
    for hs in HIDDEN_SIZES:
        hs_entries = [c for c in all_classifications
                      if c['hidden_size'] == hs]
        per_hidden[hs] = {
            'total': len(hs_entries),
            'categories': dict(Counter(
                c['final_category'] for c in hs_entries
            )),
        }

    summary = {
        'total_variables': len(all_classifications),
        'category_counts': dict(category_counts),
        'per_level': per_level,
        'per_hidden_size': per_hidden,
        'classifications': all_classifications,
    }

    if save_path is not None:
        save_results_json(summary, save_path)
        logger.info("Saved classification results to %s", save_path)

    return summary


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_classification_summary(results):
    """Print formatted summary of zombie test results.

    Parameters
    ----------
    results : dict
        Output from classify_all().
    """
    total = results.get('total_variables', 0)
    counts = results.get('category_counts', {})

    print("\n" + "=" * 70)
    print("  L5PC DESCARTES -- Variable Classification Summary")
    print("=" * 70)
    print(f"\n  Total variables probed: {total}\n")

    # Category breakdown
    print("  Category Breakdown:")
    print("  " + "-" * 50)
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        pct = (n / max(total, 1)) * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat:<28s}  {n:4d}  ({pct:5.1f}%)  {bar}")
    print()

    # Per-level breakdown
    per_level = results.get('per_level', {})
    for level in ['A', 'B', 'C']:
        if level not in per_level:
            continue
        ldata = per_level[level]
        print(f"  Level {level} ({ldata['total']} variables):")
        cats = ldata.get('categories', {})
        for cat in CATEGORIES:
            n = cats.get(cat, 0)
            if n > 0:
                print(f"    {cat:<28s}  {n:4d}")
        print()

    # Per-hidden-size breakdown
    per_hs = results.get('per_hidden_size', {})
    if per_hs:
        print("  Per Hidden Size:")
        print("  " + "-" * 50)
        for hs in sorted(per_hs.keys(), key=lambda x: int(x)):
            hsdata = per_hs[hs]
            cats = hsdata.get('categories', {})
            n_mandatory = sum(
                cats.get(c, 0) for c in CATEGORIES
                if c.startswith('MANDATORY')
            )
            n_zombie = cats.get('ZOMBIE', 0)
            print(f"    h={hs:>3s}:  {hsdata['total']:3d} vars  "
                  f"| {n_zombie:3d} zombie  | {n_mandatory:3d} mandatory")
        print()

    # Key findings
    n_mandatory = sum(counts.get(c, 0) for c in CATEGORIES
                      if c.startswith('MANDATORY'))
    n_zombie = counts.get('ZOMBIE', 0)
    n_byproduct = counts.get('LEARNED_BYPRODUCT', 0)
    n_voltage = counts.get('VOLTAGE_REENCODING', 0)

    print("  Key Findings:")
    print("  " + "-" * 50)
    if n_zombie > 0:
        print(f"    {n_zombie} variables are ZOMBIE "
              "(not represented beyond chance)")
    if n_voltage > 0:
        print(f"    {n_voltage} variables are VOLTAGE_REENCODING "
              "(trivially from V)")
    if n_byproduct > 0:
        print(f"    {n_byproduct} variables are LEARNED_BYPRODUCT "
              "(present but not used)")
    if n_mandatory > 0:
        print(f"    {n_mandatory} variables are MANDATORY "
              "(causally used by the network)")
    print("=" * 70 + "\n")
