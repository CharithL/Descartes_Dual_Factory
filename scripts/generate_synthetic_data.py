#!/usr/bin/env python3
"""Generate synthetic trial data for pipeline validation (no NEURON needed).

Creates trial files with the same structure as real simulations:
  - inputs:        (T, n_syn)        synaptic current traces
  - output:        (T,)              somatic voltage
  - level_A_gates: (T, n_gate_vars)  ion channel gate variables
  - level_B_cond:  (T, n_cond_vars)  effective conductances
  - level_C_emerge: (n_emerge,)      emergent properties

Also writes variable_names.json for probe target naming.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-trials 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    N_TRIALS, T_STEPS, TOTAL_SYN, BAHL_TRIAL_DIR,
    CHANNEL_SPECS, BAHL_REGIONS,
)


def generate_synthetic_trials(n_trials=N_TRIALS, output_dir=None, seed=42):
    """Generate synthetic trials mimicking biophysical simulation output."""
    if output_dir is None:
        output_dir = str(BAHL_TRIAL_DIR)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # --- Build variable name lists ---
    # Level A: gate variables per channel per compartment
    gate_names = []
    for chan, spec in CHANNEL_SPECS.items():
        for region in BAHL_REGIONS:
            for gate in spec['gates']:
                gate_names.append(f'{gate}_{chan}_{region}')
    n_gates = len(gate_names)

    # Level B: effective conductances per channel per compartment
    cond_names = []
    for chan in CHANNEL_SPECS:
        for region in BAHL_REGIONS:
            cond_names.append(f'geff_{chan}_{region}')
    n_cond = len(cond_names)

    # Level C: emergent properties
    emerge_names = [
        'spike_count', 'burst_ratio', 'mean_isi_ms',
        'ca_integral_nexus', 'bac_flag', 'first_spike_latency_ms',
    ]
    n_emerge = len(emerge_names)

    # Save variable names metadata
    meta = {
        'level_A': gate_names,
        'level_B': cond_names,
        'level_C': emerge_names,
    }
    meta_path = output_dir / 'variable_names.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved variable names: {len(gate_names)} A, "
          f"{len(cond_names)} B, {len(emerge_names)} C")

    # --- Generate each trial ---
    conditions = ['subthreshold', 'tonic', 'burst', 'bac', 'mixed']
    cond_weights = [50, 100, 100, 100, 150]  # rough match to STIM_CONDITIONS

    for trial_idx in range(n_trials):
        # Pick condition
        cond_idx = rng.choice(len(conditions),
                              p=np.array(cond_weights) / sum(cond_weights))
        condition = conditions[cond_idx]

        # Base firing rate depends on condition
        if condition == 'subthreshold':
            base_rate = rng.uniform(0.0, 0.05)
        elif condition == 'tonic':
            base_rate = rng.uniform(0.1, 0.4)
        elif condition == 'burst':
            base_rate = rng.uniform(0.3, 0.7)
        elif condition == 'bac':
            base_rate = rng.uniform(0.2, 0.5)
        else:  # mixed
            base_rate = rng.uniform(0.0, 0.6)

        # --- Synaptic inputs: (T, TOTAL_SYN) ---
        # Poisson-like synaptic currents
        inputs = rng.exponential(scale=base_rate + 0.01,
                                 size=(T_STEPS, TOTAL_SYN)).astype(np.float32)

        # --- Somatic voltage: (T,) ---
        # Leaky-integrate-and-fire-like trace with spikes
        v = np.full(T_STEPS, -70.0, dtype=np.float32)
        drive = np.mean(inputs, axis=1)
        for t in range(1, T_STEPS):
            dv = -0.05 * (v[t-1] + 70) + drive[t] * 100 + rng.normal(0, 0.5)
            v[t] = v[t-1] + dv
            if v[t] > 0:
                v[t] = 40.0  # spike peak
            elif v[t-1] > 0:
                v[t] = -75.0  # reset
            v[t] = np.clip(v[t], -90, 45)

        output = v

        # --- Level A gates: (T, n_gates) ---
        # Gate variables are in [0, 1], correlated with voltage
        v_norm = (v + 70) / 110  # normalise to ~[0, 1]
        gates = np.zeros((T_STEPS, n_gates), dtype=np.float32)
        for g in range(n_gates):
            tau = rng.uniform(0.5, 50.0)
            alpha = min(1.0, 0.5 / tau)
            noise = rng.normal(0, 0.02, size=T_STEPS)
            target = np.clip(v_norm + noise, 0, 1)
            gates[0, g] = 0.5
            for t in range(1, T_STEPS):
                gates[t, g] = (1 - alpha) * gates[t-1, g] + alpha * target[t]
            gates[:, g] = np.clip(gates[:, g], 0, 1)

        # --- Level B conductances: (T, n_cond) ---
        # g_eff = gbar * prod(gate^exp), approximated as correlated traces
        cond = np.zeros((T_STEPS, n_cond), dtype=np.float32)
        for c_idx in range(n_cond):
            gbar = rng.uniform(0.001, 0.5)
            # Use 1-2 gate columns to build conductance
            g_cols = rng.choice(n_gates, size=min(2, n_gates), replace=False)
            prod = np.ones(T_STEPS, dtype=np.float32)
            for gc in g_cols:
                prod *= gates[:, gc]
            cond[:, c_idx] = gbar * prod

        # --- Level C emergent: (n_emerge,) ---
        spike_times = np.where(np.diff((v > 0).astype(int)) > 0)[0]
        n_spikes = len(spike_times)
        isis = np.diff(spike_times) * 0.5  # in ms
        burst_r = float(np.sum(isis < 10) / max(len(isis), 1)) if len(isis) > 0 else 0.0
        mean_isi = float(np.mean(isis)) if len(isis) > 0 else 0.0
        ca_integral = float(rng.exponential(0.5))
        bac = 1.0 if (condition == 'bac' and n_spikes > 5) else 0.0
        first_lat = float(spike_times[0] * 0.5) if n_spikes > 0 else 1000.0

        emerge = np.array([
            float(n_spikes), burst_r, mean_isi,
            ca_integral, bac, first_lat,
        ], dtype=np.float32)

        # --- Save trial ---
        save_path = output_dir / f'trial_{trial_idx:03d}.npz'
        np.savez_compressed(
            save_path,
            inputs=inputs, output=output,
            level_A_gates=gates, level_B_cond=cond,
            level_B_curr=cond * 0.1,  # placeholder
            level_C_emerge=emerge,
        )

        if trial_idx % 50 == 0 or trial_idx == n_trials - 1:
            print(f"  Trial {trial_idx:3d}/{n_trials}  "
                  f"[{condition:12s}]  spikes={n_spikes:3d}")

    print(f"\n  Generated {n_trials} synthetic trials in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic trial data for pipeline validation')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help='Number of trials to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Synthetic Trial Data (no NEURON needed)")
    print("=" * 60)

    generate_synthetic_trials(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("=" * 60)
