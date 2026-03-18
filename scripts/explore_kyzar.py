#!/usr/bin/env python3
"""
explore_kyzar.py

DESCARTES Circuit 6 Phase 0.2/0.3: Explore Kyzar dataset and apply quality gates.
Processes all NWB files from DANDI 000469, extracts neuron counts per region,
trial info, and behavioral performance. Outputs summary table and quality gates.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('explore_kyzar')

# Region mapping for DESCARTES I/O
INPUT_REGIONS = ['hippocampus', 'amygdala']
OUTPUT_REGIONS = ['dorsal_anterior_cingulate_cortex', 'pre_supplementary_motor_area', 'vmPFC']

def normalize_region(location_str):
    """Map NWB electrode location to standardized region name."""
    loc = location_str.lower().replace(' ', '_')
    if 'hippocampus' in loc:
        return 'hippocampus'
    elif 'amygdala' in loc:
        return 'amygdala'
    elif 'anterior_cingulate' in loc or 'dacc' in loc:
        return 'dACC'
    elif 'supplementary_motor' in loc or 'pre_sma' in loc or 'presma' in loc:
        return 'preSMA'
    elif 'prefrontal' in loc or 'vmpfc' in loc or 'vmPFC' in loc:
        return 'vmPFC'
    else:
        return loc


def is_limbic(region):
    return region in ['hippocampus', 'amygdala']


def is_frontal(region):
    return region in ['dACC', 'preSMA', 'vmPFC']


def _to_python(obj):
    """Convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def explore_nwb(nwb_path):
    """Extract key info from one NWB file."""
    import pynwb

    io = pynwb.NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()

    info = {
        'path': str(nwb_path),
        'subject_id': str(nwb.subject.subject_id) if nwb.subject else 'unknown',
        'session_id': str(nwb.session_id),
        'description': str(nwb.experiment_description)[:200] if nwb.experiment_description else '',
    }

    # Check if Sternberg task
    desc = (nwb.experiment_description or '').lower()
    info['is_sternberg'] = 'sternberg' in desc

    # Units (neurons)
    unit_info = []
    region_counts = {}

    if nwb.units is not None and len(nwb.units) > 0:
        for i in range(len(nwb.units)):
            spikes = nwb.units['spike_times'][i]
            n_spikes = len(spikes)

            # Get brain region from electrode
            electrode_idx = nwb.units['electrodes'][i].index[0]
            raw_location = nwb.electrodes['location'][electrode_idx]
            region = normalize_region(raw_location)

            # Compute mean firing rate
            if n_spikes >= 2:
                duration = spikes[-1] - spikes[0]
                fr = n_spikes / duration if duration > 0 else 0
            else:
                fr = 0

            snr = nwb.units['waveforms_mean_snr'][i] if 'waveforms_mean_snr' in nwb.units.colnames else 0

            unit_info.append({
                'unit_id': i,
                'region': region,
                'raw_location': raw_location,
                'n_spikes': int(n_spikes),
                'firing_rate': float(fr),
                'snr': float(snr),
            })

            region_counts[region] = region_counts.get(region, 0) + 1

    info['n_units'] = len(unit_info)
    info['region_counts'] = region_counts
    info['units'] = unit_info

    # Count limbic vs frontal
    n_limbic = sum(1 for u in unit_info if is_limbic(u['region']))
    n_frontal = sum(1 for u in unit_info if is_frontal(u['region']))
    info['n_limbic'] = n_limbic
    info['n_frontal'] = n_frontal

    # Trials
    if nwb.trials is not None:
        n_trials = len(nwb.trials)
        info['n_trials'] = n_trials
        info['trial_columns'] = list(nwb.trials.colnames)

        # Trial durations
        starts = np.array([nwb.trials['start_time'][i] for i in range(n_trials)])
        stops = np.array([nwb.trials['stop_time'][i] for i in range(n_trials)])
        durations = stops - starts
        info['trial_duration_mean'] = float(np.mean(durations))
        info['trial_duration_std'] = float(np.std(durations))
    else:
        info['n_trials'] = 0

    # Events
    if 'events' in nwb.acquisition:
        events = nwb.acquisition['events']
        event_data = events.data[:]
        unique_events = np.unique(event_data)
        info['event_codes'] = {str(e): int(np.sum(event_data == e)) for e in unique_events}
    else:
        info['event_codes'] = {}

    # Has continuous spike timestamps?
    info['has_spike_timestamps'] = any(u['n_spikes'] > 0 for u in unit_info)

    io.close()
    return info


def apply_quality_gates(session_info, min_fr=0.5):
    """Apply quality gates to one session."""
    gates = {}

    # Gate 1: Sternberg task
    gates['gate1_sternberg'] = session_info['is_sternberg']

    # Gate 2: >= 5 neurons in both limbic and frontal
    # Filter by firing rate first
    good_units = [u for u in session_info['units'] if u['firing_rate'] >= min_fr]
    n_limbic_good = sum(1 for u in good_units if is_limbic(u['region']))
    n_frontal_good = sum(1 for u in good_units if is_frontal(u['region']))
    gates['gate2_neurons'] = n_limbic_good >= 5 and n_frontal_good >= 5
    gates['n_limbic_good'] = n_limbic_good
    gates['n_frontal_good'] = n_frontal_good

    # Gate 3: >= 50 trials (we can't easily check correctness without
    # full event parsing, so just check trial count)
    gates['gate3_trials'] = session_info['n_trials'] >= 50

    # Gate 4: Firing rate filter already applied above
    gates['gate4_applied'] = True

    gates['overall_pass'] = all([
        gates['gate1_sternberg'],
        gates['gate2_neurons'],
        gates['gate3_trials'],
    ])

    return gates


def main():
    parser = argparse.ArgumentParser(
        description='Explore Kyzar DANDI 000469 dataset')
    parser.add_argument('--data-dir', required=True,
                        help='Path to kyzar_raw/ with NWB files')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-fr', type=float, default=0.5,
                        help='Minimum firing rate threshold (Hz)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NWB files
    nwb_files = sorted(data_dir.glob('**/*.nwb'))
    log.info("Found %d NWB files", len(nwb_files))

    all_info = []
    for nwb_path in nwb_files:
        log.info("Processing %s...", nwb_path.name)
        try:
            info = explore_nwb(nwb_path)
            gates = apply_quality_gates(info, args.min_fr)
            info['quality_gates'] = gates
            all_info.append(info)
        except Exception as exc:
            log.warning("  FAILED: %s", exc)
            all_info.append({'path': str(nwb_path), 'error': str(exc)})

    # Summary table
    log.info("\n" + "=" * 110)
    log.info("KYZAR DATASET SUMMARY")
    log.info("=" * 110)
    log.info("%-35s %4s %4s %4s %4s %4s %4s %6s %6s %5s %6s",
             'File', 'HPC', 'AMY', 'dACC', 'SMA', 'vmPF', 'Tot',
             'Limb', 'Front', 'Trial', 'Pass')
    log.info("-" * 110)

    n_pass = 0
    for info in all_info:
        if 'error' in info:
            log.info("%-35s  ERROR: %s", Path(info['path']).name, info['error'])
            continue

        rc = info['region_counts']
        gates = info['quality_gates']
        hpc = rc.get('hippocampus', 0)
        amy = rc.get('amygdala', 0)
        dacc = rc.get('dACC', 0)
        sma = rc.get('preSMA', 0)
        vmpfc = rc.get('vmPFC', 0)
        tot = info['n_units']
        limb_g = gates['n_limbic_good']
        front_g = gates['n_frontal_good']
        passed = 'PASS' if gates['overall_pass'] else 'FAIL'
        if gates['overall_pass']:
            n_pass += 1

        fname = Path(info['path']).name
        log.info("%-35s %4d %4d %4d %4d %4d %4d %6d %6d %5d %6s",
                 fname[:35], hpc, amy, dacc, sma, vmpfc, tot,
                 limb_g, front_g, info['n_trials'], passed)

    log.info("-" * 110)
    log.info("%d of %d sessions pass all quality gates",
             n_pass, len([i for i in all_info if 'error' not in i]))

    # Unique regions found
    all_regions = set()
    for info in all_info:
        if 'error' not in info:
            all_regions.update(info['region_counts'].keys())
    log.info("Unique brain regions: %s", sorted(all_regions))

    # Save
    with open(output_dir / 'exploration.json', 'w') as f:
        json.dump(_to_python(all_info), f, indent=2)
    log.info("Saved: %s", output_dir / 'exploration.json')

    # Save quality gates summary
    passing = [{'subject': i['subject_id'], 'session': i['session_id'],
                'n_limbic': i['quality_gates']['n_limbic_good'],
                'n_frontal': i['quality_gates']['n_frontal_good'],
                'n_trials': i['n_trials']}
               for i in all_info
               if 'error' not in i and i['quality_gates']['overall_pass']]
    with open(output_dir / 'quality_gates.json', 'w') as f:
        json.dump(_to_python({
            'total_sessions': len([i for i in all_info if 'error' not in i]),
            'passing_sessions': n_pass,
            'passing': passing
        }), f, indent=2)
    log.info("Saved: %s", output_dir / 'quality_gates.json')


if __name__ == '__main__':
    main()
