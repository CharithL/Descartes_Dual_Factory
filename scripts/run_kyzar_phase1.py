#!/usr/bin/env python3
"""
run_phase1.py

DESCARTES Circuit 6 Phase 1: Preprocess all passing Kyzar Sternberg
sessions and compute 18 biological probe target variables.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kyzar.preprocessing import process_session
from kyzar.bio_variables import compute_session_bio_targets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('phase1')

# 14 passing Sternberg sessions (ses-2 only)
PASSING_SUBJECTS = [
    '1', '2', '3', '4', '5', '7', '8',
    '10', '11', '13', '14', '15', '18', '21',
]


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Kyzar Phase 1: Preprocessing')
    parser.add_argument('--data-dir', required=True,
                        help='Path to kyzar_raw/ with NWB files')
    parser.add_argument('--output-dir', required=True,
                        help='Base path for processed output')
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='Specific subject IDs (default: all passing)')
    parser.add_argument('--dt-ms', type=int, default=10)
    parser.add_argument('--sigma-ms', type=int, default=30)
    parser.add_argument('--min-fr', type=float, default=0.5)
    args = parser.parse_args()

    subjects = args.subjects or PASSING_SUBJECTS
    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)

    log.info("Phase 1: Processing %d sessions", len(subjects))
    t0 = time.time()

    results = []
    for sub_id in subjects:
        nwb_path = data_dir / f'sub-{sub_id}' / f'sub-{sub_id}_ses-2_ecephys+image.nwb'

        if not nwb_path.exists():
            log.warning("NWB not found: %s", nwb_path)
            continue

        out_dir = output_base / f'session_sub{sub_id}_ses2'

        # Task 1.1: Spike extraction, binning, trial segmentation
        try:
            log.info("=" * 60)
            log.info("Subject %s: Task 1.1 (preprocessing)", sub_id)
            meta = process_session(
                nwb_path, out_dir,
                dt_ms=args.dt_ms, sigma_ms=args.sigma_ms,
                min_fr=args.min_fr)

            # Task 1.2: Biological probe targets
            log.info("Subject %s: Task 1.2 (bio variables)", sub_id)
            var_names = compute_session_bio_targets(
                out_dir, dt_ms=args.dt_ms)
            log.info("  %d variables: %s", len(var_names),
                     ', '.join(var_names[:5]) + '...')

            results.append({
                'subject': sub_id,
                'status': 'OK',
                'n_trials': meta['n_trials'],
                'n_limbic': meta['n_limbic'],
                'n_frontal': meta['n_frontal'],
                'n_variables': len(var_names),
            })

        except Exception as exc:
            log.error("Subject %s FAILED: %s", sub_id, exc)
            import traceback
            traceback.print_exc()
            results.append({
                'subject': sub_id,
                'status': 'FAILED',
                'error': str(exc),
            })

    dt = time.time() - t0

    # Summary
    log.info("\n" + "=" * 60)
    log.info("PHASE 1 COMPLETE (%.1f min)", dt / 60)
    log.info("=" * 60)

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    log.info("%d/%d sessions processed successfully", n_ok, len(results))

    for r in results:
        if r['status'] == 'OK':
            log.info("  sub-%s: %d trials, %d limbic, %d frontal, %d bio vars",
                     r['subject'], r['n_trials'], r['n_limbic'],
                     r['n_frontal'], r['n_variables'])
        else:
            log.info("  sub-%s: FAILED - %s", r['subject'],
                     r.get('error', '?'))


if __name__ == '__main__':
    main()
