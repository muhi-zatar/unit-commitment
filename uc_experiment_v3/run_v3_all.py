"""
Top-level driver: run phases 1->5 in order.

Usage:
    python uc_experiment_v3/run_v3_all.py                 # all phases
    python uc_experiment_v3/run_v3_all.py --phase 1       # just one phase
    python uc_experiment_v3/run_v3_all.py --quick         # tiny smoke run
    python uc_experiment_v3/run_v3_all.py --analyze-only  # just regenerate plots

Empirical wall-clock: a few minutes total on a modern workstation. HiGHS
solves nearly every cell at the root node, so the originally-estimated
4-10 h is far off — expect minutes. Some PGLib instances are infeasible
under our adapter's lossy conversion; those rows show status=Infeasible
in the CSV and are excluded from the analysis plots.
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_5_ENABLED = True


PHASES = {
    '1': ('run_phase1_symmetry_sweep.py',  'analyze_phase1.py'),
    '2': ('run_phase2_detect_symmetry.py', 'analyze_phase2.py'),
    '3': ('run_phase3_mode_ablation.py',   'analyze_phase3.py'),
    '4': ('run_phase4_solver_bakeoff.py',  'analyze_phase4.py'),
    '5': ('run_phase5_pglib.py',           'analyze_phase5.py'),
}


def run_script(name, extra_args):
    path = os.path.join(THIS_DIR, name)
    cmd = [sys.executable, path] + list(extra_args)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    default_phases = ['1', '2', '3', '4'] + (['5'] if PHASE_5_ENABLED else [])
    parser.add_argument('--phase', choices=list(PHASES) + ['all'], default='all',
                        help=f'Default "all" runs phases {default_phases}.')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--skip-scip', action='store_true')
    args = parser.parse_args()

    if args.phase == 'all':
        phases = ['1', '2', '3', '4'] + (['5'] if PHASE_5_ENABLED else [])
    else:
        phases = [args.phase]
    sweep_args = []
    if args.quick:
        sweep_args.append('--quick')
    if args.workers is not None:
        sweep_args.extend(['--workers', str(args.workers)])

    for p in phases:
        sweep_script, analyze_script = PHASES[p]
        if not args.analyze_only:
            extra = list(sweep_args)
            if p in ('4', '5') and args.skip_scip:
                extra.append('--skip-scip')
            rc = run_script(sweep_script, extra)
            if rc != 0:
                print(f"!!! phase {p} sweep returned {rc}; continuing", flush=True)
        # Analysis always runs (cheap, regenerates plots)
        rc = run_script(analyze_script, [])
        if rc != 0:
            print(f"!!! phase {p} analysis returned {rc}; continuing", flush=True)



if __name__ == '__main__':
    main()
