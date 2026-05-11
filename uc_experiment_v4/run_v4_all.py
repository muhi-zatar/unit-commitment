"""
V4 top-level driver: run synth + pglib phases, then analyse.
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


PHASES = {
    '1': ('run_phase1_synth_v4.py', 'analyze_v4.py'),
    '2': ('run_phase2_pglib_v4.py', 'analyze_v4.py'),
}


def run_script(name, extra):
    path = os.path.join(THIS_DIR, name)
    cmd = [sys.executable, path] + list(extra)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['1', '2', 'all'], default='all')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--pglib-root', default=None,
                        help='Override PGLib root for phase 2.')
    args = parser.parse_args()

    phases = ['1', '2'] if args.phase == 'all' else [args.phase]
    extra = []
    if args.quick:
        extra.append('--quick')
    if args.workers is not None:
        extra.extend(['--workers', str(args.workers)])

    for p in phases:
        sweep, _ = PHASES[p]
        if not args.analyze_only:
            phase_extra = list(extra)
            if p == '2' and args.pglib_root:
                phase_extra.extend(['--pglib-root', args.pglib_root])
            rc = run_script(sweep, phase_extra)
            if rc != 0:
                print(f"!!! phase {p} sweep returned {rc}; continuing", flush=True)

    # Analysis reads both CSVs and produces a unified report
    run_script('analyze_v4.py', [])


if __name__ == '__main__':
    main()
