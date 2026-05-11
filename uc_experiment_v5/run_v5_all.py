"""
V5 top-level driver: network-size sweep + analysis.
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(name, extra):
    path = os.path.join(THIS_DIR, name)
    cmd = [sys.executable, path] + list(extra)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--skip-scip', action='store_true')
    parser.add_argument('--networks', nargs='+', default=None)
    parser.add_argument('--rts-path', default=None)
    args = parser.parse_args()

    extra = []
    if args.quick: extra.append('--quick')
    if args.workers is not None: extra.extend(['--workers', str(args.workers)])
    if args.skip_scip: extra.append('--skip-scip')
    if args.networks: extra.extend(['--networks'] + args.networks)
    if args.rts_path: extra.extend(['--rts-path', args.rts_path])

    if not args.analyze_only:
        rc = run_script('run_phase1_network_sweep.py', extra)
        if rc != 0:
            print(f"!!! sweep returned {rc}; continuing to analysis", flush=True)

    run_script('analyze_v5.py', [])


if __name__ == '__main__':
    main()
