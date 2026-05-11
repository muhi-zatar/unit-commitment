"""
V5 Phase 1: SCUC network-size sweep with HiGHS (+ SCIP if installed).

Networks (small -> large): case30, rts_gmlc, ieee118, case300.
Per-network horizon caps to keep wall-clock reasonable:
  case30   -> T=24
  rts_gmlc -> T=24 (real UC data)
  ieee118  -> T=12 (synth)
  case300  -> T=8  (synth, large)

Seeds default to 1 since most loaders are deterministic; multi-seed only
matters for MATPOWER+synth where the daily load curve has small noise.

Output: phase1_v5_results.csv (atomic checkpoint per run).
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
for p in (PARENT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from v5_common import _v5_worker, checkpoint_csv, force_single_thread_env  # noqa: E402

OUTPUT_CSV = os.path.join(THIS_DIR, 'phase1_v5_results.csv')

# (network, horizon, time_limit) — large networks get tight horizon + budget.
NETWORK_SPECS = [
    ('case30',   24, 300.0),
    ('rts_gmlc', 24, 600.0),
    ('ieee118',  12, 600.0),
    ('case300',   8, 900.0),
]
SEEDS = [42]
SOLVERS = ['highs', 'scip']


def _has_scip():
    try:
        import pyscipopt  # noqa: F401
        return True
    except ImportError:
        return False


def build_tasks(quick=False, networks=None, solvers=None,
                seeds=None, rts_path=None):
    networks = networks or [s[0] for s in NETWORK_SPECS]
    solvers = solvers or SOLVERS
    seeds = seeds or SEEDS
    specs = {s[0]: s for s in NETWORK_SPECS}
    tasks = []
    for net in networks:
        if net not in specs:
            print(f"WARNING: unknown network {net!r}; skipping", flush=True)
            continue
        _, horizon, time_limit = specs[net]
        if quick:
            horizon = min(6, horizon)
            time_limit = 60.0
        for solver in solvers:
            for seed in seeds:
                tasks.append({
                    'network': net,
                    'solver': solver,
                    'horizon': horizon,
                    'seed': seed,
                    'time_limit': time_limit,
                    'gap_tol': 0.001,
                    'rts_path': rts_path,
                })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--networks', nargs='+', default=None)
    parser.add_argument('--solvers', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--skip-scip', action='store_true')
    parser.add_argument('--rts-path', default=None,
                        help='Override RTS-GMLC SourceData path.')
    args = parser.parse_args()

    force_single_thread_env()

    have_scip = (not args.skip_scip) and _has_scip()
    solvers = args.solvers
    if solvers is None:
        solvers = ['highs', 'scip'] if have_scip else ['highs']
    else:
        if 'scip' in solvers and not have_scip:
            print("WARNING: pyscipopt not installed; dropping scip from sweep")
            solvers = [s for s in solvers if s != 'scip']

    tasks = build_tasks(quick=args.quick, networks=args.networks,
                        solvers=solvers, seeds=args.seeds,
                        rts_path=args.rts_path)
    print(f"V5 Phase 1: {len(tasks)} runs ({len(solvers)} solver(s), "
          f"{args.workers} workers).", flush=True)
    if not tasks:
        sys.exit(0)
    for t in tasks:
        print(f"  {t['network']:10s} solver={t['solver']:5s} "
              f"T={t['horizon']:2d} seed={t['seed']}")

    rows = []
    t0 = time.perf_counter()
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            r = _v5_worker(task)
            rows.append(r); checkpoint_csv(rows, args.output)
            print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id'):28s} "
                  f"solver={r.get('solver'):5s} "
                  f"nodes={r.get('mip_node_count')} "
                  f"time={r.get('mip_total_time_s')} "
                  f"status={r.get('status')}", flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_v5_worker, tasks), 1):
                rows.append(r); checkpoint_csv(rows, args.output)
                print(f"[{i:3d}/{len(tasks)}] {r.get('instance_id'):28s} "
                      f"solver={r.get('solver'):5s} "
                      f"nodes={r.get('mip_node_count')} "
                      f"time={r.get('mip_total_time_s')} "
                      f"status={r.get('status')}", flush=True)

    print(f"V5 Phase 1 done in {time.perf_counter() - t0:.0f}s "
          f"({len(rows)} rows).", flush=True)


if __name__ == '__main__':
    main()
