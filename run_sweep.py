"""
UC experiment sweep.

Sweeps:
  - n_gen in {10, 25, 50, 100, 200, 400}
  - horizon in {24, 72}  (day-ahead, 3-day-ahead)
  - formulation in {three_bin, three_bin_tight, perspective}
  - 3 random replicates per cell

For each cell we record total time, LP relax time, B&C time, node count,
LP iterations, gap, and structural counts (vars, constrs, nnz).
"""

import time
import sys
import pandas as pd
from pathlib import Path

from uc_instance import generate_uc_instance
from uc_solver import solve_uc


def run_cell(n_gen, horizon, formulation, seed, time_limit, gap_tol):
    """Run a single cell of the experiment."""
    try:
        inst = generate_uc_instance(n_gen, horizon, seed=seed)
        r = solve_uc(inst, formulation=formulation,
                     time_limit=time_limit, gap_tol=gap_tol)
        r['seed'] = seed
        r['error'] = None
        return r
    except Exception as e:
        return {
            'n_gen': n_gen,
            'horizon': horizon,
            'formulation': formulation,
            'seed': seed,
            'error': str(e),
            'converged': False,
        }


def run_sweep(grid, output_csv, time_limit, gap_tol, verbose=True):
    """Run the full experimental grid with checkpointing."""
    rows = []
    sweep_t0 = time.perf_counter()
    total = sum(len(reps) for reps in grid.values())
    cell_idx = 0

    for (n_gen, horizon, formulation), seeds in grid.items():
        for seed in seeds:
            cell_idx += 1
            elapsed = time.perf_counter() - sweep_t0
            if verbose:
                print(f"[{cell_idx:3d}/{total}] N_g={n_gen:3d} T={horizon:2d} "
                      f"{formulation:18s} seed={seed}  (elapsed: {elapsed:.0f}s)",
                      flush=True)

            cell_t0 = time.perf_counter()
            r = run_cell(n_gen, horizon, formulation, seed, time_limit, gap_tol)
            cell_dt = time.perf_counter() - cell_t0

            if verbose:
                lp = r.get('lp_relax_time_s', 0)
                mip = r.get('mip_total_time_s', 0)
                nodes = r.get('mip_node_count', 0)
                conv = r.get('converged', False)
                print(f"      ... LP={lp:.2f}s, MIP={mip:.2f}s, nodes={nodes}, "
                      f"converged={conv} (cell wall: {cell_dt:.1f}s)",
                      flush=True)

            rows.append(r)
            # Checkpoint every cell
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    if verbose:
        print(f"\nSwept {len(rows)} cells in {time.perf_counter() - sweep_t0:.1f}s")
    return pd.DataFrame(rows)


def main():
    # Conservative grid that should finish in ~10-15 min
    # Adjust n_replicates and large-N cells based on available compute
    n_gens = [10, 25, 50, 100, 200]
    horizons = [24, 72]
    formulations = ['three_bin', 'three_bin_tight', 'perspective']
    n_replicates = 2

    grid = {}
    for ng in n_gens:
        for T in horizons:
            for form in formulations:
                seeds = [42 + r * 17 for r in range(n_replicates)]
                grid[(ng, T, form)] = seeds

    output_csv = '/home/claude/uc_experiment/uc_results.csv'
    df = run_sweep(grid, output_csv, time_limit=60.0, gap_tol=0.001,
                   verbose=True)

    print("\n=== Summary ===")
    summary = df[df['converged']].groupby(['n_gen', 'horizon', 'formulation']).agg({
        'mip_total_time_s': ['mean', 'min', 'max'],
        'mip_node_count': 'mean',
    }).round(3)
    print(summary)


if __name__ == '__main__':
    main()
