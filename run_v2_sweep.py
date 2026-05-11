"""
UC v2 sweep: how does branching scale with fleet symmetry?

Key axes:
  fleet_size     : total number of generators (scaling)
  symmetry_level : 'diverse' / 'sym_groups_of_4' / 'sym_groups_of_8'
  formulation    : 'three_bin' / 'three_bin_tight'
  plateau        : 1.0 (random within type) / 0.0 (identical within type)

Measures:
  - MIP wall time
  - B&C node count
  - LP gap
  - LP relaxation time

Uses GLPK because HiGHS's symmetry detection collapses these instances to 1 node.
"""

import time
import sys
import pandas as pd
from pathlib import Path

from uc_hard_instance import generate_hard_uc, TYPE_TEMPLATES
from instance_adapter import to_formulation_format
from uc_glpk_solver import solve_uc_glpk


# Fleet recipes parameterized by total size and symmetry pattern
def make_fleet_recipe(n_total, pattern):
    """
    pattern: 'diverse' | 'sym4' | 'sym8'
        diverse: scattered across 3-4 types, ~no two identical
        sym4   : grouped into clusters of 4
        sym8   : grouped into clusters of 8
    """
    if pattern == 'diverse':
        # 30% baseload, 50% midmerit, 20% peaker; small per-type counts
        n_b = max(1, int(0.3 * n_total))
        n_m = max(1, int(0.5 * n_total))
        n_p = n_total - n_b - n_m
        return [(n_b, 'baseload_med'), (n_m, 'midmerit'), (n_p, 'peaker')]
    elif pattern == 'sym4':
        # Force groups of 4 of each type
        n_groups = n_total // 4
        # Distribute groups: 1/3 baseload, 1/2 midmerit, 1/6 peaker
        n_b_grp = max(1, n_groups // 3)
        n_m_grp = max(1, n_groups // 2)
        n_p_grp = n_groups - n_b_grp - n_m_grp
        recipe = []
        if n_b_grp > 0:
            recipe.append((4 * n_b_grp, 'baseload_med'))
        if n_m_grp > 0:
            recipe.append((4 * n_m_grp, 'midmerit'))
        if n_p_grp > 0:
            recipe.append((4 * n_p_grp, 'peaker'))
        return recipe
    elif pattern == 'sym8':
        n_groups = max(1, n_total // 8)
        # 1/3 baseload, 1/2 midmerit, 1/6 peaker
        n_b_grp = max(1, n_groups // 3)
        n_m_grp = max(1, n_groups // 2)
        n_p_grp = max(1, n_groups - n_b_grp - n_m_grp)
        return [(8 * n_b_grp, 'baseload_med'),
                (8 * n_m_grp, 'midmerit'),
                (8 * n_p_grp, 'peaker')]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def run_single_experiment(n_total, horizon, pattern, plateau, formulation,
                           seed, time_limit=120):
    recipe = make_fleet_recipe(n_total, pattern)
    hard = generate_hard_uc(recipe, horizon=horizon, reserve_frac=0.15,
                            plateau_factor=plateau, seed=seed)
    inst = to_formulation_format(hard)

    r = solve_uc_glpk(inst, formulation=formulation, time_limit=time_limit)
    r['n_total_intended'] = n_total
    r['fleet_pattern'] = pattern
    r['plateau'] = plateau
    r['symmetry_score'] = hard['symmetry_score']
    r['n_gen_actual'] = hard['n_gen']
    r['fleet_recipe'] = str(recipe)
    r['seed'] = seed
    return r


def main():
    # Tuned for GLPK's slowness: maxes out around 16 generators × 8 horizon
    network_sizes = [8, 12, 16]
    horizons = [6, 8]
    patterns = ['diverse', 'sym4', 'sym8']
    plateaus = [1.0, 0.0]
    formulations = ['three_bin']  # drop tight to halve runtime; can add later
    n_replicates = 2
    time_limit = 45  # tighter so stuck cells don't blow the budget

    output_csv = '/home/claude/uc_experiment_v2/uc_v2_results.csv'
    rows = []
    sweep_t0 = time.perf_counter()
    total_cells = (len(network_sizes) * len(horizons) * len(patterns) *
                   len(plateaus) * len(formulations) * n_replicates)
    cell = 0

    for n_total in network_sizes:
        for horizon in horizons:
            for pattern in patterns:
                for plateau in plateaus:
                    for formulation in formulations:
                        for replicate in range(n_replicates):
                            cell += 1
                            seed = 42 + replicate * 17
                            elapsed = time.perf_counter() - sweep_t0
                            print(f"[{cell:3d}/{total_cells}] "
                                  f"N={n_total} T={horizon} "
                                  f"{pattern} p={plateau} {formulation} rep={replicate} "
                                  f"(elapsed: {elapsed:.0f}s)",
                                  flush=True)
                            try:
                                r = run_single_experiment(
                                    n_total, horizon, pattern, plateau,
                                    formulation, seed, time_limit=time_limit)
                                rows.append(r)
                                print(f"   -> nodes={r['mip_node_count']}, "
                                      f"time={r['mip_total_time_s']:.2f}s, "
                                      f"gap={r['root_lp_gap_pct']:.3f}%",
                                      flush=True)
                            except Exception as e:
                                print(f"   ERROR: {e}", flush=True)
                                rows.append({
                                    'n_total_intended': n_total, 'horizon': horizon,
                                    'fleet_pattern': pattern, 'plateau': plateau,
                                    'formulation': formulation, 'seed': seed,
                                    'error': str(e), 'converged': False,
                                })
                            # Checkpoint
                            pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\nDone: {len(df)} rows, total wall: {time.perf_counter() - sweep_t0:.1f}s")


if __name__ == '__main__':
    main()
