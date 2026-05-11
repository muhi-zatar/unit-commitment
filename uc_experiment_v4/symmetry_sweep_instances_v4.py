"""
V4 synthetic generator: wraps v3's make_symmetry_sweep_instance and adds
the four extra fields the v4 builder reads.

Fields added:
  - ramp_startup_limit, ramp_shutdown_limit  (for F2)
  - startup_categories  (for F3): list of (lag_lo, lag_hi, cost) per unit,
                                  3 brackets — hot, warm, cold
  - must_run            (for F4): boolean array; ~20% of units, chosen from
                                  the smallest-pmin units in the diverse
                                  block to avoid breaking feasibility at
                                  low-demand hours. Identical block is
                                  never marked must-run (would collapse
                                  the symmetry signal).
"""

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
V3_DIR = os.path.join(PARENT, 'uc_experiment_v3')
if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

from symmetry_sweep_instances import make_symmetry_sweep_instance  # noqa: E402
from instance_adapter import to_formulation_format  # noqa: E402


# Fraction of diverse-block units to mark as must-run.
DEFAULT_MUSTRUN_FRAC = 0.20

# Startup category brackets (lag_lo, lag_hi, cost_multiplier)
DEFAULT_STARTUP_CATS = [
    (1, 3, 1.0),       # hot start (within min_dn+small window)
    (4, 8, 1.5),       # warm
    (9, 9999, 2.5),    # cold (residual)
]


def make_symmetry_sweep_instance_v4(N_g, k_identical, horizon, seed,
                                    reserve_frac=0.15,
                                    identical_type='midmerit',
                                    mustrun_frac=DEFAULT_MUSTRUN_FRAC,
                                    startup_cats=DEFAULT_STARTUP_CATS):
    """
    Build a V4 instance: v3's symmetry-sweep instance + the four extra
    fields the v4 builder needs.

    Returns a formulation-format inst dict (same shape v3 uses), already
    passed through instance_adapter.to_formulation_format.
    """
    hard = make_symmetry_sweep_instance(N_g, k_identical, horizon, seed,
                                        reserve_frac=reserve_frac,
                                        identical_type=identical_type)
    inst = to_formulation_format(hard)

    n_gen = inst['n_gen']
    pmin = inst['P_min']
    pmax = inst['P_max']

    # F2: separate startup/shutdown ramp limits. Make them generous enough
    # that off units can be started, but tighter than full ramp_up so the
    # feature has a measurable effect.
    inst['ramp_startup_limit']  = pmin + 0.4 * (pmax - pmin)
    inst['ramp_shutdown_limit'] = pmin + 0.4 * (pmax - pmin)

    # F3: lag-dependent startup costs. Each unit gets the same bracket
    # structure with the unit-specific cost_su as the hot-start baseline.
    cost_su = inst['cost_su']
    inst['startup_categories'] = [
        [(int(lo), int(hi), float(cost_su[i] * mult))
         for (lo, hi, mult) in startup_cats]
        for i in range(n_gen)
    ]

    # F4: must-run. Identical block occupies indices [0, k_identical);
    # diverse block is [k_identical, n_gen). Mark the smallest-pmin units
    # in the diverse block to minimise feasibility risk at low-demand hours.
    must_run = np.zeros(n_gen, dtype=bool)
    n_mustrun = max(1, int(round(mustrun_frac * (n_gen - k_identical))))
    diverse_idx = np.arange(k_identical, n_gen)
    # Sort diverse units by pmin, smallest first
    pmin_order = diverse_idx[np.argsort(pmin[diverse_idx])]
    picked = pmin_order[:n_mustrun]
    must_run[picked] = True
    inst['must_run'] = must_run

    # Forward the V3 metadata for analysis
    inst['k_identical'] = k_identical
    inst['N_g'] = N_g
    return inst


if __name__ == '__main__':
    from itertools import product
    from uc_experiment_v4.uc_hard_solver_v4 import solve_uc_v4

    print(f"{'feats':6s} {'N_g':>4s} {'k':>3s} {'status':>10s} "
          f"{'nodes':>6s} {'time':>7s} {'gap%':>7s}")
    print('-' * 60)
    for N_g, k in [(12, 1), (12, 6), (12, 12)]:
        inst = make_symmetry_sweep_instance_v4(N_g, k, horizon=12, seed=42)
        # Show 4 representative configs: baseline, F1 only, F4 only, all
        for combo, label in [((0,0,0,0), 'base'),
                             ((1,0,0,0), 'F1'),
                             ((0,0,0,1), 'F4'),
                             ((1,1,1,1), 'all')]:
            feats = dict(zip(['F1','F2','F3','F4'], [bool(x) for x in combo]))
            r = solve_uc_v4(inst, features=feats, time_limit=20)
            print(f"{label:6s} {N_g:>4d} {k:>3d} {r['status']:>10s} "
                  f"{r['mip_node_count']:>6d} "
                  f"{r['mip_total_time_s']:>6.2f}s "
                  f"{r['root_lp_gap_pct']:>6.3f}%")
