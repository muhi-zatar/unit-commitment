"""
Diagnose: what really creates branching in UC?

Hypothesis: branching is forced when the LP relaxation places fractional
values on u_it because no integer commitment exactly meets the requirement.

This happens when (demand at hour t) / (capacity per identical unit) is
non-integer. With 8 identical 100 MW units and demand=350, the LP wants
to commit 3.5 units -> branches on u.

Let's construct exactly that.
"""

import numpy as np
import highspy

from uc_hard_instance import generate_hard_uc, FLEET_RECIPES, _sample_unit, TYPE_TEMPLATES
from instance_adapter import to_formulation_format
from uc_hard_solver import solve_uc_with_settings


def make_targeted_hard_inst(n_identical, unit_pmax, target_demand_pattern, horizon=24, seed=0):
    """Construct instance with n_identical units, each unit_pmax MW.
    target_demand_pattern: list of len horizon, each value is the demand."""
    rng = np.random.default_rng(seed)

    # All units identical
    pmax = np.full(n_identical, float(unit_pmax))
    pmin = pmax * 0.30  # 30% min load
    a_cost = np.full(n_identical, 0.0005)
    b_cost = np.full(n_identical, 30.0)
    c_cost = np.full(n_identical, 200.0)
    su_cost = np.full(n_identical, 5000.0)
    sd_cost = np.full(n_identical, 1500.0)
    min_up = np.full(n_identical, 4, dtype=int)
    min_dn = np.full(n_identical, 4, dtype=int)
    ramp = pmax * 0.4

    demand = np.array(target_demand_pattern, dtype=float)
    reserve = 0.10 * demand

    initial_on = np.zeros(n_identical, dtype=int)
    initial_on[:n_identical // 2] = 1  # half on initially

    return {
        'n_gen': n_identical,
        'horizon': horizon,
        'pmax': pmax, 'pmin': pmin,
        'a_cost': a_cost, 'b_cost': b_cost, 'c_cost': c_cost,
        'su_cost': su_cost, 'sd_cost': sd_cost,
        'min_up': min_up, 'min_dn': min_dn,
        'ramp': ramp,
        'demand': demand, 'reserve': reserve,
        'initial_on': initial_on,
        'initial_uptime': np.where(initial_on == 1, min_up, 0),
        'initial_dntime': np.where(initial_on == 0, min_dn, 0),
        'reserve_frac': 0.10,
        'symmetry_score': n_identical * (n_identical - 1) / 2,
        'plateau_factor': 0.0,
    }


if __name__ == '__main__':
    print("=== Targeted instances designed to force branching ===\n")

    # 8 identical 100 MW units; demand pattern oscillates around fractional commits
    n_id = 8
    unit_pmax = 100.0
    horizon = 24

    # Pattern 1: demand sits right between integer commitment counts
    # If a unit produces 30-100 MW, then 4 units produce 120-400 MW
    # Demand of 250 forces between 3 and 4 units, with 3 units at 83 MW each (above pmin)
    # vs 4 units at 62.5 MW each (above pmin). LP will prefer fractional.
    pattern_fractional = [250 + 30 * np.sin(t * np.pi / 12) for t in range(horizon)]

    print("Test 1: 8 identical units, fractional-commit demand pattern")
    hard = make_targeted_hard_inst(n_id, unit_pmax, pattern_fractional, horizon=horizon)
    inst = to_formulation_format(hard)
    print(f"  n_gen={inst['n_gen']}, total_cap={hard['pmax'].sum():.0f}, "
          f"peak_demand={hard['demand'].max():.0f}")

    for name, settings in [
        ('default',     dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=True)),
        ('no_symmetry', dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=False)),
        ('all_off',     dict(presolve='off', heuristics_effort=0.0,  detect_symmetry=False)),
    ]:
        r = solve_uc_with_settings(inst, formulation='three_bin',
                                    time_limit=60, **settings)
        print(f"  {name:14s}: time={r['mip_total_time_s']:6.2f}s, "
              f"nodes={r['mip_node_count']:7d}, gap={r['root_lp_gap_pct']:.3f}%")

    print()
    # Pattern 2: scale up the symmetry to 20 identical units
    n_id = 20
    pattern2 = [600 + 80 * np.sin(t * np.pi / 12) for t in range(horizon)]
    print("Test 2: 20 identical units, fractional-commit pattern")
    hard = make_targeted_hard_inst(n_id, unit_pmax, pattern2, horizon=horizon)
    inst = to_formulation_format(hard)
    print(f"  n_gen={inst['n_gen']}, total_cap={hard['pmax'].sum():.0f}, "
          f"peak_demand={hard['demand'].max():.0f}")
    for name, settings in [
        ('default',     dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=True)),
        ('no_symmetry', dict(presolve='on',  heuristics_effort=0.05, detect_symmetry=False)),
        ('all_off',     dict(presolve='off', heuristics_effort=0.0,  detect_symmetry=False)),
    ]:
        r = solve_uc_with_settings(inst, formulation='three_bin',
                                    time_limit=60, **settings)
        print(f"  {name:14s}: time={r['mip_total_time_s']:6.2f}s, "
              f"nodes={r['mip_node_count']:7d}, gap={r['root_lp_gap_pct']:.3f}%")
