"""
V4 PGLib-UC adapter.

Calls v3's load_pglib_instance to get the baseline inst dict, then re-attaches
the raw fields v3 discarded so the V4 builder can use them:

  - startup_categories  : from the PGLib `startup` list (lag/cost pairs),
                          packed into 3 brackets (hot/warm/cold)
  - ramp_startup_limit  : from PGLib `ramp_startup_limit` field
  - ramp_shutdown_limit : from PGLib `ramp_shutdown_limit` field
  - must_run            : from PGLib `must_run` field (rare in bundled cases)
  - cost_a, cost_b are also recomputed via least-squares on the full
                          piecewise_production table, so F1 with 4 segments
                          uses a tighter convex fit than the v3 endpoint
                          linearisation.

The function reuses the V3 net_renewables flag and returns the same
formulation-format dict shape.
"""

import json
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

from pglib_uc_adapter import load_pglib_instance  # noqa: E402


# PGLib's startup list has 1-3 lag/cost entries. We bin them into 3 fixed
# brackets so every unit has the same n_startup_cats (required by the v4
# builder). Brackets are by lag (hours offline):
#   hot:  [1, 3]
#   warm: [4, 12]
#   cold: [13, 9999]
# We use the PGLib startup cost whose lag falls in each bracket (or the
# closest available; PGLib's costs are monotone in lag).
DEFAULT_BRACKETS = [(1, 3), (4, 12), (13, 9999)]


def _pack_startup_brackets(startup_list, brackets=DEFAULT_BRACKETS):
    """
    startup_list: list of {'lag': int, 'cost': float} from PGLib JSON.
    Returns list of (lag_lo, lag_hi, cost) tuples, one per bracket.
    """
    if not startup_list:
        return [(lo, hi, 0.0) for (lo, hi) in brackets]

    lags = np.array([s['lag'] for s in startup_list])
    costs = np.array([s['cost'] for s in startup_list])

    out = []
    for (lo, hi) in brackets:
        # Find the PGLib lag entry that falls inside [lo, hi]; if none,
        # use the smallest-lag entry whose lag >= lo, else the largest.
        in_bracket = (lags >= lo) & (lags <= hi)
        if in_bracket.any():
            idx = np.argmax(in_bracket)  # first True
            cost = float(costs[idx])
        else:
            ge = lags >= lo
            if ge.any():
                cost = float(costs[np.argmax(ge)])
            else:
                cost = float(costs[-1])  # fall back to largest available
        out.append((int(lo), int(hi), cost))
    return out


def _fit_convex_pwl(breakpoints, n_segments=4):
    """
    Convert PGLib piecewise_production breakpoints to (cost_a, cost_b, cost_nl)
    via least-squares fit cost(P) = a*P^2 + b*P + c on the breakpoints.

    breakpoints: list of {'mw': float, 'cost': float}
    Returns (a, b, c).
    """
    if len(breakpoints) < 2:
        return 0.0, 0.0, float(breakpoints[0]['cost']) if breakpoints else 0.0
    mw = np.array([bp['mw'] for bp in breakpoints], dtype=float)
    cost = np.array([bp['cost'] for bp in breakpoints], dtype=float)
    # Use degree-2 polyfit; coefficients are [a, b, c]
    coeffs = np.polyfit(mw, cost, deg=min(2, len(mw) - 1))
    if len(coeffs) == 3:
        a, b, c = coeffs
    elif len(coeffs) == 2:
        b, c = coeffs
        a = 0.0
    else:
        a = 0.0; b = 0.0; c = float(coeffs[0])
    return float(a), float(b), float(c)


def load_pglib_instance_v4(json_path, horizon=None, net_renewables=False):
    """
    Load a PGLib-UC case in V4 format. Same return shape as
    pglib_uc_adapter.load_pglib_instance plus must_run, startup_categories,
    ramp_startup_limit, ramp_shutdown_limit, and refit cost_a.
    """
    inst = load_pglib_instance(json_path,
                               horizon=horizon,
                               net_renewables=net_renewables)

    with open(json_path) as f:
        raw = json.load(f)

    therm = raw['thermal_generators']
    gen_names = list(therm.keys())
    n_gen = len(gen_names)

    must_run = np.zeros(n_gen, dtype=bool)
    ramp_su = np.zeros(n_gen)
    ramp_sd = np.zeros(n_gen)
    cost_a = np.zeros(n_gen)
    cost_b = np.zeros(n_gen)
    cost_nl = np.zeros(n_gen)
    startup_categories = []

    for i, name in enumerate(gen_names):
        g = therm[name]
        must_run[i] = bool(g.get('must_run', 0))
        ramp_su[i] = float(g.get('ramp_startup_limit',
                                 g['ramp_up_limit']))
        ramp_sd[i] = float(g.get('ramp_shutdown_limit',
                                 g['ramp_down_limit']))
        a, b, c = _fit_convex_pwl(g['piecewise_production'])
        cost_a[i] = a
        cost_b[i] = b
        cost_nl[i] = c
        startup_categories.append(_pack_startup_brackets(g.get('startup', [])))

    inst['must_run'] = must_run
    inst['ramp_startup_limit'] = ramp_su
    inst['ramp_shutdown_limit'] = ramp_sd
    inst['startup_categories'] = startup_categories
    # Overwrite the linearised cost with the quadratic fit so F1 has
    # something to work with.
    inst['cost_a'] = cost_a
    inst['cost_b'] = cost_b
    inst['cost_nl'] = cost_nl
    return inst


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python pglib_uc_adapter_v4.py <pglib_case.json>")
        sys.exit(1)
    inst = load_pglib_instance_v4(sys.argv[1], horizon=24)
    print(f"n_gen     = {inst['n_gen']}")
    print(f"horizon   = {inst['horizon']}")
    print(f"must_run  = {inst['must_run'].sum()} / {inst['n_gen']}")
    print(f"ramp_su   range = {inst['ramp_startup_limit'].min():.1f} .. "
          f"{inst['ramp_startup_limit'].max():.1f}")
    print(f"cost_a    range = {inst['cost_a'].min():.6f} .. "
          f"{inst['cost_a'].max():.6f}")
    print(f"startup_cats[0] = {inst['startup_categories'][0]}")
