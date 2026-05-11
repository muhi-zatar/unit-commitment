"""
PGLib-UC JSON -> our formulation-format dict.

Maps the schema documented at https://github.com/power-grid-lib/pglib-uc
into the field names uc_formulations.build_three_bin expects.

Lossy conversions and the reasoning:

  - piecewise_production (convex piecewise linear) -> a single linear
    approximation: cost = cost_nl + cost_b * P, fit by matching cost at
    P_min and P_max. cost_a is set to 0. Phase 4 already linearizes the
    quadratic (n_pwl_segments=1), so this is consistent.

  - startup (off-time-dependent costs) -> the hot-start cost (smallest).
    Our three-bin formulation only has a single startup cost.

  - ramp_up / ramp_down -> max(ramp_up_limit, ramp_startup_limit) so off
    units can reach P_min in one hour (avoids spurious infeasibility).
    Symmetric on the shutdown side.

  - renewable_generators -> by default IGNORED (demand passed through raw).
    PGLib's renewables are dispatchable within [min, max], which requires
    curtailment variables our three-bin formulation does not have. Netting
    out renewable_maximum makes many cases infeasible (the pmin floor of
    initially-on thermals exceeds net demand). Pass `net_renewables=True`
    to subtract the renewable lower bound instead; this is closer to the
    PGLib obj but may still over-curtail at some hours.

  - must_run flag -> ignored. Most must_run units will be committed by
    economics anyway. A future improvement would add u[i,t] == 1
    constraints; for branching-behavior benchmarks the relaxation is fine.

  - reserves -> taken as-is from the JSON; this is an MW value, not a fraction.

Output dict mirrors instance_adapter.to_formulation_format.
"""

import json
import os

import numpy as np


def load_pglib_instance(json_path, horizon=None, net_renewables=False):
    """
    Load a PGLib-UC JSON file and return a formulation-format inst dict.

    Parameters
    ----------
    json_path : str
        Path to a PGLib-UC JSON case file.
    horizon : int or None
        Truncate the time series to this many hours. None = keep full horizon.

    Returns
    -------
    inst : dict
        Same shape as instance_adapter.to_formulation_format output, plus
        'pglib_path' and 'pglib_name' metadata.
    """
    with open(json_path) as f:
        d = json.load(f)

    T_full = int(d['time_periods'])
    T = int(horizon) if horizon is not None else T_full
    T = min(T, T_full)

    therm = d['thermal_generators']
    gen_names = list(therm.keys())
    n_gen = len(gen_names)

    pmin = np.zeros(n_gen)
    pmax = np.zeros(n_gen)
    cost_nl = np.zeros(n_gen)
    cost_b = np.zeros(n_gen)
    cost_su = np.zeros(n_gen)
    ramp_up = np.zeros(n_gen)
    ramp_dn = np.zeros(n_gen)
    min_up = np.zeros(n_gen, dtype=int)
    min_dn = np.zeros(n_gen, dtype=int)
    init_status = np.zeros(n_gen, dtype=int)
    init_hours = np.zeros(n_gen, dtype=int)
    P_init = np.zeros(n_gen)

    for i, name in enumerate(gen_names):
        g = therm[name]
        pmn = float(g['power_output_minimum'])
        pmx = float(g['power_output_maximum'])
        pmin[i] = pmn
        pmax[i] = pmx

        # Linearize the piecewise cost curve: match cost at pmin and pmax.
        pw = g['piecewise_production']
        if len(pw) >= 2:
            c0 = float(pw[0]['cost']);  m0 = float(pw[0]['mw'])
            cN = float(pw[-1]['cost']); mN = float(pw[-1]['mw'])
            if mN > m0:
                slope = (cN - c0) / (mN - m0)
            else:
                slope = 0.0
            cost_b[i] = slope
            cost_nl[i] = c0 - slope * m0
        elif len(pw) == 1:
            cost_b[i] = 0.0
            cost_nl[i] = float(pw[0]['cost'])
        else:
            cost_b[i] = 0.0
            cost_nl[i] = 0.0

        # Startup: pick hot-start (smallest) cost.
        starts = g.get('startup', [])
        if starts:
            cost_su[i] = float(min(s['cost'] for s in starts))
        else:
            cost_su[i] = 0.0

        ru = float(g['ramp_up_limit'])
        rd = float(g['ramp_down_limit'])
        rsu = float(g.get('ramp_startup_limit', ru))
        rsd = float(g.get('ramp_shutdown_limit', rd))
        ramp_up[i] = max(ru, rsu)
        ramp_dn[i] = max(rd, rsd)

        min_up[i] = max(1, int(g['time_up_minimum']))
        min_dn[i] = max(1, int(g['time_down_minimum']))

        on0 = int(g['unit_on_t0'])
        init_status[i] = on0
        init_hours[i] = int(g['time_up_t0'] if on0 == 1 else g['time_down_t0'])
        P_init[i] = float(g.get('power_output_t0', 0.0))

    # Demand and reserves
    demand = np.array(d['demand'][:T], dtype=float)
    reserves = np.array(d['reserves'][:T], dtype=float)

    # Renewables: by default leave demand alone (see module docstring).
    # If net_renewables, subtract the renewable lower bound — what they're
    # forced to produce — rather than the upper bound.
    if net_renewables:
        renew = d.get('renewable_generators', {}) or {}
        for r in renew.values():
            rmin = np.array(r['power_output_minimum'][:T], dtype=float)
            demand = demand - rmin
        demand = np.clip(demand, 0.0, None)

    inst = {
        'n_gen': n_gen,
        'horizon': T,
        'demand': demand,
        'load': demand,
        'reserve': reserves,
        'P_min': pmin,
        'P_max': pmax,
        'cost_nl': cost_nl,
        'cost_su': cost_su,
        'cost_b': cost_b,
        'cost_a': np.zeros(n_gen),
        'min_up': min_up,
        'min_dn': min_dn,
        'ramp_up': ramp_up,
        'ramp_dn': ramp_dn,
        'init_status': init_status,
        'init_hours': init_hours,
        'P_init': P_init,
        # Metadata
        'pglib_path': json_path,
        'pglib_name': os.path.splitext(os.path.basename(json_path))[0],
        'pglib_dataset': os.path.basename(os.path.dirname(json_path)),
        'symmetry_score': None,
        'reserve_frac': None,
    }
    return inst


def categorize_size(n_gen):
    if n_gen <= 30:
        return 'small'
    if n_gen <= 100:
        return 'medium'
    return 'large'


def discover_instances(pglib_root, datasets=('ca', 'ferc', 'rts_gmlc'),
                       max_size='medium'):
    """
    Walk a PGLib-UC checkout and return [(json_path, name, n_gen, size_tag)].

    Filters out 'large' instances by default to keep Phase 5 tractable.
    """
    order = {'small': 0, 'medium': 1, 'large': 2}
    max_rank = order[max_size]
    rows = []
    for ds in datasets:
        d = os.path.join(pglib_root, ds)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json'):
                continue
            path = os.path.join(d, fn)
            try:
                with open(path) as f:
                    raw = json.load(f)
                n_gen = len(raw.get('thermal_generators', {}))
                T = int(raw.get('time_periods', 0))
            except Exception:
                continue
            size = categorize_size(n_gen)
            if order[size] > max_rank:
                continue
            rows.append({
                'path': path,
                'name': os.path.splitext(fn)[0],
                'dataset': ds,
                'n_gen': n_gen,
                'time_periods': T,
                'size_tag': size,
            })
    return rows


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pglib_uc_adapter.py <path/to/pglib_case.json>")
        sys.exit(1)
    inst = load_pglib_instance(sys.argv[1])
    print(f"name        = {inst['pglib_name']}")
    print(f"n_gen       = {inst['n_gen']}")
    print(f"horizon     = {inst['horizon']}")
    print(f"total_pmax  = {inst['P_max'].sum():.1f}")
    print(f"peak demand = {inst['demand'].max():.1f}")
    print(f"peak res    = {inst['reserve'].max():.1f}")
    print(f"size_tag    = {categorize_size(inst['n_gen'])}")
