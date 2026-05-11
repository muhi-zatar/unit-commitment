"""
Synthetic UC instance generator.

Produces a UC problem with N_g generators and T time periods, complete with
- Generator characteristics (cost coefficients, capacity, ramps, min up/down)
- Load profile (with realistic intra-day shape and multi-day variation)
- Reserve requirement
"""

import numpy as np


def generate_uc_instance(n_gen, horizon, seed=0):
    """
    Generate a synthetic UC instance.

    Generator mix is roughly:
      - ~25% baseload (large, cheap, slow ramps, long min up/down)
      - ~50% mid-merit (medium size and cost, moderate ramps)
      - ~25% peakers (small, expensive, fast ramps, short min up/down)
    """
    rng = np.random.default_rng(seed)

    n_baseload = max(1, int(0.25 * n_gen))
    n_peaker = max(1, int(0.25 * n_gen))
    n_mid = n_gen - n_baseload - n_peaker

    P_max = np.zeros(n_gen)
    P_min = np.zeros(n_gen)
    cost_a = np.zeros(n_gen)
    cost_b = np.zeros(n_gen)
    cost_nl = np.zeros(n_gen)
    cost_su = np.zeros(n_gen)
    ramp_up = np.zeros(n_gen)
    ramp_dn = np.zeros(n_gen)
    min_up = np.zeros(n_gen, dtype=int)
    min_dn = np.zeros(n_gen, dtype=int)
    init_status = np.zeros(n_gen, dtype=int)
    init_hours = np.zeros(n_gen, dtype=int)

    idx = 0
    for k in range(n_baseload):
        P_max[idx] = rng.uniform(300, 600)
        P_min[idx] = 0.4 * P_max[idx]
        cost_a[idx] = rng.uniform(0.001, 0.005)
        cost_b[idx] = rng.uniform(15, 25)
        cost_nl[idx] = rng.uniform(500, 1500)
        cost_su[idx] = rng.uniform(5000, 15000)
        ramp_up[idx] = 0.15 * P_max[idx]
        ramp_dn[idx] = 0.15 * P_max[idx]
        min_up[idx] = int(rng.integers(8, 16))
        min_dn[idx] = int(rng.integers(6, 12))
        init_status[idx] = 1
        init_hours[idx] = int(rng.integers(10, 30))
        idx += 1

    for k in range(n_mid):
        P_max[idx] = rng.uniform(100, 300)
        P_min[idx] = 0.3 * P_max[idx]
        cost_a[idx] = rng.uniform(0.005, 0.02)
        cost_b[idx] = rng.uniform(25, 40)
        cost_nl[idx] = rng.uniform(200, 600)
        cost_su[idx] = rng.uniform(1000, 4000)
        ramp_up[idx] = 0.5 * P_max[idx]
        ramp_dn[idx] = 0.5 * P_max[idx]
        min_up[idx] = int(rng.integers(3, 8))
        min_dn[idx] = int(rng.integers(2, 6))
        init_status[idx] = int(rng.integers(0, 2))
        init_hours[idx] = int(rng.integers(2, 10))
        idx += 1

    for k in range(n_peaker):
        P_max[idx] = rng.uniform(20, 100)
        P_min[idx] = 0.2 * P_max[idx]
        cost_a[idx] = rng.uniform(0.01, 0.04)
        cost_b[idx] = rng.uniform(50, 90)
        cost_nl[idx] = rng.uniform(50, 200)
        cost_su[idx] = rng.uniform(100, 1000)
        ramp_up[idx] = P_max[idx]
        ramp_dn[idx] = P_max[idx]
        min_up[idx] = int(rng.integers(1, 4))
        min_dn[idx] = int(rng.integers(1, 3))
        init_status[idx] = 0
        init_hours[idx] = int(rng.integers(1, 5))
        idx += 1

    P_init = np.where(init_status > 0, 0.7 * P_max, 0.0)

    # 24-hour load shape
    base_shape_24 = np.array([
        0.65, 0.60, 0.58, 0.57, 0.58, 0.62,
        0.70, 0.80, 0.88, 0.92, 0.94, 0.95,
        0.95, 0.94, 0.93, 0.93, 0.95, 0.98,
        1.00, 0.98, 0.93, 0.85, 0.78, 0.70,
    ])

    total_capacity = P_max.sum()
    # Peak load is ~75% of capacity; small fleets need more headroom
    # because of the reserve requirement and limited diversity
    if n_gen < 30:
        peak_factor = 0.65
    elif n_gen < 100:
        peak_factor = 0.75
    else:
        peak_factor = 0.80
    peak_load = peak_factor * total_capacity

    n_days = (horizon + 23) // 24
    daily_var = 1.0 + 0.05 * rng.standard_normal(n_days)
    load = np.zeros(horizon)
    for t in range(horizon):
        d = t // 24
        h = t % 24
        load[t] = peak_load * base_shape_24[h] * daily_var[d]
    load *= 1.0 + 0.01 * rng.standard_normal(horizon)

    reserve = 0.05 * load + P_max.max() * 0.5

    return {
        'n_gen': n_gen,
        'horizon': horizon,
        'P_max': P_max,
        'P_min': P_min,
        'cost_a': cost_a,
        'cost_b': cost_b,
        'cost_nl': cost_nl,
        'cost_su': cost_su,
        'ramp_up': ramp_up,
        'ramp_dn': ramp_dn,
        'min_up': min_up,
        'min_dn': min_dn,
        'init_status': init_status,
        'init_hours': init_hours,
        'P_init': P_init,
        'load': load,
        'reserve': reserve,
    }


if __name__ == '__main__':
    for ng, T in [(10, 24), (50, 24), (100, 72), (200, 72)]:
        inst = generate_uc_instance(ng, T, seed=42)
        cap = inst['P_max'].sum()
        print(f"N_g={ng:3d}, T={T:3d}: total_cap={cap:6.0f} MW, "
              f"peak_load={inst['load'].max():6.0f} MW, "
              f"min_load={inst['load'].min():6.0f} MW")
