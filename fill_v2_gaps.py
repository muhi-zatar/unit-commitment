"""Fill the missing cells from the dead v2 sweep."""

import time
import pandas as pd
from pathlib import Path

from run_v2_sweep import run_single_experiment


def main():
    csv_path = '/home/claude/uc_experiment_v2/uc_v2_results.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} existing rows")

    seen = set()
    for _, r in df.iterrows():
        if pd.notna(r.get('seed')):
            seen.add((int(r['n_total_intended']), int(r['horizon']),
                      r['fleet_pattern'], float(r['plateau']),
                      r['formulation'], int(r['seed'])))

    # Cells to fill: N=12 missing (T=8, sym8, plateau=0)
    todo = [
        (12, 8, 'sym8', 0.0, 'three_bin', 42),
        (12, 8, 'sym8', 0.0, 'three_bin', 59),
    ]

    rows = df.to_dict('records')
    sweep_t0 = time.perf_counter()

    for (n_total, horizon, pattern, plateau, formulation, seed) in todo:
        key = (n_total, horizon, pattern, plateau, formulation, seed)
        if key in seen:
            print(f"Already done: {key}")
            continue
        elapsed = time.perf_counter() - sweep_t0
        print(f"Running: N={n_total} T={horizon} {pattern} p={plateau} seed={seed}  (elapsed: {elapsed:.0f}s)",
              flush=True)
        try:
            r = run_single_experiment(n_total, horizon, pattern, plateau,
                                       formulation, seed, time_limit=120)
            rows.append(r)
            print(f"   -> nodes={r['mip_node_count']}, "
                  f"time={r['mip_total_time_s']:.2f}s, "
                  f"gap={r['root_lp_gap_pct']:.3f}%", flush=True)
        except Exception as e:
            print(f"   ERROR: {e}", flush=True)
            rows.append({
                'n_total_intended': n_total, 'horizon': horizon,
                'fleet_pattern': pattern, 'plateau': plateau,
                'formulation': formulation, 'seed': seed,
                'error': str(e), 'converged': False,
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nDone. Total rows: {len(rows)}, wall: {time.perf_counter() - sweep_t0:.1f}s")


if __name__ == '__main__':
    main()
