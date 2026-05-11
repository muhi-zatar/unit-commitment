"""
Phase 4 analysis: HiGHS vs SCIP comparison.

Reads phase4_results.csv. Writes:
  plots/phase4_time_scatter.png
  plots/phase4_nodes_scatter.png
Prints a median-ratio table by k_identical.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(THIS_DIR, 'plots')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(THIS_DIR, 'phase4_results.csv'))
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct',
              'k_identical', 'N_g', 'horizon'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    solvers = df['solver'].unique()
    print(f"Solvers present: {solvers}")
    if 'scip' not in solvers:
        print("No SCIP rows — analysis falls back to HiGHS-only summary.")
        hs = df[df['solver'] == 'highs']
        print(hs.groupby('k_identical').agg(
            highs_med_nodes=('mip_node_count', 'median'),
            highs_med_time=('mip_total_time_s', 'median')).to_string())
        return

    h = df[df['solver'] == 'highs'].set_index('instance_id')
    s = df[df['solver'] == 'scip'].set_index('instance_id')
    common = h.index.intersection(s.index)
    print(f"Paired instances: {len(common)}")

    pairs = pd.DataFrame({
        'k_identical': h.loc[common, 'k_identical'].values,
        'highs_nodes': h.loc[common, 'mip_node_count'].values,
        'scip_nodes':  s.loc[common, 'mip_node_count'].values,
        'highs_time':  h.loc[common, 'mip_total_time_s'].values,
        'scip_time':   s.loc[common, 'mip_total_time_s'].values,
    }, index=common)

    # Scatter: time
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pairs['highs_time'].clip(lower=0.01),
               pairs['scip_time'].clip(lower=0.01),
               c=pairs['k_identical'], cmap='viridis', s=28, alpha=0.85)
    lo = max(0.01, min(pairs['highs_time'].min(), pairs['scip_time'].min()))
    hi = max(pairs['highs_time'].max(), pairs['scip_time'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('HiGHS time (s)'); ax.set_ylabel('SCIP time (s)')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'phase4_time_scatter.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")

    # Scatter: nodes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pairs['highs_nodes'].clip(lower=1),
               pairs['scip_nodes'].clip(lower=1),
               c=pairs['k_identical'], cmap='viridis', s=28, alpha=0.85)
    lo = 1; hi = max(pairs['highs_nodes'].max(), pairs['scip_nodes'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('HiGHS nodes'); ax.set_ylabel('SCIP nodes')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'phase4_nodes_scatter.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")

    # Median ratios by k
    print("\n=== Median ratios (scip / highs) by k_identical ===")
    summary = pairs.groupby('k_identical').apply(lambda g: pd.Series({
        'highs_med_nodes': g['highs_nodes'].median(),
        'scip_med_nodes':  g['scip_nodes'].median(),
        'nodes_ratio_scip_over_highs': (g['scip_nodes'].median()
                                        / max(g['highs_nodes'].median(), 1)),
        'highs_med_time': g['highs_time'].median(),
        'scip_med_time':  g['scip_time'].median(),
        'time_ratio_scip_over_highs': (g['scip_time'].median()
                                       / max(g['highs_time'].median(), 1e-6)),
    }))
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == '__main__':
    main()
