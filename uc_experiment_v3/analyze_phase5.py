"""
Phase 5 analysis: PGLib-UC HiGHS vs SCIP (or HiGHS-only if SCIP missing).

Reads phase5_results.csv. Writes:
  plots/phase5_time_scatter.png  (if both solvers present)
  plots/phase5_nodes_scatter.png (if both solvers present)
  plots/phase5_highs_time_vs_size.png
Prints a per-dataset summary table.
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
    parser.add_argument('--csv', default=os.path.join(THIS_DIR, 'phase5_results.csv'))
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct',
              'n_gen', 'horizon'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    solvers = df['solver'].unique()
    print(f"Solvers present: {solvers}")
    print(f"Total rows: {len(df)}")

    # Per-dataset summary
    print("\n=== Per-dataset, per-solver summary (medians) ===")
    summary = df.groupby(['solver', 'dataset', 'size_tag']).agg(
        n_instances=('instance_id', 'nunique'),
        med_nodes=('mip_node_count', 'median'),
        med_time=('mip_total_time_s', 'median'),
        med_gap=('root_lp_gap_pct', 'median'),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # HiGHS-only time-vs-size plot
    h = df[df['solver'] == 'highs']
    if len(h):
        fig, ax = plt.subplots(figsize=(7, 5))
        for ds, grp in h.groupby('dataset'):
            ax.scatter(grp['n_gen'], grp['mip_total_time_s'].clip(lower=0.01),
                       s=30, alpha=0.7, label=ds)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('n_gen (thermal)')
        ax.set_ylabel('HiGHS wall time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, 'phase5_highs_time_vs_size.png')
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"wrote {out}")

    # Paired scatter if both solvers present
    if 'scip' in solvers and 'highs' in solvers:
        h = df[df['solver'] == 'highs'].set_index('instance_id')
        s = df[df['solver'] == 'scip'].set_index('instance_id')
        common = h.index.intersection(s.index)
        print(f"\nPaired instances: {len(common)}")
        if len(common):
            pairs = pd.DataFrame({
                'n_gen':       h.loc[common, 'n_gen'].values,
                'highs_nodes': h.loc[common, 'mip_node_count'].values,
                'scip_nodes':  s.loc[common, 'mip_node_count'].values,
                'highs_time':  h.loc[common, 'mip_total_time_s'].values,
                'scip_time':   s.loc[common, 'mip_total_time_s'].values,
            }, index=common)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pairs['highs_time'].clip(lower=0.01),
                       pairs['scip_time'].clip(lower=0.01),
                       c=pairs['n_gen'], cmap='viridis', s=28, alpha=0.85)
            lo = max(0.01, min(pairs['highs_time'].min(), pairs['scip_time'].min()))
            hi = max(pairs['highs_time'].max(), pairs['scip_time'].max())
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('HiGHS time (s)'); ax.set_ylabel('SCIP time (s)')
            ax.grid(True, alpha=0.3, which='both')
            fig.tight_layout()
            out = os.path.join(PLOTS_DIR, 'phase5_time_scatter.png')
            fig.savefig(out, dpi=140); plt.close(fig)
            print(f"wrote {out}")

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(pairs['highs_nodes'].clip(lower=1),
                       pairs['scip_nodes'].clip(lower=1),
                       c=pairs['n_gen'], cmap='viridis', s=28, alpha=0.85)
            lo = 1; hi = max(pairs['highs_nodes'].max(), pairs['scip_nodes'].max())
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('HiGHS nodes'); ax.set_ylabel('SCIP nodes')
            ax.grid(True, alpha=0.3, which='both')
            fig.tight_layout()
            out = os.path.join(PLOTS_DIR, 'phase5_nodes_scatter.png')
            fig.savefig(out, dpi=140); plt.close(fig)
            print(f"wrote {out}")


if __name__ == '__main__':
    main()
