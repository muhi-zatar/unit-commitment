"""
Phase 2 analysis: paired detect_symmetry on/off comparison.

Reads phase2_results.csv. Writes plots/phase2_sym_scatter.png and prints
a paired table.
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


def pair_runs(df):
    """Return one row per (instance_id) with sym_off / sym_on side by side."""
    keep = ['instance_id', 'N_g', 'k_identical', 'horizon', 'seed',
            'config_label', 'mip_node_count', 'mip_total_time_s',
            'root_lp_gap_pct', 'status']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    off = df[df['config_label'] == 'sym_off'].rename(columns={
        'mip_node_count': 'n_nodes_off',
        'mip_total_time_s': 'time_off_s',
        'root_lp_gap_pct': 'gap_off_pct',
        'status': 'status_off',
    }).drop(columns=['config_label'])
    on = df[df['config_label'] == 'sym_on'].rename(columns={
        'mip_node_count': 'n_nodes_on',
        'mip_total_time_s': 'time_on_s',
        'root_lp_gap_pct': 'gap_on_pct',
        'status': 'status_on',
    })[['instance_id', 'n_nodes_on', 'time_on_s', 'gap_on_pct', 'status_on']]

    paired = off.merge(on, on='instance_id', how='inner')
    paired['ratio_nodes'] = paired['n_nodes_off'] / paired['n_nodes_on'].replace(0, np.nan)
    paired['ratio_time'] = paired['time_off_s'] / paired['time_on_s'].replace(0, np.nan)
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(THIS_DIR, 'phase2_results.csv'))
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct',
              'k_identical', 'N_g', 'horizon'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    paired = pair_runs(df)
    print(f"Paired {len(paired)} instances.")
    print(paired[['instance_id', 'k_identical', 'horizon',
                  'n_nodes_off', 'n_nodes_on', 'ratio_nodes',
                  'time_off_s', 'time_on_s', 'ratio_time']]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Scatter
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if len(paired):
        x = paired['n_nodes_off'].replace(0, np.nan)
        y = paired['n_nodes_on'].replace(0, np.nan)
        ax.scatter(x, y, c='C1', s=24, alpha=0.7)
        lo = min(x.min(), y.min()) if x.notna().any() else 1
        hi = max(x.max(), y.max()) if x.notna().any() else 10
        lo = max(lo, 1)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('MIP nodes (detect_symmetry = OFF)')
    ax.set_ylabel('MIP nodes (detect_symmetry = ON)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'phase2_sym_scatter.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    # High-symmetry tail criterion
    paired['high_sym'] = paired['k_identical'] >= (paired['N_g'] / 2)
    tail = paired[paired['high_sym']]
    if len(tail):
        med_ratio = tail['ratio_nodes'].median()
        print(f"\nHigh-symmetry tail median ratio (off/on) = {med_ratio:.2f}x "
              f"(>=2.0 means detect_symmetry helps)")


if __name__ == '__main__':
    main()
