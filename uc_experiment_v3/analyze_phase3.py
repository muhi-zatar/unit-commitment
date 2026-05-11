"""
Phase 3 analysis: bar charts of nodes and time per config.

Reads phase3_results.csv; writes:
  plots/phase3_nodes_by_config.png
  plots/phase3_time_by_config.png
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

CONFIG_ORDER = ['default', 'no_symmetry', 'no_heuristics', 'no_presolve', 'raw']


def bar(df, ycol, ylabel, fname, log_y=True):
    summary = df.groupby('config_label')[ycol].agg(['median', 'min', 'max'])
    summary = summary.reindex([c for c in CONFIG_ORDER if c in summary.index])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary.index, summary['median'].clip(lower=1e-3), color='C2', alpha=0.85)
    # Error bars: min/max relative to median
    yerr = np.vstack([
        (summary['median'] - summary['min']).clip(lower=0).values,
        (summary['max'] - summary['median']).clip(lower=0).values,
    ])
    ax.errorbar(range(len(summary)), summary['median'].clip(lower=1e-3),
                yerr=yerr, fmt='none', ecolor='black', capsize=4)
    if log_y:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Phase 3: median {ycol} per config (min/max bars)')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(THIS_DIR, 'phase3_results.csv'))
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print("=== Phase 3 summary table ===")
    summary = df.groupby('config_label').agg(
        med_nodes=('mip_node_count', 'median'),
        med_time=('mip_total_time_s', 'median'),
        med_gap=('root_lp_gap_pct', 'median'),
        n=('mip_node_count', 'size'),
    ).reindex([c for c in CONFIG_ORDER if c in df['config_label'].unique()])
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    # Acceptance criterion
    if 'default' in summary.index:
        d_nodes = summary.loc['default', 'med_nodes']
        worst = summary['med_nodes'].max()
        ratio = worst / max(d_nodes, 1e-9)
        print(f"\nWorst/default node ratio: {ratio:.1f}x "
              f"(criterion: at least one ablated config >= 10x default)")

    bar(df, 'mip_node_count', 'MIP nodes (median)',
        'phase3_nodes_by_config.png', log_y=True)
    bar(df, 'mip_total_time_s', 'MIP wall time (s, median)',
        'phase3_time_by_config.png', log_y=True)


if __name__ == '__main__':
    main()
