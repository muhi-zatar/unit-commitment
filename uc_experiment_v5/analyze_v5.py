"""
V5 analysis: log-log scaling of {nodes, time} vs network size.

Reads phase1_v5_results.csv. Writes:
  plots/v5_time_vs_size.png
  plots/v5_nodes_vs_size.png
  plots/v5_solver_compare.png (if both solvers present)
Prints scaling exponents and a per-network table.

All figures carry an "includes Pyomo build/IO overhead" annotation.
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
os.makedirs(PLOTS_DIR, exist_ok=True)

OVERHEAD_NOTE = 'timings include Pyomo build + MPS write overhead'


def _load(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None


def _numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _power_law_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    lx = np.log10(x[mask]); ly = np.log10(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept


def plot_metric_vs_size(df, ycol, ylabel, fname, log_y=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    fits = {}
    for solver, grp in df.groupby('solver'):
        x = grp['n_branch'].values.astype(float)
        y = grp[ycol].values.astype(float)
        ax.scatter(x, y, s=40, alpha=0.8, label=solver)
        fit = _power_law_fit(x, y)
        if fit is not None:
            slope, intercept = fit
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, 10**(intercept) * xs**slope, '--', alpha=0.6,
                    label=f'{solver} fit: ~ n_branch^{slope:.2f}')
            fits[solver] = (slope, intercept)
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel('n_branch (transmission size)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_title(f'V5 SCUC scaling — {OVERHEAD_NOTE}')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    return fits


def plot_solver_compare(df):
    """Bar chart: time and nodes per network, side-by-side HiGHS vs SCIP."""
    pivot_t = df.pivot_table(index='network', columns='solver',
                              values='mip_total_time_s', aggfunc='median')
    pivot_n = df.pivot_table(index='network', columns='solver',
                              values='mip_node_count', aggfunc='median')
    if pivot_t.shape[1] < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot_t.plot(kind='bar', ax=axes[0], logy=True, color=['C0', 'C3'])
    axes[0].set_ylabel('MIP wall time (s)')
    axes[0].set_title('HiGHS vs SCIP — solve time')
    axes[0].grid(True, alpha=0.3, axis='y', which='both')
    pivot_n.plot(kind='bar', ax=axes[1], logy=True, color=['C0', 'C3'])
    axes[1].set_ylabel('MIP nodes')
    axes[1].set_title('HiGHS vs SCIP — node count')
    axes[1].grid(True, alpha=0.3, axis='y', which='both')
    fig.suptitle(OVERHEAD_NOTE, fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'v5_solver_compare.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        default=os.path.join(THIS_DIR, 'phase1_v5_results.csv'))
    args = parser.parse_args()

    df = _load(args.csv)
    if df is None or len(df) == 0:
        print(f"no data at {args.csv}")
        sys.exit(0)

    df = _numeric(df, ['n_branch', 'n_bus', 'n_gen', 'horizon',
                       'mip_total_time_s', 'mip_node_count', 'mip_gap'])
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().str.startswith(
            ('optimal', 'time'))]

    print(f"\nLoaded {len(df)} rows. solvers={list(df['solver'].unique())}")
    print("\n=== Per-network summary ===")
    summary = df.groupby(['network', 'solver']).agg(
        n_bus=('n_bus', 'first'),
        n_branch=('n_branch', 'first'),
        n_gen=('n_gen', 'first'),
        horizon=('horizon', 'first'),
        med_time=('mip_total_time_s', 'median'),
        med_nodes=('mip_node_count', 'median'),
        med_gap=('mip_gap', 'median'),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Scaling fits ===")
    fits_time = plot_metric_vs_size(df, 'mip_total_time_s',
                                    'MIP wall time (s)',
                                    'v5_time_vs_size.png', log_y=True)
    fits_nodes = plot_metric_vs_size(df, 'mip_node_count',
                                     'MIP nodes',
                                     'v5_nodes_vs_size.png', log_y=True)
    if fits_time:
        for solver, (slope, _) in fits_time.items():
            print(f"  {solver:5s}: time ~ n_branch^{slope:.2f}")
    if fits_nodes:
        for solver, (slope, _) in fits_nodes.items():
            print(f"  {solver:5s}: nodes ~ n_branch^{slope:.2f}")

    if len(df['solver'].unique()) >= 2:
        plot_solver_compare(df)


if __name__ == '__main__':
    main()
