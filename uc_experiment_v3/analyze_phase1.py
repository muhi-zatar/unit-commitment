"""
Phase 1 analysis: N_nodes(k), time(k), time vs nodes.

Reads phase1_results.csv; writes:
  plots/phase1_nodes_vs_k.png
  plots/phase1_time_vs_k.png
  plots/phase1_time_vs_nodes.png
  prints summary table.
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


def _quantile_band(df, col):
    return (df[col].median(), df[col].quantile(0.25), df[col].quantile(0.75))


def plot_metric_vs_k(df, ycol, ylabel, fname, log_y=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (N_g, T), grp in df.groupby(['N_g', 'horizon']):
        agg = grp.groupby('k_identical')[ycol].agg(['median',
                                                    lambda s: s.quantile(0.25),
                                                    lambda s: s.quantile(0.75)])
        agg.columns = ['med', 'q25', 'q75']
        agg = agg.reset_index().sort_values('k_identical')
        ax.plot(agg['k_identical'], agg['med'].clip(lower=1e-6),
                marker='o', label=f'N_g={N_g}, T={T}')
        ax.fill_between(agg['k_identical'],
                        agg['q25'].clip(lower=1e-6),
                        agg['q75'].clip(lower=1e-6), alpha=0.18)
    ax.set_xlabel('k_identical')
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_time_vs_nodes(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df.dropna(subset=['mip_node_count', 'mip_total_time_s'])
    sub = sub[(sub['mip_node_count'] > 0) & (sub['mip_total_time_s'] > 0)]
    if len(sub) == 0:
        print("  no points for time-vs-nodes plot")
        return
    ax.scatter(sub['mip_node_count'], sub['mip_total_time_s'],
               s=18, alpha=0.55, c='C0')
    # Fit power law in log-log
    lx = np.log(sub['mip_node_count'].values.astype(float))
    ly = np.log(sub['mip_total_time_s'].values.astype(float))
    mask = np.isfinite(lx) & np.isfinite(ly)
    if mask.sum() >= 5:
        a, b = np.polyfit(lx[mask], ly[mask], 1)
        xs = np.logspace(np.log10(sub['mip_node_count'].min()),
                         np.log10(sub['mip_node_count'].max()), 100)
        ax.plot(xs, np.exp(b) * xs**a, 'k--',
                label=f'fit: time ~ nodes^{a:.2f}')
        ax.legend()
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('MIP node count')
    ax.set_ylabel('MIP wall time (s)')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'phase1_time_vs_nodes.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def summary_table(df):
    print("\n=== Phase 1 summary: median nodes by (N_g, T, k) ===")
    pivot = df.groupby(['N_g', 'horizon', 'k_identical'])[
        'mip_node_count'].median().unstack('k_identical')
    print(pivot.to_string(float_format=lambda x: f"{x:8.0f}" if pd.notnull(x) else "    -"))

    print("\n=== Threshold: smallest k where median nodes >= T ===")
    rows = []
    for (N_g, T), grp in df.groupby(['N_g', 'horizon']):
        med = grp.groupby('k_identical')['mip_node_count'].median()
        for thresh in (10, 100, 1000):
            hit = med[med >= thresh]
            k_first = hit.index.min() if len(hit) else None
            rows.append({'N_g': N_g, 'T': T, 'threshold': thresh,
                         'first_k': k_first})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        default=os.path.join(THIS_DIR, 'phase1_results.csv'))
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run phase 1 first.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.contains('ptimal') |
                df['status'].astype(str).str.contains('Time')]
    # numeric coercion
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct',
              'k_identical', 'N_g', 'horizon'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    plot_metric_vs_k(df, 'mip_node_count', 'MIP nodes (median, IQR band)',
                     'phase1_nodes_vs_k.png', log_y=True)
    plot_metric_vs_k(df, 'mip_total_time_s', 'MIP wall time (s)',
                     'phase1_time_vs_k.png', log_y=True)
    plot_time_vs_nodes(df)
    summary_table(df)


if __name__ == '__main__':
    main()
