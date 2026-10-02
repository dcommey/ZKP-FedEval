"""Render the manuscript figures as vector PDFs sized for an IEEE column."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito hues (CVD-checked); every series also carries a line style or marker.
BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#7f7f7f'
COLUMN = 3.5  # IEEE two-column width in inches
DENOM = 10 * 8 * 100**4  # 10 channels x 8 records x S^4

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.1,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#d9d9d9', lw=0.4)
    ax.set_axisbelow(True)


def unique_batches():
    local = json.loads((ROOT / 'results/committed-rerun/local_outcomes.json').read_text())
    key = lambda x: (x['dataset'], x['seed'], x['distribution'], x['num_clients'], x['client_id'])
    return list({key(x): x for x in local}.values())


def loss_cdf(rows):
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.55), sharey=True)
    for ax, ds in zip(axes, ['mnist', 'har']):
        for t in (0.025, 0.05, 0.1):
            ax.axvline(t, color=GREY, lw=0.5, ls=':', zorder=0)
        for split, color, style, label in [('iid', BLUE, '-', 'IID'), ('label_sorted', ORANGE, '--', 'Label sorted')]:
            losses = np.sort([int(r['loss_sum']) / DENOM for r in rows if r['dataset'] == ds and r['distribution'] == split])
            # Positive means Q < tau, so a batch counts once the cutoff exceeds its loss.
            x = np.r_[0, losses, 0.12]
            y = np.r_[0, np.arange(1, len(losses) + 1) / len(losses) * 100, 100]
            ax.step(x, y, where='post', color=color, ls=style, label=label)
        ax.set(xlim=(0, 0.12), ylim=(0, 102), xticks=[0, 0.025, 0.05, 0.075, 0.1], yticks=[0, 50, 100])
        ax.set_xticklabels(['0', '.025', '.05', '.075', '.10'])
        ax.set_title(ds.upper(), pad=2)
        ax.set_xlabel(r'Loss cutoff $T$')
        grid(ax)
    axes[0].set_ylabel('Positive batches (%)')
    axes[1].legend(frameon=False, loc='lower right', handlelength=1.8, borderaxespad=0.1)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / 'loss-cdf.pdf')
    plt.close(fig)


def leakage(rows):
    raw = np.array([int(r['loss_sum']) for r in rows], dtype=np.int64)
    exact = raw / DENOM
    qs = np.arange(1, 21)
    worst, mean = [], []
    lo = np.zeros_like(raw); hi = np.full_like(raw, DENOM)
    for _ in qs:
        mid = (lo + hi + 1) // 2
        below = raw < mid
        hi = np.where(below, mid - 1, hi); lo = np.where(below, lo, mid)
        err = np.abs((lo + hi) / 2 / DENOM - exact)
        worst.append(err.max()); mean.append(err.mean())
    fig, ax = plt.subplots(figsize=(COLUMN, 1.45))
    ax.semilogy(qs, worst, color=ORANGE, marker='s', ms=2.8, label='Worst batch')
    ax.semilogy(qs, mean, color=BLUE, marker='o', ms=2.8, label='Mean over 180 batches')
    ax.semilogy(qs, 0.5 ** (qs + 1), color='black', lw=0.6, ls=(0, (3, 2)), zorder=4,
                label=r'Midpoint bound $2^{-(q+1)}$')
    ax.set(xlim=(0.5, 20.5), xticks=[1, 4, 8, 12, 16, 20], ylim=(1e-7, 1),
           xlabel=r'Adaptive threshold queries $q$', ylabel='Abs. loss error')
    grid(ax)
    ax.legend(frameon=False, loc='upper right', ncol=1, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(OUT / 'leakage.pdf')
    plt.close(fig)


def utility():
    study = json.loads((ROOT / 'results/utility-study/summary.json').read_text())
    styles = {0.01: (BLUE, 'o', '1%'), 0.05: (ORANGE, 's', '5%'), 0.3: (GREEN, '^', '30%')}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.8), sharey=True)
    for ax, ds in zip(axes, ['mnist', 'har']):
        rho = next(r['spearman_accuracy_pass'] for r in study['relations'] if r['dataset'] == ds)
        for f, (color, marker, label) in styles.items():
            m = [x for x in study['models'] if x['dataset'] == ds and x['fraction'] == f]
            ax.scatter([100 * x['full_test_accuracy'] for x in m], [100 * x['pass_fraction'] for x in m],
                       color=color, marker=marker, s=16, lw=0.5, edgecolor='white', label=label, zorder=3)
        ax.set_title(f'{ds.upper()} ($\\rho_s={rho:.2f}$)', pad=2)
        ax.set_xlabel('Test accuracy (%)')
        ax.set_ylim(0, 60)
        grid(ax, 'both')
    axes[0].set_xlim(73, 78.5); axes[1].set_xlim(72, 88)
    axes[0].set_ylabel('Passing batches (%)')
    handles = [Line2D([], [], color=c, marker=mk, ls='', ms=4, label=l + ' training pool')
               for c, mk, l in styles.values()]
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.legend(handles=handles, frameon=False, loc='upper center', ncol=3, handletextpad=0.1,
               columnspacing=1.0, borderaxespad=0.0)
    fig.savefig(OUT / 'utility.pdf')
    plt.close(fig)


if __name__ == '__main__':
    rows = unique_batches()
    loss_cdf(rows)
    leakage(rows)
    utility()
    print('Wrote', ', '.join(sorted(p.name for p in OUT.glob('*.pdf'))))
