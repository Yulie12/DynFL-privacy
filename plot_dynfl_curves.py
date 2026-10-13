"""Plot CIFAR-10 DynFL ablation curves from locally saved CSV logs.
Run from the DynFL-privacy project root: python plot_dynfl_curves.py
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('out')
DEST = ROOT / 'all_curve_plots'
DEST.mkdir(parents=True, exist_ok=True)
GROUPS = {
    'Ours': ('full_dynfl', 0.461, None),
    'Clipping-only': ('full_dynfl_clipping_only', 0.514, 'cifar10_clipping_only_101_matched'),
    'No-DP': ('full_dynfl_no_dp', 0.608, 'v370_nodp_verified'),
}
COLORS = {'Ours': '#2563eb', 'Clipping-only': '#ea580c', 'No-DP': '#15803d'}

def normalize(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')

def matching_column(df, names):
    cols = {normalize(c): c for c in df.columns}
    for name in names:
        if normalize(name) in cols:
            return cols[normalize(name)]
    return None

def accuracy_from_summary(run_dir, policy):
    for f in [run_dir / 'summary_table.csv', run_dir.parent / 'summary_table.csv']:
        if not f.is_file():
            continue
        try:
            df = pd.read_csv(f)
            pc = matching_column(df, ['policy', 'policy_name'])
            ac = matching_column(df, ['final_test_accuracy', 'test_accuracy', 'final_accuracy'])
            if ac is None:
                continue
            if pc:
                df = df[df[pc].astype(str).eq(policy)]
            values = pd.to_numeric(df[ac], errors='coerce').dropna()
            if len(values):
                return float(values.iloc[-1])
        except Exception:
            continue
    return None

def candidate_files():
    patterns = ('round_metrics.csv', '*round*metrics*.csv', '*round*history*.csv', '*round*log*.csv', '*round*results*.csv')
    found = set()
    for pattern in patterns:
        found.update(ROOT.rglob(pattern))
    return sorted(found)

def choose_csv(label, policy, expected, root_hint, files):
    options = []
    for p in files:
        # Match policy by a full path component, not substring, to avoid 'full_dynfl' matching both ablations.
        if policy not in p.parts:
            continue
        if root_hint and root_hint not in p.parts:
            continue
        try:
            df = pd.read_csv(p)
            if len(df) < 70:
                continue
            round_col = matching_column(df, ['round', 'round_idx', 'round_index', 'communication_round'])
            ac = matching_column(df, ['test_accuracy', 'final_test_accuracy', 'accuracy', 'global_test_accuracy', 'acc'])
            if ac is None:
                continue
            final = float(pd.to_numeric(df[ac], errors='coerce').dropna().iloc[-1])
            final = final / 100 if final > 1.01 else final
            summary_acc = accuracy_from_summary(p.parent, policy)
            distance = abs(final - expected)
            if summary_acc is not None:
                if summary_acc > 1.01:
                    summary_acc /= 100
                distance = min(distance, abs(summary_acc - expected))
            options.append((distance, abs(len(df) - 101), p, df))
        except (ValueError, IndexError, OSError):
            continue
    options.sort(key=lambda x: (x[0], x[1], -x[2].stat().st_mtime))
    if not options:
        print(f'[MISSING] {label}: no appropriate CSV. Checked policy={policy}, root={root_hint}')
        return None
    distance, _, path, df = options[0]
    if distance > 0.015:
        print(f'[CHECK] {label}: closest run differs from expected accuracy by {distance:.3f}. Not plotting wrong run.')
        print(f'        Candidate: {path}')
        return None
    print(f'[FOUND] {label}: {path} ({len(df)} rows)')
    print('        Available columns:', ', '.join(df.columns))
    return df

def numeric_column(df, aliases):
    name = matching_column(df, aliases)
    return pd.to_numeric(df[name], errors='coerce') if name else None

def rounds(df):
    s = numeric_column(df, ['round', 'round_idx', 'round_index', 'communication_round'])
    if s is None or s.isna().all():
        return pd.Series(range(1, len(df)+1), dtype=float)
    if float(s.min()) == 0 and float(s.max()) == len(df)-1:
        s = s + 1
    return s.reset_index(drop=True)

def plot(datasets, file, title, ylabel, yaliases, xkind='round', accuracy=False):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    count = 0
    for label, df in datasets.items():
        y = numeric_column(df, yaliases)
        if y is None:
            continue
        y = y.reset_index(drop=True)
        if accuracy and y.dropna().max() <= 1.01:
            y *= 100
        if xkind == 'round':
            x = rounds(df)
            xlabel = 'Communication round'
        elif xkind == 'logical':
            x = numeric_column(df, ['logical_time', 'cumulative_logical_time', 'cumulative_logical_time_sec', 'logical_time_sec', 'total_logical_time_sec'])
            xlabel = 'Logical time (s)'
        else:
            x = numeric_column(df, ['cumulative_wall_time_sec', 'wall_time', 'cumulative_wall_time', 'wall_time_sec', 'total_wall_time_sec'])
            if x is None:
                x = numeric_column(df, ['round_wall_time_sec', 'round_wall_time'])
                if x is not None:
                    x = x.cumsum()
            xlabel = 'Wall time (s)'
        if x is None:
            continue
        x = x.reset_index(drop=True)
        good = x.notna() & y.notna()
        if not good.any():
            continue
        ax.plot(x[good], y[good], label=label, lw=2, color=COLORS[label])
        count += 1
    if not count:
        print(f'[SKIP] {file}: corresponding metric column not found')
        plt.close(fig)
        return
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(DEST / f'{file}.{ext}', dpi=250)
    plt.close(fig)
    print(f'[SAVED] {file}.png / .pdf ({count} series)')

files = candidate_files()
print(f'Found {len(files)} candidate round CSV files')
data = {}
for label, (policy, expected, root_hint) in GROUPS.items():
    df = choose_csv(label, policy, expected, root_hint, files)
    if df is not None:
        data[label] = df
if not data:
    print('No valid CSV found. Inspect the saved CSV locations with PowerShell:')
    print('Get-ChildItem out -Recurse -Filter *.csv | Select-Object -ExpandProperty FullName')
    raise SystemExit(1)

acc = ['test_accuracy', 'global_test_accuracy', 'accuracy', 'acc']
plot(data, 'accuracy_vs_round', 'CIFAR-10: test accuracy vs round', 'Test accuracy (%)', acc, accuracy=True)
plot(data, 'accuracy_vs_logical_time', 'Test accuracy vs logical time', 'Test accuracy (%)', acc, 'logical', True)
plot(data, 'accuracy_vs_wall_time', 'Test accuracy vs wall time', 'Test accuracy (%)', acc, 'wall', True)
plot(data, 'test_loss_vs_round', 'Test loss vs round', 'Test loss', ['test_loss', 'global_test_loss', 'validation_loss'])
plot(data, 'train_loss_vs_round', 'Training loss vs round', 'Training loss', ['train_loss', 'training_loss', 'mean_train_loss', 'loss'])
plot(data, 'sample_epsilon_vs_round', 'Sample epsilon vs round (not a guarantee for clipping-only)', 'Recorded sample epsilon', ['max_sample_epsilon', 'sample_epsilon', 'sample_eps'])
plot(data, 'noise_ratio_vs_round', 'Noise/signal ratio vs round', 'Noise/signal', ['update_dp_noise_to_signal_ratio', 'noise_to_signal_ratio', 'noise_signal_ratio'])
plot(data, 'clip_fraction_vs_round', 'Update clip fraction vs round', 'Fraction', ['update_dp_clip_fraction', 'clip_fraction'])
plot(data, 'he_clients_vs_round', 'HE mode selection vs round (profiled HE is not encryption)', 'Number of HE clients', ['num_he_clients', 'he_clients', 'he_count'])
print(f'\nDone. Output folder: {DEST.resolve()}')
