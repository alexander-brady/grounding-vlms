import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── Project directories ───────────────────────────────────────────────────────
# script.py now in /src/, so BASE_DIR is project root:
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
EVAL_DIR       = BASE_DIR / 'eval'
ANALYSIS_DIR   = EVAL_DIR / 'analysis'
DATA_DIR       = EVAL_DIR / 'data'
RESULTS_DIR    = EVAL_DIR / 'valid_results'

# Ensure analysis folder exists
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_merge(dfile: Path, rfile: Path) -> pd.DataFrame:
    """
    Load truth + results CSVs, inject integer idx from row order,
    cast columns, and return merged DataFrame with ['idx','truth','result'].
    """
    df_truth = (
        pd.read_csv(dfile)
          .reset_index(drop=False)
          .rename(columns={'index': 'idx'})
    )
    df_truth['truth'] = df_truth['truth'].astype(int)
    df_res = pd.read_csv(rfile).astype({'idx': int, 'result': int})
    return df_truth.join(df_res.set_index('idx'), on='idx', how='inner')


def summarize_counts(df):
    """Compute error metrics on a merged DataFrame."""
    e      = df['result'] - df['truth']
    abs_e  = e.abs()
    denom  = df['truth'].replace(0, np.nan).abs()

    return pd.Series({
        'n_examples':     len(df),
        'accuracy':       (e == 0).mean(),
        'over_rate':      (e > 0).mean(),
        'under_rate':     (e < 0).mean(),
        'mean_error':     e.mean(),
        'MAE':            abs_e.mean(),
        'MAPE (%)':       (abs_e / denom).mean() * 100,
        'sMAPE (%)':      (2 * abs_e / (df['truth'].abs() + df['result'].abs())).mean() * 100
    })


def find_largest_outliers(df, n=10, model=None, dataset=None,
                          outfile: Path = ANALYSIS_DIR / 'outliers.csv'):
    """Save top-n absolute-error outliers to a CSV file."""
    df_out   = df.assign(abs_err=(df['result'] - df['truth']).abs())
    df_top   = (
        df_out.nlargest(n, 'abs_err')
              .reset_index()
              .rename(columns={'index': 'idx'})
              [['idx', 'truth', 'result', 'abs_err']]
    )
    df_top['model']   = model
    df_top['dataset'] = dataset

    outfile.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if not outfile.exists() else 'a'
    df_top.to_csv(outfile, mode=mode, header=(mode=='w'), index=False)


def find_label_performance(df, n=10, model=None, dataset=None,
                           outfile: Path = ANALYSIS_DIR / 'label_accuracy.csv'):
    """Save top/worst-n label accuracies to a CSV file."""
    acc = (
        df.groupby('label')
          .apply(lambda g: (g['result'] == g['truth']).mean())
          .reset_index(name='accuracy')
    )

    counts = df.groupby('label').size().reset_index(name='count')
    acc = acc.merge(counts, on='label')

    acc_top = acc.nlargest(n, 'accuracy').assign(rank='top')
    acc_bot = acc.nsmallest(n, 'accuracy').assign(rank='worst')
    df_labels = pd.concat([acc_top, acc_bot], ignore_index=True)
    df_labels['model']   = model
    df_labels['dataset'] = dataset

    outfile.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if not outfile.exists() else 'a'
    df_labels.to_csv(outfile, mode=mode, header=(mode=='w'), index=False)


def plot_range_performance(data_dir=DATA_DIR, results_dir=RESULTS_DIR, n_bins=10):
    """Plot bin‐wise accuracy and MAE (log scale) for all models."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    out = ANALYSIS_DIR / 'plots'; out.mkdir(exist_ok=True, parents=True)

    for data_path in data_dir.glob('*_dataset.csv'):
        name     = data_path.stem.replace('_dataset', '')
        truth    = pd.read_csv(data_path)['truth'].astype(int)
        max_val  = truth.max()
        bw       = int(np.ceil((max_val + 1) / n_bins))
        bins     = np.arange(0, max_val + bw, bw)
        labels   = [f"{i}\u2013{i + bw - 1}" for i in bins[:-1]]
        binned   = pd.cut(truth, bins=bins, labels=labels, right=False)
        models   = sorted(results_dir.iterdir())

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2       = ax1.twinx()
        x         = np.arange(len(labels))
        width     = 0.8 / len(models)

        for i, model_dir in enumerate(models):
            rf = model_dir / f"{name}_results.csv"
            if not rf.exists(): 
                continue

            df   = load_and_merge(data_path, rf)
            grp  = df.assign(bin=binned).groupby('bin')
            acc  = grp.apply(lambda g: (g['result'] == g['truth']).mean()).reindex(labels)
            mae  = grp.apply(lambda g: (g['result'] - g['truth']).abs().mean()).reindex(labels)

            ax1.bar(x + i*width, acc, width=width, alpha=0.8, label=f"{model_dir.name} acc")
            ax2.plot(x + width*(len(models)-1)/2, mae, marker='o',
                     linestyle='-', linewidth=1, label=f"{model_dir.name} mae")

        ax1.set(ylim=(0,1),
                xticks=x + width*(len(models)-1)/2,
                xticklabels=labels,
                ylabel='Exact-match accuracy')
        ax2.set(yscale='log', ylabel='MAE (log)')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, bbox_to_anchor=(1.02,1), loc='upper left')
        plt.title(f"{name} — Range Performance (bw={bw}, bins={n_bins})")
        plt.tight_layout()
        plt.savefig(out / f"{name}_range.png")
        plt.close()


def plot_rolling_accuracy(data_dir=DATA_DIR, results_dir=RESULTS_DIR, window=50):
    """Overlay rolling accuracy curves for all models."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    out = ANALYSIS_DIR / 'plots'; out.mkdir(exist_ok=True, parents=True)

    for data_path in data_dir.glob('*_dataset.csv'):
        name       = data_path.stem.replace('_dataset', '')
        df_truth   = pd.read_csv(data_path)['truth'].astype(int)

        plt.figure(figsize=(10, 5))
        plt.title(f"Rolling Accuracy (w={window}) — {name}")
        plt.xlabel('Index'); plt.ylabel('Accuracy')
        plt.ylim(0, 1); plt.grid(True, linestyle='--', alpha=0.4)

        for model_dir in sorted(results_dir.iterdir()):
            rf = model_dir / f"{name}_results.csv"
            if not rf.exists():
                continue
            df   = load_and_merge(data_path, rf)
            roll = df['result'].eq(df['truth']).astype(int).rolling(window, min_periods=1).mean()
            plt.plot(roll, label=model_dir.name, linewidth=1)

        plt.legend(title='Model', bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out / f"{name}_rolling.png")
        plt.close()

def plot_best_stacked(data_dir=DATA_DIR, results_dir=RESULTS_DIR, n_bins=25, max_val=120):
    """Plot bin-wise stacked bar chart showing best accuracy model with its corresponding under/over prediction rates."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    out = ANALYSIS_DIR / 'plots'; out.mkdir(exist_ok=True, parents=True)

    for data_path in data_dir.glob('*_dataset.csv'):
        name     = data_path.stem.replace('_dataset', '')
        truth    = pd.read_csv(data_path)['truth'].astype(int)
        if max_val is None:
            max_val  = truth.max()
        else:
            max_val  = min(max_val, truth.max())
        bw       = int(np.ceil((max_val + 1) / n_bins))
        bins     = np.arange(0, max_val + bw, bw)
        labels   = [f"{i}\u2013{i + bw - 1}" for i in bins[:-1]]
        binned   = pd.cut(truth, bins=bins, labels=labels, right=False)
        models   = sorted(results_dir.iterdir())
        # Create Series to track best accuracy per bin and related metrics
        best_acc = pd.Series(0.0, index=labels)
        best_model = pd.Series("", index=labels)
        best_under = pd.Series(0.0, index=labels)
        best_over = pd.Series(0.0, index=labels)
        # Find best accuracy for each bin across all models
        for model_dir in models:
            rf = model_dir / f"{name}_results.csv"
            if not rf.exists(): 
                continue
            df   = load_and_merge(data_path, rf)
            grp  = df.assign(bin=binned).groupby('bin')
            # Calculate accuracy, underprediction and overprediction rates
            acc = grp.apply(lambda g: (g['result'] == g['truth']).mean()).reindex(labels)
            under = grp.apply(lambda g: (g['result'] < g['truth']).mean()).reindex(labels)
            over = grp.apply(lambda g: (g['result'] > g['truth']).mean()).reindex(labels)
            # Update best accuracy model and corresponding metrics
            is_better = acc > best_acc
            best_acc = best_acc.where(~is_better, acc)
            best_model = best_model.where(~is_better, model_dir.name)
            best_under = best_under.where(~is_better, under)
            best_over = best_over.where(~is_better, over)
        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(labels))
        width = 0.6
        # Create stacked bars
        bars_acc = ax.bar(x, best_acc, width=width, label='Accuracy (higher is better)', 
                          color='green', alpha=0.8)
        bars_under = ax.bar(x, best_under, width=width, bottom=best_acc, 
                            label='Underprediction (lower is better)', color='orange', alpha=0.8)
        bars_over = ax.bar(x, best_over, width=width, bottom=best_acc+best_under,
                           label='Overprediction (lower is better)', color='red', alpha=0.8)
        # Add model name labels to accuracy segment
        for i, (bar, model) in enumerate(zip(bars_acc, best_model)):
            height = bar.get_height()
            if height > 0.1:  # If accuracy segment is large enough
                ax.text(i, height/2, model, ha='center', va='center', 
                        rotation=90, fontsize=8, color='white', fontweight='bold')
            elif model:  # If accuracy segment is small but we have a model
                ax.text(i, 0.02, model, ha='center', va='bottom', 
                        rotation=90, fontsize=8, color='black')
        ax.set(ylim=(0,1),
               xticks=x,
               xticklabels=labels,
               ylabel='Rate')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.title(f"{name} — Best Model Performance per Bin (bw={bw}, bins={n_bins})")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(out / f"{name}_best_stacked.png")
        plt.close()
def plot_best_models_per_bin(data_dir=DATA_DIR, results_dir=RESULTS_DIR, n_bins=25, max_val=120):
    """Plot a single stacked bar chart showing best accuracy model with under/over prediction rates across all datasets."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    out = ANALYSIS_DIR / 'plots'; out.mkdir(exist_ok=True, parents=True)
    # Find global max value and create bins
    all_max_val = 0
    for data_path in data_dir.glob('*_dataset.csv'):
        truth = pd.read_csv(data_path)['truth'].astype(int)
        all_max_val = max(all_max_val, truth.max())
    if max_val is not None:
        all_max_val = min(all_max_val, max_val)
    bw = int(np.ceil((all_max_val + 1) / n_bins))
    bins = np.arange(0, all_max_val + bw, bw)
    labels = [f"{i}\u2013{i + bw - 1}" for i in bins[:-1]]    
    # Initialize DataFrame to store all results with model info
    all_data = []
    # Process each dataset and model
    datasets_processed = 0
    for data_path in data_dir.glob('*_dataset.csv'):
        truth = pd.read_csv(data_path)['truth'].astype(int)
        # Ensure values beyond max_val are capped
        if max_val is not None:
            truth = truth.clip(upper=max_val)
        binned = pd.cut(truth, bins=bins, labels=labels, right=False)
        dataset_name = data_path.stem.replace('_dataset', '')
        models_processed = 0
        for model_dir in results_dir.iterdir():
            model_name = model_dir.name
            rf = model_dir / f"{dataset_name}_results.csv"
            if not rf.exists():
                continue
            df = load_and_merge(data_path, rf)
            # Add model, dataset and bin information
            df['model'] = model_name
            df['dataset'] = dataset_name
            df['bin'] = binned
            all_data.append(df)
            models_processed += 1
        if models_processed > 0:
            datasets_processed += 1    
    if not all_data:
        print("No data found!")
        return
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    # Calculate metrics for each bin and model
    metrics = []
    for bin_name in labels:
        bin_data = combined_df[combined_df['bin'] == bin_name]
        bin_count = len(bin_data)
        if bin_count == 0:
            metrics.append({
                'bin': bin_name,
                'model': 'No data',
                'accuracy': 0.0,
                'underprediction': 0.0,
                'overprediction': 0.0,
                'count': 0
            })
            continue   
        for model in bin_data['model'].unique():
            model_bin_data = bin_data[bin_data['model'] == model]
            # Ensure these add up to 1.0
            acc = (model_bin_data['result'] == model_bin_data['truth']).mean()
            under = (model_bin_data['result'] < model_bin_data['truth']).mean()
            over = (model_bin_data['result'] > model_bin_data['truth']).mean()
            total = acc + under + over
            if abs(total - 1.0) > 0.01:
                print(f"Warning: Metrics don't sum to 1.0 for {model} in bin {bin_name}: {total}")
            metrics.append({
                'bin': bin_name,
                'model': model,
                'accuracy': acc,
                'underprediction': under,
                'overprediction': over,
                'count': len(model_bin_data)
            })
    metrics_df = pd.DataFrame(metrics)
    # Print bin distribution
    bin_counts = metrics_df.groupby('bin')['count'].sum()    
    # Find best model for each bin based on accuracy
    non_empty_metrics = metrics_df[metrics_df['count'] > 0]
    if len(non_empty_metrics) == 0:
        print("No data found for any bin!")
        return
    best_models = non_empty_metrics.loc[non_empty_metrics.groupby('bin')['accuracy'].idxmax()]
    best_models = best_models.set_index('bin').reindex(labels)    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    # Only plot non-empty bins to save space
    non_empty_bins = best_models.dropna(subset=['accuracy'])
    x = np.arange(len(non_empty_bins))
    bin_labels = non_empty_bins.index.tolist()
    width = 0.6
    # Get metrics from best models
    acc_values = non_empty_bins['accuracy'].values
    under_values = non_empty_bins['underprediction'].values
    over_values = non_empty_bins['overprediction'].values
    best_model_names = non_empty_bins['model'].values
    # Create stacked bars with distinct colors
    bars_acc = ax.bar(x, acc_values, width=width, label='Accuracy', 
                     color='forestgreen', alpha=0.9)
    bars_under = ax.bar(x, under_values, width=width, bottom=acc_values, 
                       label='Underprediction', color='royalblue', alpha=0.9)
    bars_over = ax.bar(x, over_values, width=width, bottom=acc_values + under_values, 
                      label='Overprediction', color='crimson', alpha=0.9)
    # Add best model names to the bars
    for i, (acc, model) in enumerate(zip(acc_values, best_model_names)):
        if acc > 0.15:  # Only add text if there's a visible bar
            ax.text(i, acc/2, model, ha='center', va='center', 
                   rotation=90, fontsize=8, color='white', fontweight='bold')
        elif model:
            ax.text(i, 0.02, model, ha='center', va='bottom', 
                   rotation=90, fontsize=8, color='black')
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set(ylim=(0, 1), xticks=x, xticklabels=bin_labels, ylabel='Rate')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(f"Best Model Performance per Bin Across All Datasets (bw={bw})", 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    # Add counts as a second row of labels
    counts = non_empty_bins['count'].fillna(0).astype(int)
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, -0.05, f"n={count}", ha='center', va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "best_models_per_bin.png")
    plt.close()

# backward-compatible wrappers
def plot_best_models_per_bin_all_models(**kwargs):
    return plot_best_models_per_bin(**kwargs)

def plot_best_stacked_all_models(**kwargs):
    return plot_best_stacked(**kwargs)

def plot_range_performance_all_models(**kwargs):
    return plot_range_performance(**kwargs)


def plot_rolling_accuracy_all_models(**kwargs):
    return plot_rolling_accuracy(**kwargs)


def create_results(data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    """Generate summaries, plots, and outlier/label reports for all models and datasets."""
    print("Creating results...")
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    out = ANALYSIS_DIR

    # clear old reports
    for fname in ['outliers.csv', 'label_accuracy.csv', 'summary.csv']:
        fpath = out / fname
        if fpath.exists():
            fpath.unlink()

    results = []
    for model_dir in sorted(results_dir.iterdir()):
        for rf in model_dir.glob('*_results.csv'):
            name = rf.stem.replace('_results', '')
            truth_file = DATA_DIR / f"{name}_dataset.csv"
            df    = load_and_merge(truth_file, rf)
            df    = df[df['result'] != -1]  # Filter out rows where result is -1

            find_largest_outliers(df, model=model_dir.name, dataset=name)
            find_label_performance(df, model=model_dir.name, dataset=name)

            summary = summarize_counts(df)
            summary['model'], summary['dataset'] = model_dir.name, name
            results.append(summary)

    # Sort by dataset and write to CSV
    summary_df = pd.DataFrame(results)[[
        'model','dataset','n_examples','accuracy',
        'over_rate','under_rate','mean_error',
        'MAE','MAPE (%)','sMAPE (%)'
    ]]
    summary_df = summary_df.sort_values(by=['dataset', 'model'])
    summary_df.to_csv(out / 'summary.csv', index=False)

    print("Summary saved to", out / 'summary.csv')
    plot_rolling_accuracy_all_models()
    plot_range_performance_all_models()
    # plot_best_stacked_all_models()
    # plot_best_models_per_bin_all_models()
    print("Plots saved to", out / 'plots')