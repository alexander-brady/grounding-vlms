import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

def create_unified_dataframe(data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    # Create combined dataframe with error handling and validation
    unified = pd.DataFrame()
    dataset_counts = {}
    model_counts = {}

    # Track processing statistics
    datasets_processed = 0
    datasets_skipped = 0
    models_processed = 0

    for data_path in DATA_DIR.glob('*_dataset.csv'):
        try:
            dataset_name = data_path.stem.replace('_dataset', '')
            print(f"Processing dataset: {dataset_name}")
            
            # Load dataset with error handling
            try:
                # First check what columns are available in the dataset
                all_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
                
                # Define the columns we want to extract
                required_columns = ['prompt', 'truth', 'label']
                
                # Handle file_name/image_url column variation
                file_column = None
                if 'file_name' in all_columns:
                    file_column = 'file_name'
                elif 'image_url' in all_columns:
                    file_column = 'image_url'
                else:
                    print(f"Warning: Neither 'file_name' nor 'image_url' found in {dataset_name}")
                    datasets_skipped += 1
                    continue
                
                columns_to_extract = [file_column] + required_columns
                
                # Load the dataset with the correct columns
                df = pd.read_csv(data_path, usecols=columns_to_extract)
                
                # Check if all required columns exist
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: Missing columns in {dataset_name}: {missing_columns}")
                    datasets_skipped += 1
                    continue
                    
                # Rename the file column consistently to file_name
                if file_column != 'file_name':
                    df = df.rename(columns={file_column: 'file_name'})
                
                # Create and validate index
                df = df.reset_index(drop=False).rename(columns={'index': 'idx'})
                
                # Check for duplicate indices
                if df['idx'].duplicated().any():
                    print(f"Warning: Duplicate indices found in {dataset_name}. Creating new unique indices.")
                    df = df.reset_index(drop=True)
                    df['idx'] = df.index
                
                df = df.astype({
                    'idx': int,
                    'file_name': str,
                    'prompt': str,
                    'truth': int,
                    'label': str
                }, errors='raise')  # Catch type conversion errors
                
                df['dataset'] = dataset_name
                dataset_counts[dataset_name] = len(df)
                
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
                datasets_skipped += 1
                continue
            
            # Process model results
            model_results = []
            dataset_models_processed = 0
            
            for model_dir in sorted(RESULTS_DIR.iterdir()):
                if not model_dir.is_dir():
                    continue
                    
                resultFile = model_dir / f'{dataset_name}_results.csv'
                if not resultFile.exists():
                    print(f"Information: Results for {dataset_name} not found in {model_dir.name}")
                    continue
                    
                try:
                    rd_df = pd.read_csv(resultFile, usecols=['idx', 'result'])
                    
                    # Validate model results
                    if rd_df.empty:
                        print(f"Warning: Empty results for {model_dir.name} on {dataset_name}")
                        continue
                        
                    rd_df['model'] = model_dir.stem
                    rd_df = rd_df.astype({
                        'idx': int,
                        'result': int,
                        'model': str
                    })
                    
                    # Check if model indices match dataset indices
                    missing_indices = set(df['idx']) - set(rd_df['idx'])
                    if missing_indices:
                        print(f"Warning: Model {model_dir.name} is missing {len(missing_indices)} indices for {dataset_name}")
                    rd_df = rd_df[rd_df['result']!=-1]  # Filter out invalid results
                    
                    model_results.append(rd_df)
                    dataset_models_processed += 1
                    model_counts[model_dir.stem] = model_counts.get(model_dir.stem, 0) + 1
                    
                except Exception as e:
                    print(f"Error processing {model_dir.name} for {dataset_name}: {str(e)}")
                    continue
            
            if model_results:
                # Combine all model results
                all_results = pd.concat(model_results, ignore_index=True)
                
                # Merge with dataset info with appropriate error handling
                try:
                    final_df = df.merge(all_results.rename(columns={'result': 'model_result'}), 
                                        on='idx', how='left', validate='one_to_many')
                    
                    # Add to unified dataframe
                    unified = pd.concat([unified, final_df], ignore_index=True)
                    models_processed += dataset_models_processed
                    datasets_processed += 1
                    
                except pd.errors.MergeError as e:
                    print(f"Merge error for {dataset_name}: {str(e)}")
            else:
                print(f"No valid model results found for {dataset_name}")
        
        except Exception as e:
            print(f"Unexpected error processing {data_path}: {str(e)}")

    # Check if we have data to save
    if not unified.empty:
        # Ensure output directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create output path with safeguards
        output_path = RESULTS_DIR / "combinedResults.csv"
        
        # Backup existing file if it exists
        if output_path.exists():
            backup_path = RESULTS_DIR / f"combinedResults_backup.csv"
            os.rename(output_path, backup_path)
            print(f"Existing file backed up to: {backup_path}")
        
        # Save the unified results
        unified.to_csv(output_path, index=False)
        print(f"Combined results saved to: {output_path}")
        
        # Display basic statistics
        print(f"\nSummary Statistics:")
        print(f"Combined dataset size: {len(unified)} rows")
        print(f"Datasets processed successfully: {datasets_processed} of {datasets_processed + datasets_skipped}")
        print(f"Models evaluated: {unified['model'].nunique()} unique models")
        print(f"Total dataset-model combinations: {len(unified)}")
        
        # Show dataset distribution
        print("\nRows per dataset:")
        dataset_distribution = unified.groupby('dataset').size()
        for dataset, count in dataset_distribution.items():
            print(f"  - {dataset}: {count} rows")
            
        # Show model distribution
        print("\nDatasets evaluated by each model:")
        for model, count in model_counts.items():
            print(f"  - {model}: {count} datasets")
    else:
        print("Error: No data was processed successfully. No output file created.")




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
    """Plot bin-wise accuracy and MAE (log scale) for all models."""
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

def plot_model_accuracies_by_bin(csv_path, bin_width, value_range, log_scale=False, minData=3, output_path='accuracy_plot.pdf'):
    # Unpack the value_range tuple into min_value and max_value
    min_value, max_value = value_range
    
    # Load data
    df = pd.read_csv(csv_path)

    # Ensure numeric types for comparison
    df['truth'] = pd.to_numeric(df['truth'], errors='coerce')
    df['model_result'] = pd.to_numeric(df['model_result'], errors='coerce')

    # Filter out rows with NaNs in numeric columns
    df = df.dropna(subset=['truth', 'model_result'])

    # Filter rows based on the specified value range
    df = df[(df['truth'] >= min_value) & (df['truth'] <= max_value)]

    # Create bin edges based on the range
    bins = np.arange(min_value, max_value + bin_width, bin_width)
    df['bin'] = pd.cut(df['truth'], bins=bins, include_lowest=True)

    # Determine model accuracies in each bin
    df['correct'] = df['truth'] == df['model_result']
    # Fix deprecated observed parameter warning
    grouped = df.groupby(['model', 'bin'], observed=False)['correct'].agg(['mean', 'count']).reset_index()

    # Pivot the data to plot by model
    pivot_mean = grouped.pivot(index='bin', columns='model', values='mean')
    pivot_count = grouped.pivot(index='bin', columns='model', values='count')

    # Calculate bin centers for plotting (keep this for the actual plot positioning)
    bin_centers = [(interval.left + interval.right) / 2 for interval in pivot_mean.index]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Fix deprecated get_cmap warning
    cmap = plt.colormaps['tab10']
    num_models = len(pivot_mean.columns)

    for i, model in enumerate(pivot_mean.columns):
        # Copy the data and set NaN for bins with insufficient data
        plot_data = pivot_mean[model].copy()
        mask = pivot_count[model] < minData
        plot_data[mask] = np.nan
        
        # Plot the data - matplotlib will automatically skip NaN values but connect the line
        ax.plot(bin_centers, plot_data,
                label=model,
                marker='o',
                markersize=2,
                linewidth=1,
                color=cmap(i % cmap.N))

    ax.set_xlabel('Truth Value Bins')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Model Accuracy by Truth Value Bin ({min_value}-{max_value}), Bin Width: {bin_width}')
    ax.legend()
    ax.grid(True)

    if log_scale:
        ax.set_yscale('log')

    # Adjust x-axis ticks and labels (every 5th bin)
    tick_indices = np.arange(0, len(bin_centers), 5)
    ax.set_xticks([bin_centers[i] for i in tick_indices if i < len(bin_centers)])
    # Format labels to show inclusive bounds [lower, upper]
    ax.set_xticklabels([f"[{interval.left:.0f},{(interval.right-1):.0f}]" 
                        for interval in [pivot_mean.index[i] for i in tick_indices if i < len(bin_centers)]],
                        rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_label_performance(csv_path, minData=5, maxTruth=20):

    df = pd.read_csv(csv_path)
    #get rid of too high values (model too inaccurate anyhow here)
    df = df[df['truth'] <= maxTruth]

    # First calculate counts for each truth-label combination (across all models)
    truth_label_counts = df.groupby(['truth', 'label']).size().reset_index(name='count')

    # Filter to only include combinations with at least minData values
    valid_combinations = truth_label_counts[truth_label_counts['count'] >= minData]

    # Apply the filter to the original dataframe
    df_filtered = df.merge(valid_combinations[['truth', 'label']], on=['truth', 'label'])

    # Group by truth and label (not model) to get aggregate metrics
    grouped_by_truth_label = df_filtered.groupby(['truth', 'label'])

    # Calculate accuracy metrics across all models for each truth-label combination
    label_accuracy = grouped_by_truth_label.apply(lambda g: pd.Series({
        'count': len(g),
        'exact_match_rate': (g['truth'] == g['model_result']).mean(),
        'within_one_rate': (abs(g['truth'] - g['model_result']) <= 1).mean(),
        'mean_abs_error': abs(g['truth'] - g['model_result']).mean(),
        'std_dev': g['model_result'].std(),
        'num_models': g['model'].nunique()  # Count unique models in this group
    })).reset_index()

    # Sort by truth value for easier analysis
    label_accuracy = label_accuracy.sort_values(['truth', 'label'])

    print(f"Accuracy metrics aggregated by truth-label combinations (with minimum {minData} samples):")
    # Optionally save the results to CSV
    output_path = ANALYSIS_DIR / "truth_label_accuracy.csv"
    label_accuracy.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    # Create plot directory if it doesn't exist
    plots_dir = ANALYSIS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    # Determine total number of possible truth values
    possible_truth_values = maxTruth + 1  # 0 to maxTruth inclusive
    min_required_coverage = 2*possible_truth_values // 3  # Has at least 2/3 of the truth bins covered
    # Count how many different truth values each label has data for
    label_truth_coverage = label_accuracy.groupby('label')['truth'].nunique()
    # Filter to only include labels with data for at least half of the truth values
    valid_labels = label_truth_coverage[label_truth_coverage >= min_required_coverage].index.tolist()
    # Calculate overall performance only for valid labels
    label_overall_performance = label_accuracy[label_accuracy['label'].isin(valid_labels)].groupby('label')['exact_match_rate'].mean().sort_values(ascending=False)
    # Get top 3 and bottom 3 labels from the filtered set for the plot
    top_labels = label_overall_performance.head(3).index.tolist()
    bottom_labels = label_overall_performance.tail(3).index.tolist()
    # Calculate average performance for each truth value across all labels
    overall_truth_performance = label_accuracy.groupby('truth')['exact_match_rate'].mean().reset_index()
    # Create a new plot
    plt.figure(figsize=(14, 8))
    # Use positive colors for top performers
    top_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    # Use negative colors for bottom performers
    bottom_colors = ['#d62728', '#9467bd', '#8c564b']  # Red, Purple, Brown
    # Plot the top 3 performing labels
    for i, label in enumerate(top_labels):
        label_data = label_accuracy[label_accuracy['label'] == label].sort_values('truth')
        plt.plot(
            label_data['truth'], 
            label_data['exact_match_rate'],
            color=top_colors[i],
            linestyle='-',
            linewidth=2.5,
            marker='o',
            markersize=7,
            label=f"{label} (top {i+1})"
        )
    # Plot the bottom 3 performing labels
    for i, label in enumerate(bottom_labels):
        label_data = label_accuracy[label_accuracy['label'] == label].sort_values('truth')
        plt.plot(
            label_data['truth'], 
            label_data['exact_match_rate'],
            color=bottom_colors[i],
            linestyle='--',
            linewidth=2,
            marker='s',
            markersize=6,
            label=f"{label} (bottom {i+1})"
        )
    # Plot the overall average performance in black
    plt.plot(
        overall_truth_performance['truth'],
        overall_truth_performance['exact_match_rate'],
        color='black',
        linestyle='-',
        linewidth=3,
        marker='*',
        markersize=8,
        label='Overall Average'
    )
    # Add a horizontal reference line at 50%
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    # Set plot properties
    plt.xlabel('Ground Truth Value', fontsize=12)
    plt.ylabel('Exact Match Rate', fontsize=12)
    plt.title(f'Label Performance by Ground Truth Value (0-{maxTruth})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    # Set x-axis to show integer values only
    plt.xticks(range(0, maxTruth+1))
    # Add legend with better positioning
    plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=10)
    plt.tight_layout()
    # Save the plot
    output_path = plots_dir / "label_performance_by_truth_value.pdf"
    plt.savefig(output_path)
    print(f"Performance comparison plot saved to: {output_path}")
    # ----- Create a detailed performance table for top and bottom labels -----
    # Get top 15 and bottom 15 labels for detailed analysis
    top15_labels = label_overall_performance.head(15).index.tolist()
    bottom15_labels = label_overall_performance.tail(15).index.tolist()
    selected_labels = top15_labels + bottom15_labels
    # Create a summary DataFrame with aggregate metrics for these labels
    summary_data = []
    for label in selected_labels:
        # Get all data for this label
        label_data = label_accuracy[label_accuracy['label'] == label]
        # Calculate aggregate metrics
        avg_accuracy = label_data['exact_match_rate'].mean()
        avg_within_one = label_data['within_one_rate'].mean()
        avg_error = label_data['mean_abs_error'].mean()
        avg_std = label_data['std_dev'].mean()
        total_samples = label_data['count'].sum()
        truth_coverage = label_data['truth'].nunique()
        coverage_percent = (truth_coverage / (maxTruth + 1)) * 100
        # Store in list
        summary_data.append({
            'label': label,
            'avg_accuracy': avg_accuracy,
            'avg_within_one': avg_within_one,
            'avg_error': avg_error,
            'avg_std': avg_std,
            'total_samples': total_samples,
            'truth_coverage': f"{truth_coverage}/{maxTruth+1} ({coverage_percent:.1f}%)"
        })
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    # Mark top/bottom labels
    summary_df['category'] = 'Middle'
    summary_df.loc[summary_df['label'].isin(top15_labels), 'category'] = 'Top'
    summary_df.loc[summary_df['label'].isin(bottom15_labels), 'category'] = 'Bottom'
    # Sort by category first (top, then bottom), then by accuracy
    summary_df['category_order'] = summary_df['category'].map({'Top': 0, 'Bottom': 1, 'Middle': 2})
    summary_df = summary_df.sort_values(['category_order', 'avg_accuracy'], ascending=[True, False])
    summary_df = summary_df.drop(columns=['category_order'])
    # Format the numeric columns
    summary_df['avg_accuracy'] = summary_df['avg_accuracy'].map('{:.3f}'.format)
    summary_df['avg_within_one'] = summary_df['avg_within_one'].map('{:.3f}'.format)
    summary_df['avg_error'] = summary_df['avg_error'].map('{:.2f}'.format)
    summary_df['avg_std'] = summary_df['avg_std'].map('{:.2f}'.format)
    # Create a figure and axis for the table
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')
    # Create the table
    table_data = summary_df[['label', 'category', 'avg_accuracy', 'avg_within_one', 'avg_error', 'avg_std', 'total_samples', 'truth_coverage']]
    table = ax.table(
        cellText=table_data.values,
        colLabels=['Label', 'Category', 'Accuracy', 'Within ±1', 'Mean Error', 'Std Dev', 'Samples', 'Truth Coverage'],
        loc='center',
        cellLoc='center'
    )
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Color the header row
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#D8E9F0')
        table[(0, i)].set_text_props(weight='bold')
    # Color the rows based on category
    for i in range(len(table_data)):
        category = table_data.iloc[i]['category']
        if category == 'Top':
            color = '#E5F5E0'  # Light green
        elif category == 'Bottom':
            color = '#FEE0D2'  # Light red
        else:
            color = 'white'
            
        for j in range(len(table_data.columns)):
            table[(i+1, j)].set_facecolor(color)
    # Add a title
    plt.title('Top and Bottom 15 Label Performance', fontsize=14, y=1.05)
    # Save the table as PDF
    table_output_path = plots_dir / "label_performance_table.pdf"
    plt.tight_layout()
    plt.savefig(table_output_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Performance table saved to: {table_output_path}")

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

    create_unified_dataframe(data_dir, results_dir)
    plot_model_accuracies_by_bin(
        csv_path=RESULTS_DIR / 'combinedResults.csv',
        bin_width=2,
        value_range=(0, 100),
        log_scale=False,
        minData=10,
        output_path=ANALYSIS_DIR / 'plots' / 'accuracy_plot.pdf'
    )
    plot_label_performance(
        csv_path=RESULTS_DIR / 'combinedResults.csv',
        minData=5,
        maxTruth=20
    )

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
    plot_best_stacked_all_models()
    plot_best_models_per_bin_all_models()
    print("Plots saved to", out / 'plots')