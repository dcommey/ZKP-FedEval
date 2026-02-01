#!/usr/bin/env python3
"""
Analysis script for ZKP Federated Evaluation results.
Generates visualization plots and performance summary tables.
"""

import os
import sys
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap # For wrapping long titles

OUTPUT_DIR = 'analysis_output'

PLOT_SETTINGS = {
    'style': 'seaborn-v0_8-ticks', # Use a style suitable for papers/reports
    'figsize': (6, 4), # Slightly smaller default, adjust as needed
    'dpi': 300,
    'formats': ['pdf'], # Save plots only in PDF format
    'errorbar': 'sd', # Show standard deviation as error bars in seaborn plots
    'errorbar_capsize': 5 # Cap size for error bars
}

def save_plot(fig, filename_base):
    """Saves the plot in PDF format."""
    filepath = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    try:
        fig.savefig(filepath, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight', format='pdf')
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig)

def save_latex_table(df, filename, caption, label):
    """Saves DataFrame as LaTeX table with proper formatting and metadata."""
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.tex")
    try:
        latex_string = df.to_latex(index=True, caption=caption, label=label, escape=True, float_format="%.2f")
        with open(filepath, 'w') as f:
            f.write(latex_string)
        print(f"LaTeX table saved to {filepath}")
    except Exception as e:
        print(f"Error saving LaTeX table {filepath}: {e}")

def load_results(filepath):
    """Loads results from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        return []

def preprocess_data(results):
    """Preprocesses the results data."""
    # Ensure precision and constraint_count are present, handle missing
    for r in results:
        if 'precision' not in r.get('params', {}):
             # Try to infer from SCALE_FACTOR if possible, else default or mark as unknown
             # This part is tricky as SCALE_FACTOR isn't saved per run by default
             # Defaulting to 6 for older results
             r['params']['precision'] = r['params'].get('precision', 6)
        if 'constraint_count' not in r:
            r['constraint_count'] = r.get('constraint_count', -1) # Default if missing

    df = pd.json_normalize(results)
    return df

def plot_time_vs_clients(df):
    """Generates plot comparing client proof generation and server verification times across different client counts."""
    print("Generating plot: Time vs. Number of Clients")
    grouped = df.groupby(['params.dataset', 'params.num_clients'])[[
        'avg_client_proof_gen_time_s',
        'avg_server_verify_time_s',
    ]].agg(['mean', 'std']).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped.rename(columns={
        'params.dataset': 'Dataset',
        'params.num_clients': 'Number of Clients',
        'avg_client_proof_gen_time_s_mean': 'Avg. Client Proof Gen Time (s)',
        'avg_client_proof_gen_time_s_std': 'Avg. Client Proof Gen Time Std',
        'avg_server_verify_time_s_mean': 'Avg. Server Verify Time (s)',
        'avg_server_verify_time_s_std': 'Avg. Server Verify Time Std'
    }, inplace=True)

    plt.style.use(PLOT_SETTINGS['style'])
    fig, axes = plt.subplots(1, 2, figsize=(PLOT_SETTINGS['figsize'][0]*2, PLOT_SETTINGS['figsize'][1]), sharey=False)

    sns.lineplot(data=grouped, x='Number of Clients', y='Avg. Client Proof Gen Time (s)',
                 hue='Dataset', style='Dataset', markers=True, dashes=False,
                 errorbar=PLOT_SETTINGS['errorbar'], err_style="bars", ax=axes[0])
    axes[0].set_title('Client Proof Generation Time')
    axes[0].set_xlabel('Number of Clients')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(title='Dataset')
    axes[0].set_xticks(df['params.num_clients'].unique())

    sns.lineplot(data=grouped, x='Number of Clients', y='Avg. Server Verify Time (s)',
                 hue='Dataset', style='Dataset', markers=True, dashes=False,
                 errorbar=PLOT_SETTINGS['errorbar'], err_style="bars", ax=axes[1])
    axes[1].set_title('Server Verification Time (per Proof)')
    axes[1].set_xlabel('Number of Clients')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(title='Dataset')
    axes[1].set_xticks(df['params.num_clients'].unique())

    plt.suptitle('Computation Time vs. Number of Clients (Mean ± Std Dev over Seeds)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "time_vs_clients")

def plot_comm_cost_vs_clients(df):
    """Analyzes communication overhead scaling with number of participating clients."""
    print("Generating plot: Communication Cost vs. Number of Clients")
    grouped = df.groupby(['params.dataset', 'params.num_clients'])[[
        'avg_comm_cost_proof_kib',
        'total_comm_cost_upload_kib'
    ]].agg(['mean', 'std']).reset_index()
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped.rename(columns={
        'params.dataset': 'Dataset',
        'params.num_clients': 'Number of Clients',
        'avg_comm_cost_proof_kib_mean': 'Avg. Proof Size (KiB)',
        'avg_comm_cost_proof_kib_std': 'Avg. Proof Size Std',
        'total_comm_cost_upload_kib_mean': 'Total Upload Cost (KiB)',
        'total_comm_cost_upload_kib_std': 'Total Upload Cost Std'
    }, inplace=True)

    plt.style.use(PLOT_SETTINGS['style'])
    fig, axes = plt.subplots(1, 2, figsize=(PLOT_SETTINGS['figsize'][0]*2, PLOT_SETTINGS['figsize'][1]), sharey=False)

    sns.lineplot(data=grouped, x='Number of Clients', y='Avg. Proof Size (KiB)',
                 hue='Dataset', style='Dataset', markers=True, dashes=False,
                 errorbar=PLOT_SETTINGS['errorbar'], err_style="bars", ax=axes[0])
    axes[0].set_title('Average Proof Size')
    axes[0].set_xlabel('Number of Clients')
    axes[0].set_ylabel('Size (KiB)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(title='Dataset')
    axes[0].set_xticks(df['params.num_clients'].unique())
    min_proof_size = grouped['Avg. Proof Size (KiB)'].min()
    max_proof_size = grouped['Avg. Proof Size (KiB)'].max()
    if max_proof_size > 0 and (max_proof_size - min_proof_size) / max_proof_size < 0.05:
         axes[0].set_ylim(min_proof_size * 0.95, max_proof_size * 1.05)

    sns.lineplot(data=grouped, x='Number of Clients', y='Total Upload Cost (KiB)',
                 hue='Dataset', style='Dataset', markers=True, dashes=False,
                 errorbar=PLOT_SETTINGS['errorbar'], err_style="bars", ax=axes[1])
    axes[1].set_title('Total Upload Communication Cost')
    axes[1].set_xlabel('Number of Clients')
    axes[1].set_ylabel('Cost (KiB)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(title='Dataset')
    axes[1].set_xticks(df['params.num_clients'].unique())

    plt.suptitle('Communication Cost vs. Number of Clients (Mean ± Std Dev over Seeds)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "comm_cost_vs_clients")

def plot_validation_vs_threshold(df):
    """Analyzes relationship between loss threshold and validation success rate."""
    print("Generating plot: Validation Rate vs. Loss Threshold")
    grouped = df.groupby(['params.dataset', 'params.loss_threshold'])[[
        'valid_proof_percentage'
    ]].agg(['mean', 'std']).reset_index()
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped.rename(columns={
        'params.dataset': 'Dataset',
        'params.loss_threshold': 'Loss Threshold',
        'valid_proof_percentage_mean': 'Valid Proof Rate (%)',
        'valid_proof_percentage_std': 'Valid Proof Rate Std'
    }, inplace=True)

    plt.style.use(PLOT_SETTINGS['style'])
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'])

    sns.lineplot(data=grouped, x='Loss Threshold', y='Valid Proof Rate (%)',
                 hue='Dataset', style='Dataset', markers=True, dashes=False,
                 errorbar=PLOT_SETTINGS['errorbar'], err_style="bars", ax=ax)

    ax.set_title('Valid Proof Rate vs. Loss Threshold (Mean ± Std Dev over Seeds)')
    ax.set_xlabel('Loss Threshold')
    ax.set_ylabel('Valid Proofs (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Dataset')
    ax.set_ylim(-5, 105)
    ax.set_xticks(df['params.loss_threshold'].unique())

    plt.tight_layout()
    save_plot(fig, "validation_vs_threshold")

def generate_summary_table(df):
    """Generates LaTeX table with key performance metrics averaged across experiment runs."""
    print("Generating summary LaTeX table.")
    summary_df = df.groupby(['params.dataset', 'params.num_clients']).agg(
        avg_client_time_mean=('avg_client_proof_gen_time_s', 'mean'),
        avg_client_time_std=('avg_client_proof_gen_time_s', 'std'),
        avg_verify_time_mean=('avg_server_verify_time_s', 'mean'),
        avg_verify_time_std=('avg_server_verify_time_s', 'std'),
        avg_comm_kib_mean=('avg_comm_cost_proof_kib', 'mean'),
        avg_comm_kib_std=('avg_comm_cost_proof_kib', 'std'),
        valid_perc_mean=('valid_proof_percentage', 'mean'),
        valid_perc_std=('valid_proof_percentage', 'std')
    ).reset_index()

    summary_df.columns = ['Dataset', 'Num Clients',
                           'Client Time (s)', 'Client Time Std',
                           'Verify Time (s)', 'Verify Time Std',
                           'Comm Cost (KiB)', 'Comm Cost Std',
                           'Valid Proofs (%)', 'Valid Proofs Std']

    summary_table = summary_df[['Dataset', 'Num Clients', 'Client Time (s)',
                                'Verify Time (s)', 'Comm Cost (KiB)', 'Valid Proofs (%)']]

    caption = "Summary of Performance Metrics (Mean over Seeds)"
    label = "tab:performance_summary"
    save_latex_table(summary_table.set_index(['Dataset', 'Num Clients']), "summary_table", caption, label)

def plot_scalability(df, dataset='mnist', threshold=1.0):
    """Generates plot showing total server verification time vs. number of clients."""
    print(f"Generating plot: Scalability (Total Server Time vs. Clients) for {dataset}, T={threshold}")

    # Filter for the specific dataset and threshold used in scalability runs
    df_filtered = df[(df['params.dataset'] == dataset) & (df['params.loss_threshold'] == threshold)].copy()

    if df_filtered.empty:
        print(f"Warning: No data found for scalability plot (dataset={dataset}, threshold={threshold}). Skipping plot.", file=sys.stderr)
        return

    # Group by number of clients and calculate mean/std of total server time
    grouped = df_filtered.groupby('params.num_clients')['total_server_verify_time_s'].agg(['mean', 'std']).reset_index()
    grouped.rename(columns={
        'params.num_clients': 'Number of Clients',
        'mean': 'Total Server Verify Time (s)',
        'std': 'Total Server Verify Time Std'
    }, inplace=True)

    plt.style.use(PLOT_SETTINGS['style'])
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'])

    # Plotting the mean line
    sns.lineplot(data=grouped, x='Number of Clients', y='Total Server Verify Time (s)',
                 marker='o', ax=ax, label=f'{dataset} (T={threshold})')

    # Adding error bars (std deviation)
    ax.errorbar(grouped['Number of Clients'], grouped['Total Server Verify Time (s)'],
                yerr=grouped['Total Server Verify Time Std'], fmt='none', color='gray',
                capsize=PLOT_SETTINGS['errorbar_capsize'], label='Std Dev' if not ax.get_legend() else "") # Avoid duplicate legend label

    ax.set_title('Server Scalability: Total Verification Time')
    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Total Time (seconds)')
    ax.grid(True, linestyle='--', alpha=0.6)
    # Ensure legend includes Std Dev if error bars are plotted
    handles, labels = ax.get_legend_handles_labels()
    if 'Std Dev' not in labels and not grouped['Total Server Verify Time Std'].isnull().all():
         # Add a dummy entry for the legend if needed
         from matplotlib.lines import Line2D
         handles.append(Line2D([0], [0], color='gray', lw=0, marker='_')) # Represent std dev bar
         labels.append('Std Dev')
    ax.legend(handles=handles, labels=labels, title='Configuration')


    # Ensure x-axis ticks match the client counts tested
    ax.set_xticks(grouped['Number of Clients'].unique())

    plt.tight_layout()
    save_plot(fig, "scalability_total_server_time")

def generate_precision_table(df, output_dir="analysis_output"):
    """Generates a Markdown table summarizing fixed-point precision results."""
    print("Generating table: Fixed-Point Precision Analysis")

    # Filter out runs where constraint count might be invalid (-1) if needed
    df_filtered = df[df['constraint_count'] > 0].copy()

    if df_filtered.empty or 'params.precision' not in df_filtered.columns:
        print("Warning: No data found or 'params.precision' column missing for precision table. Skipping.", file=sys.stderr)
        return

    # Group by dataset and precision
    grouped = df_filtered.groupby(['params.dataset', 'params.precision'])[[
        'constraint_count',
        'avg_client_proof_gen_time_s'
    ]].agg({
        'constraint_count': 'first', # Should be the same for all runs with same precision
        'avg_client_proof_gen_time_s': ['mean', 'std']
    }).reset_index()

    # Flatten multi-index columns and rename
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped.rename(columns={
        'params.dataset': 'Dataset',
        'params.precision': 'Precision (dec. places)',
        'constraint_count_first': 'Circuit Constraints',
        'avg_client_proof_gen_time_s_mean': 'Proof Gen. Time (s)',
        'avg_client_proof_gen_time_s_std': 'Proof Gen. Time Std'
    }, inplace=True)

    # Add placeholder for Match with FP (%) - Requires separate calculation
    grouped['Match with FP (%)'] = 'N/A' # Placeholder

    # Reorder columns for the table
    grouped = grouped[[
        'Dataset', 'Precision (dec. places)', 'Circuit Constraints',
        'Proof Gen. Time (s)', 'Match with FP (%)' # Removed Std for brevity, matching example
    ]]

    # Format Proof Gen Time
    grouped['Proof Gen. Time (s)'] = grouped['Proof Gen. Time (s)'].round(2)

    # Sort for consistent table output
    grouped.sort_values(by=['Dataset', 'Precision (dec. places)'], inplace=True)

    # Generate Markdown Table
    md_table = grouped.to_markdown(index=False)

    # Add Caption and Label for Markdown (as comments or surrounding text)
    md_output = f"<!-- Table: Effect of Fixed-Point Precision -->\n"
    md_output += f"**Table: Effect of Fixed-Point Precision on Circuit Properties and Accuracy**\n\n"
    md_output += md_table
    md_output += "\n\n*Note: 'Match with FP (%)' requires separate analysis and is shown as N/A.*\n"
    md_output += "<!-- End Table -->\n"


    # Save Markdown Table
    output_path = os.path.join(output_dir, "precision_summary_table.md")
    try:
        with open(output_path, 'w') as f:
            f.write(md_output)
        print(f"Precision summary table saved to {output_path}")
    except IOError as e:
        print(f"Error saving precision table: {e}", file=sys.stderr)

def main_analysis(args):
    """Main function to perform analysis."""
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}", file=sys.stderr)
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    results = load_results(args.results_file)
    if not results:
        print("No results loaded. Exiting.", file=sys.stderr)
        return

    df = preprocess_data(results)

    # Generate standard plots and tables
    plot_time_vs_clients(df)
    plot_comm_cost_vs_clients(df)
    plot_validation_vs_threshold(df)
    generate_summary_table(df)

    # --- Generate New Plots/Tables ---
    plot_scalability(df) # Assumes default dataset/threshold or adjust as needed
    generate_precision_table(df, args.output_dir)

    print("\nAnalysis complete. Plots and tables saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ZKP Federated Evaluation Results")
    parser.add_argument("results_file", help="Path to the JSON results file.")
    parser.add_argument("-o", "--output_dir", default="analysis_output", help="Directory to save plots and tables.")
    args = parser.parse_args()
    main_analysis(args)