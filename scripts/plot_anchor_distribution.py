#!/usr/bin/env python3
"""
Plot anchor count per file and cumulative anchor count curve.
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot anchor count per file and cumulative curve.')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with semantic anchors')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading anchors from {args.input}...")
    df = pd.read_csv(args.input, usecols=['file_path'])
    print(f"Loaded {len(df):,} anchors.")

    # Count anchors per file
    anchor_counts = df['file_path'].value_counts().sort_values(ascending=False)
    anchor_counts = anchor_counts.reset_index()
    anchor_counts.columns = ['file_path', 'anchor_count']

    # Cumulative sum
    anchor_counts['cumulative_anchors'] = anchor_counts['anchor_count'].cumsum()
    anchor_counts['file_index'] = range(1, len(anchor_counts)+1)

    # Save summary CSV
    summary_csv = os.path.join(args.output, 'anchor_count_per_file.csv')
    anchor_counts.to_csv(summary_csv, index=False)
    print(f"Anchor count summary saved to: {summary_csv}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(anchor_counts['file_index'], anchor_counts['anchor_count'], label='Anchors per file', color='tab:blue', alpha=0.7)
    ax1.set_xlabel('File (sorted by anchor count)')
    ax1.set_ylabel('Anchors per file', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(anchor_counts['file_index'], anchor_counts['cumulative_anchors'], label='Cumulative anchors', color='tab:orange')
    ax2.set_ylabel('Cumulative anchors', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Anchor Count per File and Cumulative Anchor Curve')
    fig.tight_layout()
    plot_path = os.path.join(args.output, 'anchor_distribution.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
