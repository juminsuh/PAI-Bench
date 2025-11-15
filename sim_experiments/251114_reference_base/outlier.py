"""
Find outliers in CLIP reference similarities using IQR method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_outliers_iqr(csv_path: str, folder_name: str, output_dir: str = "./analyze"):
    """
    Find outliers for a specific folder using IQR method.
    
    Args:
        csv_path: Path to the clip_reference_similarities.csv file
        folder_name: Name of the folder to analyze (e.g., "BrunoMars")
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("=" * 80)
    print(f"Loading data from {csv_path}")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    
    # Filter for specific folder
    folder_df = df[df['folder'] == folder_name].copy()
    
    if len(folder_df) == 0:
        print(f"ERROR: No data found for folder '{folder_name}'")
        print(f"Available folders: {df['folder'].unique().tolist()}")
        return
    
    print(f"\nAnalyzing folder: {folder_name}")
    print(f"Total images: {len(folder_df)}")
    
    # Extract similarities
    similarities = np.array(folder_df['avg_similarity'])
    
    # Basic statistics
    mean = np.mean(similarities)
    std = np.std(similarities)
    median = np.median(similarities)
    
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    print(f"Mean:   {mean:.4f}")
    print(f"Std:    {std:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Min:    {np.min(similarities):.4f}")
    print(f"Max:    {np.max(similarities):.4f}")
    
    # IQR (Interquartile Range) method
    Q1 = np.percentile(similarities, 25)
    Q3 = np.percentile(similarities, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("\n" + "=" * 80)
    print("IQR ANALYSIS")
    print("=" * 80)
    print(f"Q1 (25th percentile): {Q1:.4f}")
    print(f"Q3 (75th percentile): {Q3:.4f}")
    print(f"IQR (Q3 - Q1):        {IQR:.4f}")
    print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
    print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")
    
    # Find outliers
    outliers_mask = (similarities < lower_bound) | (similarities > upper_bound)
    num_outliers = outliers_mask.sum()
    
    print(f"\nNumber of outliers: {num_outliers} ({num_outliers/len(similarities)*100:.2f}%)")
    
    if num_outliers > 0:
        # Get outlier details
        outliers_df = folder_df[outliers_mask][['reference_image', 'reference_path', 
                                                  'avg_similarity', 'min_similarity', 
                                                  'max_similarity', 'std_similarity']].copy()
        outliers_df = outliers_df.sort_values('avg_similarity')
        
        print("\n" + "=" * 80)
        print("OUTLIERS (IQR method)")
        print("=" * 80)
        print(outliers_df.to_string(index=False))
        
        # Classify outliers
        low_outliers = outliers_df[outliers_df['avg_similarity'] < lower_bound]
        high_outliers = outliers_df[outliers_df['avg_similarity'] > upper_bound]
        
        print(f"\nLow outliers (below {lower_bound:.4f}): {len(low_outliers)}")
        print(f"High outliers (above {upper_bound:.4f}): {len(high_outliers)}")
        
        # Save outliers to CSV
        output_csv = output_path / f'outliers_iqr_{folder_name}.csv'
        outliers_df.to_csv(output_csv, index=False)
        print(f"\n✓ Outliers saved to: {output_csv}")
        
        # Create visualization
        create_outlier_plot(similarities, outliers_mask, Q1, Q3, lower_bound, upper_bound,
                           folder_name, output_path)
        
    else:
        print("\n✓ No outliers detected!")
    
    return folder_df, outliers_mask if num_outliers > 0 else None


def create_outlier_plot(similarities, outliers_mask, Q1, Q3, lower_bound, upper_bound,
                       folder_name, output_path):
    """
    Create visualization of similarity distribution with outliers highlighted.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(similarities, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(Q1, color='green', linestyle='--', linewidth=2, label=f'Q1 ({Q1:.4f})')
    ax1.axvline(Q3, color='green', linestyle='--', linewidth=2, label=f'Q3 ({Q3:.4f})')
    ax1.axvline(lower_bound, color='red', linestyle='--', linewidth=2, 
                label=f'Lower bound ({lower_bound:.4f})')
    ax1.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
                label=f'Upper bound ({upper_bound:.4f})')
    ax1.axvline(np.mean(similarities), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(similarities):.4f})')
    ax1.set_xlabel('Average Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Average Similarities - {folder_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    box_data = [similarities[~outliers_mask], similarities[outliers_mask]]
    bp = ax2.boxplot(box_data, labels=['Normal', 'Outliers'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Average Similarity')
    ax2.set_title(f'Box Plot Comparison - {folder_name}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_fig = output_path / f'outliers_plot_{folder_name}.png'
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_fig}")
    plt.close()


def analyze_all_folders(csv_path: str, output_dir: str = "./analyze"):
    """
    Analyze all folders in the CSV and generate outlier reports for each.
    """
    df = pd.read_csv(csv_path)
    folders = df['folder'].unique()
    
    print("=" * 80)
    print(f"ANALYZING ALL FOLDERS ({len(folders)} total)")
    print("=" * 80)
    
    summary_results = []
    
    for folder in sorted(folders):
        print(f"\n{'='*80}")
        print(f"Processing: {folder}")
        print(f"{'='*80}")
        
        folder_df = df[df['folder'] == folder]
        similarities = np.array(folder_df['avg_similarity'])
        
        # Calculate IQR
        Q1 = np.percentile(similarities, 25)
        Q3 = np.percentile(similarities, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers_mask = (similarities < lower_bound) | (similarities > upper_bound)
        num_outliers = outliers_mask.sum()
        
        summary_results.append({
            'folder': folder,
            'total_images': len(similarities),
            'num_outliers': num_outliers,
            'outlier_percentage': num_outliers / len(similarities) * 100,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        print(f"Outliers: {num_outliers} ({num_outliers/len(similarities)*100:.2f}%)")
        
        # Save individual folder outliers if any exist
        if num_outliers > 0:
            outliers_df = folder_df[outliers_mask][['reference_image', 'reference_path', 
                                                      'avg_similarity']].copy()
            outliers_df = outliers_df.sort_values('avg_similarity')
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_csv = output_path / f'outliers_iqr_{folder}.csv'
            outliers_df.to_csv(output_csv, index=False)
    
    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values('outlier_percentage', ascending=False)
    
    output_path = Path(output_dir)
    summary_csv = output_path / 'outliers_summary_all_folders.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY - ALL FOLDERS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Summary saved to: {summary_csv}")
    
    return summary_df


def main():
    # Configuration
    csv_path = "/home/jiyoon/PAI_Bench/utils/reference_base/results/clip_reference_similarities.csv"
    folder_name = "BrunoMars"
    output_dir = "/home/jiyoon/PAI_Bench/utils/reference_base/outlier_results"
    
    # Analyze specific folder
    print("=" * 80)
    print("OPTION 1: Analyze Single Folder")
    print("=" * 80)
    folder_df, outliers_mask = find_outliers_iqr(csv_path, folder_name, output_dir)
    
    # Uncomment below to analyze all folders
    # print("\n\n" + "=" * 80)
    # print("OPTION 2: Analyze All Folders")
    # print("=" * 80)
    # summary_df = analyze_all_folders(csv_path, output_dir)


if __name__ == "__main__":
    main()