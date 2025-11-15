"""
Find outlier image pairs within folders using IQR method on pairwise DINO similarities
"""

from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from itertools import combinations
import matplotlib.pyplot as plt

class DINOPairwiseAnalyzer:
    def __init__(self):
        """Initialize analyzer without loading the model since we only use pre-computed features"""
        pass
    
    def load_feature(self, feature_path: Path) -> torch.Tensor:
        """Load a single feature tensor from disk"""
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    
    def get_all_images_from_features(self, feature_dir: str, crawled_path: str) -> Dict[str, List[tuple]]:
        """
        Get all image paths by scanning the feature directory and matching with actual image files.
        
        Args:
            feature_dir: Directory containing feature files
            crawled_path: Path to original crawled directory
        
        Returns:
            Dictionary mapping folder names to lists of (image_path, feature_path) tuples
        """
        feature_dir_path = Path(feature_dir)
        crawled_dir_path = Path(crawled_path)
        
        if not feature_dir_path.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        
        folder_images = {}
        
        # Scan feature directory for folders
        for folder_path in sorted(feature_dir_path.iterdir()):
            if not folder_path.is_dir():
                continue
            
            folder_name = folder_path.name
            image_feature_pairs = []
            
            # Get all .pkl files in this folder
            for feature_file in sorted(folder_path.glob('*.pkl')):
                # Get the base name without extension
                base_name = feature_file.stem
                
                # Find corresponding image file in the crawled directory
                crawled_folder = crawled_dir_path / folder_name
                if crawled_folder.exists():
                    # Try common image extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
                        image_file = crawled_folder / f"{base_name}{ext}"
                        if image_file.exists():
                            image_feature_pairs.append((str(image_file), str(feature_file)))
                            break
            
            if image_feature_pairs:
                folder_images[folder_name] = image_feature_pairs
        
        return folder_images
    
    def dino_score_from_features(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Calculate cosine similarity between two feature vectors"""
        score = torch.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1)
        return score.item()
    
    def calculate_all_pairwise_similarities(self, folder_name: str, image_feature_pairs: List[tuple],
                                           output_dir: str = "./pairwise_results"):
        """
        Calculate DINO similarity for ALL pairs within a folder and save to CSV.
        
        Args:
            folder_name: Name of the folder
            image_feature_pairs: List of (image_path, feature_path) tuples
            output_dir: Directory to save results
        
        Returns:
            DataFrame with all pairwise similarities
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing folder: {folder_name}")
        print(f"  Total images: {len(image_feature_pairs)}")
        
        # Load all features
        print(f"  Loading DINO features...")
        features = {}
        for img_path, feat_path in image_feature_pairs:
            try:
                features[img_path] = self.load_feature(Path(feat_path))
            except Exception as e:
                print(f"    Error loading {feat_path}: {e}")
        
        print(f"  Loaded {len(features)} features")
        
        if len(features) < 2:
            print(f"  Warning: Not enough features to compute pairs")
            return None
        
        # Calculate all pairwise similarities
        image_paths = list(features.keys())
        total_pairs = len(image_paths) * (len(image_paths) - 1) // 2
        
        print(f"  Calculating {total_pairs} pairwise similarities...")
        
        results = []
        for img1, img2 in tqdm(combinations(image_paths, 2), 
                              total=total_pairs,
                              desc=f"  Computing pairs for {folder_name}"):
            try:
                similarity = self.dino_score_from_features(features[img1], features[img2])
                results.append({
                    'folder': folder_name,
                    'image1': Path(img1).name,
                    'image2': Path(img2).name,
                    'image1_path': img1,
                    'image2_path': img2,
                    'dino_similarity': similarity
                })
            except Exception as e:
                print(f"    Error computing similarity: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('dino_similarity')
        
        # Save to CSV
        output_csv = output_path / f'pairwise_dino_similarities_{folder_name}.csv'
        df.to_csv(output_csv, index=False)
        print(f"  ✓ Saved {len(df)} pairs to: {output_csv}")
        
        return df
    
    def find_outlier_pairs(self, df: pd.DataFrame, folder_name: str, output_dir: str = "./pairwise_results"):
        """
        Find outlier pairs using IQR method.
        
        Args:
            df: DataFrame with pairwise similarities
            folder_name: Name of the folder
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        
        similarities = np.array(df['dino_similarity'])
        
        # Basic statistics
        mean = np.mean(similarities)
        std = np.std(similarities)
        median = np.median(similarities)
        
        print("\n" + "=" * 80)
        print(f"DINO STATISTICS FOR {folder_name}")
        print("=" * 80)
        print(f"Total pairs:  {len(similarities)}")
        print(f"Mean:         {mean:.4f}")
        print(f"Std:          {std:.4f}")
        print(f"Median:       {median:.4f}")
        print(f"Min:          {np.min(similarities):.4f}")
        print(f"Max:          {np.max(similarities):.4f}")
        
        # IQR (Interquartile Range) method
        Q1 = np.percentile(similarities, 25)
        Q3 = np.percentile(similarities, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"\nIQR Analysis:")
        print(f"Q1 (25th percentile):       {Q1:.4f}")
        print(f"Q3 (75th percentile):       {Q3:.4f}")
        print(f"IQR (Q3 - Q1):              {IQR:.4f}")
        print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
        print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")
        
        # Find outliers
        outliers_mask = (similarities < lower_bound) | (similarities > upper_bound)
        num_outliers = outliers_mask.sum()
        
        print(f"\nOutliers detected: {num_outliers} ({num_outliers/len(similarities)*100:.2f}%)")
        
        if num_outliers > 0:
            # Get outlier details
            outliers_df = df[outliers_mask].copy()
            outliers_df = outliers_df.sort_values('dino_similarity')
            
            # Classify outliers
            low_outliers = outliers_df[outliers_df['dino_similarity'] < lower_bound]
            high_outliers = outliers_df[outliers_df['dino_similarity'] > upper_bound]
            
            print(f"  Low outliers (below {lower_bound:.4f}):  {len(low_outliers)}")
            print(f"  High outliers (above {upper_bound:.4f}): {len(high_outliers)}")
            
            print("\n" + "=" * 80)
            print(f"OUTLIER PAIRS FOR {folder_name}")
            print("=" * 80)
            print(outliers_df[['image1', 'image2', 'dino_similarity']].to_string(index=False))
            
            # Save outliers to CSV
            outlier_csv = output_path / f'outlier_pairs_dino_iqr_{folder_name}.csv'
            outliers_df.to_csv(outlier_csv, index=False)
            print(f"\n✓ Outlier pairs saved to: {outlier_csv}")
            
            # Create visualization
            self.create_pair_outlier_plot(similarities, outliers_mask, Q1, Q3, 
                                         lower_bound, upper_bound, folder_name, output_path)
        else:
            print("\n✓ No outlier pairs detected!")
        
        return outliers_df if num_outliers > 0 else None
    
    def create_pair_outlier_plot(self, similarities, outliers_mask, Q1, Q3, 
                                lower_bound, upper_bound, folder_name, output_path):
        """
        Create visualization of pairwise similarity distribution with outliers highlighted.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(Q1, color='green', linestyle='--', linewidth=2, label=f'Q1 ({Q1:.4f})')
        ax1.axvline(Q3, color='green', linestyle='--', linewidth=2, label=f'Q3 ({Q3:.4f})')
        ax1.axvline(lower_bound, color='red', linestyle='--', linewidth=2, 
                    label=f'Lower bound ({lower_bound:.4f})')
        ax1.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
                    label=f'Upper bound ({upper_bound:.4f})')
        ax1.axvline(np.mean(similarities), color='orange', linestyle='-', linewidth=2, 
                    label=f'Mean ({np.mean(similarities):.4f})')
        ax1.set_xlabel('Pairwise DINO Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Pairwise DINO Similarities - {folder_name}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        box_data = [similarities[~outliers_mask], similarities[outliers_mask]]
        bp = ax2.boxplot(box_data, labels=['Normal Pairs', 'Outlier Pairs'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        if len(box_data[1]) > 0:
            bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Pairwise DINO Similarity')
        ax2.set_title(f'Box Plot - {folder_name}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_fig = output_path / f'outlier_pairs_dino_plot_{folder_name}.png'
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_fig}")
        plt.close()
    
    def analyze_all_folders(self, feature_dir: str, crawled_path: str, 
                           output_dir: str = "./pairwise_results"):
        """
        Analyze all folders: calculate pairwise DINO similarities and find outlier pairs.
        """
        # Get all folders
        folder_images = self.get_all_images_from_features(feature_dir, crawled_path)
        folder_names = sorted(folder_images.keys())
        
        print("=" * 80)
        print(f"ANALYZING PAIRWISE DINO SIMILARITIES FOR ALL FOLDERS")
        print("=" * 80)
        print(f"Total folders: {len(folder_names)}")
        print(f"Folders: {folder_names}")
        
        summary_results = []
        
        for folder_name in folder_names:
            print("\n" + "=" * 80)
            
            # Calculate all pairwise similarities
            df = self.calculate_all_pairwise_similarities(
                folder_name, folder_images[folder_name], output_dir
            )
            
            if df is None or len(df) == 0:
                continue
            
            # Find outlier pairs
            outliers_df = self.find_outlier_pairs(df, folder_name, output_dir)
            
            # Store summary
            similarities = np.array(df['dino_similarity'])
            Q1 = np.percentile(similarities, 25)
            Q3 = np.percentile(similarities, 75)
            IQR = Q3 - Q1
            outliers_mask = (similarities < Q1 - 1.5 * IQR) | (similarities > Q3 + 1.5 * IQR)
            
            summary_results.append({
                'folder': folder_name,
                'total_pairs': len(similarities),
                'num_outlier_pairs': outliers_mask.sum(),
                'outlier_percentage': outliers_mask.sum() / len(similarities) * 100,
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'median_similarity': np.median(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.sort_values('outlier_percentage', ascending=False)
        
        output_path = Path(output_dir)
        summary_csv = output_path / 'pairwise_dino_outliers_summary_all_folders.csv'
        summary_df.to_csv(summary_csv, index=False)
        
        print("\n" + "=" * 80)
        print("SUMMARY - ALL FOLDERS (DINO)")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\n✓ Summary saved to: {summary_csv}")
        
        return summary_df


def main():
    # Configuration
    feature_dir = "/data2/jiyoon/PAI-Bench/data/crawled/dino_features"
    crawled_path = "/data2/jiyoon/PAI-Bench/data/crawled/imgs"
    output_dir = "/home/jiyoon/PAI_Bench/utils/diagonal/results"
    # Analyze specific folder
    folder_name = "BrunoMars"
    
    print("=" * 80)
    print("OPTION 1: Analyze Single Folder (DINO)")
    print("=" * 80)
    
    analyzer = DINOPairwiseAnalyzer()
    folder_images = analyzer.get_all_images_from_features(feature_dir, crawled_path)
    
    if folder_name in folder_images:
        # Calculate pairwise similarities
        df = analyzer.calculate_all_pairwise_similarities(
            folder_name, folder_images[folder_name], output_dir
        )
        
        # Find outlier pairs
        if df is not None:
            outliers_df = analyzer.find_outlier_pairs(df, folder_name, output_dir)
    else:
        print(f"ERROR: Folder '{folder_name}' not found")
        print(f"Available folders: {list(folder_images.keys())}")
    
    # Uncomment below to analyze all folders
    # print("\n\n" + "=" * 80)
    # print("OPTION 2: Analyze All Folders (DINO)")
    # print("=" * 80)
    # summary_df = analyzer.analyze_all_folders(feature_dir, crawled_path, output_dir)


if __name__ == "__main__":
    main()