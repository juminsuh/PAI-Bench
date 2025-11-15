"""
DINO scorer for celeb face angle dataset with individual feature file caching
"""

from pathlib import Path
import pickle
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

class DINOScorer:
    def __init__(self):
        pass
    
    def get_feature_path(self, image_path: str, feature_dir: str) -> Path:
        """
        Generate feature file path for an image.
        Maintains folder structure within feature directory.
        """
        img_path = Path(image_path)
        feature_dir_path = Path(feature_dir)
        
        # Create subfolder structure matching the image folder
        relative_parts = img_path.parts
        # Find the index where 'crawled' appears to maintain structure after it
        try:
            crawled_idx = relative_parts.index('crawled')
            relative_path = Path(*relative_parts[crawled_idx+1:])
        except ValueError:
            # If 'crawled' not in path, just use the last two parts (folder/filename)
            relative_path = Path(img_path.parent.name) / img_path.name
        
        # Replace image extension with .pkl
        feature_file = relative_path.with_suffix('.pkl')
        return feature_dir_path / feature_file
    
    def load_feature(self, feature_path: Path) -> torch.Tensor:
        """Load a single feature tensor from disk"""
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    
    
    def load_features_for_images(self, image_paths: List[str], feature_dir: str) -> Dict[str, torch.Tensor]:
        """
        Load features for a list of images from disk.
        
        Args:
            image_paths: List of image file paths
            feature_dir: Directory containing feature files
        
        Returns:
            Dictionary mapping image paths to feature tensors
        """
        features = {}
        
        for img_path in image_paths:
            feature_path = self.get_feature_path(img_path, feature_dir)
            if feature_path.exists():
                features[img_path] = self.load_feature(feature_path)
            else:
                print(f"Warning: Feature not found for {img_path}")
        
        return features
    
    def get_existing_features_mapping(self, feature_dir: str) -> Dict[str, Dict[str, Path]]:
        """
        Get mapping of existing feature files without trying to match to actual images.
        
        Args:
            feature_dir: Directory containing pre-extracted feature files
        
        Returns:
            Dictionary mapping folder names to dict of {image_identifier: feature_path}
        """
        feature_dir_path = Path(feature_dir)
        
        if not feature_dir_path.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        
        # Find all .pkl files in feature directory
        feature_files = list(feature_dir_path.rglob("*.pkl"))
        
        # Group by folder name
        folder_features = {}
        
        for feature_file in feature_files:
            # Extract folder name
            relative_path = feature_file.relative_to(feature_dir_path)
            folder_name = relative_path.parts[0]  # First part is the folder name
            
            # Use the base filename as identifier
            image_identifier = relative_path.name.replace('.pkl', '')
            
            if folder_name not in folder_features:
                folder_features[folder_name] = {}
            folder_features[folder_name][image_identifier] = feature_file
        
        print(f"Found pre-extracted features for {len(folder_features)} folders")
        for folder_name, features in folder_features.items():
            print(f"  {folder_name}: {len(features)} feature files")
        
        return folder_features
    
    def detect_inter_folder_outliers(self, feature_dir: str, output_csv: str = "dino_outliers.csv") -> pd.DataFrame:
        """
        Detect outliers in inter-folder similarity scores using IQR method.
        
        Args:
            feature_dir: Directory containing pre-extracted feature files
            output_csv: Path to save outlier results CSV
        
        Returns:
            DataFrame with outlier information
        """
        print("=" * 80)
        print("OUTLIER DETECTION: Loading features and calculating individual pair scores")
        print("=" * 80)
        
        # Get existing features mapping
        folder_features = self.get_existing_features_mapping(feature_dir)
        folder_names = sorted(folder_features.keys())
        n = len(folder_names)
        
        # Collect all inter-folder pair scores
        all_pair_scores = []
        pair_details = []
        
        for i in range(n):
            folder_i = folder_names[i]
            features_i = {}
            # Load all features for folder i
            for img_id, feature_path in folder_features[folder_i].items():
                features_i[img_id] = self.load_feature(feature_path)
            
            for j in range(n):
                if i == j:  # Skip intra-folder (diagonal) comparisons
                    continue
                    
                folder_j = folder_names[j]
                features_j = {}
                # Load all features for folder j
                for img_id, feature_path in folder_features[folder_j].items():
                    features_j[img_id] = self.load_feature(feature_path)
                
                # Calculate all pairwise scores between folders i and j
                for img_id1, feat1 in features_i.items():
                    for img_id2, feat2 in features_j.items():
                        try:
                            score = self.dino_score_from_features(feat1, feat2)
                            all_pair_scores.append(score)
                            pair_details.append({
                                'image1_id': img_id1,
                                'image2_id': img_id2,
                                'folder1': folder_i,
                                'folder2': folder_j,
                                'similarity_score': score
                            })
                        except Exception as e:
                            print(f"Error processing pair {img_id1} vs {img_id2}: {e}")
                            continue
        
        # Calculate IQR-based outliers
        scores_array = np.array(all_pair_scores)
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f"\nOutlier Detection Statistics:")
        print(f"Total inter-folder pairs: {len(all_pair_scores)}")
        print(f"Q1 (25th percentile): {q1:.4f}")
        print(f"Q3 (75th percentile): {q3:.4f}")
        print(f"IQR: {iqr:.4f}")
        print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
        print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")
        
        # Mark outliers
        outlier_data = []
        for detail in pair_details:
            score = detail['similarity_score']
            is_outlier = score < lower_bound or score > upper_bound
            outlier_type = 'none'
            if score < lower_bound:
                outlier_type = 'low'
            elif score > upper_bound:
                outlier_type = 'high'
                
            outlier_data.append({
                'image1_full': f"{detail['folder1']}/{detail['image1_id']}",
                'image2_full': f"{detail['folder2']}/{detail['image2_id']}",
                'similarity_score': score,
                'is_outlier': is_outlier,
                'outlier_type': outlier_type,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(outlier_data)
        outlier_count = df['is_outlier'].sum()
        high_outliers = (df['outlier_type'] == 'high').sum()
        low_outliers = (df['outlier_type'] == 'low').sum()
        
        print(f"\nOutlier Results:")
        print(f"Total outliers found: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
        print(f"High outliers (> {upper_bound:.4f}): {high_outliers}")
        print(f"Low outliers (< {lower_bound:.4f}): {low_outliers}")
        
        # Save to CSV
        try:
            df.to_csv(output_csv, index=False)
            print(f"Outlier results saved to: {output_csv}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            print(f"Output path: {output_csv}")
            print(f"DataFrame shape: {df.shape}")
            raise
        
        # Save only outliers to separate CSV
        outliers_only_df = df[df['is_outlier'] == True]
        outliers_only_csv = output_csv.replace('.csv', '_outliers_only.csv')
        try:
            outliers_only_df.to_csv(outliers_only_csv, index=False)
            print(f"Outliers-only results saved to: {outliers_only_csv}")
            print(f"Total outliers saved: {len(outliers_only_df)}")
        except Exception as e:
            print(f"Error saving outliers-only CSV: {e}")
            raise
        
        # Save top 20 highest similarity score outliers (remove duplicates)
        # Create a column for sorting pairs to remove duplicates (A,B) and (B,A)
        outliers_only_df['pair_key'] = outliers_only_df.apply(
            lambda row: tuple(sorted([row['image1_full'], row['image2_full']])), axis=1
        )
        
        # Remove duplicate pairs and get top 20
        unique_outliers = outliers_only_df.drop_duplicates(subset=['pair_key', 'similarity_score'])
        top_20_outliers = unique_outliers.nlargest(20, 'similarity_score')
        
        # Drop the helper column before saving
        top_20_outliers = top_20_outliers.drop('pair_key', axis=1)
        
        top_20_csv = output_csv.replace('.csv', '_top20_high_outliers_unique.csv')
        try:
            top_20_outliers.to_csv(top_20_csv, index=False)
            print(f"Top 20 unique highest similarity outliers saved to: {top_20_csv}")
            if len(top_20_outliers) > 0:
                print(f"Highest score: {top_20_outliers['similarity_score'].max():.4f}")
                print(f"Lowest of top 20: {top_20_outliers['similarity_score'].min():.4f}")
        except Exception as e:
            print(f"Error saving top 20 unique outliers CSV: {e}")
            raise
        
        return df
    
    def dino_score_from_features(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Calculate cosine similarity between two feature vectors"""
        score = torch.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1)
        return score.item()
    
    def calculate_intra_folder_similarity_cached(self, features: Dict[str, torch.Tensor]) -> float:
        """Calculate average DINO similarity between all unique pairs within a folder using cached features"""
        image_ids = list(features.keys())
        
        if len(image_ids) < 2:
            print(f"Warning: Less than 2 images in folder")
            return 0.0
        
        total_score = 0.0
        num_pairs = 0
        
        # Calculate similarity for all unique pairs
        for id1, id2 in combinations(image_ids, 2):
            try:
                score = self.dino_score_from_features(features[id1], features[id2])
                total_score += score
                num_pairs += 1
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        if num_pairs == 0:
            return 0.0
        
        avg_score = total_score / num_pairs
        return avg_score
    
    def calculate_inter_folder_similarity_cached(self, features1: Dict[str, torch.Tensor], 
                                                 features2: Dict[str, torch.Tensor]) -> float:
        """Calculate average DINO similarity between all pairs across two folders using cached features"""
        ids1 = list(features1.keys())
        ids2 = list(features2.keys())
        
        if not ids1 or not ids2:
            print(f"Warning: Empty folder(s)")
            return 0.0
        
        total_score = 0.0
        num_pairs = 0
        
        # Calculate similarity for all pairs between folders
        for id1 in ids1:
            for id2 in ids2:
                try:
                    score = self.dino_score_from_features(features1[id1], features2[id2])
                    total_score += score
                    num_pairs += 1
                except Exception as e:
                    print(f"Error processing pair: {e}")
                    continue
        
        if num_pairs == 0:
            return 0.0
        
        avg_score = total_score / num_pairs
        return avg_score
    
    def create_similarity_heatmap(self, crawled_path: str, output_path: str = "dino_similarity_heatmap.png",
                                 feature_dir: Optional[str] = None):
        """
        Create 10x10 heatmap of DINO similarities between all folder pairs using pre-extracted features.
        
        Args:
            crawled_path: Path to directory containing celebrity folders
            output_path: Path to save heatmap image
            feature_dir: Directory containing pre-extracted feature files
        """
        if feature_dir is None:
            feature_dir = str(Path(output_path).parent / "features")
        
        # Load existing features
        print("=" * 80)
        print("STEP 1: Loading Pre-extracted Features")
        print("=" * 80)
        folder_features = self.get_existing_features_mapping(feature_dir)
        
        folder_names = sorted(folder_features.keys())
        n = len(folder_names)
        
        print(f"\nFound {n} folders: {folder_names}")
        
        if n != 10:
            print(f"Warning: Expected 10 folders, found {n}")
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        print("\n" + "=" * 80)
        print("STEP 2: Loading Features and Calculating Similarities")
        print("=" * 80)
        
        # Load features for each folder and calculate similarities
        for i in tqdm(range(n), desc="Processing folders"):
            folder_i = folder_names[i]
            features_i = {}
            # Load all features for folder i
            for img_id, feature_path in folder_features[folder_i].items():
                features_i[img_id] = self.load_feature(feature_path)
            
            for j in range(n):
                folder_j = folder_names[j]
                features_j = {}
                # Load all features for folder j
                for img_id, feature_path in folder_features[folder_j].items():
                    features_j[img_id] = self.load_feature(feature_path)
                
                if i == j:
                    # Diagonal: intra-folder similarity
                    similarity_matrix[i, j] = self.calculate_intra_folder_similarity_cached(features_i)
                    print(f"{folder_i} (intra): {similarity_matrix[i, j]:.4f}")
                else:
                    # Off-diagonal: inter-folder similarity
                    similarity_matrix[i, j] = self.calculate_inter_folder_similarity_cached(
                        features_i, features_j)
        
        # Create heatmap
        print("\n" + "=" * 80)
        print("STEP 3: Creating Heatmap")
        print("=" * 80)
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   xticklabels=folder_names,
                   yticklabels=folder_names,
                   square=True,
                   cbar_kws={'label': 'DINO Similarity Score'})
        
        plt.title('DINO Similarity Matrix Between Celebrity Folders')
        plt.xlabel('Celebrity Folders')
        plt.ylabel('Celebrity Folders')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save heatmap
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
        
        # Also save the similarity matrix as CSV
        csv_path = str(output_path_obj).replace('.png', '_matrix.csv')
        df = pd.DataFrame(similarity_matrix, 
                         index=folder_names, 
                         columns=folder_names)
        df.to_csv(csv_path)
        print(f"Similarity matrix saved to {csv_path}")
        
        plt.close()
        
        return similarity_matrix, folder_names


def main():
    # Configuration
    crawled_path = "/data2/jiyoon/PAI-Bench/data/crawled/imgs"
    output_heatmap = "/home/jiyoon/PAI_Bench/utils/features/dino_similarity_heatmap.png"
    feature_dir = "/data2/jiyoon/PAI-Bench/data/crawled/dino_features"
    
    # Create scorer and generate heatmap
    scorer = DINOScorer()
    similarity_matrix, folder_names = scorer.create_similarity_heatmap(
        crawled_path, output_heatmap, feature_dir=feature_dir
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("\nSimilarity Matrix:")
    for i, name1 in enumerate(folder_names):
        for j, name2 in enumerate(folder_names):
            print(f"{name1} vs {name2}: {similarity_matrix[i, j]:.4f}")


def outlier_detection_main():
    """
    Main function specifically for outlier detection in inter-folder similarity scores.
    """
    # Configuration
    crawled_path = "/data2/jiyoon/PAI-Bench/data/crawled/imgs"
    feature_dir = "/data2/jiyoon/PAI-Bench/data/crawled/dino_features"
    output_csv = "/home/jiyoon/PAI-Bench/sim_experiments/251117_negative_outliers/outlier_results/dino_outliers.csv"
    # Create output directory
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Create scorer
    scorer = DINOScorer()
    
    # Run outlier detection
    outlier_df = scorer.detect_inter_folder_outliers(
        feature_dir, output_csv
    )
    
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION COMPLETED")
    print("=" * 80)
    
    return outlier_df


if __name__ == "__main__":
    # Run outlier detection instead of heatmap generation
    outlier_detection_main()