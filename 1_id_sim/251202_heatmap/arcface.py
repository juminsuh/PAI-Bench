"""
ArcFace scorer for calculating reference-based similarities using pre-computed features
Fixed to properly detect actual image filenames from feature files
"""

from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

class ArcFaceScorer:
    # --- Utils ---
    # initialize scorer without loading the model since we only use pre-computed features
    def __init__(self):
        pass
    
    # load a single feature array from disk
    def load_feature(self, feature_path: Path) -> np.ndarray:
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    
    # calculate cosine sim between two features
    def arcface_score_from_features(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if abs(norm1 - 1.0) < 1e-6 and abs(norm2 - 1.0) < 1e-6:
            # if vectors are alr normalized, use dot product directly
            similarity = dot_product
        else:
            similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    
    def load_folder_features(self, folder_path: str, feature_base_dir: str) -> Dict[str, np.ndarray]:
        """
        Load features for images 001.jpg - 025.jpg from a specific folder and make a lookup table for easy access.
        
        Args:
            folder_path: Path to image folder
            feature_base_dir: Directory containing pre-computed features (full path to the specific folder's features)
            
        Returns:
            Dictionary mapping image numbers to feature arrays
        """
        feature_folder = Path(feature_base_dir)
        
        features = {}
        
        for i in range(1, 26):  # 001.jpg to 025.jpg
            img_num = f"{i:03d}"
            feature_file = feature_folder / f"{img_num}.pkl"
            
            if feature_file.exists():
                try:
                    features[img_num] = self.load_feature(feature_file)
                except Exception as e:
                    print(f"Error loading feature {feature_file}: {e}")
            else:
                print(f"Warning: Feature file not found: {feature_file}")
        
        return features


    def calculate_cross_folder_similarities(self, folder1_path: str, folder2_path: str, 
                                          feature_dir1: str, feature_dir2: str, output_csv: str,
                                          output_heatmap: str = None):
        """
        Calculate similarities between corresponding images from two folders and create heatmap.
        
        Args:
            folder1_path: Path to first image folder
            folder2_path: Path to second image folder
            feature_dir1: Directory containing pre-computed features for folder1
            feature_dir2: Directory containing pre-computed features for folder2
            output_csv: Path to save CSV results
            output_heatmap: Path to save heatmap image (optional)
            
        Returns:
            tuple: (df, summary) - DataFrame with results and summary statistics
        """
        
        # --- Load features ---
        features1 = self.load_folder_features(folder1_path, feature_dir1)
        features2 = self.load_folder_features(folder2_path, feature_dir2)
        
        print(f"Loaded {len(features1)} features from folder 1")
        print(f"Loaded {len(features2)} features from folder 2")
        
        # --- Calculate sim score for each image pair ---
        all_results = []
        folder1_name = Path(folder1_path).name
        folder2_name = Path(folder2_path).name
        
        # create 25x25 similarity matrix
        similarity_matrix = np.zeros((25, 25))
        valid_pairs = []
        
        for i in range(1, 26):  # folder2 images (y-axis)
            for j in range(1, 26):  # folder1 images (x-axis)
                img_num_folder1 = f"{j:03d}"
                img_num_folder2 = f"{i:03d}"
                
                if img_num_folder1 in features1 and img_num_folder2 in features2:
                    similarity = self.arcface_score_from_features(features1[img_num_folder1], features2[img_num_folder2])
                    similarity_matrix[i-1, j-1] = similarity
                    valid_pairs.append((img_num_folder1, img_num_folder2, similarity))
                    
                    all_results.append({
                        'folder1_image': f"{folder1_name}_{img_num_folder1}",
                        'folder2_image': f"{folder2_name}_{img_num_folder2}",
                        'similarity': similarity
                    })
                else:
                    similarity_matrix[i-1, j-1] = np.nan
        
        if not valid_pairs:
            print("ERROR: No valid image pairs found!")
            return None, None
        
        
        # --- Save heatmap ---
        plt.figure(figsize=(20, 16))
        
        # Create labels
        x_labels = [f"{i:03d}" for i in range(1, 26)]  # folder1 images
        y_labels = [f"{i:03d}" for i in range(1, 26)]  # folder2 images
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=x_labels,
                   yticklabels=y_labels,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'ArcFace Similarity'},
                   mask=np.isnan(similarity_matrix))
        
        plt.title(f'ArcFace Similarities: {folder1_name} (x-axis) vs {folder2_name} (y-axis)')
        plt.xlabel(f'{folder1_name} Image Numbers')
        plt.ylabel(f'{folder2_name} Image Numbers')
        plt.tight_layout()
        
        if output_heatmap:
            output_path = Path(output_heatmap)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {output_heatmap} ")
        plt.show()
        

        # --- Save sim scores in csv ---
        if not all_results:
            print("ERROR: No results to save!")
            return None, None
        
        df = pd.DataFrame(all_results)
        
        # sort by folder1_image and folder2_image
        df = df.sort_values(['folder1_image', 'folder2_image'], ascending=True)
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Total image pairs processed: {len(all_results)}")
        

        # --- Save statistics in csv ---
        # print summary statistics
        print("\nSimilarity Statistics:")
        print(f"Total pairs: {len(df)}")
        print(f"Mean similarity: {df['similarity'].mean():.4f}")
        print(f"Std similarity: {df['similarity'].std():.4f}")
        print(f"Min similarity: {df['similarity'].min():.4f}")
        print(f"Max similarity: {df['similarity'].max():.4f}")
        
        summary_data = {
            'total_pairs': [len(df)],
            'mean_similarity': [df['similarity'].mean()],
            'std_similarity': [df['similarity'].std()],
            'min_similarity': [df['similarity'].min()],
            'max_similarity': [df['similarity'].max()]
        }
        summary = pd.DataFrame(summary_data)
        
        # save summary
        summary_csv = str(output_csv).replace('.csv', '_summary.csv')
        summary.to_csv(summary_csv)
        print(f"\nSummary saved to {summary_csv}")
        
        return df, summary


def main():
    # config 
    folder1_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/1"  
    folder2_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/2"  
    feature_dir1 = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/arcface/1"  # path to pre-computed features for folder1
    feature_dir2 = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/arcface/2"  # path to pre-computed features for folder2
    output_csv = "/home/jiyoon/PAI-Bench/sim_experiments/251202_heatmap/results/arcface_similarity.csv"
    output_heatmap = "/home/jiyoon/PAI-Bench/sim_experiments/251202_heatmap/results/arcface_heatmap.png"
    
    
    scorer = ArcFaceScorer()
    df, summary = scorer.calculate_cross_folder_similarities(
        folder1_path, folder2_path, feature_dir1, feature_dir2, output_csv, output_heatmap
    )
    
    if df is not None:
        print("CROSS-FOLDER ANALYSIS COMPLETED!")
    else:
        print("FAILED - No results generated")

if __name__ == "__main__":
    main()