"""
CLIP-I scorer for calculating reference-based similarities using pre-computed features
Fixed to properly detect actual image filenames from feature files
"""

from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns

class CLIPScorer:
    # --- Utils ---
    # initialize scorer without loading the model since we only use pre-computed features
    def __init__(self):
        pass
    
    # load a single feature tensor from disk
    def load_feature(self, feature_path: Path) -> torch.Tensor:
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    
    # calculate cosine sim between two features
    def clip_score_from_features(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        score = torch.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1)
        return score.item()
    

    def load_folder_features(self, folder_path: str, feature_base_dir: str) -> Dict[str, torch.Tensor]:
        """
        Load features for images 001.jpg - 025.jpg from a specific folder.
        
        Args:
            folder_path: Path to image folder
            feature_base_dir: Base directory containing pre-computed features
            
        Returns:
            Dictionary mapping image numbers to feature tensors
        """
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        feature_folder = Path(feature_base_dir) / folder_name
        
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

    # --- Calculate CLIP score and save results ----
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

        print("=" * 80)
        print("CROSS-FOLDER SIMILARITY ANALYSIS")
        print("=" * 80)
        
        # load features
        features1 = self.load_folder_features(folder1_path, feature_dir1)
        features2 = self.load_folder_features(folder2_path, feature_dir2)
        
        print(f"Loaded {len(features1)} features from folder 1")
        print(f"Loaded {len(features2)} features from folder 2")
        
        # calculate sim score for each image pair
        all_results = []
        similarities_for_heatmap = []
        image_numbers = []
        
        folder1_name = Path(folder1_path).name
        folder2_name = Path(folder2_path).name
        
        for i in range(1, 26):
            img_num = f"{i:03d}"
            
            if img_num in features1 and img_num in features2:
                # Calculate similarity between corresponding images
                similarity = self.clip_score_from_features(features1[img_num], features2[img_num])
                similarities_for_heatmap.append(similarity)
                image_numbers.append(img_num)
                
                # Add result for each folder
                all_results.append({
                    'folder_pair': f"{folder1_name}_vs_{folder2_name}",
                    'image_number': img_num,
                    'folder1': folder1_name,
                    'folder2': folder2_name,
                    'similarity': similarity
                })
            else:
                print(f"Warning: Missing features for image {img_num}")
        
        if not similarities_for_heatmap:
            print("ERROR: No valid image pairs found!")
            return None, None
        
        # Create similarity matrix (2 rows: one for each folder)
        similarity_matrix = np.array([similarities_for_heatmap, similarities_for_heatmap])
        
        # Create heatmap
        plt.figure(figsize=(15, 4))
        
        # Create labels
        folder_labels = [Path(folder1_path).name, Path(folder2_path).name]
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   xticklabels=image_numbers,
                   yticklabels=folder_labels,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'CLIP Similarity'})
        
        plt.title(f'CLIP Similarities between Corresponding Images\n{folder_labels[0]} vs {folder_labels[1]}')
        plt.xlabel('Image Number')
        plt.ylabel('Folder')
        plt.tight_layout()
        
        # Save heatmap if path provided
        if output_heatmap:
            output_path = Path(output_heatmap)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {output_heatmap}")
        
        plt.show()
        
        # Save results to CSV
        print("\n" + "=" * 80)
        print("STEP 3: Saving Results")
        print("=" * 80)
        
        if not all_results:
            print("ERROR: No results to save!")
            return None, None
        
        df = pd.DataFrame(all_results)
        
        # Sort by image number
        df = df.sort_values(['image_number'], ascending=True)
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Total image pairs processed: {len(all_results)}")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        summary = df.groupby('folder_pair')['similarity'].agg(['count', 'mean', 'std', 'min', 'max'])
        summary.columns = ['num_pairs', 'mean_similarity', 'std_similarity', 'min_similarity', 'max_similarity']
        print(summary)
        
        # Save summary
        summary_csv = str(output_csv).replace('.csv', '_summary.csv')
        summary.to_csv(summary_csv)
        print(f"\nSummary saved to {summary_csv}")
        
        return df, summary


def main():
    # config 
    folder1_path = "path/to/folder1"  
    folder2_path = "path/to/folder2"  
    feature_dir1 = "path/to/feature_dir1"  # path to pre-computed features for folder1
    feature_dir2 = "path/to/feature_dir2"  # path to pre-computed features for folder2
    output_csv = "/home/jiyoon/PAI-Bench/sim_experiments/251202_positive_pairs/cross_folder_similarities.csv"
    output_heatmap = "/home/jiyoon/PAI-Bench/sim_experiments/251202_positive_pairs/similarity_heatmap.png"
    
    
    # calculate CLIP similarity
    scorer = CLIPScorer()
    df, summary = scorer.calculate_cross_folder_similarities(
        folder1_path, folder2_path, feature_dir1, feature_dir2, output_csv, output_heatmap
    )
    

    # print statistics
    if df is not None:
        print("\n" + "=" * 80)
        print("CROSS-FOLDER ANALYSIS COMPLETED!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED - No results generated")
        print("=" * 80)

if __name__ == "__main__":
    main()