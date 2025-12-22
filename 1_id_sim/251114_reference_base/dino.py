"""
DINO scorer for calculating reference-based similarities using pre-computed features
"""

from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

class DINOScorer:
    def __init__(self):
        """Initialize scorer without loading the model since we only use pre-computed features"""
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
                print(f"  Found {len(image_feature_pairs)} image-feature pairs in {folder_name}")
        
        return folder_images
    
    def dino_score_from_features(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Calculate cosine similarity between two feature vectors"""
        score = torch.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1)
        return score.item()
    
    def calculate_reference_based_similarities(self, crawled_path: str, output_csv: str,
                                              feature_dir: str):
        """
        Calculate DINO similarity using each image in a folder as reference.
        For each reference image, calculate average similarity to all other images in the same folder.
        
        Args:
            crawled_path: Path to directory containing celebrity folders
            output_csv: Path to save CSV results
            feature_dir: Directory containing pre-computed feature files
        """
        # Get all images from feature directory
        print("=" * 80)
        print("STEP 1: Scanning Feature Directory")
        print("=" * 80)
        folder_images = self.get_all_images_from_features(feature_dir, crawled_path)
        
        folder_names = sorted(folder_images.keys())
        
        print(f"\nFound {len(folder_names)} folders with features")
        
        if not folder_names:
            print("ERROR: No folders with features found!")
            return None, None
        
        # Collect results for all reference images
        all_results = []
        
        print("\n" + "=" * 80)
        print("STEP 2: Calculating Reference-Based Similarities")
        print("=" * 80)
        
        for folder_name in folder_names:
            print(f"\nProcessing folder: {folder_name}")
            image_feature_pairs = folder_images[folder_name]
            
            if len(image_feature_pairs) < 2:
                print(f"Warning: Folder {folder_name} has less than 2 images, skipping...")
                continue
            
            # Load all features for this folder
            print(f"  Loading {len(image_feature_pairs)} features...")
            features = {}
            for img_path, feat_path in image_feature_pairs:
                try:
                    features[img_path] = self.load_feature(Path(feat_path))
                except Exception as e:
                    print(f"  Error loading feature {feat_path}: {e}")
                    continue
            
            if len(features) < 2:
                print(f"Warning: Could not load enough features for folder {folder_name}, skipping...")
                continue
            
            print(f"  Successfully loaded {len(features)} features")
            
            image_paths = list(features.keys())
            
            # Use each image as reference
            for ref_path in tqdm(image_paths, desc=f"  Processing references in {folder_name}"):
                ref_feature = features[ref_path]
                similarities = []
                
                # Calculate similarity to all other images in the folder
                for other_path in image_paths:
                    if other_path == ref_path:  # Skip self-comparison
                        continue
                    
                    other_feature = features[other_path]
                    similarity = self.dino_score_from_features(ref_feature, other_feature)
                    similarities.append(similarity)
                
                # Calculate average similarity for this reference
                if similarities:
                    avg_similarity = np.mean(similarities)
                    min_similarity = np.min(similarities)
                    max_similarity = np.max(similarities)
                    std_similarity = np.std(similarities)
                    
                    all_results.append({
                        'folder': folder_name,
                        'reference_image': Path(ref_path).name,
                        'reference_path': ref_path,
                        'num_comparisons': len(similarities),
                        'avg_similarity': avg_similarity,
                        'min_similarity': min_similarity,
                        'max_similarity': max_similarity,
                        'std_similarity': std_similarity
                    })
        
        # Save results to CSV
        print("\n" + "=" * 80)
        print("STEP 3: Saving Results")
        print("=" * 80)
        
        if not all_results:
            print("ERROR: No results to save! All folders were skipped.")
            return None, None
        
        df = pd.DataFrame(all_results)
        
        # Sort by folder and avg_similarity
        df = df.sort_values(['folder', 'avg_similarity'], ascending=[True, False])
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Total reference images processed: {len(all_results)}")
        
        # Print summary statistics per folder
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS PER FOLDER")
        print("=" * 80)
        
        summary = df.groupby('folder')['avg_similarity'].agg(['count', 'mean', 'std', 'min', 'max'])
        summary.columns = ['num_images', 'mean_avg_sim', 'std_avg_sim', 'min_avg_sim', 'max_avg_sim']
        print(summary)
        
        # Save summary
        summary_csv = str(output_csv).replace('.csv', '_summary.csv')
        summary.to_csv(summary_csv)
        print(f"\nSummary saved to {summary_csv}")
        
        return df, summary


def main():
    # Configuration
    crawled_path = "/data2/jiyoon/PAI-Bench/data/crawled/imgs"
    output_csv = "/home/jiyoon/PAI_Bench/utils/features/dino_reference_similarities.csv"
    feature_dir = "/data2/jiyoon/PAI-Bench/data/crawled/dino_features"  # Directory with pre-computed features
    
    # Create scorer and calculate reference-based similarities
    scorer = DINOScorer()
    df, summary = scorer.calculate_reference_based_similarities(
        crawled_path, output_csv, feature_dir=feature_dir
    )
    
    if df is not None:
        print("\n" + "=" * 80)
        print("DONE!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED - No results generated")
        print("=" * 80)


if __name__ == "__main__":
    main()