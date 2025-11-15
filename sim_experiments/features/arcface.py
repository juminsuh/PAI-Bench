"""
ArcFace scorer for celeb face angle dataset with individual feature file caching
"""

from pathlib import Path
import pickle
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

class ArcFaceScorer:
    def __init__(self, model_name: str = 'antelopev2', device: str = 'cuda'):
        self.device = device
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320), det_thresh=0.3)

    def img_feat(self, img: Image.Image) -> np.ndarray:
        img_array = np.array(img)
        if img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        faces = self.app.get(img_array)
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Use the first (largest) detected face
        face = faces[0]
        embedding = face.normed_embedding
        return embedding
    
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
    
    def save_feature(self, feature: np.ndarray, feature_path: Path):
        """Save a single feature array to disk"""
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_path, 'wb') as f:
            pickle.dump(feature, f)
    
    def load_feature(self, feature_path: Path) -> np.ndarray:
        """Load a single feature array from disk"""
        with open(feature_path, 'rb') as f:
            return pickle.load(f)
    
    def extract_and_save_features(self, folder_path: str, feature_dir: str) -> List[str]:
        """
        Extract features for all images in a folder and save each as individual file.
        
        Args:
            folder_path: Path to folder containing images
            feature_dir: Directory to save individual feature files
        
        Returns:
            List of image paths that were processed
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        processed_images = []
        
        print(f"Processing {len(image_files)} images in {folder.name}")
        for img_file in tqdm(image_files, desc=f"Extracting ArcFace features from {folder.name}"):
            try:
                feature_path = self.get_feature_path(str(img_file), feature_dir)
                
                # Skip if feature already exists
                if feature_path.exists():
                    processed_images.append(str(img_file))
                    continue
                
                # Extract and save feature
                img = Image.open(img_file).convert('RGB')
                feat = self.img_feat(img)
                self.save_feature(feat, feature_path)
                processed_images.append(str(img_file))
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Processed {len(processed_images)} images from {folder.name}")
        return processed_images
    
    def extract_all_features(self, crawled_path: str, feature_dir: str) -> Dict[str, List[str]]:
        """
        Extract features for all folders in crawled directory.
        
        Args:
            crawled_path: Path to directory containing celebrity folders
            feature_dir: Directory to save individual feature files
        
        Returns:
            Dictionary mapping folder names to lists of image paths
        """
        crawled_dir = Path(crawled_path)
        
        if not crawled_dir.exists():
            raise FileNotFoundError(f"Crawled directory not found: {crawled_path}")
        
        folders = [f for f in crawled_dir.iterdir() if f.is_dir()]
        folders = sorted(folders, key=lambda x: x.name)
        
        folder_images = {}
        
        for folder in folders:
            image_paths = self.extract_and_save_features(str(folder), feature_dir)
            folder_images[folder.name] = image_paths
        
        return folder_images
    
    def load_features_for_images(self, image_paths: List[str], feature_dir: str) -> Dict[str, np.ndarray]:
        """
        Load features for a list of images from disk.
        
        Args:
            image_paths: List of image file paths
            feature_dir: Directory containing feature files
        
        Returns:
            Dictionary mapping image paths to feature arrays
        """
        features = {}
        
        for img_path in image_paths:
            feature_path = self.get_feature_path(img_path, feature_dir)
            if feature_path.exists():
                features[img_path] = self.load_feature(feature_path)
            else:
                print(f"Warning: Feature not found for {img_path}")
        
        return features
    
    def arcface_score_from_features(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        score = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return float(score)
    
    def calculate_intra_folder_similarity_cached(self, features: Dict[str, np.ndarray]) -> float:
        """Calculate average ArcFace similarity between all unique pairs within a folder using cached features"""
        image_paths = list(features.keys())
        
        if len(image_paths) < 2:
            print(f"Warning: Less than 2 images in folder")
            return 0.0
        
        total_score = 0.0
        num_pairs = 0
        
        # Calculate similarity for all unique pairs
        for path1, path2 in combinations(image_paths, 2):
            try:
                score = self.arcface_score_from_features(features[path1], features[path2])
                total_score += score
                num_pairs += 1
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        if num_pairs == 0:
            return 0.0
        
        avg_score = total_score / num_pairs
        return avg_score
    
    def calculate_inter_folder_similarity_cached(self, features1: Dict[str, np.ndarray], 
                                                 features2: Dict[str, np.ndarray]) -> float:
        """Calculate average ArcFace similarity between all pairs across two folders using cached features"""
        paths1 = list(features1.keys())
        paths2 = list(features2.keys())
        
        if not paths1 or not paths2:
            print(f"Warning: Empty folder(s)")
            return 0.0
        
        total_score = 0.0
        num_pairs = 0
        
        # Calculate similarity for all pairs between folders
        for path1 in paths1:
            for path2 in paths2:
                try:
                    score = self.arcface_score_from_features(features1[path1], features2[path2])
                    total_score += score
                    num_pairs += 1
                except Exception as e:
                    print(f"Error processing pair: {e}")
                    continue
        
        if num_pairs == 0:
            return 0.0
        
        avg_score = total_score / num_pairs
        return avg_score
    
    def create_similarity_heatmap(self, crawled_path: str, output_path: str = "arcface_similarity_heatmap.png",
                                 feature_dir: Optional[str] = None):
        """
        Create 10x10 heatmap of ArcFace similarities between all folder pairs using cached features.
        
        Args:
            crawled_path: Path to directory containing celebrity folders
            output_path: Path to save heatmap image
            feature_dir: Directory to save/load individual feature files
        """
        if feature_dir is None:
            feature_dir = str(Path(output_path).parent / "arcface_features")
        
        # Extract all features (or skip if already exist)
        print("=" * 80)
        print("STEP 1: Extracting/Loading ArcFace Features")
        print("=" * 80)
        folder_images = self.extract_all_features(crawled_path, feature_dir)
        
        folder_names = sorted(folder_images.keys())
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
            features_i = self.load_features_for_images(folder_images[folder_i], feature_dir)
            
            for j in range(n):
                folder_j = folder_names[j]
                features_j = self.load_features_for_images(folder_images[folder_j], feature_dir)
                
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
                   cbar_kws={'label': 'ArcFace Similarity Score'})
        
        plt.title('ArcFace Similarity Matrix Between Celebrity Folders')
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
    output_heatmap = "/home/jiyoon/PAI_Bench/utils/features/arcface_similarity_heatmap.png"
    feature_dir = "/data2/jiyoon/PAI-Bench/data/crawled/arcface_features"  # Directory to save individual features
    model_name = "antelopev2"
    device = "cuda"
    
    # Create scorer and generate heatmap
    scorer = ArcFaceScorer(model_name, device)
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


if __name__ == "__main__":
    main()