"""
Calculate CLIP scores from pre-extracted image and text features
"""

import pickle
import json
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm


def load_features(feature_path: str) -> torch.Tensor:
    """Load features from pickle file"""
    try:
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        return features
    except Exception as e:
        print(f"Error loading {feature_path}: {e}")
        return None


def calculate_clip_scores_from_features(image_features_dir: str, text_features_dir: str, output_path: str = "clip_scores_results.json"):
    """Calculate CLIP scores from pre-extracted features"""
    
    image_dir = Path(image_features_dir)
    text_dir = Path(text_features_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image features directory not found: {image_features_dir}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Text features directory not found: {text_features_dir}")
    
    image_files = sorted(list(image_dir.glob("*.pkl")))
    text_files = list(text_dir.glob("*.pkl"))
    
    print(f"Found {len(image_files)} image feature files")
    print(f"Found {len(text_files)} text feature files")
    
    scores = {}
    valid_pairs = 0
    failed_pairs = []
    
    for img_file in tqdm(image_files, desc="Calculating CLIP scores"):
        img_id = img_file.stem
        text_file = text_dir / f"{img_id}.pkl"
        
        if not text_file.exists():
            print(f"Warning: No matching text features for image {img_id}")
            failed_pairs.append(img_id)
            continue
        
        img_features = load_features(str(img_file))
        text_features = load_features(str(text_file))
        
        if img_features is None or text_features is None:
            failed_pairs.append(img_id)
            continue
        
        try:
            if isinstance(img_features, torch.Tensor):
                img_features = img_features.numpy()
            if isinstance(text_features, torch.Tensor):
                text_features = text_features.numpy()
            
            clip_score = np.dot(img_features, text_features)
            scores[img_id] = float(clip_score)
            valid_pairs += 1
            
        except Exception as e:
            print(f"Error calculating score for {img_id}: {e}")
            failed_pairs.append(img_id)
    
    if scores:
        score_values = list(scores.values())
        mean_score = np.mean(score_values)
        std_score = np.std(score_values)
        min_score = np.min(score_values)
        max_score = np.max(score_values)
        
        print(f"\nCLIP Score Results:")
        print(f"Valid pairs: {valid_pairs}")
        print(f"Failed pairs: {len(failed_pairs)}")
        print(f"Mean CLIP score: {mean_score:.4f}")
        print(f"Std CLIP score: {std_score:.4f}")
        print(f"Min CLIP score: {min_score:.4f}")
        print(f"Max CLIP score: {max_score:.4f}")
        
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[0]))
        
        results = {
            "scores": sorted_scores,
            "statistics": {
                "valid_pairs": valid_pairs,
                "failed_pairs": len(failed_pairs),
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score
            },
            "failed_pairs": failed_pairs
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    else:
        print("No valid CLIP scores calculated!")


def main():
    image_features_dir = "/data2/jiyoon/PAI-Bench/data/251222/img_features/clip"
    text_features_dir = "/data2/jiyoon/PAI-Bench/data/251222/text_features/clip"
    output_path = "/home/jiyoon/PAI-Bench/prompt_alignment/251222_textimg/results/clip.json"
    
    calculate_clip_scores_from_features(image_features_dir, text_features_dir, output_path)


if __name__ == "__main__":
    main()