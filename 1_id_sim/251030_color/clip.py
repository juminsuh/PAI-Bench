"""
CLIP-I scorer for comparing original images with their dark/light variants
"""

from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import open_clip

class ColorVariantCLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze(0)
    
    def clip_score(self, img1_path: str, img2_path: str) -> float:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        feat1 = self.img_feat(img1)
        feat2 = self.img_feat(img2)
        
        score = torch.cosine_similarity(feat1, feat2, dim=0)
        return score.item()
    
    def find_image_triplets(self, folder_path: str) -> List[Tuple[str, str, str]]:
        """Find triplets of (original, dark, light) images"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        all_files = [f for f in folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions]
        
        triplets = []
        
        # Group files by their base number
        base_files = {}
        for file in all_files:
            name = file.stem
            
            # Check if it's a base file (just number)
            if re.match(r'^\d+$', name):
                base_num = name
                if base_num not in base_files:
                    base_files[base_num] = {}
                base_files[base_num]['original'] = str(file)
            
            # Check if it's a dark variant
            elif name.endswith('_dark'):
                base_num = name[:-5]  # Remove '_dark'
                if base_num not in base_files:
                    base_files[base_num] = {}
                base_files[base_num]['dark'] = str(file)
            
            # Check if it's a light variant
            elif name.endswith('_light'):
                base_num = name[:-6]  # Remove '_light'
                if base_num not in base_files:
                    base_files[base_num] = {}
                base_files[base_num]['light'] = str(file)
        
        # Create triplets where all three variants exist
        for base_num, variants in base_files.items():
            if all(key in variants for key in ['original', 'dark', 'light']):
                triplets.append((
                    variants['original'],
                    variants['dark'], 
                    variants['light']
                ))
        
        return triplets
    
    def compare_variants(self, folder_path: str, output_csv_dark: str, output_csv_light: str):
        """Compare original images with their dark and light variants"""
        triplets = self.find_image_triplets(folder_path)
        
        if not triplets:
            raise ValueError(f"No complete image triplets found in folder: {folder_path}")
        
        print(f"Found {len(triplets)} complete image triplets")
        
        dark_results = []
        light_results = []
        
        total_dark_score = 0.0
        total_light_score = 0.0
        
        for original_path, dark_path, light_path in tqdm(triplets, desc="Calculating CLIP scores"):
            try:
                # Compare original with dark variant
                dark_score = self.clip_score(original_path, dark_path)
                dark_results.append({
                    'original_image': original_path,
                    'variant_image': dark_path,
                    'clip_score': dark_score
                })
                total_dark_score += dark_score
                
                # Compare original with light variant
                light_score = self.clip_score(original_path, light_path)
                light_results.append({
                    'original_image': original_path,
                    'variant_image': light_path,
                    'clip_score': light_score
                })
                total_light_score += light_score
                
            except Exception as e:
                print(f"Error processing triplet {original_path}: {e}")
                continue
        
        # Save dark variant results
        if dark_results:
            avg_dark_score = total_dark_score / len(dark_results)
            dark_results.append({
                'original_image': 'AVERAGE',
                'variant_image': 'AVERAGE',
                'clip_score': avg_dark_score
            })
            
            df_dark = pd.DataFrame(dark_results)
            df_dark.to_csv(output_csv_dark, index=False)
            print(f"Dark variant results saved to {output_csv_dark}")
            print(f"Average CLIP score (original vs dark): {avg_dark_score:.4f}")
        
        # Save light variant results
        if light_results:
            avg_light_score = total_light_score / len(light_results)
            light_results.append({
                'original_image': 'AVERAGE',
                'variant_image': 'AVERAGE',
                'clip_score': avg_light_score
            })
            
            df_light = pd.DataFrame(light_results)
            df_light.to_csv(output_csv_light, index=False)
            print(f"Light variant results saved to {output_csv_light}")
            print(f"Average CLIP score (original vs light): {avg_light_score:.4f}")
        
        return {
            'dark_avg': avg_dark_score if dark_results else 0,
            'light_avg': avg_light_score if light_results else 0,
            'num_comparisons': len(triplets)
        }


def main():
    images_folder = "/data2/jiyoon/PAI-Bench/data/color_changed/TomHolland"
    output_csv_dark = "/data2/jiyoon/PAI-Bench/results/color/TomHolland/clip_dark.csv"
    output_csv_light = "/data2/jiyoon/PAI-Bench/results/color/TomHolland/clip_light.csv"
    model_name = "ViT-B-32"
    pretrained = "openai"
    device = "cuda"
    
    # Create output directory if it doesn't exist
    Path(output_csv_dark).parent.mkdir(parents=True, exist_ok=True)
    
    scorer = ColorVariantCLIPScorer(model_name, pretrained, device)
    results = scorer.compare_variants(images_folder, output_csv_dark, output_csv_light)
    
    print(f"\nSummary:")
    print(f"Number of image triplets processed: {results['num_comparisons']}")
    print(f"Average CLIP score (original vs dark): {results['dark_avg']:.4f}")
    print(f"Average CLIP score (original vs light): {results['light_avg']:.4f}")


if __name__ == "__main__":
    main()