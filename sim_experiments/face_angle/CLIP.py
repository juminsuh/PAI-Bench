"""
CLIP-I scorer for celeb face angle dataset
"""

from pathlib import Path
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import open_clip

class CLIPScorer:
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
    
    def compare_images_in_folder(self, reference_img_path: str, folder_path: str, output_csv: str):
        folder = Path(folder_path)
        reference_path = Path(reference_img_path)
        
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_img_path}")
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        results = []
        total_score = 0.0
        
        for img_file in tqdm(image_files, desc="Calculating CLIP scores"):
            try:
                score = self.clip_score(str(reference_path), str(img_file))
                results.append({
                    'image0': str(reference_path),
                    'image1': str(img_file),
                    'clip_score': score
                })
                total_score += score
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        if results:
            avg_score = total_score / len(results)
            results.append({
                'image0': 'AVERAGE',
                'image1': 'AVERAGE', 
                'clip_score': avg_score
            })
            
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
            print(f"Average CLIP score: {avg_score:.4f}")
        else:
            print("No valid comparisons could be made")


def main():
    reference_image = "/data2/jiyoon/PAI-Bench/data/crawled/cropped/TomHolland/000001.jpg"
    images_folder = "/data2/jiyoon/PAI-Bench/data/crawled/cropped/TomHolland"
    output_csv = "/data2/jiyoon/PAI-Bench/results/crawled/TomHolland_clip.csv"
    model_name = "ViT-B-32"
    pretrained = "openai"
    device = "cuda"
    
    scorer = CLIPScorer(model_name, pretrained, device)
    scorer.compare_images_in_folder(reference_image, images_folder, output_csv)


if __name__ == "__main__":
    main()