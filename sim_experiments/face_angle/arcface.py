"""
ArcFace similarity scorer for celeb face angle dataset
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
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from insightface.app import FaceAnalysis

class ArcFaceScorer:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        # Use smaller detection size and lower threshold for cropped faces
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320), det_thresh=0.3)

    def img_feat(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        
        # Try to get face embedding from cropped image
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError(f"No face detected in cropped image: {img_path}")
        
        # Use the first (and likely only) face in the cropped image
        embedding = faces[0].normed_embedding
        return embedding
    
    def arcface_score(self, img1_path: str, img2_path: str) -> float:
        feat1 = self.img_feat(img1_path)
        feat2 = self.img_feat(img2_path)
        
        # Calculate cosine similarity using numpy
        score = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return float(score)
    
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
        
        for img_file in tqdm(image_files, desc="Calculating ArcFace scores"):
            try:
                score = self.arcface_score(str(reference_path), str(img_file))
                results.append({
                    'image0': str(reference_path),
                    'image1': str(img_file),
                    'arcface_score': score
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
                'arcface_score': avg_score
            })
            
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
            print(f"Average ArcFace score: {avg_score:.4f}")
        else:
            print("No valid comparisons could be made")


def main():
    reference_image = "/data2/jiyoon/PAI-Bench/data/crawled/cropped/TomHolland/000001.jpg"
    images_folder = "/data2/jiyoon/PAI-Bench/data/crawled/cropped/TomHolland"
    output_csv = "/data2/jiyoon/PAI-Bench/results/crawled/TomHolland_arcface.csv"
    device = "cuda"
    
    scorer = ArcFaceScorer(device=device)
    scorer.compare_images_in_folder(reference_image, images_folder, output_csv)


if __name__ == "__main__":
    main()