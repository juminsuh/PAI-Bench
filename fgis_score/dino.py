"""
1. Code that extracts region-wise dino emb and saves it as pkl file
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

ATTRIBUTES = [
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',
    'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip',
    'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
]


# extract DINO emb of specific region
def extract_region_embedding(original_image, binary_mask, processor, model, image_number, device):
    try:
        img_array = np.array(original_image)
        mask_array = np.array(binary_mask)
        
        binary = mask_array > 127
        if not binary.any():
            return None
        
        # find bounding boxes
        coords = np.argwhere(binary)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        masked_image = img_array.copy()
        masked_image[~binary] = 255  # make background white
        
        cropped = masked_image[y_min:y_max+1, x_min:x_max+1]
        pil_cropped = Image.fromarray(cropped)
        
        # extract DINO embedding
        if pil_cropped.mode != 'RGB':
            pil_cropped = pil_cropped.convert('RGB')
            
        inputs = processor(images=pil_cropped, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
    
        return embedding

    except:
        print(f"ERROR - Skipping {image_number}.jpg")
        return None



# extract DINO emb for all face regions
def extract_embeddings_for_folder(images_dir, masks_dir, processor, model, device):
    results = []
    
    # load image files 
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg', 'JPG'))])
    print(f"Processing folder: {len(image_files)} images")
    

    for image_file in tqdm(image_files, desc=f"Extracting embeddings"):
        # img file path
        image_number = os.path.splitext(image_file)[0] 
        image_path = os.path.join(images_dir, image_file)
        original_image = Image.open(image_path).convert('RGB')
        
        # mask file path
        mask_image_dir = os.path.join(masks_dir, image_number)
        mask_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        
        if not os.path.exists(mask_image_dir):
            print(f"Mask directory not found: {mask_image_dir}")
            continue
        
        # save metadata
        row_data = {
            'image_id': image_number
        }
        
        # iterate each face region
        for mask_file in mask_files:
            mask_path = os.path.join(mask_image_dir, mask_file) 
            number = os.path.splitext(mask_file)[0].split('_')[-1]
                        
            if os.path.exists(mask_path):
                binary_mask = Image.open(mask_path).convert('L')
                binary_mask_array = np.array(binary_mask)

                # if there is no region (all pixels of the mask is 0) -> skip and set embedding as None 
                if np.all(binary_mask_array < 127): 
                    row_data[number] = None
                else:
                    embedding = extract_region_embedding(
                        original_image, 
                        binary_mask, 
                        processor, 
                        model,
                        image_number, 
                        device
                    )
                    row_data[number] = embedding
        print(f"{image_number}.jpg completed!")
        results.append(row_data)
    
    df = pd.DataFrame(results)
    
    return df


# extract emb from all imgs and binary masks in folder
def extract_all_embeddings(images_dir, masks_dir, output_path="face_embeddings.pkl"):
    # load DINO
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # extract emb
    df = extract_embeddings_for_folder(
        images_dir, 
        masks_dir,
        processor, 
        model, 
        device
    )
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # save embs
    df.to_pickle(output_path)
    print(f"Saved to {output_path}")
    
    # save metadata as .csv
    meta_df = df[['image_id']].copy()
    csv_path = output_path.replace('.pkl', '_meta.csv')
    meta_df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")
    
    return df


if __name__ == "__main__":
    # config
    images_dir = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/cropped/1"
    masks_dir = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/1/binary_mask_output"
    output_dir = f"/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/1/embs"
    os.mkdirs(output_dir, exist_ok = True)
    
    # Extract individual embeddings for each image file (001.pkl, 002.pkl, ..., 025.pkl)
    extract_single_image_embeddings(
        images_dir, 
        masks_dir, 
        output_dir
    )
    print("🎉 INDIVIDUAL EXTRACTION COMPLETE!")
