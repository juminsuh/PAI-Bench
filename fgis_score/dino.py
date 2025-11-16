"""
Code that extracts region-wise dino embedding
and save it as pkl file
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


def extract_region_embedding(original_image, binary_mask, processor, model, image_number, device):
    """
    Extract dino embedding of specific region from original image, using binary mask of each region
    
    Args:
        original_image: PIL Image (원본 이미지)
        binary_mask: PIL Image (binary mask)
        processor: DINO image processor
        model: DINO model
        device: torch device
    
    Returns:
        numpy array: DINO embedding (768-dim for dinov2-base)
    """
    try:
        # convert image and mask into numpy array
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
        masked_image[~binary] = 255  # make background be white (1)
        
        cropped = masked_image[y_min:y_max+1, x_min:x_max+1]
        pil_cropped = Image.fromarray(cropped)
        
        # extract DINO embedding
        if pil_cropped.mode != 'RGB':
            pil_cropped = pil_cropped.convert('RGB')
        ## code for double-check
        # save_dir = "./pil_cropped"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, "image.jpg")
        # pil_cropped.save(save_path)
            
        inputs = processor(images=pil_cropped, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
    
        return embedding

    except:
        print(f"❌ ERROR - Skipping {image_number}.jpg")
        return None


def extract_embeddings_for_celeb(celeb_name, images_dir, masks_dir, processor, model, device):
    """
    Extract DINO embedding for all face regions
    
    Args:
        celeb_name: 연예인 이름 (e.g., "BrunoMars")
        images_dir: 원본 이미지 디렉토리 (e.g., "./assets/images/BrunoMars")
        masks_dir: Binary mask 디렉토리 (e.g., "./assets/binary_mask_output/BrunoMars")
        processor: DINO processor
        model: DINO model
        device: torch device
    
    Returns:
        DataFrame: 각 행은 이미지, 각 열은 face region의 임베딩
    """
    results = []
    
    # load image files 
    image_files = sorted([f for f in os.listdir(images_dir) # all files in ./assets/images/BrunoMars
                         if f.endswith(('.jpg', '.png', '.jpeg', 'JPG'))])

    print(f"\n🎭 Processing {celeb_name}: {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc=f"Extracting embeddings"):

        image_number = os.path.splitext(image_file)[0] # 000001
        
        image_path = os.path.join(images_dir, image_file)
        original_image = Image.open(image_path).convert('RGB')
        
        mask_image_dir = os.path.join(masks_dir, image_number) # ./assets/binary_mask_output/BrunoMars/000001
        mask_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        
        if not os.path.exists(mask_image_dir):
            print(f"⚠️  Mask directory not found: {mask_image_dir}")
            continue
        
        # save metadata
        row_data = {
            'celeb': celeb_name,
            'image_id': image_number
        }
        
        # process for each face region
        for mask_file in mask_files:
            mask_path = os.path.join(mask_image_dir, mask_file) 
            number = os.path.splitext(mask_file)[0].split('_')[-1]
            # print(f"📌 number: {number}")
            # number = os.path.splitext(mask_file)[0] # 1
                        
            # if there is no region (all pixels of the mask is 0) -> skip and set embedding as None 
            if os.path.exists(mask_path):
                binary_mask = Image.open(mask_path).convert('L')
                binary_mask_array = np.array(binary_mask)
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
        print(f"✅ {image_number}.jpg completed!")
        results.append(row_data)
    
    df = pd.DataFrame(results)
    
    return df


def extract_all_embeddings(base_images_dir, base_masks_dir, output_path="face_embeddings.pkl"):
    """
    Iterate the process for all celebs
    
    Args:
        base_images_dir: 이미지 베이스 디렉토리 (e.g., "./assets/images")
        base_masks_dir: 마스크 베이스 디렉토리 (e.g., "./assets/binary_mask_output")
        output_path: 저장할 파일 경로
    """
    # DINO 모델 로드
    print("🔄 Loading DINO model...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    print(f"\n📋 Processing {celeb}'s images...")
    
    images_dir = os.path.join(base_images_dir, celeb) # ./assets/images/BrunoMars
    masks_dir = os.path.join(base_masks_dir, celeb) # ./assets/binary_mask_output/BrunoMars
    
    df = extract_embeddings_for_celeb(
        celeb, 
        images_dir, 
        masks_dir,
        processor, 
        model, 
        device
    )
    
    print(f"\n📊 DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # save as .pkl
    df.to_pickle(output_path)
    print(f"💾 Saved to {output_path}")
    
    # save metadata as .csv
    meta_df = df[['celeb', 'image_id']].copy()
    csv_path = output_path.replace('.pkl', '_meta.csv')
    meta_df.to_csv(csv_path, index=False)
    print(f"💾 Metadata saved to {csv_path}")
    
    return df


# ============= main =============

if __name__ == "__main__":
    # set directories
    celeb = "TomHolland"
    base_images_dir = "/data2/jiyoon/PAI-Bench/data/crawled/imgs"
    base_masks_dir =  "/data2/jiyoon/PAI-Bench/data/crawled/fgis/binary_mask_output"
    output_path = f"/data2/jiyoon/PAI-Bench/data/crawled/fgis/embeddings/{celeb}.pkl"
    
    df = extract_all_embeddings(
        base_images_dir, 
        base_masks_dir, 
        output_path
    )
    
    print("\n" + "="*50)
    print("✨ Extraction Complete!")
    print("="*50)

    # print info
    print(f"\nDataFrame Info:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Celebrities: {df['celeb'].unique().tolist()}")
    print(f"  - Total images: {len(df)}")
