"""
Code to crop faces using YOLOv8
- input: folder path containing face imgs
- output: folder path to save cropped face imgs
"""

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import os

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

folder_path = "/data2/jiyoon/PAI-Bench/data/crawled/orig/EunwooCha"
output_dir = "/data2/jiyoon/PAI-Bench/data/crawled/cropped/EunwooCha"
os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")
        
        try:
            image = Image.open(image_path)
            output = model(image)
            results = Detections.from_ultralytics(output[0])
            
            print(f"Found {len(results.xyxy)} faces in {filename}")
            
            # crop and save only the first detected face
            # if len(results.xyxy) > 0:
            #     bbox = results.xyxy[0]
            #     x1, y1, x2, y2 = bbox.astype(int)
            #     cropped_image = image.crop((x1, y1, x2, y2))
                
            #     # convert RGBA to RGB if necessary
            #     if cropped_image.mode == 'RGBA':
            #         cropped_image = cropped_image.convert('RGB')
                
            #     # save cropped image
            #     base_name = os.path.splitext(filename)[0]
            #     crop_filename = f"{base_name}.jpg"
            #     crop_path = os.path.join(output_dir, crop_filename)
            #     cropped_image.save(crop_path)
                
            #     print(f"Saved first face to: {crop_path}")
            # else:
            #     print(f"No faces detected in {filename}")

            for i,bbox in enumerate(results.xyxy):
                bbox = x1, y1, x2, y2 = bbox.astype(int)
                cropped_image = image.crop((x1, y1, x2, y2))
                
                # convert RGBA to RGB if necessary
                if cropped_image.mode == 'RGBA':
                    cropped_image = cropped_image.convert('RGB')
                
                # save cropped image
                base_name = os.path.splitext(filename)[0]
                crop_filename = f"{base_name}_{i}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cropped_image.save(crop_path)
                
                print(f"Saved cropped face to: {crop_path}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue