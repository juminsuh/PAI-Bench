"""
ArcFace Feature Extractor
Extracts ArcFace features from images and saves them as pickle files
"""

from pathlib import Path
import pickle
from typing import List, Optional
import numpy as np

from PIL import Image
from tqdm import tqdm
import insightface


class ArcFaceFeatureExtractor:
    # load pretrained ArcFace model
    def __init__(self, model_name: str = 'antelopev2', device: str = 'cuda'):
        self.device = device
        self.app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320), det_thresh=0.3)


    # get all img files from input folder 
    def get_image_files(self, input_folder: str) -> List[Path]:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)

        
    # extract ArcFace features from single img (direct from cropped face)
    def extract_image_features(self, image_path: str) -> np.ndarray:
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Get face analysis results (should find exactly one face in cropped image)
            faces = self.app.get(image_array)
            
            if len(faces) == 0:
                print(f"Warning: No face detected in {image_path}")
                return None
            
            # Use the first (and should be only) detected face
            face = faces[0]
            embedding = face.embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    

    # extract ArcFace features for all imgs
    def extract_folder_features(self, input_folder: str, output_folder: str):
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # load img files
        image_files = self.get_image_files(input_folder)
        
        if not image_files:
            print("ERROR: No image files found in input folder!")
            return
        
        # extract features
        success_count = 0
        failed_files = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            features = self.extract_image_features(str(image_file))
            
            if features is not None:
                # save features as pickle file
                feature_filename = image_file.stem + ".pkl"
                feature_path = output_path / feature_filename
                
                try:
                    with open(feature_path, 'wb') as f:
                        pickle.dump(features, f)
                    success_count += 1
                except Exception as e:
                    print(f"Error saving features for {image_file.name}: {e}")
                    failed_files.append(image_file.name)
            else:
                failed_files.append(image_file.name)
        

        # print summary
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(f"Total images processed: {len(image_files)}")
        print(f"Successfully extracted: {success_count}")
        print(f"Failed extractions: {len(failed_files)}")
        
        if failed_files:
            print(f"\nFailed files: {failed_files}")
        
        print(f"\nFeatures saved to: {output_folder}")
        

    # extract ArcFace features using batch processing
    def extract_batch_folders(self, input_base_folder: str, output_base_folder: str):
        input_base = Path(input_base_folder)
        output_base = Path(output_base_folder)
        
        if not input_base.exists():
            raise FileNotFoundError(f"Input base folder not found: {input_base_folder}")
        
        # find all subdirectories
        subfolders = [f for f in input_base.iterdir() if f.is_dir()]
        
        if not subfolders:
            print("No subfolders found. Processing as single folder.")
            self.extract_folder_features(str(input_base), str(output_base))
            return
        
        print(f"Found {len(subfolders)} folders to process")
        
        for subfolder in subfolders:
            print(f"\nProcessing folder: {subfolder.name}")
            input_folder = str(subfolder)
            output_folder = str(output_base / subfolder.name)
            
            self.extract_folder_features(input_folder, output_folder)


def main():
    # config
    input_folder = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/2"
    output_folder = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/arcface/2"
    model_name = "antelopev2"
    device = "cuda"
    batch_mode = False  # Set to True for batch processing multiple folders
    

    extractor = ArcFaceFeatureExtractor(model_name=model_name, device=device)
    
    if batch_mode:
        extractor.extract_batch_folders(input_folder, output_folder)
    else:
        extractor.extract_folder_features(input_folder, output_folder)
    

    print("ARCFACE FEATURE EXTRACTION COMPLETED!")

if __name__ == "__main__":
    main()