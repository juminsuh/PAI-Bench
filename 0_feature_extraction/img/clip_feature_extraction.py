"""
CLIP Feature Extractor
Extracts CLIP features from images and saves them as pickle files
"""

from pathlib import Path
import pickle
from typing import List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from tqdm import tqdm


class CLIPFeatureExtractor:
    # load pretrained CLIP model
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading CLIP model: {model_name} ({pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()


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

        
    # Extract CLIP features from single img    
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # extract features
                image_features = self.model.encode_image(image_input)

                # normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().squeeze(0)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    

    # extract CLIP features for all imgs
    def extract_folder_features(self, input_folder: str, output_folder: str):
        
        # create output directory
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
        

    # extract CLIP features using batch processing
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
    # Configuration
    input_folder = "/data2/jiyoon/PAI-Bench/data/251222/cropped/generated_gemini2.5Flash"
    output_folder = "/data2/jiyoon/PAI-Bench/data/251222/img_features/clip"
    model_name = "ViT-B-32"
    pretrained = "openai"
    device = "cuda"
    batch_mode = True 
    
    extractor = CLIPFeatureExtractor(model_name=model_name, pretrained=pretrained, device=device)
    
    if batch_mode:
        extractor.extract_batch_folders(input_folder, output_folder)
    else:
        extractor.extract_folder_features(input_folder, output_folder)
    
    print("FEATURE EXTRACTION COMPLETED!")

if __name__ == "__main__":
    main()