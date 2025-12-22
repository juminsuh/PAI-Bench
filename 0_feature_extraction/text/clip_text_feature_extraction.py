"""
CLIP Text Feature Extractor 
Extracts CLIP features from text prompts and saves them as pickle files
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional
import torch
import open_clip
from tqdm import tqdm


class CLIPTextFeatureExtractor:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading CLIP model: {model_name} ({pretrained}) on {self.device}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def load_prompts(self, json_path: str) -> Dict[str, str]:
        """Load prompts from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {item['id']: item['prompt'] for item in data}

    def extract_text_features(self, text: str) -> torch.Tensor:
        """Extract CLIP features from text"""
        try:
            text_tokens = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().squeeze(0)
            
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return None

    def extract_all_text_features(self, json_path: str, output_folder: str):
        """Extract features for all texts in JSON file"""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        prompts_dict = self.load_prompts(json_path)
        print(f"Loaded {len(prompts_dict)} text prompts")
        
        success_count = 0
        failed_texts = []
        
        for text_id, prompt in tqdm(prompts_dict.items(), desc="Extracting text features"):
            features = self.extract_text_features(prompt)
            
            if features is not None:
                feature_filename = f"{text_id}.pkl"
                feature_path = output_path / feature_filename
                
                try:
                    with open(feature_path, 'wb') as f:
                        pickle.dump(features, f)
                    success_count += 1
                except Exception as e:
                    print(f"Error saving features for {text_id}: {e}")
                    failed_texts.append(text_id)
            else:
                failed_texts.append(text_id)

        print(f"\nText feature extraction summary:")
        print(f"Total texts: {len(prompts_dict)}")
        print(f"Successfully extracted: {success_count}")
        print(f"Failed: {len(failed_texts)}")
        if failed_texts:
            print(f"Failed text IDs: {failed_texts}")
        print(f"Features saved to: {output_folder}")


def main():
    json_path = "/data2/jiyoon/PAI-Bench/data/datasets_final/generation_prompts.json"
    output_folder = "/data2/jiyoon/PAI-Bench/data/251222/text_features/clip"
    
    extractor = CLIPTextFeatureExtractor(model_name="ViT-B-32", pretrained="openai", device="cuda")
    extractor.extract_all_text_features(json_path, output_folder)
    
    print("TEXT FEATURE EXTRACTION COMPLETED!")

if __name__ == "__main__":
    main()