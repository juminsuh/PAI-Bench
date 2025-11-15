
from pathlib import Path
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

ALT_EXTS = [".png", ".PNG", ".jpg", ".jpeg", ".webp", ".bmp"]
ANCHOR_EXTS = [".png", ".jpg", ".jpeg"] 

def resolve_ffhq_anchor(ffhq_root: Path, filename_stem: str, bucket_size: int, recursive_fallback: bool = False) -> Optional[Path]:
    """
    Find FFHQ anchor image using filename stem (without extension).
    Priority:
      1) Flat layout ({ffhq_root}/{filename_stem}.{ext})
      2) (Optional) Recursive search
    """
    # 1) Flat layout
    for ext in ANCHOR_EXTS:
        p = ffhq_root / f"{filename_stem}{ext}"
        if p.exists():
            return p

    # 2) (Optional) Recursive search - disabled by default as it can be slow
    if recursive_fallback:
        for ext in ANCHOR_EXTS:
            hits = list(ffhq_root.rglob(f"{filename_stem}{ext}"))
            if hits:
                return hits[0]

    return None

def resolve_target_image(folder: Path, target_name: Optional[str]) -> Optional[Path]:
    if target_name:
        base = target_name.strip()
        # With extension
        if Path(base).suffix:
            p = folder / base
            if p.exists():
                return p
        else:
            # Without extension
            for ext in ALT_EXTS:
                p = folder / f"{base}{ext}"
                if p.exists():
                    return p
        # Prefix matching
        cands = sorted([q for q in folder.glob(f"{base}*") if q.suffix.lower() in ALT_EXTS])
        if cands:
            return cands[0]
        return None

    # Auto selection
    imgs = sorted([q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in ALT_EXTS])
    return imgs[0] if imgs else None

class ArcFaceScorer:
    def __init__(self, device: str):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        # Use ResNet50 as backbone for face feature extraction
        from torchvision.models import resnet50
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        # Remove the final classification layer to get feature embeddings
        self.model.fc = torch.nn.Identity()
        
        # Face recognition typically uses these preprocessing parameters
        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),  # Standard face recognition input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def extract_features(self, img: Image.Image) -> torch.Tensor:
        """Extract normalized face features using ResNet50 backbone"""
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model(x)
        # L2 normalize the features (standard in face recognition)
        features = F.normalize(features, p=2, dim=1)
        return features.squeeze(0)
    
    def calculate_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two feature vectors
        Returns similarity score (higher = more similar)
        """
        # Convert to numpy for cosine similarity calculation
        feat1_np = feat1.cpu().numpy().reshape(1, -1)
        feat2_np = feat2.cpu().numpy().reshape(1, -1)
        
        similarity = cosine_similarity(feat1_np, feat2_np)[0, 0]
        return float(similarity)

def main():
    target_folder = "/data2/jiyoon/PAI-Bench/data/faceshape_keep/survey_cropped"
    out_file =  "/data2/jiyoon/PAI-Bench/results/ArcFace/cropped/faceshape_keep.csv"
    ffhq_root_path = "/data2/jiyoon/PAI-Bench/data/ffhq_50/survey_cropped"
    bucket_size = 1000
    target_name = ""
    device = "cuda"

    root_dir = Path(target_folder)
    ffhq_root = Path(ffhq_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ root does not exist: {ffhq_root}"

    arcface_scorer = ArcFaceScorer(device)
    print(f"[INFO] device={arcface_scorer.device}, model=ArcFace-ResNet50")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | bucket_size={bucket_size} | target_name={target_name or '(auto)'}")
    
    # ArcFace calculates similarity directly between image pairs

    rows: List[Dict] = []
    
    # Check if we have subdirectories or individual files
    all_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    all_files = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    
    if all_dirs:
        # Original behavior: process subdirectories
        print(f"[INFO] total_subdirs={len(all_dirs)}")
        items_to_process = all_dirs
        process_mode = "subdirs"
    else:
        # New behavior: process individual files
        print(f"[INFO] total_files={len(all_files)}")
        items_to_process = all_files
        process_mode = "files"

    # Anchor feature cache (to handle duplicate filenames)
    anchor_feat_cache: Dict[str, Optional[torch.Tensor]] = {}

    for item in tqdm(items_to_process, desc="Evaluating", ncols=100):
        if process_mode == "subdirs":
            # Original logic for subdirectories
            filename_stem = item.name
            
            row = {
                "folder": item.name,
                "filename_stem": filename_stem,
                "anchor_path": "",
                "target_path": "",
                "arcface_similarity": np.nan,
                "notes": ""
            }
            
            # Target image
            target_path = resolve_target_image(item, target_name or None)
            if not target_path:
                row["notes"] = "target_missing"; rows.append(row); continue
            row["target_path"] = str(target_path)
        else:
            # New logic for individual files
            filename_stem = item.stem  # filename without extension
            
            row = {
                "folder": item.name,
                "filename_stem": filename_stem,
                "anchor_path": "",
                "target_path": str(item),
                "arcface_similarity": np.nan,
                "notes": ""
            }
            target_path = item

        # Anchor image
        anchor_path = resolve_ffhq_anchor(ffhq_root, filename_stem, bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # Anchor feature extraction
        if filename_stem not in anchor_feat_cache:
            try:
                ref_img = Image.open(anchor_path).convert("RGB")
                anchor_feat_cache[filename_stem] = arcface_scorer.extract_features(ref_img)
            except Exception:
                anchor_feat_cache[filename_stem] = None
        anchor_feat = anchor_feat_cache[filename_stem]
        if anchor_feat is None:
            row["notes"] = "anchor_open_or_feat_error"; rows.append(row); continue

        # Target feature extraction and similarity calculation
        try:
            target_img = Image.open(target_path).convert("RGB")
            target_feat = arcface_scorer.extract_features(target_img)
            similarity = arcface_scorer.calculate_similarity(anchor_feat, target_feat)
            row["arcface_similarity"] = float(similarity)
        except Exception:
            row["notes"] = "target_open_or_feat_error"
        rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    out = Path(out_file)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["arcface_similarity"].notna().any():
            s = df["arcface_similarity"].dropna()
            print(f"[STAT] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
        else:
            print("[STAT] No valid ArcFace similarity values.")
    else:
        print("[DEBUG] No target items to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()