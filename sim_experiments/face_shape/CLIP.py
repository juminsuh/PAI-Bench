
"""
CLIP image-image similarity:
- <target-folder>  vs  FFHQ/{bucket}/{id5}.{ext}

Usage: python CLIP_id.py
(Modify the hardcoded values in main() function as needed)
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

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/PAI-Bench/data/faceshape_change/survey_cropped"
    out_file =  "/data2/jiyoon/PAI-Bench/results/CLIP/cropped/faceshape_change.csv"
    ffhq_root_path = "/data2/jiyoon/PAI-Bench/data/ffhq_50/survey_cropped"
    bucket_size = 1000
    target_name = ""
    clip_model = "ViT-L-14"
    clip_pretrained = "openai"
    device = "cuda"

    root_dir = Path(target_folder)
    ffhq_root = Path(ffhq_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ root does not exist: {ffhq_root}"

    clip = CLIPScorer(clip_model, clip_pretrained, device)
    print(f"[INFO] device={clip.device}, model={clip_model}:{clip_pretrained}")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | bucket_size={bucket_size} | target_name={target_name or '(auto)'}")

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

    # Anchor embedding cache (to handle duplicate filenames)
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
                "clip_i": np.nan,
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
                "clip_i": np.nan,
                "notes": ""
            }
            target_path = item

        # Anchor image
        anchor_path = resolve_ffhq_anchor(ffhq_root, filename_stem, bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # Anchor embedding
        if filename_stem not in anchor_feat_cache:
            try:
                ref_img = Image.open(anchor_path).convert("RGB")
                anchor_feat_cache[filename_stem] = clip.img_feat(ref_img)
            except Exception:
                anchor_feat_cache[filename_stem] = None
        a_feat = anchor_feat_cache[filename_stem]
        if a_feat is None:
            row["notes"] = "anchor_open_or_feat_error"; rows.append(row); continue

        # Target embedding and similarity
        try:
            gen_img = Image.open(target_path).convert("RGB")
            g_feat = clip.img_feat(gen_img)
            row["clip_i"] = float((a_feat @ g_feat).item())
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
        if df["clip_i"].notna().any():
            s = df["clip_i"].dropna()
            print(f"[STAT] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
        else:
            print("[STAT] No valid CLIP values.")
    else:
        print("[DEBUG] No target items to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()