#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

# Process BLIP file
blip_file = "/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_ITM/results/blip_Gemini2.5Flash.json"
with open(blip_file, 'r') as f:
    blip_data = json.load(f)

# Extract probabilities for IQR calculation
probabilities = [item["probability"] for item in blip_data["scores"].values()]
q1 = np.percentile(probabilities, 25)
q3 = np.percentile(probabilities, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"BLIP Gemini2.5Flash - Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
print(f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
print("\nBLIP Outliers:")

blip_outliers = []
for item_id, item_data in blip_data["scores"].items():
    score = item_data["probability"]
    if score < lower_bound or score > upper_bound:
        blip_outliers.append({
            "id": item_id,
            "prompt": item_data["prompt"],
            "score": score
        })
        print(f"ID: {item_id}, Score: {score:.4f}")
        print(f"  Prompt: {item_data['prompt']}")

# Process CLIP files
clip_files = [
    "/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_ITM/results/clip_Gemini2.5Flash.json",
    "/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_ITM/results/clip_GPT5.1.json"
]

for clip_file in clip_files:
    if Path(clip_file).exists():
        with open(clip_file, 'r') as f:
            clip_data = json.load(f)
        
        # Extract scores for IQR calculation
        scores = [item["score"] for item in clip_data["scores"].values()]
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filename = Path(clip_file).name
        print(f"\n{filename} - Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        print(f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
        print(f"\n{filename} Outliers:")
        
        clip_outliers = []
        for item_id, item_data in clip_data["scores"].items():
            score = item_data["score"]
            if score < lower_bound or score > upper_bound:
                clip_outliers.append({
                    "id": item_id,
                    "prompt": item_data["prompt"],
                    "score": score
                })
                print(f"ID: {item_id}, Score: {score:.4f}")
                print(f"  Prompt: {item_data['prompt']}")

print("\nDone!")