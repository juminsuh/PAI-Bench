#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

def calculate_iqr_outliers(scores):
    """Calculate IQR outliers from a list of scores"""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound, q1, q3, iqr

def process_file(file_path):
    """Process a single result file and extract outliers"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    scores_data = data.get('scores', {})
    if not scores_data:
        return []
    
    # Determine score field and extract scores
    first_item = next(iter(scores_data.values()))
    if 'probability' in first_item:
        score_field = 'probability'
    elif 'score' in first_item:
        score_field = 'score'
    else:
        print(f"Warning: Cannot determine score field in {file_path}")
        return []
    
    scores = [item[score_field] for item in scores_data.values()]
    lower_bound, upper_bound, q1, q3, iqr = calculate_iqr_outliers(scores)
    
    filename = Path(file_path).name
    print(f"\n{filename} - Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    print(f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
    print(f"\n{filename} Outliers:")
    
    outliers = []
    for item_id, item_data in scores_data.items():
        score = item_data[score_field]
        if score < lower_bound or score > upper_bound:
            outliers.append({
                "id": item_id,
                "prompt": item_data.get("prompt", "Unknown"),
                "score": score
            })
            print(f"ID: {item_id}, Score: {score:.4f}")
            print(f"  Prompt: {item_data.get('prompt', 'Unknown')}")
    
    return outliers

# Find all JSON files in results directory
results_dir = Path("/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_ITM/results")
json_files = list(results_dir.glob("*.json"))

all_outliers = {}

for json_file in json_files:
    if json_file.name != "iqr_outliers.json":  # Skip the output file
        outliers = process_file(json_file)
        if outliers:
            all_outliers[json_file.stem] = outliers

# Save results to JSON file
output_file = results_dir / "iqr_outliers.json"
with open(output_file, 'w') as f:
    json.dump(all_outliers, f, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"Total files processed: {len(json_files)}")
print(f"Files with outliers: {len(all_outliers)}")
print("Done!")