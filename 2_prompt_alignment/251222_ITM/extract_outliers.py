"""
Extract IQR outliers from result files and output id, prompt, and score.
"""

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
    
    return lower_bound, upper_bound


def extract_outliers_from_file(file_path):
    """Extract outliers from a single result file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    scores_data = data.get('scores', {})
    
    if not scores_data:
        return []
    
    # Determine the score field name based on file structure
    first_item = next(iter(scores_data.values()))
    if 'probability' in first_item:
        score_field = 'probability'
    elif 'score' in first_item:
        score_field = 'score'
    else:
        print(f"Warning: Cannot determine score field in {file_path}")
        return []
    
    # Extract scores for IQR calculation
    scores = [item[score_field] for item in scores_data.values()]
    
    # Calculate IQR bounds
    lower_bound, upper_bound = calculate_iqr_outliers(scores)
    
    # Find outliers
    outliers = []
    for item_id, item_data in scores_data.items():
        score = item_data[score_field]
        if score < lower_bound or score > upper_bound:
            outliers.append({
                'id': item_id,
                'prompt': item_data.get('prompt', 'Unknown'),
                'score': score
            })
    
    return outliers


def main():
    """Extract outliers from all result files"""
    results_dir = Path("/home/jiyoon/PAI-Bench/2_prompt_alignment/251222_ITM/results")
    
    all_outliers = {}
    
    for json_file in results_dir.glob("*.json"):
        print(f"\nProcessing {json_file.name}...")
        
        outliers = extract_outliers_from_file(json_file)
        
        if outliers:
            all_outliers[json_file.stem] = outliers
            print(f"Found {len(outliers)} outliers")
            
            for outlier in outliers:
                print(f"  ID: {outlier['id']}, Score: {outlier['score']:.4f}")
                print(f"    Prompt: {outlier['prompt']}")
        else:
            print(f"No outliers found")
    
    # Save all outliers to a summary file
    output_file = results_dir / "iqr_outliers_summary.json"
    with open(output_file, 'w') as f:
        json.dump(all_outliers, f, indent=2)
    
    print(f"\nOutliers summary saved to: {output_file}")
    
    # Print summary
    total_outliers = sum(len(outliers) for outliers in all_outliers.values())
    print(f"\nTotal outliers across all files: {total_outliers}")


if __name__ == "__main__":
    main()