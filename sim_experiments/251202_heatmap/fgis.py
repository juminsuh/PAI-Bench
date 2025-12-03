"""
FGIS Similarity Calculator for Two Image Folders
Creates a 25x25 heatmap of similarities between corresponding images
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Load single pkl file
def load_embedding_file(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
        return df
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def calculate_fgis_similarity(df1_row, df2_row):
    """
    Calculate FGIS similarity between two images using their face region embeddings
    
    Args:
        df1_row: DataFrame row from first image
        df2_row: DataFrame row from second image
        
    Returns:
        float: Average cosine similarity across valid face regions
    """
    # Get feature columns (exclude 'image_id')
    feature_cols = [col for col in df1_row.index if col != 'image_id' and col.isdigit()]

    
    # calculate similairy between all regions
    feature_similarities = []

    for feature_col in feature_cols:
        emb1 = df1_row[feature_col]
        emb2 = df2_row[feature_col]
        
        # compute similarity if both embs exist
        if emb1 is not None and emb2 is not None:
            if isinstance(emb1, (list, np.ndarray)) and isinstance(emb2, (list, np.ndarray)):
                emb1 = np.array(emb1).reshape(1, -1)
                emb2 = np.array(emb2).reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                feature_similarities.append(similarity)
    
    # return average similarity across all valid regions
    if len(feature_similarities) > 0:
        return np.mean(feature_similarities)
    else:
        return np.nan



def create_similarity_heatmap(folder1_emb_dir, folder2_emb_dir, output_csv, output_heatmap=None):
    """
    Create 25x25 similarity heatmap between two folders
    
    Args:
        folder1_emb_dir: Directory containing pkl files for folder 1 (001.pkl - 025.pkl)
        folder2_emb_dir: Directory containing pkl files for folder 2 (001.pkl - 025.pkl)
        output_csv: Path to save CSV results
        output_heatmap: Path to save heatmap image (optional)
        
    Returns:
        tuple: (similarity_matrix, df_results)
    """
   
    # Initialize 25x25 similarity matrix
    similarity_matrix = np.zeros((25, 25))
    all_results = []
    
    # --- Load emb folders ---
    folder1_name = Path(folder1_emb_dir).parent.name
    folder2_name = Path(folder2_emb_dir).parent.name
    
    # --- Calculate fgis sim ---
    for i in range(1, 26):  # folder2 imgs (y-axis)
        for j in range(1, 26):  # folder1 imgs (x-axis)
            img_num_folder1 = f"{j:03d}"
            img_num_folder2 = f"{i:03d}"
            
            # load embedding files
            pkl1_path = os.path.join(folder1_emb_dir, f"{img_num_folder1}.pkl")
            pkl2_path = os.path.join(folder2_emb_dir, f"{img_num_folder2}.pkl")
            
            if os.path.exists(pkl1_path) and os.path.exists(pkl2_path):
                df1 = load_embedding_file(pkl1_path)
                df2 = load_embedding_file(pkl2_path)
                
                if df1 is not None and df2 is not None:
                    # calculate fgis sim
                    similarity = calculate_fgis_similarity(df1.iloc[0], df2.iloc[0])
                    similarity_matrix[i-1, j-1] = similarity
                    
                    # Store result
                    all_results.append({
                        'folder1_image': f"{folder1_name}_{img_num_folder1}",
                        'folder2_image': f"{folder2_name}_{img_num_folder2}",
                        'similarity': similarity
                    })
                else:
                    similarity_matrix[i-1, j-1] = np.nan
            else:
                print(f"Warning: Missing files - {pkl1_path} or {pkl2_path}")
                similarity_matrix[i-1, j-1] = np.nan
    

    # --- Save heatmap ---
    plt.figure(figsize=(20, 16))

    x_labels = [f"{i:03d}" for i in range(1, 26)]  # folder1 images
    y_labels = [f"{i:03d}" for i in range(1, 26)]  # folder2 images
    
    sns.heatmap(similarity_matrix, 
               xticklabels=x_labels,
               yticklabels=y_labels,
               annot=True, 
               fmt='.3f',
               cmap='viridis',
               cbar_kws={'label': 'FGIS Similarity'},
               mask=np.isnan(similarity_matrix))
    
    plt.title(f'FGIS Similarities: {folder1_name} (x-axis) vs {folder2_name} (y-axis)')
    plt.xlabel(f'{folder1_name} Image Numbers')
    plt.ylabel(f'{folder2_name} Image Numbers')
    plt.tight_layout()
    
    if output_heatmap:
        output_path = Path(output_heatmap)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_heatmap}")
    plt.show()
    
    
    if all_results:
        # --- Save sim scores in csv ---
        df_results = pd.DataFrame(all_results)
        
        # sort by folder1_image and folder2_image
        df_results = df_results.sort_values(['folder1_image', 'folder2_image'], ascending=True)
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Total image pairs processed: {len(all_results)}")
        
        # --- Save statistics in csv ---
        # print summary statistics
        valid_similarities = df_results['similarity'].dropna()
        if len(valid_similarities) > 0:
            print("\nSimilarity Statistics:")
            print(f"Valid pairs: {len(valid_similarities)}")
            print(f"Mean similarity: {valid_similarities.mean():.4f}")
            print(f"Std similarity: {valid_similarities.std():.4f}")
            print(f"Min similarity: {valid_similarities.min():.4f}")
            print(f"Max similarity: {valid_similarities.max():.4f}")
            
            # save summary
            summary_data = {
                'total_pairs': [len(df_results)],
                'valid_pairs': [len(valid_similarities)],
                'mean_similarity': [valid_similarities.mean()],
                'std_similarity': [valid_similarities.std()],
                'min_similarity': [valid_similarities.min()],
                'max_similarity': [valid_similarities.max()]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = str(output_csv).replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}")
        return similarity_matrix, df_results
    else:
        print("ERROR: No results to save!")
        return None, None


def main():
    # config
    folder1_emb_dir = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/1/embs"
    folder2_emb_dir = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/2/embs"
    output_csv = "/home/jiyoon/PAI-Bench/sim_experiments/251202_heatmap/results/fgis_similarity.csv"
    output_heatmap = "/home/jiyoon/PAI-Bench/sim_experiments/251202_heatmap/results/fgis_heatmap.png"
    

    similarity_matrix, df_results = create_similarity_heatmap(
        folder1_emb_dir, 
        folder2_emb_dir, 
        output_csv, 
        output_heatmap
    )
    
    if similarity_matrix is not None:
        print("FGIS SIMILARITY ANALYSIS COMPLETED!")
    else:
        print("FAILED - No results generated")


if __name__ == "__main__":
    main()