import pandas as pd
import numpy as np
from itertools import combinations
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Configuration - centralized file paths
EMBEDDINGS_DIR = "/data2/jiyoon/PAI-Bench/data/crawled/fgis/embeddings"
OUTPUT_DIR = "/home/jiyoon/PAI-Bench/sim_experiments/251117_negative_outliers"
SIMILARITY_RESULTS_DIR = "./similarity_results"
OUTLIER_RESULTS_DIR = f"{OUTPUT_DIR}/outlier_results"

# CUDA configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def cosine_similarity_cuda(emb1, emb2):
    """
    Calculate cosine similarity using PyTorch on CUDA for better performance
    """
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1).float().to(DEVICE)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2).float().to(DEVICE)
    
    # Ensure tensors are 1D
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    
    # Calculate cosine similarity
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1)
    return similarity.cpu().item()

### compute diagonal similarities
def compute_diagonal_sim(celeb_list):
    for celeb in tqdm(celeb_list, desc="Processing celebrities (diagonal)"):
        results = []
        
        # open .pkl file
        with open(f'{EMBEDDINGS_DIR}/{celeb}.pkl', 'rb') as f: # original image
            embeddings = pickle.load(f)
        
        # convert to df
        df = pd.DataFrame(embeddings)  
        
        # extract feature embedding columns (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
        feature_cols = [col for col in df.columns if col not in ['celeb', 'image_id']]
        image_ids = df['image_id'].tolist()
        fgis_similarities = [] # list to save nC2 fgis scores
        
        for reference_id, compare_id in tqdm(combinations(image_ids, 2), desc=f"Computing pairs for {celeb}", leave=False):
            df_reference = df[df['image_id'] == reference_id]
            df_compare = df[df['image_id'] == compare_id]
            
            reference_row = df_reference.iloc[0]
            compare_row = df_compare.iloc[0]
            feature_similarities = [] # list to save a single fgis score 
            
            for feature_col in feature_cols:
                # compare each feature
                emb_reference = reference_row[feature_col]
                emb_compare = compare_row[feature_col]
                
                # compute similarity if and only if both embeddings are not None
                if emb_reference is not None and emb_compare is not None:
                    if isinstance(emb_reference, (list, np.ndarray)) and isinstance(emb_compare, (list, np.ndarray)):
                        emb_reference = np.array(emb_reference)
                        emb_compare = np.array(emb_compare)
                        similarity = cosine_similarity_cuda(emb_reference, emb_compare) # similarity between specific regions using CUDA
                        feature_similarities.append(similarity)
            
            # average similarity between reference image and compare image
            if len(feature_similarities) > 0:
                avg_similarity = np.mean(feature_similarities)
                fgis_similarities.append(avg_similarity)
                
                # save a result for a single reference image
                results.append({
                    'celeb': celeb,
                    'reference_image_id': reference_id,
                    'compare_image_id': compare_row['image_id'],
                    'cosine_similarity': avg_similarity,
                    'num_features_compared': len(feature_similarities)
                })

        # compute the entire average similarity for each celeb
        if len(fgis_similarities) > 0:
            fgis_avg_similarity = np.mean(fgis_similarities) # diagonal value
            print(f"✅ {celeb}: 평균 유사도 = {fgis_avg_similarity:.6f} (비교 이미지 수: {len(fgis_similarities)})") 
        else:
            print(f"⚠️ {celeb}: 유사도를 계산할 이미지가 없습니다.")

        # convert the results into df
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{SIMILARITY_RESULTS_DIR}/diag/results_{celeb}.csv', index=False)

        print(f"\n{'='*60}")
        print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
        print(f"📊 DataFrame shape: {df_results.shape}")

### compute non-diagonal similarities
def compute_non_diagonal_sim(celeb_list):
    celeb_embeddings = {}
    for celeb in tqdm(celeb_list, desc="Loading embeddings"):
        with open(f"{EMBEDDINGS_DIR}/{celeb}.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        celeb_embeddings[celeb] = embeddings
        
    for celeb1, celeb2 in tqdm(combinations(celeb_list, 2), desc="Processing celebrity pairs (non-diagonal)"):
        
        # convert to df
        df1 = pd.DataFrame(celeb_embeddings[celeb1]) 
        df2 = pd.DataFrame(celeb_embeddings[celeb2]) 
        
        # extract feature embedding columns (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
        feature_cols = [col for col in df1.columns if col not in ['celeb', 'image_id']]
        image_ids1 = df1['image_id'].tolist()
        image_ids2 = df2['image_id'].tolist()
        fgis_similarities = [] # list to save nC2 fgis scores
        results = []
        
        # celeb1 (50) vs celeb2 (50)
        for id1 in tqdm(image_ids1, desc=f"{celeb1} vs {celeb2}", leave=False):
            for id2 in image_ids2:
                reference_row = df1[df1['image_id'] == id1].iloc[0]
                compare_row = df2[df2['image_id'] == id2].iloc[0]
                
                feature_similarities = [] # list to save a single fgis score
                
                for feature_col in feature_cols:
                    emb_reference = reference_row[feature_col]
                    emb_compare = compare_row[feature_col]
                    
                    if emb_reference is not None and emb_compare is not None:
                        if isinstance(emb_reference, (list, np.ndarray)) and isinstance(emb_compare, (list, np.ndarray)):
                            emb_reference = np.array(emb_reference)
                            emb_compare = np.array(emb_compare)
                            similarity = cosine_similarity_cuda(emb_reference, emb_compare)
                            feature_similarities.append(similarity)
                
                if len(feature_similarities) > 0:
                    avg_similarity = np.mean(feature_similarities)
                    fgis_similarities.append(avg_similarity)
                    
                    results.append({
                        'image_1': f"{celeb1}/{id1}",
                        'image_2': f"{celeb2}/{id2}",
                        'cosine_similarity': avg_similarity,
                        'num_features_compared': len(feature_similarities)
                    })
        
        # compute average similarity
        if len(fgis_similarities) > 0:
            fgis_avg_similarity = np.mean(fgis_similarities)
            print(f"✅ {celeb1} vs {celeb2}: 평균 유사도 = {fgis_avg_similarity:.6f} (비교 수: {len(fgis_similarities)})")
        else:
            print(f"⚠️ {celeb1} vs {celeb2}: 유사도를 계산할 이미지가 없습니다.")
        
        # save result
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{SIMILARITY_RESULTS_DIR}/non_diag/results_{celeb1}_vs_{celeb2}.csv', index=False)
        
        print(f"{'='*60}")
        print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
        print(f"📊 DataFrame shape: {df_results.shape}\n")
        
        return results


def detect_fgis_outliers(celeb_list, output_csv=f"{OUTLIER_RESULTS_DIR}/fgis_outliers.csv"):
    """
    Detect outliers in inter-celebrity FGIS similarity scores using IQR method.
    Based on CLIP's detect_inter_folder_outliers method.
    """
    print("=" * 80)
    print("FGIS OUTLIER DETECTION: Calculating inter-celebrity similarity scores")
    print("=" * 80)
    
    # Load all celebrity embeddings
    celeb_embeddings = {}
    for celeb in tqdm(celeb_list, desc="Loading celebrity embeddings"):
        with open(f"{EMBEDDINGS_DIR}/{celeb}.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        celeb_embeddings[celeb] = embeddings
    
    # Collect all inter-celebrity pair scores
    all_pair_scores = []
    pair_details = []
    
    for celeb1, celeb2 in tqdm(combinations(celeb_list, 2), desc="Processing celebrity pairs for outlier detection"):
        # convert to df
        df1 = pd.DataFrame(celeb_embeddings[celeb1]) 
        df2 = pd.DataFrame(celeb_embeddings[celeb2]) 
        
        # extract feature embedding columns
        feature_cols = [col for col in df1.columns if col not in ['celeb', 'image_id']]
        image_ids1 = df1['image_id'].tolist()
        image_ids2 = df2['image_id'].tolist()
        
        # Calculate all pairwise scores between celeb1 and celeb2
        for id1 in tqdm(image_ids1, desc=f"Processing {celeb1} vs {celeb2}", leave=False):
            for id2 in image_ids2:
                reference_row = df1[df1['image_id'] == id1].iloc[0]
                compare_row = df2[df2['image_id'] == id2].iloc[0]
                
                feature_similarities = []
                
                for feature_col in feature_cols:
                    emb_reference = reference_row[feature_col]
                    emb_compare = compare_row[feature_col]
                    
                    if emb_reference is not None and emb_compare is not None:
                        if isinstance(emb_reference, (list, np.ndarray)) and isinstance(emb_compare, (list, np.ndarray)):
                            emb_reference = np.array(emb_reference)
                            emb_compare = np.array(emb_compare)
                            similarity = cosine_similarity_cuda(emb_reference, emb_compare)
                            feature_similarities.append(similarity)
                
                if len(feature_similarities) > 0:
                    avg_similarity = np.mean(feature_similarities)
                    all_pair_scores.append(avg_similarity)
                    pair_details.append({
                        'image1_id': id1,
                        'image2_id': id2,
                        'celeb1': celeb1,
                        'celeb2': celeb2,
                        'similarity_score': avg_similarity
                    })
    
    # Calculate IQR-based outliers
    scores_array = np.array(all_pair_scores)
    q1 = np.percentile(scores_array, 25)
    q3 = np.percentile(scores_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"\nOutlier Detection Statistics:")
    print(f"Total inter-celebrity pairs: {len(all_pair_scores)}")
    print(f"Q1 (25th percentile): {q1:.4f}")
    print(f"Q3 (75th percentile): {q3:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
    print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")
    
    # Mark outliers
    outlier_data = []
    for detail in pair_details:
        score = detail['similarity_score']
        is_outlier = score < lower_bound or score > upper_bound
        outlier_type = 'none'
        if score < lower_bound:
            outlier_type = 'low'
        elif score > upper_bound:
            outlier_type = 'high'
            
        outlier_data.append({
            'image1_full': f"{detail['celeb1']}/{detail['image1_id']}",
            'image2_full': f"{detail['celeb2']}/{detail['image2_id']}",
            'similarity_score': score,
            'is_outlier': is_outlier,
            'outlier_type': outlier_type,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(outlier_data)
    outlier_count = df['is_outlier'].sum()
    high_outliers = (df['outlier_type'] == 'high').sum()
    low_outliers = (df['outlier_type'] == 'low').sum()
    
    print(f"\nOutlier Results:")
    print(f"Total outliers found: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
    print(f"High outliers (> {upper_bound:.4f}): {high_outliers}")
    print(f"Low outliers (< {lower_bound:.4f}): {low_outliers}")
    
    # Create output directory
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Outlier results saved to: {output_csv}")
    
    # Save only outliers to separate CSV
    outliers_only_df = df[df['is_outlier'] == True]
    outliers_only_csv = output_csv.replace('.csv', '_outliers_only.csv')
    outliers_only_df.to_csv(outliers_only_csv, index=False)
    print(f"Outliers-only results saved to: {outliers_only_csv}")
    print(f"Total outliers saved: {len(outliers_only_df)}")
    
    # Save top 20 highest similarity score outliers (remove duplicates)
    # Create a column for sorting pairs to remove duplicates (A,B) and (B,A)
    outliers_only_df['pair_key'] = outliers_only_df.apply(
        lambda row: tuple(sorted([row['image1_full'], row['image2_full']])), axis=1
    )
    
    # Remove duplicate pairs and get top 20
    unique_outliers = outliers_only_df.drop_duplicates(subset=['pair_key', 'similarity_score'])
    top_20_outliers = unique_outliers.nlargest(20, 'similarity_score')
    
    # Drop the helper column before saving
    top_20_outliers = top_20_outliers.drop('pair_key', axis=1)
    
    top_20_csv = output_csv.replace('.csv', '_top20_high_outliers_unique.csv')
    top_20_outliers.to_csv(top_20_csv, index=False)
    print(f"Top 20 unique highest similarity outliers saved to: {top_20_csv}")
    if len(top_20_outliers) > 0:
        print(f"Highest score: {top_20_outliers['similarity_score'].max():.4f}")
        print(f"Lowest of top 20: {top_20_outliers['similarity_score'].min():.4f}")
    
    return df


def main():
    celeb_list = ['BrunoMars', 'Dicaprio', 'EunwooCha', 'FanBingbing', 'IshiharaSatomi', 'Jennie', 'JKRowling', 'Obama', 'TaylorSwift', 'TomHolland']
    compute_diagonal_sim(celeb_list=celeb_list)
    compute_non_diagonal_sim(celeb_list=celeb_list)


def outlier_detection_main():
    """
    Main function specifically for outlier detection in inter-celebrity FGIS similarity scores.
    """
    celeb_list = ['BrunoMars', 'Dicaprio', 'EunwooCha', 'FanBingbing', 'IshiharaSatomi', 'Jennie', 'JKRowling', 'Obama', 'TaylorSwift', 'TomHolland']
    output_csv = f"{OUTLIER_RESULTS_DIR}/fgis_outliers.csv"
    
    # Run outlier detection
    outlier_df = detect_fgis_outliers(celeb_list, output_csv)
    
    print("\n" + "=" * 80)
    print("FGIS OUTLIER DETECTION COMPLETED")
    print("=" * 80)
    
    return outlier_df
    
if __name__ == "__main__":
    # Run outlier detection instead of similarity matrix calculation
    outlier_detection_main()
