"""
FGIS scorer for celeb face angle dataset with outlier detection for negative pairs
Based on the CLIP outlier detection approach adapted for FGIS embeddings
"""

import pandas as pd
import numpy as np
from itertools import combinations
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import os


class FGISOutlierDetector:
    def __init__(self):
        pass
    
    def load_embeddings(self, celeb_name, embeddings_dir):
        """Load FGIS embeddings for a celebrity"""
        embeddings_path = f"{embeddings_dir}/{celeb_name}.pkl"
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        return pd.DataFrame(embeddings)
    
    def calculate_fgis_similarity(self, row1, row2, feature_cols):
        """
        Calculate FGIS similarity between two images using their region embeddings
        Based on the method from /home/jiyoon/PAI-Bench/sim_experiments/251117_negative_outliers/fgis.py
        """
        feature_similarities = []
        
        for feature_col in feature_cols:
            emb1 = row1[feature_col]
            emb2 = row2[feature_col]
            
            # Compute similarity if and only if both embeddings are not None
            if emb1 is not None and emb2 is not None:
                if isinstance(emb1, (list, np.ndarray)) and isinstance(emb2, (list, np.ndarray)):
                    emb1 = np.array(emb1).reshape(1, -1)
                    emb2 = np.array(emb2).reshape(1, -1)
                    similarity = cosine_similarity(emb1, emb2)[0][0]
                    feature_similarities.append(similarity)
        
        # Average similarity between the two images
        if len(feature_similarities) > 0:
            return np.mean(feature_similarities)
        else:
            return None
    
    def detect_inter_celeb_outliers(self, embeddings_dir, celeb_list, output_csv="fgis_outliers.csv"):
        """
        Detect outliers in inter-celebrity similarity scores using IQR method.
        Adapted from CLIP's detect_inter_folder_outliers method.
        """
        print("=" * 80)
        print("FGIS OUTLIER DETECTION: Loading embeddings and calculating individual pair scores")
        print("=" * 80)
        
        # Load all celebrity embeddings
        celeb_embeddings = {}
        for celeb in celeb_list:
            try:
                df = self.load_embeddings(celeb, embeddings_dir)
                celeb_embeddings[celeb] = df
                print(f"Loaded {celeb}: {len(df)} images")
            except Exception as e:
                print(f"Error loading {celeb}: {e}")
                continue
        
        # Collect all inter-celebrity pair scores
        all_pair_scores = []
        pair_details = []
        
        n = len(celeb_list)
        
        for i in range(n):
            celeb1 = celeb_list[i]
            if celeb1 not in celeb_embeddings:
                continue
                
            df1 = celeb_embeddings[celeb1]
            feature_cols = [col for col in df1.columns if col not in ['celeb', 'image_id']]
            
            for j in range(n):
                if i == j:  # Skip intra-celebrity (diagonal) comparisons
                    continue
                    
                celeb2 = celeb_list[j]
                if celeb2 not in celeb_embeddings:
                    continue
                    
                df2 = celeb_embeddings[celeb2]
                
                # Calculate all pairwise scores between celeb1 and celeb2
                for _, row1 in df1.iterrows():
                    for _, row2 in df2.iterrows():
                        try:
                            score = self.calculate_fgis_similarity(row1, row2, feature_cols)
                            if score is not None:
                                all_pair_scores.append(score)
                                pair_details.append({
                                    'image1_id': row1['image_id'],
                                    'image2_id': row2['image_id'],
                                    'celeb1': celeb1,
                                    'celeb2': celeb2,
                                    'similarity_score': score
                                })
                        except Exception as e:
                            print(f"Error processing pair {row1['image_id']} vs {row2['image_id']}: {e}")
                            continue
        
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
        
        # Save to CSV
        try:
            df.to_csv(output_csv, index=False)
            print(f"Outlier results saved to: {output_csv}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            print(f"Output path: {output_csv}")
            print(f"DataFrame shape: {df.shape}")
            raise
        
        # Save only outliers to separate CSV
        outliers_only_df = df[df['is_outlier'] == True]
        outliers_only_csv = output_csv.replace('.csv', '_outliers_only.csv')
        try:
            outliers_only_df.to_csv(outliers_only_csv, index=False)
            print(f"Outliers-only results saved to: {outliers_only_csv}")
            print(f"Total outliers saved: {len(outliers_only_df)}")
        except Exception as e:
            print(f"Error saving outliers-only CSV: {e}")
            raise
        
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
        try:
            top_20_outliers.to_csv(top_20_csv, index=False)
            print(f"Top 20 unique highest similarity outliers saved to: {top_20_csv}")
            if len(top_20_outliers) > 0:
                print(f"Highest score: {top_20_outliers['similarity_score'].max():.4f}")
                print(f"Lowest of top 20: {top_20_outliers['similarity_score'].min():.4f}")
        except Exception as e:
            print(f"Error saving top 20 unique outliers CSV: {e}")
            raise
        
        return df


def main():
    """
    Main function for FGIS outlier detection in inter-celebrity similarity scores.
    """
    # Configuration
    embeddings_dir = "/data2/jiyoon/PAI-Bench/data/crawled/fgis/embeddings"
    output_csv = "/home/jiyoon/PAI-Bench/fgis_score/outlier_results/fgis_outliers.csv"
    
    # Create output directory
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Celebrity list
    celeb_list = ['BrunoMars', 'Dicaprio', 'EunwooCha', 'FanBingbing', 
                  'IshiharaSatomi', 'Jennie', 'JKRowling', 'Obama', 
                  'TaylorSwift', 'TomHolland']
    
    # Create detector
    detector = FGISOutlierDetector()
    
    # Run outlier detection
    outlier_df = detector.detect_inter_celeb_outliers(
        embeddings_dir, celeb_list, output_csv
    )
    
    print("\n" + "=" * 80)
    print("FGIS OUTLIER DETECTION COMPLETED")
    print("=" * 80)
    
    return outlier_df


if __name__ == "__main__":
    main()