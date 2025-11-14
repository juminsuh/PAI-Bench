import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### outlier detection
def outlier_detection_v1(celeb):
    df = pd.read_csv(f"./similarity_results/diag/results_{celeb}.csv")
    similarities = np.array(df['cosine_similarity']) 

    # Basic statistics
    mean = np.mean(similarities)
    std = np.std(similarities)
    median = np.median(similarities)

    # IQR (Interquartile Range) method
    Q1 = np.percentile(similarities, 25)
    Q3 = np.percentile(similarities, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = (similarities < lower_bound) | (similarities > upper_bound)

    print(f"Mean: {mean:.4f}, Std: {std:.4f}, Median: {median:.4f}")
    print(f"✅ IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"Number of outliers (IQR): {outliers_iqr.sum()}")

    # detected outliers by IQR
    print("\n" + "="*80)
    print("OUTLIERS (IQR method)")
    print("="*80)
    outliers_iqr_df = df[outliers_iqr][['reference_image_id', 'compare_image_id', 'cosine_similarity']]
    print(outliers_iqr_df.to_string(index=False))

    # save as .csv
    outliers_iqr_df.to_csv(f'./analyze/outliers/outliers_iqr_{celeb}.csv', index=False)
    print(f"\nOutliers saved to CSV files!")

### analyze the case where each image is a reference image
def sim_per_id(celeb):
    df = pd.read_csv(f"./embedding/face_region_embeddings_{celeb}_meta.csv")
    image_ids = df['image_id'].tolist()

    df_sim = pd.read_csv(f"./similarity_results/diag/results_{celeb}.csv")
    sim_per_id_results = []
    for id in image_ids:
        df_id = df_sim[(df_sim['reference_image_id'] == id) | (df_sim['compare_image_id'] == id)]
        similarities = df_id['cosine_similarity'].tolist()
        avg_similarity = np.mean(similarities)
        sim_per_id_results.append(
            {"celeb": celeb,
            "reference_image_id": id,
            "avg_similarity": avg_similarity}
        )

    # save to .csv
    sim_per_id_results_df = pd.DataFrame(sim_per_id_results)
    sim_per_id_results_df.to_csv(f"./analyze/sim_per_id/sim_per_id_{celeb}.csv")

### outlier detection for individual reference image
def outlier_detection_v2(celeb):
    df = pd.read_csv(f"./analyze/sim_per_id/sim_per_id_{celeb}.csv")
    similarities = df['avg_similarity']
    
    # IQR (Interquartile Range) method
    Q1 = np.percentile(similarities, 25)
    Q3 = np.percentile(similarities, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = (similarities < lower_bound) | (similarities > upper_bound)

    print(f"IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"Number of outliers (IQR): {outliers_iqr.sum()}")

    # detected outliers by IQR
    print("\n" + "="*80)
    print("OUTLIERS (IQR method)")
    print("="*80)
    outliers_iqr_df = df[outliers_iqr][['reference_image_id', 'avg_similarity']]
    print(outliers_iqr_df.to_string(index=False))

    # save as .csv
    outliers_iqr_df.to_csv(f'./analyze/outliers_per/outliers_iqr_{celeb}.csv', index=False)
    print(f"\nOutliers saved to CSV files!")
    
def main():
    celeb = "BrunoMars"
    outlier_detection_v1(celeb=celeb)
    sim_per_id(celeb=celeb)
    outlier_detection_v2(celeb=celeb)

if __name__ == "__main__":
    main()
