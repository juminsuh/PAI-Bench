import pandas as pd
import numpy as np
from itertools import combinations
import pickle
from sklearn.metrics.pairwise import cosine_similarity

### compute diagonal similarities
def compute_diagonal_sim(celeb_list):
    for celeb in celeb_list:
        results = []
        
        # open .pkl file
        with open(f'/data2/jiyoon/PAI-Bench/data/crawled/fgis/embeddings/{celeb}.pkl', 'rb') as f: # original image
            embeddings = pickle.load(f)
        
        # convert to df
        df = pd.DataFrame(embeddings)  
        
        # extract feature embedding columns (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
        feature_cols = [col for col in df.columns if col not in ['celeb', 'image_id']]
        image_ids = df['image_id'].tolist()
        fgis_similarities = [] # list to save nC2 fgis scores
        
        for reference_id, compare_id in combinations(image_ids, 2):
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
                        emb_reference = np.array(emb_reference).reshape(1, -1)
                        emb_compare = np.array(emb_compare).reshape(1, -1)
                        similarity = cosine_similarity(emb_reference, emb_compare)[0][0] # similarity between specific regions
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
        df_results.to_csv(f'./similarity_results/diag/results_{celeb}.csv', index=False)

        print(f"\n{'='*60}")
        print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
        print(f"📊 DataFrame shape: {df_results.shape}")

### compute non-diagonal similarities
def compute_non_diagonal_sim(celeb_list):
    celeb_embeddings = {}
    for celeb in celeb_list:
        with open(f"/data2/jiyoon/PAI-Bench/data/crawled/fgis/embeddings/{celeb}.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        celeb_embeddings[celeb] = embeddings
        
    for celeb1, celeb2 in combinations(celeb_list, 2):
        
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
        for id1 in image_ids1:
            for id2 in image_ids2:
                reference_row = df1[df1['image_id'] == id1].iloc[0]
                compare_row = df2[df2['image_id'] == id2].iloc[0]
                
                feature_similarities = [] # list to save a single fgis score
                
                for feature_col in feature_cols:
                    emb_reference = reference_row[feature_col]
                    emb_compare = compare_row[feature_col]
                    
                    if emb_reference is not None and emb_compare is not None:
                        if isinstance(emb_reference, (list, np.ndarray)) and isinstance(emb_compare, (list, np.ndarray)):
                            emb_reference = np.array(emb_reference).reshape(1, -1)
                            emb_compare = np.array(emb_compare).reshape(1, -1)
                            similarity = cosine_similarity(emb_reference, emb_compare)[0][0]
                            feature_similarities.append(similarity)
                
                if len(feature_similarities) > 0:
                    avg_similarity = np.mean(feature_similarities)
                    fgis_similarities.append(avg_similarity)
                    
                    results.append({
                        'celeb1': celeb1,
                        'celeb2': celeb2,
                        'image_id1': id1,
                        'image_id2': id2,
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
        df_results.to_csv(f'./similarity_results/non_diag/results_{celeb1}_vs_{celeb2}.csv', index=False)
        
        print(f"{'='*60}")
        print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
        print(f"📊 DataFrame shape: {df_results.shape}\n")
        
def main():
    celeb_list = ['BrunoMars', 'Dicaprio', 'EunwooCha', 'FanBingbing', 'IshiharaSatomi', 'Jennie', 'JKRowling', 'Obama', 'TaylorSwift', 'TomHolland']
    compute_diagonal_sim(celeb_list=celeb_list)
    compute_non_diagonal_sim(celeb_list=celeb_list)
    
if __name__ == "__main__":
    main()
