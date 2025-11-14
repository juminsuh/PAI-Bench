import pickle
import pandas as pd

celeb = "TomHolland"
with open(f"./embedding/face_region_embeddings_{celeb}.pkl", "rb") as f:
    embedding = pickle.load(f)

df = pd.DataFrame(embedding)
df_filter = df[(df['image_id'] == 89) | (df['image_id'] == 90) | (df['image_id'] == 98)]
df_filter.to_pickle(f"./embedding/face_region_embeddings_{celeb}.pkl")