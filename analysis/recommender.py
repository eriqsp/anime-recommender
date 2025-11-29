from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib


load_dotenv()
filepath = os.getenv('FILEPATH')

titles_df = pd.read_csv(os.path.join(filepath, 'titles.csv'))

titles = titles_df['title'].to_numpy()
all_embeddings = np.load(os.path.join(filepath, 'all_embeddings.npy'))
nbrs = joblib.load(os.path.join(filepath, 'nbrs_cosine.joblib'))


def recommend_by_title(title, k=10):
    # find exact index
    try:
        idx = int(np.where(titles == title)[0][0])
    except Exception:
        print("Title not found in dataset.")
        return []
    emb = all_embeddings[idx:idx+1]  # shape (1, dim)
    dists, inds = nbrs.kneighbors(emb, n_neighbors=k+1)  # includes self
    rec_idx = [i for i in inds[0] if i != idx][:k]
    return [(int(i), titles[int(i)]) for i in rec_idx]


# Example
anime = 'Jujutsu Kaisen'
print(f"Recommendations for {anime}:", recommend_by_title(anime, k=10))
