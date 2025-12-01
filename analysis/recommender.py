from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib


load_dotenv()
filepath = os.getenv('FILEPATH')

animes_df = pd.read_csv(os.path.join(filepath, 'titles.csv'))

titles = animes_df['title'].to_numpy()
all_embeddings = np.load(os.path.join(filepath, 'all_embeddings.npy'))
nbrs = joblib.load(os.path.join(filepath, 'nbrs_cosine.joblib'))


def get_recs(title, k, sort=False) -> pd.DataFrame:
    # find exact index
    try:
        idx = int(np.where(titles == title)[0][0])
    except Exception:
        return pd.DataFrame()
    emb = all_embeddings[idx:idx+1]  # shape (1, dim)
    dists, inds = nbrs.kneighbors(emb, n_neighbors=k+1)
    rec_idx = [i for i in inds[0] if i != idx][:k]

    df = animes_df.iloc[rec_idx]
    if sort:
        df = df.sort_values(by=['average_score'], ascending=False, na_position='last')
    return df
