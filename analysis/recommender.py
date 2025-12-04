from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib
from rapidfuzz import process, fuzz


load_dotenv()
filepath = os.getenv('FILEPATH')

animes_df = pd.read_csv(os.path.join(filepath, 'titles.csv'))

titles = animes_df['title'].to_numpy()
titles_list = list(titles)
titles_lower = [t.lower() for t in titles_list]
lower_to_index = {t.lower(): i for i, t in enumerate(titles_list)}

all_embeddings = np.load(os.path.join(filepath, 'all_embeddings.npy'))
nbrs = joblib.load(os.path.join(filepath, 'nbrs_cosine.joblib'))


def get_recs(title, k, sort=False) -> pd.DataFrame:
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
    return df.reset_index(drop=True)


def find_best_match(user_query: str):
    q = user_query.strip()
    if not q:
        return None

    q_lower = q.lower()

    if q_lower in lower_to_index:
        idx = lower_to_index[q_lower]
        return titles_list[idx]

    # try to find match using rapidfuzz if exact match doesn't work
    matches = process.extract(q_lower, titles_lower, scorer=fuzz.WRatio, limit=1)
    if matches and matches[0][1] > 60:  # score out of 100
        cand = matches[0][0]
        idx = titles_lower.index(cand)
        return titles_list[idx]

    # as a last resort, try matching against canonical titles directly
    matches = process.extract(q, titles_list, scorer=fuzz.WRatio, limit=1)
    if matches and matches[0][1] > 60:
        cand = matches[0][0]
        idx = titles_list.index(cand)
        return titles_list[idx]

    return None
