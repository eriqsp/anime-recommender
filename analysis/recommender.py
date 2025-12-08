from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib
from rapidfuzz import process, fuzz


streaming_logo_map = {
    "Crunchyroll": "https://upload.wikimedia.org/wikipedia/commons/7/75/Cib-crunchyroll_%28CoreUI_Icons_v1.0.0%29_orange.svg",
    "Netflix": "https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg",
    "Hulu": "https://upload.wikimedia.org/wikipedia/commons/5/50/Hulu_logo_%282017%29.svg",
    "Funimation": "https://upload.wikimedia.org/wikipedia/commons/4/47/Funimation_2016.svg",
    "Prime Video": "https://upload.wikimedia.org/wikipedia/commons/1/11/Amazon_Prime_Video_logo.svg",
    "Disney+": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Disney_plus_icon.png",
    "HIDIVE": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Hidive_text_logo.svg"
}

load_dotenv()
filepath = os.getenv('FILEPATH')

animes_df = pd.read_csv(os.path.join(filepath, 'titles.csv'))

titles = animes_df['title'].to_numpy()
titles_list = list(titles)
titles_lower = [t.lower() for t in titles_list]
lower_to_index = {t.lower(): i for i, t in enumerate(titles_list)}

all_embeddings = np.load(os.path.join(filepath, 'all_embeddings.npy'))
nbrs = joblib.load(os.path.join(filepath, 'nbrs_cosine.joblib'))


# get recommendations
def get_recs(title, k, sort=False) -> pd.DataFrame:
    try:
        idx = int(np.where(titles == title)[0][0])
    except Exception:
        return pd.DataFrame()
    emb = all_embeddings[idx:idx+1]  # shape (1, dim)
    dists, inds = nbrs.kneighbors(emb, n_neighbors=k+1)
    rec_idx = [i for i in inds[0] if i != idx][:k]

    df = animes_df.iloc[rec_idx]

    if 'streaming' in df.columns:
        df['streaming_logos'] = df['streaming'].apply(generate_streaming_html)
    else:
        df['streaming_logos'] = "<p></p>"

    if sort:
        df = df.sort_values(by=['average_score'], ascending=False, na_position='last')
    return df.reset_index(drop=True)


# find best match on search engine
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


# it creates the html element for the streaming logos, for each anime
def generate_streaming_html(service_string, logo_size=20):
    if pd.isnull(service_string):
        return "<p></p>"

    logo_html_parts = []
    service_pairs = service_string.split(',')

    for pair in service_pairs:
        if '|' not in pair:
            continue

        stream_name, stream_url = pair.split('|', 1)

        if stream_name in streaming_logo_map:
            logo_url = streaming_logo_map[stream_name]

            logo_html = (
                f'<a href="{stream_url}" target="_blank" style="margin-right: 5px;">'
                f'<img src="{logo_url}" alt="{stream_name} Logo" height="{logo_size}" style="border-radius: 4px;"/>'
                f'</a>'
            )
            logo_html_parts.append(logo_html)

    logos_string = "".join(logo_html_parts)
    return f'<div class="streaming-logos-container">{logos_string}</div>'
