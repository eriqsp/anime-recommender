import requests
import time
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()

API_URL = "https://graphql.anilist.co"

query = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { currentPage, lastPage, hasNextPage }
    media(type: ANIME) {
      id
      title { romaji english native }
      description
      genres
      episodes
      format
      averageScore
      popularity
      coverImage { large }
      studios { nodes { name } }
      tags { name }
      seasonYear
      source
    }
  }
}
"""


def fetch_page(page, per_page=50):
    variables = {"page": page, "perPage": per_page}
    r = requests.post(API_URL, json={"query": query, "variables": variables})
    r.raise_for_status()
    return r.json()


def normalize_media(m):
    return {
        "item_id": f"anilist_{m.get('id')}",
        "title": m.get("title", {}).get("romaji") or m.get("title", {}).get("english"),
        "synopsis": m.get("description"),
        "genres": "|".join(m.get("genres", [])),
        "episodes": m.get("episodes"),
        "format": m.get("format"),
        "average_score": m.get("averageScore"),
        "popularity": m.get("popularity"),
        "image_url": m.get("coverImage", {}).get("large"),
        "studios": "|".join([s.get("name") for s in m.get("studios", {}).get("nodes", [])]) if m.get("studios") else "",
        "tags": "|".join([t.get("name") for t in m.get("tags", [])]) if m.get("tags") else "",
        "season_year": m.get("seasonYear"),
        "source": m.get("source"),
    }


all_records = []
page = 1
while True:
    data = fetch_page(page, per_page=50)
    media = data["data"]["Page"]["media"]
    for m in media:
        record = normalize_media(m)
        all_records.append(record)
        # print(record)
    page_info = data["data"]["Page"]["pageInfo"]
    print("Fetched page", page)
    if not page_info["hasNextPage"]:
        break
    page += 1
    time.sleep(3)

df = pd.DataFrame(all_records)
df.to_parquet(os.getenv('PARQUET_FILENAME'), index=False)
