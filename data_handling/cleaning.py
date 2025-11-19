import pandas as pd
import re
import html
from dotenv import load_dotenv
import os


class DataCleaner:
    def __init__(self, filename: str):
        self.df = pd.read_parquet(filename)

        self.cols_to_exclude = ['item_id', 'episodes', 'format', 'average_score', 'popularity', 'image_url', 'studios', 'season_year', 'source']

    def first_stage_df(self):
        self.df = self.df.drop_duplicates(subset="item_id")
        self.df = self.df.dropna(subset=['title'])
        self.df["synopsis"] = self.df["synopsis"].fillna("")
        self.df = self.df.drop(self.cols_to_exclude, axis=1)

        self.df["text"] = self.df.apply(self._make_text, axis=1)
        self.df["text"] = self.df["text"].astype(str).map(self._clean_text)

    @staticmethod
    def _join_list_field(val):
        if pd.isna(val) or val == "":
            return ""
        if isinstance(val, list):
            return "|".join(val)
        return str(val)

    def _make_text(self, row):
        parts = [str(row["title"])]
        if row.get("synopsis"):
            parts.append(str(row["synopsis"]))
        if row.get("genres"):
            parts.append("Genres: " + self._join_list_field(row["genres"]))
        if row.get("tags"):
            parts.append("Tags: " + self._join_list_field(row["tags"]))
        if row.get("studios"):
            parts.append("Studio: " + self._join_list_field(row["studios"]))
        return " . ".join([p for p in parts if p])

    @staticmethod
    def _clean_text(s):
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


load_dotenv()

dc = DataCleaner(os.getenv('PARQUET_FILENAME'))
dc.first_stage_df()


print()

