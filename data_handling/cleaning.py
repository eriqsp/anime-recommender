import pandas as pd
import re
import html
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataCleaner:
    def __init__(self, filename: str, add_numerical: bool = False):
        self.df = pd.read_parquet(filename)
        self.add_numerical = add_numerical

    def final_stage_df(self):
        self._first_stage_df()
        self._second_stage_df()

        self.df['text'] = self.df['text'].astype(str).str.encode('latin-1', errors='replace').str.decode('utf-8', errors='replace')

        cols = ['item_id', 'title', 'text', 'genres_multi', 'tags_multi', 'studios_multi']
        aux_cols = ['image_url', 'average_score', 'synopsis']
        if self.add_numerical:
            cols += ['episodes', 'popularity']
        return self.df[cols + aux_cols]

    # transform values into multi-hot vectors; multi-hot vectors are more useful when dealing with neural networks later
    def _second_stage_df(self):
        cols_to_multihot = ['genres', 'tags', 'studios']
        for col in cols_to_multihot:
            all_values = Counter()
            self.df[f"{col}_list"] = self.df[col].apply(self._split_vals)
            for v_list in self.df[f"{col}_list"]:
                all_values.update(v_list)
            col2id = {c: i for i, (c, _) in enumerate(all_values.most_common())}
            num_values = len(col2id)

            self.df[f'{col}_multi'] = self.df[f'{col}_list'].apply(lambda x: self._col_to_multihot(x, num_values, col2id))

    @staticmethod
    def _col_to_multihot(vlist, num_values, col2id):
        arr = np.zeros(num_values, dtype=int)
        for g in vlist:
            if g in col2id:
                arr[col2id[g]] = 1
        return arr

    @staticmethod
    def _split_vals(v):
        if pd.isna(v) or v == "":
            return []
        if isinstance(v, list):
            return v
        return str(v).split("|")

    # clean dataframe and modify numerical and categorical features
    def _first_stage_df(self):
        self.df = self.df.drop_duplicates(subset="item_id")
        self.df = self.df.dropna(subset=['title'])
        self.df["synopsis"] = self.df["synopsis"].fillna("")

        if self.add_numerical:
            num_cols = ["popularity", "episodes"]
            for c in num_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0)

            scaler = MinMaxScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

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
