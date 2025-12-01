import pandas as pd
import streamlit as st
from analysis.recommender import get_recs


# to run: streamlit run C:/Users/Erick/PycharmProjects/anime-recommender/app.py
# TODO: improve search engine: let case insensitive; add contains
# TODO: enhance app layout: colors, font, etc
# TODO: add streaming logo that indicates where to watch the show
# TODO: deployment


st.title("Anime Recommender")

query = st.text_input("Type an anime title", "")
n = st.slider("How many recommendations", 1, 20, 5)


if st.button("Recommend"):
    try:
        df = get_recs(query, n)
        for idx, row in df.iterrows():
            if not pd.isnull(row.get('average_score')):
                subheader = f"{row['title']} — score {row.get('average_score', ''):.0f}"
            else:
                subheader = f"{row['title']}"
            st.subheader(subheader)
            if row.get("image_url"):
                st.image(row["image_url"], width=200)
            st.markdown(row.get("synopsis", ""), unsafe_allow_html=True)
    except Exception as e:
        st.error("No results")
