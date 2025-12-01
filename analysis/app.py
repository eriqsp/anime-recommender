import streamlit as st
from recommender import get_recs


st.title("Anime Recommender")

query = st.text_input("Type an anime title", "")
n = st.slider("How many recommendations", 1, 20, 5)


if st.button("Recommend"):
    try:
        # recs = get_recs(query, n)
        df = get_recs(query, n)
        for idx, row in df.iterrows():
            st.subheader(f"{row['title']} — score {row.get('average_score', ''):.3f}")
            if row.get("image_url"):
                st.image(row["image_url"], width=120)
            st.write(row.get("synopsis", "")[:300])
    except Exception as e:
        st.error(f"No results: {e}")
