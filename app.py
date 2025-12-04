import pandas as pd
import streamlit as st
from analysis.recommender import get_recs, find_best_match


# to run: streamlit run C:/Users/Erick/PycharmProjects/anime-recommender/app.py
# TODO: enhance app layout: colors, font, etc
# TODO: add streaming logo that indicates where to watch the show
# TODO: deployment


st.title("Anime Recommender")

query = st.text_input("Type an anime title", key='search', value="", placeholder='Type e.g. "naruto" then press Enter')
k = st.number_input("How many recommendations", min_value=1, max_value=50, value=10, step=1)


if st.button("Search") or (query and st.session_state.get('submitted', False) is False):
    matched_title = find_best_match(query)

    if matched_title is None:
        st.warning("No close match found. Try a different spelling, or check examples in the sidebar.")
    else:
        # if fuzzy-corrected, show the message
        canonical_display = matched_title
        if query.strip().upper() != canonical_display.upper():
            st.info(f"Showing recommendations for **{canonical_display}** instead of **{query.strip()}**.")

        # call the recommender
        df = get_recs(canonical_display, k=int(k))
        if df.empty:
            st.warning("No recommendations returned by get_recs().")
        else:
            for idx, row in df.iterrows():
                st.markdown("<br></br>", unsafe_allow_html=True)
                with st.container():
                    st.markdown(
                        f"""
                        <div style="padding:15px; background:#1A1D23; border-radius:12px; margin-bottom:20px; border:1px solid #333;">
                            <h3 style="margin-bottom:10px;">{row['title']}</h3>
                        """,
                        unsafe_allow_html=True
                    )

                    cols = st.columns([1, 3])
                    with cols[0]:
                        if row.get("image_url"):
                            st.image(row["image_url"], width=150)
                    with cols[1]:
                        st.markdown(row.get("synopsis", ""), unsafe_allow_html=True)


# TODO: fix examples side bar; works only at first interaction
# st.sidebar.header("Try examples")
# examples = ["NARUTO", "Shingeki no Kyojin", "Fullmetal Alchemist"]
# for ex in examples:
#     if st.sidebar.button(ex):
#         matched_title = find_best_match(ex)
#         df = get_recs(matched_title, k=10)
#         st.write(f"### Recommendations for {matched_title}")
#         for idx, row in df.iterrows():
#             if not pd.isnull(row.get('average_score')):
#                 subheader = f"{row['title']} — score {row.get('average_score', ''):.0f}"
#             else:
#                 subheader = f"{row['title']}"
#             st.subheader(subheader)
#             if row.get("image_url"):
#                 st.image(row["image_url"], width=200)
#             st.markdown(row.get("synopsis", ""), unsafe_allow_html=True)
