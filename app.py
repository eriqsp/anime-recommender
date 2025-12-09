import streamlit as st
from analysis.recommender import get_recs, find_best_match


# app: https://anime-flow.streamlit.app/
# to run locally: streamlit run app.py


st.set_page_config(page_title="Anime Recommender", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem !important;
    max-width: 1500px;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin-left: auto;
    margin-right: auto;
}

.card-container {
    margin-bottom: 25px;
}

.streaming-logos-container {
    display: flex;
    align-items: center; 
    margin-top: 5px; 
    margin-bottom: 5px; 
}

.anime-card {
    background: #141414;
    border-radius: 12px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    height: 600px;
    width: 100%;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.anime-card * {
    box-sizing: border-box;
}

.anime-card img {
    width: 100%;
    max-height: 350px;
    object-fit: contain;
    border-radius: 8px;
}

.anime-title {
    color: white;
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 10px;
}

.anime-synopsis {
    color: #ccc;
    font-size: 0.9rem;
    margin-top: 6px;

    height: 25%;
    overflow-y: auto;
}

.anime-card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 12px rgba(255,255,255,0.25);
}
</style>
""", unsafe_allow_html=True)


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
            st.markdown('<div class="horizontal-scroll">', unsafe_allow_html=True)

            cols = st.columns(3)
            for i, row in df.iterrows():
                col = cols[i % 3]

                with col:
                    st.markdown(
                        f"""
                        <div class="card-container">
                            <div class="anime-card">
                                <img src="{row['image_url']}" />
                                <div class="anime-title">{row['title']}</div>
                                {row['streaming_logos']}
                                <div class="anime-synopsis">{row['synopsis']}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)
