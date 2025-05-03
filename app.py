import streamlit as st
from recommendation import Recommender, movies

# Initialize Recommender object
recommender = Recommender()

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

# Movie selector
movie_titles = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("Pick a movie you like:", movie_titles)

# Recommendation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🎯 Recommend Based on Selected Movie"):
        recommendations = recommender.recommend_on_movie(selected_movie)
        st.subheader("Top Recommendations:")
        
        for i, title in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {title}**")