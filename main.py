import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Movie Recommender",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    df = joblib.load("df.joblib")
    indices = joblib.load("indices.joblib")
    tfidf_matrix = joblib.load("tfidf_matrix.joblib")
    return df, indices, tfidf_matrix

df, indices, tfidf_matrix = load_data()

# ================= TMDB CONFIG =================
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# ================= STYLES =================
st.markdown("""
<style>
html, body {
    background-color: #0b0b0b;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    text-align: center;
    padding: 40px 20px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e50914, #f40612);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #b3b3b3;
    font-size: 1.1rem;
}

.stTextInput input {
    background-color: #141414;
    color: white;
    border-radius: 10px;
    border: 1px solid #333;
    padding: 12px;
}

.stButton>button {
    background: linear-gradient(90deg, #e50914, #f40612);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.recommendation-card {
    background: linear-gradient(145deg, #141414, #1c1c1c);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    transition: all 0.3s ease;
}
.recommendation-card:hover {
    transform: scale(1.08);
}

.movie-title {
    font-size: 1.1rem;
    font-weight: 700;
}
.movie-genre {
    font-size: 0.8rem;
    color: #b3b3b3;
}
.similarity {
    font-size: 0.75rem;
    color: #46d369;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>🎬 Smart Movie Recommender</h1>
    <p>Discover movies you’ll love — powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ================= SEARCH =================
search = st.text_input("🔍 Search a movie")

selected_movie = None
if search:
    filtered = df[df["title"].str.contains(search, case=False, na=False)]
    if not filtered.empty:
        selected_movie = st.selectbox(
            "Matching results",
            filtered["title"].values
        )

# ================= POSTER FUNCTION =================
@st.cache_data(show_spinner=False)
def get_movie_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": False
        }
        r = requests.get(url, params=params, timeout=5).json()

        if r.get("results"):
            poster_path = r["results"][0].get("poster_path")
            if poster_path:
                return POSTER_BASE_URL + poster_path
    except:
        pass

    return "https://via.placeholder.com/500x750?text=No+Poster"

# ================= RECOMMENDER =================
def recommend(title, top_n=6):
    idx = indices[title]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    scores = sorted(
        list(enumerate(cosine_sim)),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    movie_indices = [i[0] for i in scores]
    similarities = [round(s[1] * 100, 2) for s in scores]

    result_df = df.iloc[movie_indices].copy()
    result_df["similarity"] = similarities

    return result_df

# ================= BUTTON =================
results = None
if selected_movie and st.button("🎯 Get Recommendations"):
    results = recommend(selected_movie)

# ================= DISPLAY =================
if results is not None:
    st.subheader("🔥 Recommended for you")

    cols = st.columns(3)
    for i, row in results.iterrows():
        poster = get_movie_poster(row["title"])

        with cols[i % 3]:
            st.markdown(f"""
                <div class="recommendation-card">
                    <img src="{poster}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row.get('genres','General')}</div>
                    <div class="similarity">🎯 Match: {row['similarity']}%</div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("More like this", key=row["title"]):
                results = recommend(row["title"])
                st.experimental_rerun()
               