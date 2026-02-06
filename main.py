import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load files
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)

st.set_page_config(page_title="Recommendation System", layout="centered")

st.title("🔍 Movies Recommendation System")
st.write("Select an item and get similar recommendations")

# Dropdown
title = st.selectbox("Choose an item", df['title'].values)

# Recommendation logic
def recommend(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    item_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[item_indices]

if st.button("Recommend"):
    results = recommend(title)
    st.subheader("✨ Recommended Items")
    for i, r in enumerate(results, 1):
        st.write(f"{i}. {r}")

st.markdown("---")
st.caption("Built with Streamlit & TF-IDF")
