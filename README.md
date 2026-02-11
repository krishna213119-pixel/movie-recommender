# 🎬 Movie Recommender System

A Content-Based Movie Recommendation System built using Machine Learning and deployed with Streamlit.

[![Live Demo](https://img.shields.io/badge/Live-App-brightgreen?style=for-the-badge)](https://movie-recommender-jymqnushfvnnxgticccj2x.streamlit.app/)

---

## 🚀 Live Application
👉 https://movie-recommender-jymqnushfvnnxgticccj2x.streamlit.app/

---

## 📌 Features
- Content-Based Filtering
- TF-IDF Vectorization
- Cosine Similarity
- Interactive Streamlit UI
- Instant Movie Recommendations

---

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## ⚙️ How It Works
1. Movie overviews are converted into numerical vectors using TF-IDF.
2. Cosine similarity is calculated between movies.
3. The system recommends movies with the highest similarity scores.

---

## 📂 Project Structure
movie-recommender/
│── app.py  
│── df.joblib  
│── tfidf.joblib  
│── tfidf_matrix.joblib  
│── indices.joblib  
│── requirements.txt  
│── README.md  

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author
Krishna
