# 🎬 Movie Recommendation System

A machine learning-based Movie Recommendation System built using both **Collaborative Filtering** and **Content-Based Filtering** techniques. This system also features a clean and interactive **Streamlit web interface**.

Here is the link of my google colab notebook for more help - https://colab.research.google.com/drive/1ticTNv8aOGBBpFhM85gjh50HMnSiF0AP?usp=sharing

---

## 🚀 Features

- 📊 Recommends movies using:
  - **Collaborative Filtering** (based on similar user preferences)
  - **Content-Based Filtering** (similar content/genres)
- 🧠 Personalized recommendations using history
- 🌐 Interactive web interface using **Streamlit**
- 🛠️ Uses **K-Nearest Neighbors** with cosine similarity

---

## 🧰 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Streamlit
- CSV files (movies, ratings, users)

---

## 📁 Dataset Files

- `movies.csv` — Movie IDs, Titles, Genres  
- `ratings.csv` — User Ratings for Movies  
- `users.csv` — User Metadata (optional)

---

## 🧠 Recommendation Approaches

### 1️⃣ Collaborative Filtering

> "Users who liked this movie also liked..."

- Pivot the rating matrix (movies × users)
- Use cosine similarity with **K-Nearest Neighbors**
- Recommend movies based on other users’ preferences

```python
from sklearn.neighbors import NearestNeighbors

# Pivot table from ratings
rating_pivot = ratings.pivot_table(values='rating', columns='userId', index='movieId').fillna(0)

# Train model
model = NearestNeighbors(metric='cosine')
model.fit(rating_pivot)
```

### 2️⃣ Content-Based Filtering

> "You might also like this because it's similar in genre or theme..."

(Currently planned; can be implemented using TF-IDF on movie genres or metadata)

---

## 🧑‍💻 How to Use

```python
# Create an instance
r = Recommender()

# Get recommendations from a movie
r.recommend_on_movie("Toy Story (1995)")

# Get recommendations based on watched history
r.recommend_on_history()
```

---

## 🌐 Streamlit Web App

### 🎯 Features

- Dropdown to select a movie  
- Button to get instant recommendations  
- Easy to expand for history-based suggestions or poster support

### ▶️ How to Run the Streamlit App

1. Install Streamlit:

```bash
pip install streamlit
```

2. Create a file `app.py` with the following content:

```python
import streamlit as st
import pandas as pd
from recommender import Recommender

r = Recommender()

st.title("🎬 Movie Recommendation System")
movie_list = pd.read_csv("movies.csv", sep=";", encoding="latin-1")["title"].tolist()
movie = st.selectbox("Select a movie", movie_list)

if st.button("Get Recommendations"):
    recommendations = r.recommend_on_movie(movie)
    if recommendations:
        st.subheader("Recommended Movies:")
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.warning("No recommendations found.")
```

3. Launch the app:

```bash
streamlit run app.py
```

---

## 🖼️ Visual Guide

### Types of Recommendation Systems  
![Types](https://miro.medium.com/max/998/1*O_GU8xLVlFx8WweIzKNCNw.png)

### Collaborative Filtering Overview  
![Collaborative](https://miro.medium.com/max/1313/1*Qkv3n2Wt9xBmvel_Ee9QGA.png)

### Content-Based Filtering Idea  
![Content](https://miro.medium.com/max/792/1*P63ZaFHlssabl34XbJgong.jpeg)

---

## 📦 Setup Instructions

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

pip install numpy pandas scikit-learn streamlit
```

Ensure the following files are present in your directory:
- `movies.csv`
- `ratings.csv`
- `users.csv`
- `recommender.py`

---

## 📈 Future Plans

- [x] Streamlit Interface  
- [ ] Add posters using TMDb API  
- [ ] Full content-based filtering using TF-IDF  
- [ ] Deploy on Hugging Face Spaces or Streamlit Cloud  
- [ ] Add login system & feedback integration  

---

## 🙋 Author

**Ankit Rewar**  
📫 Connect via GitHub Issues or PRs for suggestions

---

## 📜 License

This project is licensed under the MIT License.
