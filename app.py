# ===========================
# Movie Recommendation System - Streamlit App
# ===========================
# This Streamlit application provides three recommendation methods:
# 1. Content-Based: Recommends movies similar to the input movie based on genres
# 2. Collaborative: Recommends movies based on user ratings using SVD algorithm
# 3. Hybrid: Combines both content-based and collaborative filtering

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# LOAD DATA AND MODELS (Cached for performance)
# ===========================
@st.cache_resource
def load_models_and_data():
    """
    Load all required data files and trained models.
    Results are cached to avoid reloading on every interaction.
    """
    project_dir = Path(__file__).resolve().parent
    movie_candidates = [
        project_dir / 'movies_final.csv',
        project_dir / 'movies_cleaned.csv',
    ]
    ratings_path = project_dir / 'ratings_cleaned.csv'
    svd_path = project_dir / 'svd_model.pkl'
    cosine_path = project_dir / 'cosine_sim.pkl'
    indices_path = project_dir / 'movie_indices.pkl'

    movies_path = next((path for path in movie_candidates if path.exists()), None)

    missing_files = []
    if movies_path is None:
        missing_files.append('movies_final.csv or movies_cleaned.csv')
    if not ratings_path.exists():
        missing_files.append('ratings_cleaned.csv')
    if not svd_path.exists():
        missing_files.append('svd_model.pkl')

    if missing_files:
        st.error('❌ Missing required file(s): ' + ', '.join(missing_files))
        st.markdown("""
        ### 📥 What to copy from Colab

        Your notebook exports these files for the Streamlit app:
        - `movies_final.csv` or `movies_cleaned.csv`
        - `ratings_cleaned.csv`
        - `svd_model.pkl`
        - `cosine_sim.pkl` and `movie_indices.pkl` are optional but supported

        Put the exported files in this folder and rerun the app:
        `c:\\Users\\tejam\\Documents\\Projects on 6th\\ML Pro\\Movie recommendation\\`
        """)
        st.stop()

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    svd_model = joblib.load(svd_path)

    if 'genres_clean' not in movies_df.columns:
        movies_df['genres_clean'] = movies_df['genres'].fillna('').replace('(no genres listed)', '')
        movies_df['genres_clean'] = movies_df['genres_clean'].str.replace('|', ' ', regex=False)
    if 'genres' not in movies_df.columns and 'genres_clean' in movies_df.columns:
        movies_df['genres'] = movies_df['genres_clean']

    if cosine_path.exists():
        cosine_sim = joblib.load(cosine_path)
    else:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['genres_clean'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    if indices_path.exists():
        movie_indices = joblib.load(indices_path)
    else:
        movie_indices = pd.Series(movies_df.index, index=movies_df['title'].fillna('')).drop_duplicates()

    return movies_df, ratings_df, svd_model, cosine_sim, movie_indices

# Load data
movies_df, ratings_df, svd_model, cosine_sim, movie_indices = load_models_and_data()

# ===========================
# RECOMMENDATION FUNCTIONS
# ===========================

def get_movie_index(movie_name):
    """
    Get the index of a movie by name (case-insensitive search).
    Returns the index if found, else returns None.
    """
    movie_name = movie_name.strip()
    if not movie_name:
        return None

    if isinstance(movie_indices, pd.Series):
        matches = movie_indices.index[movie_indices.index.astype(str).str.lower() == movie_name.lower()]
        if len(matches) > 0:
            return int(movie_indices.loc[matches[0]])

    movie_list = movies_df[movies_df['title'].astype(str).str.lower() == movie_name.lower()]
    if not movie_list.empty:
        return movie_list.index[0]
    return None

def content_based_recommendations(movie_name, n_recommendations=5):
    """
    Content-Based Filtering: Recommends movies similar to the input movie
    based on genre similarity using cosine similarity.
    """
    movie_idx = get_movie_index(movie_name)
    
    if movie_idx is None:
        return None, "❌ Movie not found in database"
    
    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    
    # Sort by similarity score (descending) and exclude the input movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movies
    recommendations = movies_df.iloc[movie_indices][['movieId', 'title', 'genres']]
    return recommendations, None

def collaborative_recommendations(user_id, n_recommendations=5):
    """
    Collaborative Filtering: Recommends movies based on the user's rating patterns
    and similar users using the trained SVD model.
    """
    # Check if user exists in the dataset
    if user_id not in ratings_df['userId'].values:
        return None, f"❌ User ID {user_id} not found in database"
    
    # Get movies the user has already rated
    user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
    
    # Get all movies
    all_movies = movies_df['movieId'].values
    
    # Get unrated movies
    unrated_movies = [m for m in all_movies if m not in user_rated_movies]
    
    if len(unrated_movies) == 0:
        return None, "⚠️ User has rated all movies"
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies[:100]:  # Limit to speed up computation
        predicted_rating = svd_model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    
    # Sort by predicted rating (descending) and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [pred[0] for pred in predictions[:n_recommendations]]
    
    # Return recommended movies
    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']]
    return recommendations, None

def hybrid_recommendations(movie_name, user_id, n_recommendations=5, content_weight=0.5):
    """
    Hybrid Recommendation: Combines content-based and collaborative filtering.
    Blends both methods using weighted scores.
    """
    # Get content-based recommendations
    content_recs, content_error = content_based_recommendations(movie_name, n_recommendations * 2)
    if content_error:
        return None, content_error
    
    # Get collaborative recommendations
    collab_recs, collab_error = collaborative_recommendations(user_id, n_recommendations * 2)
    if collab_error:
        return None, collab_error
    
    # Combine recommendations (weighted average)
    content_set = set(content_recs['movieId'].values)
    collab_set = set(collab_recs['movieId'].values)
    
    # Movies recommended by both methods get higher priority
    common_movies = content_set & collab_set
    content_only = content_set - collab_set
    collab_only = collab_set - content_set
    
    # Combine and take top N
    hybrid_movies = list(common_movies) + list(content_only) + list(collab_only)
    hybrid_movies = hybrid_movies[:n_recommendations]
    
    recommendations = movies_df[movies_df['movieId'].isin(hybrid_movies)][['movieId', 'title', 'genres']]
    return recommendations, None

# ===========================
# STREAMLIT UI
# ===========================

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("🎬", use_column_width=False)
with col2:
    st.title("🎬 Movie Recommendation System")
    st.markdown("*Discover movies you'll love based on your preferences*")

st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Recommendation type selection
    st.subheader("1. Choose Recommendation Type")
    recommendation_type = st.selectbox(
        "Select recommendation method:",
        ["Content-Based 🎭", "Collaborative 👥", "Hybrid 🔄"],
        help="Content-Based: Similar movies | Collaborative: User preferences | Hybrid: Both methods"
    )
    
    # Number of recommendations
    st.subheader("2. Adjust Settings")
    n_recommendations = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many movie recommendations to show"
    )

# Main content area
st.subheader("🔍 Movie Search & Recommendations")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    movie_name = st.text_input(
        "Enter movie name:",
        placeholder="e.g., The Matrix, Inception, Titanic",
        help="Type the name of a movie you like"
    )

with col2:
    user_id = st.number_input(
        "Enter User ID:",
        min_value=1,
        value=1,
        step=1,
        help="Your unique user ID for personalized recommendations"
    )

# Recommendation button
st.divider()
if st.button("🎯 Get Recommendations", use_container_width=True, type="primary"):
    
    # Determine which recommendation method to use
    if "Content-Based" in recommendation_type:
        if not movie_name.strip():
            st.error("❌ Please enter a movie name for content-based recommendation")
        else:
            st.info("🔄 Finding similar movies based on genres...")
            recommendations, error = content_based_recommendations(movie_name, n_recommendations)
            
            if error:
                st.error(error)
            else:
                st.success(f"✅ Found {len(recommendations)} recommendations based on '{movie_name}'!")
                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )
    
    elif "Collaborative" in recommendation_type:
        st.info("🔄 Finding movies based on your preferences...")
        recommendations, error = collaborative_recommendations(user_id, n_recommendations)
        
        if error:
            st.error(error)
        else:
            st.success(f"✅ Found {len(recommendations)} personalized recommendations for User {user_id}!")
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )
    
    elif "Hybrid" in recommendation_type:
        if not movie_name.strip():
            st.error("❌ Please enter a movie name for hybrid recommendation")
        else:
            st.info("🔄 Blending content-based and collaborative filtering...")
            recommendations, error = hybrid_recommendations(movie_name, user_id, n_recommendations)
            
            if error:
                st.error(error)
            else:
                st.success(f"✅ Found {len(recommendations)} hybrid recommendations!")
                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

# Footer with information
st.divider()
st.markdown("""
### 📊 How It Works:

**🎭 Content-Based Filtering:**
- Analyzes movie genres and characteristics
- Recommends movies similar to your input movie
- Good for discovering movies in your favorite genres

**👥 Collaborative Filtering:**
- Uses SVD (Singular Value Decomposition) algorithm
- Analyzes patterns from all users' ratings
- Recommends movies similar users have enjoyed
- Best for personalized recommendations

**🔄 Hybrid Approach:**
- Combines both content-based and collaborative methods
- Leverages strengths of both approaches
- Provides more diverse and accurate recommendations
""")

# Dataset statistics
with st.expander("📈 Dataset Statistics"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", len(movies_df))
    with col2:
        st.metric("Total Users", ratings_df['userId'].nunique())
    with col3:
        st.metric("Total Ratings", len(ratings_df))
    with col4:
        avg_rating = ratings_df['rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}")

st.markdown("---")
st.markdown("*Built with Streamlit, Scikit-Learn, and SVD Collaborative Filtering*")
