import streamlit as st
import pandas as pd
from module_hybrid import HybridRecommender

# 1. Page Configuration
st.set_page_config(
    page_title="Intelligent Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# 2. Cache Model Loading
@st.cache_resource
def load_system():
    return HybridRecommender()

st.title("🎬 Movie Recommendation System")
st.markdown("Welcome! Select a recommendation mode below to view predictions customized for that algorithm.")

# Initialize System
with st.spinner("Initializing system and models, please wait..."):
    try:
        system = load_system()
        df_movies = pd.read_csv('merged_movies.csv')
    except Exception as e:
        st.error("❌ Data or model loading failed. Please ensure `data_prep.py` was run first!")
        st.error(f"Error details: {e}")
        st.stop()

# 3. Sidebar Configuration
st.sidebar.header("⚙️ Recommendation Settings")

# Algorithm Selection
rec_mode = st.sidebar.selectbox(
    " Select Recommendation Model",
    ["Hybrid Recommendation (SVD + NLP)", "Content-Based NLP (Content Similarity Only)", "Collaborative Filtering SVD (Ratings Only)"]
)

user_id = st.sidebar.number_input(" Enter/Select User ID", min_value=1, max_value=600, value=1, step=1)

movie_list = df_movies['title_x'].tolist()
selected_movie_title = st.sidebar.selectbox(" Select a movie you like/are watching", movie_list)

selected_movie_row = df_movies[df_movies['title_x'] == selected_movie_title].iloc[0]
selected_movie_id = selected_movie_row['movieId']

# 4. Dynamically set Dataset Info, Weights, and Display Columns based on mode
if rec_mode == "Content-Based NLP (Content Similarity Only)":
    alpha = 0.0
    active_dataset_info = "📄 **Dataset Used**: `merged_movies.csv` (Movie Metadata: Genres, Director, Cast)"
    display_cols = ['title', 'genres', 'director', 'content_sim']
    column_config = {
        "title": "Title",
        "genres": "Genres",
        "director": "Director",
        "content_sim": st.column_config.NumberColumn("NLP Text Similarity", format="%.4f")
    }

elif rec_mode == "Collaborative Filtering SVD (Ratings Only)":
    alpha = 1.0
    active_dataset_info = "📊 **Dataset Used**: `ratings.csv` (User Interaction & Historical Ratings)"
    display_cols = ['title', 'genres', 'svd_pred_rating']
    column_config = {
        "title": "Title",
        "genres": "Genres",
        "svd_pred_rating": st.column_config.NumberColumn("SVD Predicted Rating", format="%.2f ⭐")
    }

else:  # Hybrid Mode
    alpha = st.sidebar.slider("⚖️ SVD Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.sidebar.caption("💡 Alpha closer to 1.0 favors [Collaborative Filtering], closer to 0.0 favors [NLP Content Similarity]")
    active_dataset_info = "🔗 **Datasets Used**: `merged_movies.csv` + `ratings.csv` (Combined Metadata & Rating Interactions)"
    display_cols = ['title', 'genres', 'director', 'hybrid_score', 'svd_pred_rating', 'content_sim']
    column_config = {
        "title": "Title",
        "genres": "Genres",
        "director": "Director",
        "hybrid_score": st.column_config.NumberColumn("Hybrid Score", format="%.4f"),
        "svd_pred_rating": st.column_config.NumberColumn("SVD Predicted Rating", format="%.2f ⭐"),
        "content_sim": st.column_config.NumberColumn("NLP Text Similarity", format="%.4f")
    }

top_n = st.sidebar.slider("📊 Number of Recommendations", min_value=3, max_value=20, value=5, step=1)

# Display active dataset banner
st.info(active_dataset_info)

# 5. Main UI: Seed Movie Info
st.subheader("📌 Current Seed Movie Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Title**: {selected_movie_row['title_x']}")
with col2:
    st.info(f"**Genres**: {selected_movie_row['genres']}")
with col3:
    st.info(f"**Director**: {selected_movie_row['director'] if selected_movie_row['director'] else 'Unknown'}")

st.markdown("---")

# 6. Generate Recommendations
if st.button("🚀 Generate Recommendations", type="primary"):
    with st.spinner("Calculating recommendations..."):
        results = system.recommend(
            user_id=user_id,
            seed_movie_id=selected_movie_id,
            top_n=top_n,
            alpha=alpha
        )
        
        if not results.empty:
            st.success(f"🎉 Displaying recommendations using **{rec_mode}**:")
            
            # Show ONLY columns relevant to the selected algorithm
            st.dataframe(
                results[display_cols],
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ No matching recommendations found, please try another seed movie.")