"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
        Option 3: Hybrid Movie Recommender System - Interactive GUI
========================================================================================
Description:
    Streamlit Web Application for Hybrid Movie Recommendations.
    Features:
      1. Dual Recommender Engine: Movie-to-Movie Discovery & User-Personalized Recommendations.
      2. Dynamic Alpha Control (Content-Based vs Collaborative Filtering Balance).
      3. Interactive User Satisfaction Questionnaire & Real-Time Analytics.
========================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import module_hybrid

# ======================================================================================
# PAGE CONFIGURATION & HIGH-END CINEMATIC THEME
# ======================================================================================
st.set_page_config(
    page_title="CineMatch AI | Hybrid Recommender Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Glassmorphic Cinema Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        color: white;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF, #E0E7FF, #C7D2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.02rem;
        color: #C7D2FE;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    .movie-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .movie-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.3rem;
    }
    
    .badge-genre {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .badge-tag {
        display: inline-block;
        background: rgba(236, 72, 153, 0.15);
        color: #F472B6;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
    }
    
    .score-container {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    
    .score-box {
        flex: 1;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        padding: 0.4rem 0.6rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .score-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6366F1;
    }
    
    .score-label {
        font-size: 0.7rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# DATA INITIALIZATION & CACHING
# ======================================================================================
@st.cache_data(show_spinner="🎬 Loading MovieLens Dataset...")
def get_cached_dataset():
    return module_hybrid.load_dataset('movies_dataset.csv')

@st.cache_resource(show_spinner="⚡ Building Hybrid TF-IDF & Matrix Structures...")
def get_cached_structures(data):
    return module_hybrid.build_engine_structures(data)

try:
    data = get_cached_dataset()
    structures = get_cached_structures(data)
except Exception as e:
    st.error(f"Error loading system components: {e}")
    st.stop()


# ======================================================================================
# SIDEBAR CONTROLS & HYPERPARAMETER TUNING
# ======================================================================================
with st.sidebar:
    st.markdown("### 🎛️ Hybrid Engine Settings")
    
    recommendation_mode = st.radio(
        "Recommendation Discovery Mode",
        ["🎯 Movie-to-Movie Discovery", "👤 User-Personalized Discovery"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚖️ Hybrid Weighting (Alpha)")
    alpha = st.slider(
        "Balance (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="0.0 = Pure Collaborative Filtering (User co-ratings) | 1.0 = Pure Content-Based Filtering (Genres & Tags TF-IDF)"
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"**Content (α):** `{int(alpha*100)}%`")
    with col_b:
        st.caption(f"**Collaborative:** `{int((1-alpha)*100)}%`")
        
    st.markdown("---")
    st.markdown("#### 🔍 Filter Criteria")
    
    # Extract unique genres
    raw_genres = set()
    for g_str in structures['movie_stats']['genres'].dropna():
        for g in g_str.split('|'):
            if g.strip() and g.strip() != '(no genres listed)':
                raw_genres.add(g.strip())
    all_genres = ['All'] + sorted(list(raw_genres))
    
    genre_filter = st.selectbox("Filter by Genre", all_genres, index=0)
    
    min_ratings = st.slider(
        "Minimum Rating Count",
        min_value=0,
        max_value=100,
        value=15,
        step=5,
        help="Filters out low-rated movies to maintain high quality and statistical confidence."
    )
    
    top_n = st.slider("Top-N Recommendations", min_value=3, max_value=25, value=8, step=1)


# ======================================================================================
# TOP HEADER
# ======================================================================================
st.markdown("""
<div class="main-header">
    <h1>🎬 CineMatch AI | Hybrid Recommender Studio</h1>
    <p>An intelligent hybrid movie recommendation platform fusing <b>Content-Based TF-IDF Semantics</b> and <b>Item/User Collaborative Filtering Dynamics</b>.</p>
</div>
""", unsafe_allow_html=True)


# ======================================================================================
# MAIN NAVIGATION TABS (REVIEWS & SATISFACTION ONLY)
# ======================================================================================
tab1, tab2 = st.tabs([
    "🍿 Interactive Recommender",
    "📝 User Satisfaction Questionnaire"
])


# ======================================================================================
# TAB 1: INTERACTIVE HYBRID RECOMMENDER
# ======================================================================================
with tab1:
    if recommendation_mode == "🎯 Movie-to-Movie Discovery":
        st.markdown("### 🎯 Find Movies Similar to Your Favorite Film")
        st.write("Search for any title, keyword, or genre. The hybrid algorithm balances **thematic metadata** (genres & tags) with **viewer co-rating correlation**.")
        
        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Enter Movie Title or Search Term:", value="Toy Story", placeholder="e.g., Matrix, Inception, Harry Potter, Toy Story...")
        with col_btn:
            st.write("")
            st.write("")
            search_trigger = st.button("🔍 Search Candidates")
            
        candidate_matches = module_hybrid.search_movies(search_query, structures['movie_stats'], max_results=8)
        
        if not candidate_matches:
            st.warning(f"No movies found matching '{search_query}'. Please try a different query.")
            selected_movie = None
        else:
            selected_movie = st.selectbox("Select Target Movie:", candidate_matches, index=0)
            
        if selected_movie:
            target_meta = structures['movie_stats'][structures['movie_stats']['title'] == selected_movie].iloc[0]
            
            # Selected Movie Showcase Card
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 1.2rem; margin-top: 1rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.8rem; text-transform: uppercase; color: #A5B4FC; font-weight: 700;">Selected Target Film</span>
                        <h2 style="margin: 0.2rem 0; color: #FFFFFF; font-size: 1.6rem;">{target_meta['title']}</h2>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.4rem; font-weight: 800; color: #FBBF24;">★ {target_meta['avg_rating']}</span>
                        <span style="color: #94A3B8; font-size: 0.85rem;">/ 5.0 ({target_meta['num_of_ratings']} ratings)</span>
                    </div>
                </div>
                <div style="margin-top: 0.6rem;">
                    {" ".join([f'<span class="badge-genre">{g}</span>' for g in target_meta['genres'].split('|') if g])}
                </div>
                {f'<div style="margin-top: 0.4rem;"><span style="font-size: 0.8rem; color: #CBD5E1;">Tags: </span>' + " ".join([f'<span class="badge-tag">{t}</span>' for t in target_meta['tags'].split('|') if t]) + '</div>' if target_meta['tags'] else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation Generator
            with st.spinner("Generating Hybrid Recommendations..."):
                recs, err = module_hybrid.get_hybrid_recommendations(
                    selected_movie,
                    structures,
                    alpha=alpha,
                    min_ratings=min_ratings,
                    genre_filter=genre_filter,
                    top_n=top_n
                )
                
            if err:
                st.error(err)
            elif recs is None or recs.empty:
                st.info("No recommendations found with the current filter settings. Try lowering the 'Minimum Rating Count' or setting Genre to 'All'.")
            else:
                st.markdown(f"### ✨ Top {len(recs)} Hybrid Recommendations")
                
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    col = cols[i % 2]
                    with col:
                        genres_html = " ".join([f'<span class="badge-genre">{g}</span>' for g in str(row['genres']).split('|') if g])
                        tags_list = [t for t in str(row['tags']).split('|') if t][:5]
                        tags_html = " ".join([f'<span class="badge-tag">#{t}</span>' for t in tags_list])
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="flex: 1;">
                                    <span style="font-size: 0.75rem; color: #818CF8; font-weight: 700;">RANK #{i+1}</span>
                                    <div class="movie-title">{row['title']}</div>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 1.2rem; font-weight: 800; color: #FBBF24;">★ {row['avg_rating']}</span>
                                    <div style="font-size: 0.7rem; color: #94A3B8;">{row['num_of_ratings']} reviews</div>
                                </div>
                            </div>
                            <div style="margin-top: 0.4rem; margin-bottom: 0.4rem;">
                                {genres_html}
                            </div>
                            <div class="score-container">
                                <div class="score-box" style="border-color: rgba(99, 102, 241, 0.4);">
                                    <div class="score-val" style="color: #818CF8;">{row['hybrid_score']}%</div>
                                    <div class="score-label">Hybrid Match</div>
                                </div>
                                <div class="score-box">
                                    <div class="score-val" style="color: #34D399;">{row['cb_score']}%</div>
                                    <div class="score-label">Content (TF-IDF)</div>
                                </div>
                                <div class="score-box">
                                    <div class="score-val" style="color: #F472B6;">{row['cf_score']}%</div>
                                    <div class="score-label">Collaborative (CF)</div>
                                </div>
                            </div>
                            {f'<div style="margin-top: 0.3rem;">{tags_html}</div>' if tags_html else ''}
                        </div>
                        """, unsafe_allow_html=True)

    else:
        # Personalized User Discovery Mode
        st.markdown("### 👤 Personalized User Recommendations")
        st.write("Generates custom hybrid recommendations based on a user's historical rating profile.")
        
        user_list = sorted(data['userId'].unique())
        selected_user = st.selectbox("Select Target User ID:", user_list, index=0)
        
        u_ratings = data[data['userId'] == selected_user]
        u_liked = u_ratings[u_ratings['rating'] >= 4.0].sort_values(by='rating', ascending=False)
        
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            st.metric("Total Rated Movies", len(u_ratings))
        with col_u2:
            st.metric("Average Rating Given", f"{u_ratings['rating'].mean():.2f} ★")
        with col_u3:
            st.metric("Favorites (≥ 4.0 ★)", len(u_liked))
            
        with st.expander("⭐ View User's Top Rated Movies"):
            st.dataframe(u_liked[['title', 'rating', 'genres', 'tags']].head(10))
            
        with st.spinner("Calculating Personalized Hybrid Recommendations..."):
            user_recs, u_err = module_hybrid.get_user_personalized_recommendations(
                selected_user,
                structures,
                data,
                alpha=alpha,
                min_ratings=min_ratings,
                top_n=top_n
            )
            
        if u_err:
            st.error(u_err)
        elif user_recs is None or user_recs.empty:
            st.info("No recommendations found for this user with current filters.")
        else:
            st.markdown(f"### 🎯 Personalized Recommendations for User #{selected_user}")
            cols = st.columns(2)
            for i, (_, row) in enumerate(user_recs.iterrows()):
                col = cols[i % 2]
                with col:
                    genres_html = " ".join([f'<span class="badge-genre">{g}</span>' for g in str(row['genres']).split('|') if g])
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-size: 0.75rem; color: #818CF8; font-weight: 700;">SUGGESTION #{i+1}</span>
                                <div class="movie-title">{row['title']}</div>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 1.2rem; font-weight: 800; color: #FBBF24;">★ {row['avg_rating']}</span>
                                <div style="font-size: 0.7rem; color: #94A3B8;">{row['num_of_ratings']} reviews</div>
                            </div>
                        </div>
                        <div style="margin-top: 0.4rem; margin-bottom: 0.4rem;">{genres_html}</div>
                        <div class="score-container">
                            <div class="score-box" style="border-color: rgba(99, 102, 241, 0.4);">
                                <div class="score-val" style="color: #818CF8;">{row['hybrid_score']}%</div>
                                <div class="score-label">Personal Match</div>
                            </div>
                            <div class="score-box">
                                <div class="score-val" style="color: #34D399;">{row['cb_score']}%</div>
                                <div class="score-label">Taste Profile</div>
                            </div>
                            <div class="score-box">
                                <div class="score-val" style="color: #F472B6;">{row['cf_score']}%</div>
                                <div class="score-label">CF Prediction</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ======================================================================================
# TAB 2: USER SATISFACTION QUESTIONNAIRE
# ======================================================================================
with tab2:
    st.markdown("### 📝 User Satisfaction Questionnaire & Feedback")
    st.write("Gather subjective qualitative and quantitative user feedback to assess user experience, serendipity, and recommendation satisfaction.")
    
    col_form, col_stats = st.columns([1.2, 1])
    
    with col_form:
        st.markdown("#### 📋 Submit Your Evaluation")
        with st.form("satisfaction_form", clear_on_submit=True):
            eval_name = st.text_input("Evaluator Name / Identifier:", placeholder="e.g., Student Evaluator 1")
            
            st.markdown("**1. Recommendation Relevance (1 = Irrelevant, 5 = Highly Relevant)**")
            q_rel = st.slider("How relevant were the recommended movies to your selected title/tastes?", 1, 5, 5, key="q_rel")
            
            st.markdown("**2. Novelty & Discovery (1 = Too Obvious, 5 = Great Discoveries)**")
            q_nov = st.slider("Did the system help you discover interesting new or unexpected movies?", 1, 5, 4, key="q_nov")
            
            st.markdown("**3. Catalog Diversity (1 = Monotonous, 5 = Well-Balanced)**")
            q_div = st.slider("Was there a balanced variety of genres and movie types in the results?", 1, 5, 4, key="q_div")
            
            st.markdown("**4. UI Ease of Use (1 = Confusing, 5 = Very Intuitive)**")
            q_ui = st.slider("How intuitive and responsive was the Streamlit GUI interface and alpha slider?", 1, 5, 5, key="q_ui")
            
            st.markdown("**5. Overall Satisfaction (1 = Unsatisfied, 5 = Highly Satisfied)**")
            q_overall = st.slider("What is your overall satisfaction with the CineMatch Hybrid Recommender?", 1.0, 5.0, 4.8, step=0.1, key="q_overall")
            
            q_comments = st.text_area("Qualitative Comments & Suggestions:", placeholder="e.g., The hybrid weighting slider makes it easy to switch between similar movies and serendipitous recommendations...")
            
            submit_btn = st.form_submit_button("🚀 Submit Feedback")
            
            if submit_btn:
                module_hybrid.save_survey_response(eval_name, q_rel, q_nov, q_div, q_ui, q_overall, q_comments)
                st.success("✅ Thank you! Your feedback has been recorded successfully.")
                st.rerun()

    with col_stats:
        st.markdown("#### 📊 Real-Time Survey Analytics")
        survey_df = module_hybrid.load_survey_responses()
        
        if not survey_df.empty:
            avg_rel = survey_df['Relevance (1-5)'].mean()
            avg_nov = survey_df['Novelty (1-5)'].mean()
            avg_div = survey_df['Diversity (1-5)'].mean()
            avg_ui = survey_df['UI Usability (1-5)'].mean()
            avg_overall = survey_df['Overall Satisfaction (1-5)'].mean()
            
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Total Responses", len(survey_df))
            with s2:
                st.metric("Mean Satisfaction", f"{avg_overall:.2f} / 5.0")
                
            dimension_means = pd.DataFrame({
                'Dimension': ['Relevance', 'Novelty', 'Diversity', 'UI Usability', 'Overall'],
                'Average Score (out of 5.0)': [avg_rel, avg_nov, avg_div, avg_ui, avg_overall]
            }).set_index('Dimension')
            
            st.bar_chart(dimension_means)
            
            with st.expander("📋 View All Questionnaire Responses"):
                st.dataframe(survey_df)
        else:
            st.info("No survey responses recorded yet. Fill out the form on the left to submit the first review!")
