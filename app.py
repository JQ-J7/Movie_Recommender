"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
        Option 3: Hybrid Movie Recommender System - Interactive GUI
========================================================================================
Description:
    State-of-the-Art Streamlit Web Application for Hybrid Movie Recommendations.
    Uses module_hybrid.py with merged_movies_ratings.csv.
    Features:
      1. Dual Engine: Movie-to-Movie Discovery & User-Personalized Recommendations.
      2. Dynamic Alpha Control (Content-Based vs Collaborative Filtering Balance).
      3. Modern Glassmorphic Cinema UI/UX.
      4. Comprehensive Model Performance Evaluation (RMSE, MSE, MAE, Precision, Recall, F1).
      5. Interactive User Satisfaction Questionnaire & Real-Time Analytics.
      6. In-Depth Background Study & Project Architecture Documentation.
========================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import module_hybrid

# ======================================================================================
# PAGE CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(
    page_title="CineMatch AI | Hybrid Recommender Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Cinematic Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        color: white;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF, #E0E7FF, #C7D2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #C7D2FE;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .movie-card {
        background: rgba(30, 41, 59, 0.7);
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
    
    .doc-section {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #6366F1;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# DATA INITIALIZATION & CACHING
# ======================================================================================
@st.cache_data(show_spinner="🎬 Loading MovieLens Dataset...")
def get_cached_dataset():
    return module_hybrid.load_dataset('merged_movies_ratings.csv')

@st.cache_resource(show_spinner="⚡ Building Hybrid TF-IDF & Matrix Structures...")
def get_cached_structures(data):
    return module_hybrid.build_engine_structures(data)

@st.cache_data(show_spinner="📊 Running 80/20 Train-Test Model Evaluation...")
def get_cached_evaluation(data, alpha):
    return module_hybrid.evaluate_models(data, alpha=alpha)

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
    try:
        st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=400&auto=format&fit=crop", use_column_width=True)
    except Exception:
        st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=400&auto=format&fit=crop")
        
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
        help="0.0 = Pure Collaborative Filtering (User ratings correlation) | 1.0 = Pure Content-Based Filtering (Genres & Tags TF-IDF)"
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
    
    st.markdown("---")
    st.caption("Developed for **TARUMT AI Project (Session 202605)**")
    st.caption("Module: **Hybrid Recommender System (GUI)**")


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
# MAIN NAVIGATION TABS
# ======================================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🍿 Interactive Recommender",
    "📊 Model Benchmarks & Evaluation",
    "📝 User Satisfaction Survey",
    "📖 Background Study & System Docs"
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
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ======================================================================================
with tab2:
    st.markdown("### 📊 Model Performance & Statistical Evaluation (80/20 Train-Test)")
    st.write("Comprehensive benchmarking assessing predictive rating accuracy and classification quality across baseline, collaborative filtering, content proxy, and hybrid models.")
    
    eval_df, n_train, n_test = get_cached_evaluation(data, alpha)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Dataset Volume</div>
            <div class="stat-number">{len(data):,}</div>
            <div style="color: #64748B; font-size: 0.75rem;">Total MovieLens Ratings</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Training Partition</div>
            <div class="stat-number">{n_train:,}</div>
            <div style="color: #64748B; font-size: 0.75rem;">80% Training Split</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Testing Partition</div>
            <div class="stat-number">{n_test:,}</div>
            <div style="color: #64748B; font-size: 0.75rem;">20% Holdout Split</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        hybrid_f1 = eval_df[eval_df['Model / Architecture'].str.contains('Hybrid')]['F1-Score (%)'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Hybrid F1-Score</div>
            <div class="stat-number">{hybrid_f1}%</div>
            <div style="color: #64748B; font-size: 0.75rem;">Top-N Classification</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Comprehensive Evaluation Benchmark Table")
    st.dataframe(
        eval_df.style.highlight_min(subset=['MSE', 'RMSE', 'MAE'], color='#1E3A8A')
                     .highlight_max(subset=['Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Accuracy (%)'], color='#065F46')
    )
    
    st.markdown("---")
    st.markdown("#### 📈 Visual Metric Comparisons")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("##### 📉 Rating Prediction Errors (Lower is Better)")
        error_chart_data = eval_df.set_index('Model / Architecture')[['RMSE', 'MSE', 'MAE']]
        st.bar_chart(error_chart_data)
        
    with chart_col2:
        st.markdown("##### 🎯 Classification Metrics (%) (Higher is Better)")
        class_chart_data = eval_df.set_index('Model / Architecture')[['Precision (%)', 'Recall (%)', 'F1-Score (%)']]
        st.bar_chart(class_chart_data)
        
    with st.expander("📚 Metric Definitions & Academic Explanation"):
        st.markdown("""
        - **Root Mean Squared Error (RMSE)**: Penalizes large prediction errors more severely by taking the square root of the average squared errors:
          $$\\text{RMSE} = \\sqrt{\\frac{1}{|T|} \\sum_{(u, i) \\in T} (r_{u, i} - \\hat{r}_{u, i})^2}$$
        - **Mean Squared Error (MSE)**: The average squared difference between estimated ratings and actual ratings.
        - **Mean Absolute Error (MAE)**: Measures average absolute magnitude of errors without penalizing outliers disproportionately.
        - **Precision@K**: The proportion of recommended items that are actually relevant (Rating $\\ge 3.5$):
          $$\\text{Precision} = \\frac{TP}{TP + FP}$$
        - **Recall@K**: The proportion of relevant items successfully recommended:
          $$\\text{Recall} = \\frac{TP}{TP + FN}$$
        - **F1-Score**: Harmonic mean of Precision and Recall, providing a balanced assessment of recommendation quality.
        """)


# ======================================================================================
# TAB 3: USER SATISFACTION QUESTIONNAIRE
# ======================================================================================
with tab3:
    st.markdown("### 📝 User Satisfaction Questionnaire & Feedback")
    st.write("Gather subjective qualitative and quantitative user feedback to assess user experience, serendipity, and satisfaction as required in part (d)(iii).")
    
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


# ======================================================================================
# TAB 4: BACKGROUND STUDY & ACADEMIC DOCUMENTATION
# ======================================================================================
with tab4:
    st.markdown("### 📖 Background Study & System Documentation")
    st.write("Comprehensive academic report addressing the requirement specifications (Parts a, b, c, and d).")
    
    st.markdown("""
    <div class="doc-section">
        <h3>1. Real-Life Problem Scenario (Requirement a)</h3>
        <p>
            In modern on-demand digital entertainment platforms (e.g., Netflix, Disney+, Prime Video, Letterboxd), users are faced with tens of thousands of media titles—a phenomenon known as <b>choice overload</b> or the <b>paradox of choice</b>.
        </p>
        <p>
            Without an intelligent recommender system, user engagement drops due to decision fatigue. The objective of this project is to build an automated recommendation engine that continuously learns user tastes and provides relevant, novel, and diverse movie suggestions.
        </p>
    </div>
    
    <div class="doc-section">
        <h3>2. Background Study & System Architecture (Requirement b)</h3>
        <h4>A. Pure Content-Based Filtering (CBF) Limitations</h4>
        <ul>
            <li><b>Overspecialization:</b> Recommends only items identical to what the user already consumed (creates an "information bubble").</li>
            <li><b>Cold-Start for New Users:</b> Relies solely on metadata and cannot leverage wisdom of the crowd.</li>
        </ul>
        
        <h4>B. Pure Collaborative Filtering (CF) Limitations</h4>
        <ul>
            <li><b>Cold-Start for New Items:</b> Unrated new movies cannot be recommended because they have zero user interaction vectors.</li>
            <li><b>Sparsity Problem:</b> In large catalogs, the User-Item matrix is over 98% sparse, reducing correlation accuracy.</li>
        </ul>
        
        <h4>C. The Proposed Solution: Weighted Hybrid Recommender System</h4>
        <p>
            Our Hybrid Recommender harmonizes both paradigms using a dynamic weighted linear combination:
        </p>
        <div style="background: rgba(15, 23, 42, 0.8); padding: 1rem; border-radius: 8px; font-family: monospace; color: #818CF8; font-size: 1rem; text-align: center;">
            Score<sub>Hybrid</sub>(u, i) = α · Score<sub>Content</sub>(u, i) + (1 - α) · Score<sub>Collaborative</sub>(u, i)
        </div>
        <p style="margin-top: 0.8rem;">
            Where:
            <ul>
                <li><b>α (Alpha ∈ [0, 1]):</b> User-tunable weight parameter controlling the bias between semantic feature similarity and collaborative crowd patterns.</li>
                <li><b>Score<sub>Content</sub>:</b> TF-IDF vectorization with unigram/bigram tokenization on genres and user tags, measured via Cosine Similarity.</li>
                <li><b>Score<sub>Collaborative</sub>:</b> Pearson Correlation Coefficient across co-rated user rating vectors with bias adjustments.</li>
            </ul>
        </p>
    </div>
    
    <div class="doc-section">
        <h3>3. Expected Functionalities and Benefits (Requirement b.iii)</h3>
        <ul>
            <li><b>Mitigates Cold-Start:</b> New movies with no ratings can still be discovered via their genre/tag TF-IDF signatures (α → 1.0).</li>
            <li><b>Serendipity & Discovery:</b> Collaborative filtering introduces cross-genre unexpected gems that users with similar tastes enjoyed (α → 0.0).</li>
            <li><b>Interactive Control:</b> The Streamlit GUI empowers users to tune their own discovery preference in real time.</li>
            <li><b>Dual Recommendation Modes:</b> Supports both title-to-title exploratory matching and user-profile personalized recommendations.</li>
        </ul>
    </div>
    
    <div class="doc-section">
        <h3>4. System Evaluation Summary (Requirement d)</h3>
        <p>
            The system is rigorously benchmarked on an 80/20 train-test split across predictive accuracy metrics (RMSE, MSE, MAE) and ranking classification metrics (Precision, Recall, F1-Score), complemented by an interactive 5-point Likert scale user satisfaction survey.
        </p>
    </div>
    """, unsafe_allow_html=True)
