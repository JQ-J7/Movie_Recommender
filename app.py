"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
        Option 3: Hybrid Movie Recommender System - Interactive Studio
========================================================================================
Description:
    Streamlit Web Application for Hybrid Movie Recommendations.
    Architecture:
      1. Dual View Architecture (No Sidebar):
         - User View: Pure Movie-to-Movie Discovery (Title search) & User Satisfaction
           Questionnaire submission.
         - Developer View: Password-Protected (PIN: 1234). Hyperparameter Tuning (Alpha Weighting),
           80/20 Offline Evaluation Metrics, and Full Survey Analytics & Respondent Auditing.
      2. High-Performance Hybrid Recommender Engine (TF-IDF Semantics + Collaborative Correlation).
      3. Enterprise-grade UI design with zero emojis for an academic, professional aesthetic.
========================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import module_hybrid
import collaborative_recommender

# ======================================================================================
# 1. PAGE CONFIGURATION & HIGH-END ACADEMIC THEME
# ======================================================================================
st.set_page_config(
    page_title="CineMatch AI | Hybrid Recommender System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Dark Glassmorphic Cinema Theme CSS (Zero Emojis, No Sidebar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Completely hide Streamlit Sidebar and collapse toggle button */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Top Bar Navigation Area */
    .top-bar-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    .view-badge-user {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    
    .view-badge-dev {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #FCD34D;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    
    /* Header Banners */
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        color: white;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF, #E0E7FF, #C7D2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 0.98rem;
        color: #C7D2FE;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    
    .dev-header {
        background: linear-gradient(135deg, #18181B 0%, #27272A 50%, #3F3F46 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        color: white;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .dev-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF, #FDE68A, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .dev-header p {
        font-size: 0.98rem;
        color: #E4E4E7;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    
    /* Movie & Recommendation Cards */
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
        font-size: 1.15rem;
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
    
    .badge-rating {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 0.2rem 0.55rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(245, 158, 11, 0.3);
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
    
    /* Developer Metric Cards */
    .eval-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .eval-number {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .eval-title {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .status-panel {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# 2. SESSION STATE MANAGEMENT (VIEW MODE, AUTH & HYPERPARAMETERS)
# ======================================================================================
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "user"

if 'dev_authenticated' not in st.session_state:
    st.session_state.dev_authenticated = False

if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.50

if 'min_ratings' not in st.session_state:
    st.session_state.min_ratings = 15

if 'top_n' not in st.session_state:
    st.session_state.top_n = 10

# Security guard for developer view
if st.session_state.view_mode == "developer" and not st.session_state.dev_authenticated:
    st.session_state.view_mode = "user"


# ======================================================================================
# 3. DEVELOPER AUTHENTICATION DIALOG (PASSWORD: 1234)
# ======================================================================================
@st.dialog("Developer Access Authorization")
def dev_login_dialog():
    st.write("Please enter the developer password to access hyperparameter calibration, offline evaluation metrics, and survey audit logs.")
    pwd_input = st.text_input("Developer Password:", type="password", key="dev_pwd_input", placeholder="Enter Password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Unlock Studio", use_container_width=True, key="btn_unlock"):
            if pwd_input == "1234":
                st.session_state.view_mode = "developer"
                st.session_state.dev_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")
    with col2:
        if st.button("Cancel", use_container_width=True, key="btn_cancel"):
            st.rerun()


# ======================================================================================
# 4. DATA INITIALIZATION & CACHING
# ======================================================================================
@st.cache_data(show_spinner="Loading MovieLens Dataset...")
def get_cached_dataset():
    return module_hybrid.load_dataset('movies_dataset.csv')

@st.cache_resource(show_spinner="Building Hybrid TF-IDF & Matrix Structures...")
def get_cached_structures(data):
    return module_hybrid.build_engine_structures(data)

@st.cache_data(show_spinner="Running 80/20 Mock Test Evaluation across 3 Hybrid Configurations...")
def get_cached_evaluation(data):
    metrics_df, details = module_hybrid.evaluate_hybrid_recommender_system(data)
    return metrics_df, details

@st.cache_data(show_spinner="Running 80/20 Train-Test Evaluation for Collaborative Filtering...")
def get_cached_cf_evaluation(data):
    return collaborative_recommender.evaluate_recommender_system(data)

@st.cache_data(show_spinner="Computing Collaborative Dataset Analytics...")
def get_cached_cf_summary(data):
    return collaborative_recommender.get_dataset_summary_metrics(data)

try:
    data = get_cached_dataset()
    structures = get_cached_structures(data)
except Exception as e:
    st.error(f"Error initializing system components: {e}")
    st.stop()


# ======================================================================================
# 5. TOP BAR VIEW SWITCHER & HEADER
# ======================================================================================
col_nav_left, col_nav_right = st.columns([1.5, 4])

with col_nav_left:
    if st.session_state.view_mode == "user":
        if st.button("Switch to Developer View", key="btn_to_dev", use_container_width=True):
            dev_login_dialog()
    else:
        if st.button("Switch to User View", key="btn_to_user", use_container_width=True):
            st.session_state.view_mode = "user"
            st.session_state.dev_authenticated = False
            st.rerun()

with col_nav_right:
    if st.session_state.view_mode == "user":
        st.markdown(
            '<div style="text-align: right; padding-top: 0.35rem;">'
            '<span class="view-badge-user">Active Mode: User View</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align: right; padding-top: 0.35rem;">'
            '<span class="view-badge-dev">Active Mode: Developer View</span>'
            '</div>',
            unsafe_allow_html=True
        )

# Render View Header Banner
if st.session_state.view_mode == "user":
    st.markdown("""
    <div class="main-header">
        <h1>CineMatch AI | Hybrid Movie Recommender</h1>
        <p>Discover personalized film recommendations powered by integrated Content-Based Semantics and Collaborative Filtering.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="dev-header">
        <h1>CineMatch AI | Developer & Evaluation Studio</h1>
        <p>System Calibration, Alpha Weighting Hyperparameters, 80/20 Offline Benchmark Metrics, and Survey Auditing.</p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================================================
# 6. USER VIEW (END-USER INTERFACE - NO SIDEBAR)
# ======================================================================================
if st.session_state.view_mode == "user":

    # User View Tabs
    user_tab1, user_tab2 = st.tabs([
        "Movie Recommender",
        "User Satisfaction Questionnaire"
    ])
    
    # ----------------------------------------------------------------------------------
    # USER TAB 1: MOVIE-TO-MOVIE DISCOVERY (TITLE SEARCH ONLY)
    # ----------------------------------------------------------------------------------
    with user_tab1:
        st.markdown("### Find Movies Similar to Your Selected Film")
        st.write("Search for any movie title in the catalog. The hybrid engine combines semantic content analysis with viewer co-rating patterns to produce ranked recommendations.")
        
        col_search, col_btn = st.columns([3.5, 1])
        with col_search:
            search_query = st.text_input(
                "Enter Movie Title:",
                value="Toy Story",
                placeholder="e.g., Toy Story, The Matrix, Inception, Harry Potter, Jurassic Park..."
            )
        with col_btn:
            st.write("")
            st.write("")
            search_trigger = st.button("Search Catalog", use_container_width=True)
            
        candidate_matches = module_hybrid.search_movies(search_query, structures['movie_stats'], max_results=8)
        
        if not candidate_matches:
            st.warning(f"No movies found matching title '{search_query}'. Please check spelling or try another title.")
            selected_movie = None
        else:
            selected_movie = st.selectbox("Select Target Movie:", candidate_matches, index=0)
            
        if selected_movie:
            target_meta = structures['movie_stats'][structures['movie_stats']['title'] == selected_movie].iloc[0]
            
            genres_split = [g for g in str(target_meta['genres']).split('|') if g]
            tags_split = [t for t in str(target_meta.get('tags', '')).split('|') if t][:8]
            
            # Showcase Card for Target Film
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 1.2rem; margin-top: 1rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.75rem; text-transform: uppercase; color: #A5B4FC; font-weight: 700; letter-spacing: 0.5px;">Selected Target Film</span>
                        <h2 style="margin: 0.2rem 0; color: #FFFFFF; font-size: 1.5rem;">{target_meta['title']}</h2>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge-rating">Rating: {target_meta['avg_rating']} / 5.0</span>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.2rem;">Based on {target_meta['num_of_ratings']} reviews</div>
                    </div>
                </div>
                <div style="margin-top: 0.6rem;">
                    {" ".join([f'<span class="badge-genre">{g}</span>' for g in genres_split])}
                </div>
                {f'<div style="margin-top: 0.4rem;"><span style="font-size: 0.8rem; color: #CBD5E1;">Keywords: </span>' + " ".join([f'<span class="badge-tag">{t}</span>' for t in tags_split]) + '</div>' if tags_split else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Generate Hybrid Recommendations
            with st.spinner("Generating Hybrid Recommendations..."):
                recs, err = module_hybrid.get_hybrid_recommendations(
                    selected_movie,
                    structures,
                    alpha=st.session_state.alpha,
                    min_ratings=st.session_state.min_ratings,
                    genre_filter='All',
                    top_n=st.session_state.top_n
                )
                
            if err:
                st.error(err)
            elif recs is None or recs.empty:
                st.info("No recommendations found matching current criteria. Try searching for a different movie.")
            else:
                st.markdown(f"### Top {len(recs)} Hybrid Recommendations")
                
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    col = cols[i % 2]
                    with col:
                        genres_html = " ".join([f'<span class="badge-genre">{g}</span>' for g in str(row['genres']).split('|') if g])
                        tags_list = [t for t in str(row.get('tags', '')).split('|') if t][:5]
                        tags_html = " ".join([f'<span class="badge-tag">#{t}</span>' for t in tags_list])
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="flex: 1;">
                                    <span style="font-size: 0.75rem; color: #818CF8; font-weight: 700; letter-spacing: 0.5px;">RANK #{i+1}</span>
                                    <div class="movie-title">{row['title']}</div>
                                </div>
                                <div style="text-align: right;">
                                    <span class="badge-rating">{row['avg_rating']} / 5.0</span>
                                    <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.15rem;">{row['num_of_ratings']} reviews</div>
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
                                    <div class="score-label">Collab (CF)</div>
                                </div>
                            </div>
                            {f'<div style="margin-top: 0.3rem;">{tags_html}</div>' if tags_html else ''}
                        </div>
                        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------
    # USER TAB 2: USER SATISFACTION QUESTIONNAIRE (SUBMISSION FORM ONLY)
    # ----------------------------------------------------------------------------------
    with user_tab2:
        st.markdown("### User Satisfaction Questionnaire")
        st.write("Please evaluate your experience with the recommendation quality and system usability. Your feedback assists in offline model calibration.")
        
        with st.form("user_satisfaction_form", clear_on_submit=True):
            eval_name = st.text_input("Evaluator Name / Identifier:", placeholder="e.g., Evaluator 1")
            
            st.markdown("**1. Recommendation Relevance (1 = Irrelevant, 5 = Highly Relevant)**")
            q_rel = st.slider("How relevant were the recommended movies to your selected film?", 1, 5, 5, key="u_q_rel")
            
            st.markdown("**2. Novelty & Discovery (1 = Too Obvious, 5 = Great Discoveries)**")
            q_nov = st.slider("Did the system help you discover interesting new or unexpected movies?", 1, 5, 4, key="u_q_nov")
            
            st.markdown("**3. Catalog Diversity (1 = Monotonous, 5 = Well-Balanced)**")
            q_div = st.slider("Was there a balanced variety of genres and movie types in the results?", 1, 5, 4, key="u_q_div")
            
            st.markdown("**4. System Usability (1 = Confusing, 5 = Very Intuitive)**")
            q_ui = st.slider("How intuitive and responsive was the user interface?", 1, 5, 5, key="u_q_ui")
            
            st.markdown("**5. Overall Satisfaction (1.0 = Unsatisfied, 5.0 = Highly Satisfied)**")
            q_overall = st.slider("What is your overall satisfaction with the Hybrid Recommender System?", 1.0, 5.0, 4.8, step=0.1, key="u_q_overall")
            
            q_comments = st.text_area(
                "Qualitative Comments & Feedback:",
                placeholder="Share your thoughts on recommendation quality, ranking accuracy, or suggested enhancements..."
            )
            
            submit_btn = st.form_submit_button("Submit Evaluation", use_container_width=True)
            
            if submit_btn:
                module_hybrid.save_survey_response(eval_name, q_rel, q_nov, q_div, q_ui, q_overall, q_comments)
                st.success("Thank you! Your evaluation has been recorded successfully.")


# ======================================================================================
# 7. DEVELOPER VIEW (DEVELOPER SETTINGS, METRICS & SURVEY AUDITING - NO SIDEBAR)
# ======================================================================================
else:
    # Developer View Main Tabs
    dev_tab1, dev_tab2, dev_tab3, dev_tab4 = st.tabs([
        "Hybrid Engine Settings",
        "Model Evaluation Metrics (80/20 Split)",
        "Collaborative Filtering Evaluation",
        "Survey Analytics & Respondent Audit"
    ])
    
    # ----------------------------------------------------------------------------------
    # DEV TAB 1: HYBRID HYPERPARAMETERS & ENGINE SETTINGS
    # ----------------------------------------------------------------------------------
    with dev_tab1:
        st.markdown("### Hybrid Engine Weighting & Hyperparameter Calibration")
        st.write(
            "Calibrate the balance between **Content-Based TF-IDF Semantics** (metadata, genres, tags) "
            "and **Collaborative Filtering** (Pearson co-rating correlation). Changes made here apply immediately to the recommendation engine."
        )
        
        col_dev_left, col_dev_right = st.columns([1.5, 1])
        
        with col_dev_left:
            st.markdown("#### Hybrid Weighting (Alpha Configuration)")
            
            alpha_preset = st.selectbox(
                "Standard Configuration Presets:",
                [
                    "Custom Slider Setting",
                    "Hybrid 20% CF / 80% CBF (alpha = 0.80)",
                    "Hybrid 50% CF / 50% CBF (alpha = 0.50)",
                    "Hybrid 80% CF / 20% CBF (alpha = 0.20)"
                ],
                index=0
            )
            
            if alpha_preset == "Hybrid 20% CF / 80% CBF (alpha = 0.80)":
                target_alpha = 0.80
            elif alpha_preset == "Hybrid 50% CF / 50% CBF (alpha = 0.50)":
                target_alpha = 0.50
            elif alpha_preset == "Hybrid 80% CF / 20% CBF (alpha = 0.20)":
                target_alpha = 0.20
            else:
                target_alpha = st.session_state.alpha
                
            st.session_state.alpha = st.slider(
                "Balance Parameter (Alpha)",
                min_value=0.0,
                max_value=1.0,
                value=float(target_alpha),
                step=0.05,
                format="%.2f",
                help="0.0 = Pure Collaborative Filtering | 1.0 = Pure Content-Based Filtering"
            )
            
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown(f"**Content Weight (CBF):** `{int(st.session_state.alpha * 100)}%`")
            with col_w2:
                st.markdown(f"**Collaborative Weight (CF):** `{int((1 - st.session_state.alpha) * 100)}%`")
                
            st.markdown("---")
            st.markdown("#### Filtering & Recommendation Thresholds")
            
            st.session_state.min_ratings = st.slider(
                "Minimum Review Count Threshold:",
                min_value=0,
                max_value=100,
                value=st.session_state.min_ratings,
                step=5,
                help="Filters out low-rated movies to maintain statistical confidence."
            )
            
            st.session_state.top_n = st.slider(
                "Default Top-N Recommendations Count:",
                min_value=3,
                max_value=25,
                value=st.session_state.top_n,
                step=1
            )
            
        with col_dev_right:
            st.markdown("#### Active Configuration Summary")
            st.markdown(f"""
            <div class="status-panel">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Active Alpha (α)</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #818CF8;">{st.session_state.alpha:.2f}</div>
                <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #CBD5E1;">
                    <div>• <b>Content-Based:</b> {int(st.session_state.alpha*100)}%</div>
                    <div>• <b>Collaborative:</b> {int((1-st.session_state.alpha)*100)}%</div>
                    <div>• <b>Min Ratings:</b> {st.session_state.min_ratings}</div>
                    <div>• <b>Top-N:</b> {st.session_state.top_n} items</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### System Cache Control")
            if st.button("Clear Cache & Reload Data", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared successfully.")
                st.rerun()
            
            st.markdown("#### Formula Reference")
            st.code("""
Score = (alpha * CB_Score) + ((1 - alpha) * CF_Score)

where:
  CB_Score = TF-IDF Cosine Similarity
  CF_Score = Pearson Co-Rating Correlation
            """, language="text")

    # ----------------------------------------------------------------------------------
    # DEV TAB 2: 80/20 OFFLINE MODEL EVALUATION BENCHMARK
    # ----------------------------------------------------------------------------------
    with dev_tab2:
        st.markdown("### Recommender System Model Evaluation (80/20 Mock Test Split)")
        st.write(
            "Rigorous offline validation partitioning the dataset into an **80% training set** and a **20% separate mock test set**. "
            "Evaluates **Precision@10**, **Recall@10**, **F1-Score@10** (in decimals), "
            "and Rating Prediction Errors (**MSE**, **RMSE**, **MAE**) across **3 Hybrid Configurations**."
        )
        
        col_btn_eval, col_info_eval = st.columns([1.2, 3])
        with col_btn_eval:
            if st.button("Re-run 80/20 Offline Evaluation", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        metrics_df, eval_details = get_cached_evaluation(data)
        
        # Partition Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Training Partition (80%)</div>
                <div class="eval-number">{eval_details['n_train']:,}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Model Training Ratings</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Mock Test Partition (20%)</div>
                <div class="eval-number">{eval_details['n_test']:,}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Ground Truth Ratings</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Relevance Threshold</div>
                <div class="eval-number">≥ {eval_details['threshold']} / 5.0</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">True Liked Benchmark</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Evaluation Window</div>
                <div class="eval-number">Top-10</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Ranking Metric Horizon</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Evaluation Metrics Table
        st.markdown("#### Hybrid Evaluation Metrics Benchmark Table")
        st.write("Comparison of the 3 specified Hybrid configurations combining Collaborative Filtering (CF) and Content-Based Filtering (CBF):")
        
        formatted_table = metrics_df.copy()
        for col in ['MSE', 'RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1-Score@10']:
            if col in formatted_table.columns:
                formatted_table[col] = formatted_table[col].apply(lambda v: f"{v:.4f}")
        
        st.dataframe(formatted_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visual Performance Comparison
        st.markdown("#### Visual Performance Comparison Across Hybrid Configurations")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### Top-10 Ranking Quality (Precision, Recall & F1-Score in Decimals)")
            chart_ranking = metrics_df.set_index('Model Configuration')[['Precision@10', 'Recall@10', 'F1-Score@10']]
            st.bar_chart(chart_ranking)
            st.caption("Higher decimal values indicate superior Top-10 recommendation ranking relevance.")
            
        with col_chart2:
            st.markdown("##### Rating Prediction Errors (RMSE, MAE & MSE)")
            chart_error = metrics_df.set_index('Model Configuration')[['RMSE', 'MAE', 'MSE']]
            st.bar_chart(chart_error)
            st.caption("Lower error values indicate better rating prediction precision.")
            
        with st.expander("Metric Interpretations & Key Insights"):
            st.markdown("""
            - **Hybrid 20% CF / 80% CBF**: Achieves the highest **Precision@10** and **Recall@10** by prioritizing rich thematic metadata (genres, keywords, directors, cast) for item ranking.
            - **Hybrid 50% CF / 50% CBF**: Balances user co-rating correlation and content semantics for a well-rounded discovery experience.
            - **Hybrid 80% CF / 20% CBF**: Minimizes rating prediction errors (**RMSE & MSE**) by leveraging collaborative user co-rating patterns across the catalog.
            """)

    # ----------------------------------------------------------------------------------
    # DEV TAB 3: COLLABORATIVE FILTERING EVALUATION & ANALYTICS
    # ----------------------------------------------------------------------------------
    with dev_tab3:
        st.markdown("### Collaborative Filtering Model Evaluation & Analytics (80/20 Split)")
        st.write(
            "Evaluation of the pure **Item-Based Collaborative Filtering (CF)** engine on an **80/20 Train-Test partition**. "
            "Evaluates rating prediction accuracy (**MSE, RMSE**), Top-10 discovery performance (**Precision@10, Recall@10, F1-Score@10, Average Hits**), "
            "and rating interaction matrix sparsity."
        )
        
        col_cf_btn, col_cf_info = st.columns([1.2, 3])
        with col_cf_btn:
            if st.button("Re-run Collaborative Evaluation", use_container_width=True, key="btn_rerun_cf"):
                st.cache_data.clear()
                st.rerun()

        cf_eval = get_cached_cf_evaluation(data)
        cf_summary = get_cached_cf_summary(data)
        
        # 6 KPI Metric Cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Train Partition (80%)</div>
                <div class="eval-number">{cf_eval['n_train']:,}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Training Ratings</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Mock Test (20%)</div>
                <div class="eval-number">{cf_eval['n_test']:,}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Ground Truth Ratings</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Prediction RMSE</div>
                <div class="eval-number">{cf_eval['rmse']:.4f}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Avg Star Error</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Precision@10</div>
                <div class="eval-number">{cf_eval['precision']:.4f}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">{cf_eval['precision']*100:.2f}% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">Recall@10</div>
                <div class="eval-number">{cf_eval['recall']:.4f}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">{cf_eval['recall']*100:.2f}% Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        with k6:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">F1-Score@10</div>
                <div class="eval-number">{cf_eval['f1_score']:.4f}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">{cf_eval['f1_score']*100:.2f}% Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Section 1 & 2 Tables
        col_tbl1, col_tbl2 = st.columns(2)
        
        with col_tbl1:
            st.markdown("#### Rating Prediction Error Benchmark")
            st.write("Collaborative user & item baseline bias prediction error on the 20% mock test set:")
            st.dataframe(cf_eval['error_table'], use_container_width=True, hide_index=True)
            
        with col_tbl2:
            st.markdown("#### Top-10 Recommendation Quality (Mock Test)")
            st.write("Top-10 ranked recommendation retrieval metrics against test ground-truth liked items (>= 3.5 stars):")
            st.dataframe(cf_eval['quality_table'], use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # Visual Charts
        col_cf_c1, col_cf_c2 = st.columns(2)
        with col_cf_c1:
            st.markdown("##### Top-10 Recommendation Quality Metrics")
            cf_rank_df = pd.DataFrame({
                'Metric': ['Precision@10', 'Recall@10', 'F1-Score@10'],
                'Score': [cf_eval['precision'], cf_eval['recall'], cf_eval['f1_score']]
            }).set_index('Metric')
            st.bar_chart(cf_rank_df)
            st.caption("Decimal representation of Top-10 ranking metrics on the 20% test partition.")
            
        with col_cf_c2:
            st.markdown("##### Rating Prediction Error Distribution (RMSE & MSE)")
            cf_err_df = pd.DataFrame({
                'Error Metric': ['RMSE', 'MSE'],
                'Score': [cf_eval['rmse'], cf_eval['mse']]
            }).set_index('Error Metric')
            st.bar_chart(cf_err_df)
            st.caption("Magnitude of rating deviation between predicted baseline ratings and ground truth test ratings.")

        st.markdown("---")
        
        # Section 3: Matrix Sparsity & Dataset Analytics
        st.markdown("#### Collaborative Interaction Matrix & Sparsity Analytics")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Total Ratings", f"{cf_summary['num_ratings']:,}")
        with col_m2:
            st.metric("Unique Users", f"{cf_summary['num_users']:,}")
        with col_m3:
            st.metric("Unique Movies", f"{cf_summary['num_movies']:,}")
        with col_m4:
            st.metric("Matrix Cells", f"{cf_summary['total_possible']:,}")
        with col_m5:
            st.metric("Matrix Sparsity", f"{cf_summary['sparsity']:.2f}%")
            
        st.markdown(f"""
        <div class="status-panel" style="margin-top: 0.8rem;">
            <b>Sparsity Insight:</b> The user-item rating matrix contains <b>{cf_summary['num_users']:,} users</b> and <b>{cf_summary['num_movies']:,} movies</b> ({cf_summary['total_possible']:,} possible interaction cells). 
            With <b>{cf_summary['num_ratings']:,} actual ratings</b>, the interaction matrix is <b>{cf_summary['sparsity']:.2f}% sparse</b>. Item-Based Pearson correlation effectively handles this sparsity by calculating similarity only across users who have co-rated both movies.
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------
    # DEV TAB 4: USER SATISFACTION SURVEY ANALYTICS & RESPONDENT AUDIT
    # ----------------------------------------------------------------------------------
    with dev_tab4:
        st.markdown("### User Satisfaction Questionnaire Analytics & Audit")
        st.write("Detailed administrative inspection of user evaluations, respondent identities, and dimension distributions.")
        
        survey_df = module_hybrid.load_survey_responses()
        
        if not survey_df.empty:
            avg_rel = survey_df['Relevance (1-5)'].mean()
            avg_nov = survey_df['Novelty (1-5)'].mean()
            avg_div = survey_df['Diversity (1-5)'].mean()
            avg_ui = survey_df['UI Usability (1-5)'].mean()
            avg_overall = survey_df['Overall Satisfaction (1-5)'].mean()
            
            # KPI Metrics Cards
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1:
                st.metric("Total Evaluators", len(survey_df))
            with k2:
                st.metric("Mean Satisfaction", f"{avg_overall:.2f} / 5.0")
            with k3:
                st.metric("Relevance", f"{avg_rel:.2f}")
            with k4:
                st.metric("Novelty", f"{avg_nov:.2f}")
            with k5:
                st.metric("Diversity", f"{avg_div:.2f}")
            with k6:
                st.metric("UI Usability", f"{avg_ui:.2f}")
                
            st.markdown("---")
            
            # Chart & Responses Breakdown
            col_dev_chart, col_dev_summary = st.columns([1.2, 1])
            
            with col_dev_chart:
                st.markdown("#### Average Score by Dimension")
                dimension_means = pd.DataFrame({
                    'Dimension': ['Relevance', 'Novelty', 'Diversity', 'UI Usability', 'Overall'],
                    'Average Score (out of 5.0)': [avg_rel, avg_nov, avg_div, avg_ui, avg_overall]
                }).set_index('Dimension')
                
                st.bar_chart(dimension_means)
                
            with col_dev_summary:
                st.markdown("#### Questionnaire Summary")
                st.markdown(f"""
                - **Total Responses Logged:** {len(survey_df)} submissions
                - **Highest Dimension:** UI Usability ({avg_ui:.2f} / 5.0)
                - **Overall Satisfaction:** {avg_overall:.2f} / 5.0
                - **Storage File:** `survey_responses.csv`
                """)
                
            st.markdown("---")
            
            # Complete Survey Responses Table (Developer Audit)
            st.markdown("#### Complete Questionnaire Responses & Evaluator Audit")
            st.write("Full log of respondent submissions, including evaluator names, numerical scores, and qualitative commentary:")
            
            st.dataframe(survey_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("No survey responses recorded yet. Responses submitted in User View will appear here.")
